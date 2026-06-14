# main.py
import sys
import gc
import os
import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import gradio as gr

from src.utils import clear_page_cache
from src.agent import run_agent
from src.tools import preload_all_models
from src.voice import get_voice_interface

HISTORY_FILE = "chat_history.json"

_CSS_PATH = Path(__file__).parent / "src" / "ui" / "styles.css"


def _load_css() -> str:
    try:
        return _CSS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[WARNING] CSS file not found at {_CSS_PATH}.")
        return ""


def _aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def save_to_history(query: str, answer: str, sources: list, latency: dict):
    history_data = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history_data = json.load(f)
        except Exception:
            history_data = []

    history_data.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "sources": sources,
        "answer": answer,
        "latency": latency,
    })

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Metrics HTML builder
# ---------------------------------------------------------------------------

def _metrics_html(stt: float, agent: float, tts: float, tokens: dict | None) -> str:
    """
    Render a compact metrics strip.
    Numbers shown as:  STT  |  Agent  |  TTS  |  Tokens
    """
    total = round(stt + agent + tts, 2)

    tok_in  = tokens.get("input_tokens")  if tokens else None
    tok_out = tokens.get("output_tokens") if tokens else None
    tok_total = tokens.get("total_tokens") if tokens else None

    def _fmt_tok(v):
        return f"{v:,}" if v is not None else "—"

    tok_html = (
        f'<span class="met-item"><span class="met-label">Tokens in</span>'
        f'<span class="met-val">{_fmt_tok(tok_in)}</span></span>'
        f'<span class="met-sep">·</span>'
        f'<span class="met-item"><span class="met-label">Tokens out</span>'
        f'<span class="met-val">{_fmt_tok(tok_out)}</span></span>'
        f'<span class="met-sep">·</span>'
        f'<span class="met-item"><span class="met-label">Total tokens</span>'
        f'<span class="met-val">{_fmt_tok(tok_total)}</span></span>'
    )

    return f"""
<div class="metrics-strip">
  <span class="met-item"><span class="met-label">STT</span><span class="met-val">{stt}s</span></span>
  <span class="met-sep">·</span>
  <span class="met-item"><span class="met-label">Agent</span><span class="met-val">{agent}s</span></span>
  <span class="met-sep">·</span>
  <span class="met-item"><span class="met-label">TTS</span><span class="met-val">{tts}s</span></span>
  <span class="met-sep">·</span>
  <span class="met-item"><span class="met-label">Total</span><span class="met-val">{total}s</span></span>
  <span class="met-sep met-divider">|</span>
  {tok_html}
</div>
"""


# ---------------------------------------------------------------------------
# Streaming voice pipeline
# ---------------------------------------------------------------------------

def streaming_voice_pipeline(audio):
    """
    Full pipeline:  audio file → STT → agent → TTS stream
    Yields:         (audio_chunk, transcription, answer_text, metrics_html)
    """
    if audio is None:
        yield None, "No audio received. Please record again.", "", ""
        return

    transcription_text = ""
    final_text = ""
    metrics = ""

    stt_time   = 0.0
    agent_time = 0.0
    tts_time   = 0.0
    token_usage = None

    try:
        vi = get_voice_interface(run_agent)

        # ── 1. STT ────────────────────────────────────────────────────────
        query, stt_time = vi.transcribe_audio(audio)

        if not query or not query.strip():
            yield None, "Could not understand the audio. Please try again.", "", ""
            return

        transcription_text = query

        # ── 2. Agent ──────────────────────────────────────────────────────
        t0 = time.time()
        result = run_agent(query)
        agent_time = round(time.time() - t0, 2)

        answer     = result.get("answer", "No response generated.")
        sources    = result.get("sources", [])
        token_usage = result.get("token_usage")

        # Strip the "Source documents used: ..." line for the display text
        # (we show sources separately in the metrics strip)
        answer_body = re.sub(
            r"\n\nSource documents used:.*$", "", answer, flags=re.DOTALL
        ).strip()

        display_text = answer_body
        if sources:
            display_text += f"\n\n*Sources: {', '.join(sources)}*"
        final_text = display_text

        # Text for TTS — flatten newlines so it reads naturally
        speech_text = re.sub(r"\n+", ". ", answer_body).strip()

        # ── 3. TTS (streaming) ────────────────────────────────────────────
        tts_start = time.time()
        first_chunk = True
        for chunk in vi.speak_stream(speech_text):
            if first_chunk:
                # TTS started — update metrics with partial info
                first_chunk = False
            yield chunk, transcription_text, final_text, ""

        tts_time = round(time.time() - tts_start, 2)

        # ── 4. Final metrics update ───────────────────────────────────────
        metrics = _metrics_html(stt_time, agent_time, tts_time, token_usage)
        save_to_history(query, answer, sources, {
            "stt_time": stt_time,
            "agent_time": agent_time,
            "tts_time": tts_time,
            "total_time": round(stt_time + agent_time + tts_time, 2),
        })

        # Emit one last update with the metrics filled in
        yield None, transcription_text, final_text, metrics

    except Exception as e:
        print(f"[PIPELINE ERROR] {e}")
        yield None, transcription_text, f"An error occurred: {str(e)}", ""

    finally:
        _aggressive_cleanup()
        clear_page_cache()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_HEADER_HTML = """
<div style="padding: 1.5rem 0 0.5rem; border-bottom: 0.5px solid var(--border-color-primary, #e0e0e0);">
  <div style="font-size: 1.25rem; font-weight: 500; color: var(--body-text-color, #1a1a1a);">
    Healthcare Benefits Assistant
  </div>
  <div style="font-size: 0.85rem; color: var(--body-text-color-subdued, #666); margin-top: 4px;">
    Ask questions about your SBC and SPD documents using your voice
  </div>
</div>
"""

_SIDEBAR_HTML = """
<div style="padding: 1rem 0;">
  <div style="font-size: 0.75rem; font-weight: 500; letter-spacing: 0.06em;
              text-transform: uppercase; color: var(--body-text-color-subdued, #888);
              margin-bottom: 0.75rem;">
    Document sources
  </div>
  <div style="font-size: 0.85rem; color: var(--body-text-color, #333); line-height: 2;">
    <div style="display:flex; align-items:center; gap:8px; padding: 4px 0;">
      <span style="width:8px; height:8px; border-radius:50%; background:#1D9E75; flex-shrink:0;"></span>
      SBC — Summary of Benefits &amp; Coverage
    </div>
    <div style="display:flex; align-items:center; gap:8px; padding: 4px 0;">
      <span style="width:8px; height:8px; border-radius:50%; background:#378ADD; flex-shrink:0;"></span>
      SPD — Summary Plan Description
    </div>
  </div>

  <div style="font-size: 0.75rem; font-weight: 500; letter-spacing: 0.06em;
              text-transform: uppercase; color: var(--body-text-color-subdued, #888);
              margin-top: 1.5rem; margin-bottom: 0.75rem;">
    Example questions
  </div>
  <div style="font-size: 0.82rem; color: var(--body-text-color, #333); line-height: 2;">
    <div>What is the deductible for this plan?</div>
    <div>What services are covered under preventive care?</div>
    <div>What is the out-of-pocket maximum?</div>
    <div>Tell me about eligibility rules.</div>
    <div>How do I file a claim?</div>
    <div>What services are excluded?</div>
  </div>
</div>
"""

# Inline CSS — keeps the project self-contained even if styles.css is missing
_INLINE_CSS = """
.metrics-strip {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 4px;
  font-size: 12px;
  padding: 8px 12px;
  border-radius: 8px;
  background: var(--background-fill-secondary, #f8f8f8);
  border: 0.5px solid var(--border-color-primary, #e0e0e0);
  margin-top: 8px;
  color: var(--body-text-color-subdued, #666);
}
.met-item { display: flex; flex-direction: column; align-items: center; gap: 1px; }
.met-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; }
.met-val { font-size: 13px; font-weight: 500; color: var(--body-text-color, #333); font-variant-numeric: tabular-nums; }
.met-sep { opacity: 0.3; padding: 0 2px; }
.met-divider { font-size: 16px; }
.voice-section-label {
  font-size: 11px; font-weight: 500; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--body-text-color-subdued, #888);
  margin-bottom: 6px;
}
"""


def main(force_reindex: bool = False):
    print("\nInitialising Healthcare Benefits Assistant...\n")
    preload_all_models()

    print("Running warm-up query...")
    try:
        _ = run_agent("warmup")
    except Exception as e:
        print(f"[WARMUP] Non-fatal: {e}")
    print("System ready.\n")

    custom_css = _load_css() + _INLINE_CSS

    with gr.Blocks(
        css=custom_css,
        title="Healthcare Benefits Assistant",
        theme=gr.themes.Base(
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
            primary_hue="stone",
            neutral_hue="stone",
        ),
    ) as demo:

        gr.HTML(_HEADER_HTML)

        with gr.Row():

            # ── Sidebar ───────────────────────────────────────────────────
            with gr.Column(scale=0, min_width=220):
                gr.HTML(_SIDEBAR_HTML)

            # ── Main panel ────────────────────────────────────────────────
            with gr.Column(scale=1):

                # Record
                gr.HTML('<div class="voice-section-label">Record your question</div>')
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    show_label=False,
                    waveform_options=gr.WaveformOptions(waveform_color="#1C1C1A"),
                )
                submit_btn = gr.Button("Ask", variant="primary")

                with gr.Row():
                    # Transcription
                    with gr.Column(scale=1):
                        gr.HTML('<div class="voice-section-label">Transcription</div>')
                        transcription_box = gr.Textbox(
                            show_label=False,
                            placeholder="Your spoken words appear here…",
                            lines=3,
                            interactive=False,
                        )

                    # Answer
                    with gr.Column(scale=2):
                        gr.HTML('<div class="voice-section-label">Answer</div>')
                        answer_box = gr.Markdown(show_label=False)

                # Metrics strip
                metrics_box = gr.HTML(label=None)

                # Spoken response
                gr.HTML('<div class="voice-section-label" style="margin-top:8px;">Spoken response</div>')
                audio_output = gr.Audio(
                    show_label=False,
                    streaming=True,
                    autoplay=True,
                    interactive=False,
                    show_download_button=True,
                )

        # ── Event wiring ──────────────────────────────────────────────────
        reset_event = submit_btn.click(
            fn=lambda: (None, "", "", ""),
            inputs=None,
            outputs=[audio_output, transcription_box, answer_box, metrics_box],
        )

        run_event = reset_event.then(
            fn=streaming_voice_pipeline,
            inputs=[audio_input],
            outputs=[audio_output, transcription_box, answer_box, metrics_box],
        )

        # Allow cancellation by clicking Ask again
        submit_btn.click(
            fn=lambda: None,
            inputs=None,
            outputs=None,
            cancels=[run_event],
        )

        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            max_threads=2,
        )


if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
    main(force_reindex)