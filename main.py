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
from src.ui.templates import APP_HEADER, SIDEBAR, VOICE_HEADER

HISTORY_FILE = "chat_history.json"

# Load CSS from the external file — edit src/ui/styles.css to change the look
_CSS_PATH = Path(__file__).parent / "src" / "ui" / "styles.css"


def _load_css() -> str:
    """Read styles.css at startup so hot-editing the file takes effect on restart."""
    try:
        return _CSS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[WARNING] CSS file not found at {_CSS_PATH}. Falling back to no styles.")
        return ""


# ---------------------------------------------------------------------------
# Extra styles for the metrics strip — kept in the same Inter/stone palette
# as src/ui/styles.css so it matches the rest of the UI.
# ---------------------------------------------------------------------------
_METRICS_CSS = """
.metrics-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 16px;
    font-family: 'Inter', sans-serif;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 0.5px solid #DDDBD2;
}
.met-item {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
}
.met-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #A8A695;
}
.met-val {
    font-size: 13px;
    font-weight: 500;
    color: #1C1C1A;
    font-variant-numeric: tabular-nums;
}
.met-sep {
    color: #DDDBD2;
    padding: 0 2px;
}
.met-divider {
    font-size: 16px;
    color: #DDDBD2;
}
"""


# Helpers

def _aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _metrics_html(stt: float, agent: float, tts: float, tokens: dict | None) -> str:
    """
    Render a compact metrics strip.
    Numbers shown as: STT | Agent | TTS | Total | Tokens in/out/total
    """
    total = round(stt + agent + tts, 2)

    tok_in = tokens.get("input_tokens") if tokens else None
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
    Full pipeline: audio file → STT → agent → TTS stream
    Yields: (audio_chunk, transcription, answer_text, metrics_html)
    """
    if audio is None:
        yield None, "No audio received. Please record again.", "**Please record something.**", ""
        return

    transcription_text = "**Transcription failed.**"
    final_text = "**Processing failed.**"
    metrics = ""

    stt_time = 0.0
    agent_time = 0.0
    tts_time = 0.0
    token_usage = None

    try:
        vi = get_voice_interface(run_agent)

        # ── 1. STT ────────────────────────────────────────────────────────
        query, stt_time = vi.transcribe_audio(audio)

        if not query or not query.strip():
            yield None, "Could not understand audio.", "**Try speaking more clearly.**", ""
            return

        transcription_text = f"{query}"

        # ── 2. Agent ──────────────────────────────────────────────────────
        t0 = time.time()
        result = run_agent(query)
        agent_time = round(time.time() - t0, 2)

        answer = result.get("answer", "No response generated.")
        sources = result.get("sources", [])
        token_usage = result.get("token_usage")

        # Strip the "Source documents used: ..." line for the display text
        answer_body = re.sub(r"\n\nSource documents used:.*$", "", answer, flags=re.DOTALL).strip()

        # Text shown in the UI — keep paragraph/list structure intact for Markdown
        display_text = answer_body
        if sources:
            display_text += f"\n\n*Source documents used: {', '.join(sources)}*"
        final_text = display_text

        # Text spoken by TTS — flatten line breaks so it reads naturally
        speech_text = re.sub(r"\n+", ". ", answer_body).strip()

        if sources:
            print(f"[VOICE] Sources: {', '.join(sources)}")

        # ── 3. TTS (streaming) ────────────────────────────────────────────
        tts_start = time.time()
        for chunk in vi.speak_stream(speech_text):
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
        print(f"[VOICE ERROR] {e}")
        yield None, transcription_text, f"Error: {str(e)}", ""

    finally:
        _aggressive_cleanup()
        clear_page_cache()


# ---------------------------------------------------------------------------
# Sidebar extra: example questions (kept in the sidebar, using the same
# CSS classes as the "Powered By" block in templates.py)
# ---------------------------------------------------------------------------
SIDEBAR_EXAMPLES = """
<div class="sidebar-divider"></div>

<div class="sidebar-stack-label">Example Questions</div>
<div class="sidebar-stack-item">What is the deductible for this plan?</div>
<div class="sidebar-stack-item">What services are covered under preventive care?</div>
<div class="sidebar-stack-item">What is the out-of-pocket maximum?</div>
<div class="sidebar-stack-item">Tell me about eligibility rules.</div>
<div class="sidebar-stack-item">How do I file a claim?</div>
<div class="sidebar-stack-item">What services are excluded?</div>
"""


def main(force_reindex: bool = False):
    print("\nInitializing Healthcare Benefits Assistant...\n")
    preload_all_models()

    print("Running warm-up query...")
    try:
        _ = run_agent("warmup")
    except Exception as e:
        print(f"[WARMUP] Non-fatal: {e}")
    print("System ready.\n")

    voice_interface = get_voice_interface(run_agent)
    custom_css = _load_css() + _METRICS_CSS  # ← styles.css + metrics-strip styles

    with gr.Blocks(
        css=custom_css,
        title="Healthcare Benefits Assistant",
        theme=gr.themes.Base(
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
            primary_hue="stone",
            neutral_hue="stone",
        ),
    ) as demo:

        gr.HTML(APP_HEADER)  # ← from src/ui/templates.py

        with gr.Row(elem_id=["layout-shell"]):

            # ───────────────── Sidebar ─────────────────
            with gr.Column(elem_id="sidebar", scale=0, min_width=240):
                gr.HTML(SIDEBAR)        # ← from src/ui/templates.py
                gr.HTML(SIDEBAR_EXAMPLES)  # ← example questions, kept in sidebar

            # ── Voice Assistant ────────────────────────────────────────────
            with gr.Column(elem_id="main-content", scale=1):
                gr.HTML(VOICE_HEADER)  # ← from src/ui/templates.py

                with gr.Column(elem_id="voice-body"):

                    with gr.Column(elem_classes=["voice-card"]):
                        gr.HTML('<div class="voice-card-label">Record Question</div>')
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"], type="filepath",
                            label=None, show_label=False,
                            waveform_options=gr.WaveformOptions(waveform_color="#1C1C1A"),
                        )
                        voice_submit = gr.Button(
                            "Process voice query", variant="primary",
                            elem_id="voice-submit-btn",
                        )

                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Column(elem_classes=["voice-card"]):
                                gr.HTML('<div class="voice-card-label">Transcription</div>')
                                transcription = gr.Textbox(
                                    label=None, show_label=False,
                                    placeholder="Your spoken words appear here…",
                                    lines=3, interactive=False,
                                    elem_id="transcription-box",
                                )
                        with gr.Column(scale=2):
                            with gr.Column(elem_classes=["voice-card"]):
                                gr.HTML('<div class="voice-card-label">Agent Response</div>')
                                voice_output_text = gr.Markdown(
                                    label=None, elem_id="response-box"
                                )
                                metrics_box = gr.HTML(label=None)

                    with gr.Column(elem_classes=["voice-card"]):
                        gr.HTML('<div class="voice-card-label">Spoken Response</div>')
                        voice_output_audio = gr.Audio(
                            label=None, show_label=False, streaming=True,
                            autoplay=True, interactive=False, show_download_button=True,
                        )

                reset_event = voice_submit.click(
                    fn=lambda: (None, "", "", ""),
                    inputs=None,
                    outputs=[voice_output_audio, transcription, voice_output_text, metrics_box],
                )

                run_event = reset_event.then(
                    streaming_voice_pipeline,
                    [audio_input],
                    [voice_output_audio, transcription, voice_output_text, metrics_box],
                )

                voice_submit.click(
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