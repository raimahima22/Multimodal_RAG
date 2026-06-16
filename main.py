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

_CSS_PATH = Path(__file__).parent / "src" / "ui" / "styles.css"


def _load_css() -> str:
    try:
        return _CSS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[WARNING] CSS file not found at {_CSS_PATH}. Falling back to no styles.")
        return ""


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


def _metrics_html(stt: float, agent: float, tts_latency: float) -> str:
    """
    Render a compact metrics strip.
    TTS shows time-to-first-chunk (real processing latency), not full stream duration.
    Token usage intentionally excluded.
    """
    total = round(stt + agent + tts_latency, 2)

    return f"""
<div class="metrics-strip">
  <span class="met-item"><span class="met-label">STT</span><span class="met-val">{stt}s</span></span>
  <span class="met-sep">·</span>
  <span class="met-item"><span class="met-label">Agent</span><span class="met-val">{agent}s</span></span>
  <span class="met-sep">·</span>
  <span class="met-item"><span class="met-label">TTS</span><span class="met-val">{tts_latency}s</span></span>
  <span class="met-sep">·</span>
  <span class="met-item"><span class="met-label">Total</span><span class="met-val">{total}s</span></span>
</div>
"""

# Streaming voice pipeline

def streaming_voice_pipeline(audio):
    """
    Full pipeline: audio file → STT → agent → TTS stream
    Yields: (audio_chunk, transcription, answer_text, metrics_html)

    Timing notes:
    - stt_time   : measured by vi.transcribe_audio() — wall time for the STT call
    - agent_time : wall time for run_agent() only, excluding TTS
    - tts_latency: time from TTS start until the FIRST chunk arrives
                   (real processing latency; we don't count stream drain time
                   because that is just network/playback, not work we did)
    - Metrics are emitted alongside the first TTS chunk so the card appears
      immediately when audio starts, not after the stream finishes.
    """
    if audio is None:
        yield None, "No audio received. Please record again.", "**Please record something.**", ""
        return

    transcription_text = "**Transcription failed.**"
    final_text = "**Processing failed.**"

    stt_time = 0.0
    agent_time = 0.0
    tts_latency = 0.0

    try:
        vi = get_voice_interface(run_agent)

        # 1. STT 
        query, stt_time = vi.transcribe_audio(audio)

        if not query or not query.strip():
            yield None, "Could not understand audio.", "**Try speaking more clearly.**", ""
            return

        transcription_text = f"{query}"

        # 2. Agent 
        t0 = time.time()
        result = run_agent(query)
        agent_time = round(time.time() - t0, 2)

        answer = result.get("answer", "No response generated.")
        sources = result.get("sources", [])

        answer_body = re.sub(r"\n\nSource documents used:.*$", "", answer, flags=re.DOTALL).strip()

        display_text = answer_body
        if sources:
            display_text += f"\n\n*Source documents used: {', '.join(sources)}*"
        final_text = display_text

        speech_text = re.sub(r"\n+", ". ", answer_body).strip()

        if sources:
            print(f"[VOICE] Sources: {', '.join(sources)}")

        #  3. TTS (streaming) 
        # We only measure until the first chunk arrives — that is the true
        # TTS processing latency.  After that, streaming is just playback.
        tts_start = time.time()
        first_chunk = True
        metrics = ""

        for chunk in vi.speak_stream(speech_text):
            if first_chunk:
                # Time-to-first-chunk = real TTS latency
                tts_latency = round(time.time() - tts_start, 2)
                metrics = _metrics_html(stt_time, agent_time, tts_latency)
                first_chunk = False

            # Emit metrics alongside every chunk 
            yield chunk, transcription_text, final_text, metrics

        #  4. Persist history (uses tts_latency, not full stream time) 
        save_to_history(query, answer, sources, {
            "stt_time": stt_time,
            "agent_time": agent_time,
            "tts_latency": tts_latency,
            "total_time": round(stt_time + agent_time + tts_latency, 2),
        })

        # Final yield in case TTS produced no chunks (edge-case guard)
        if first_chunk:
            tts_latency = round(time.time() - tts_start, 2)
            metrics = _metrics_html(stt_time, agent_time, tts_latency)
            yield None, transcription_text, final_text, metrics

    except Exception as e:
        print(f"[VOICE ERROR] {e}")
        yield None, transcription_text, f"Error: {str(e)}", ""

    finally:
        _aggressive_cleanup()
        clear_page_cache()


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
    custom_css = _load_css() 

    with gr.Blocks(
        css=custom_css,
        title="Healthcare Benefits Assistant",
        theme=gr.themes.Base(
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
            primary_hue="stone",
            neutral_hue="stone",
        ),
    ) as demo:

        gr.HTML(APP_HEADER)

        with gr.Row(elem_id=["layout-shell"]):

            with gr.Column(elem_id="sidebar", scale=0, min_width=240):
                gr.HTML(SIDEBAR)
                gr.HTML(SIDEBAR_EXAMPLES)

            with gr.Column(elem_id="main-content", scale=1):
                gr.HTML(VOICE_HEADER)

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