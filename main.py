# main.py
import sys
import gc
import os
import json
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


# Helpers 

def _aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_to_history(query: str, answer: str, sources: list):
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
    })

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)


def streaming_voice_pipeline(audio):
    if audio is None:
        yield None, "No audio received. Please record again.", "**Please record something.**"
        return

    transcription_text = "**Transcription failed.**"
    final_text = "**Processing failed.**"

    try:
        vi = get_voice_interface(run_agent)
        query = vi.transcribe_audio(audio)

        if not query or not query.strip():
            yield None, "Could not understand audio.", "**Try speaking more clearly.**"
            return

        transcription_text = f"**You said:** {query}"

        result = run_agent(query)
        answer = result.get("answer", "No response generated.")
        sources = result.get("sources", [])

        clean_answer = answer.split("**Sources:")[0].strip()
        clean_answer = clean_answer.replace("**", "").replace("\n\n", ". ").replace("\n", " ")
        final_text = f"**{clean_answer}**"

        if sources:
            print(f"[VOICE] Sources: {', '.join(sources)}")

        for chunk in vi.speak_stream(clean_answer):
            yield chunk, transcription_text, final_text

    except Exception as e:
        print(f"[VOICE ERROR] {e}")
        yield None, transcription_text, f"Error: {str(e)}"

    finally:
        _aggressive_cleanup()
        clear_page_cache()



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
    custom_css = _load_css()  # ← loaded from src/ui/styles.css

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
                gr.HTML(SIDEBAR)

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

                    with gr.Column(elem_classes=["voice-card"]):
                        gr.HTML('<div class="voice-card-label">Spoken Response</div>')
                        voice_output_audio = gr.Audio(
                            label=None, show_label=False, streaming=True,
                            autoplay=True, interactive=False, show_download_button=True,
                        )
                    with gr.Column(elem_classes=["voice-card"]):
                        gr.HTML(""" 
                        <div class="voice-card-label"> 
                            Example Questions 
                        </div> 
                        <ul style="line-height:2;"> 
                            <li>What is the deductible for this plan?</li> 
                            <li>What services are covered under preventive care?</li> 
                            <li>What is the out-of-pocket maximum?</li> 
                            <li>Tell me about eligibility rules.</li> 
                        </ul> 
                        """)

                voice_submit.click(
                    streaming_voice_pipeline,
                    [audio_input],
                    [voice_output_audio, transcription, voice_output_text],
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