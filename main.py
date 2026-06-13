import sys
import gc
import os
import json
from datetime import datetime
import torch
import gradio as gr
import time

from src.utils import clear_page_cache
from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator
from src.agent import run_agent
from src.tools import preload_all_models
from src.voice import get_voice_interface

HISTORY_FILE = "chat_history.json"


# def save_to_history(query, source_input, answer):
#     history_data = []
#     if os.path.exists(HISTORY_FILE):
#         try:
#             with open(HISTORY_FILE, "r", encoding="utf-8") as f:
#                 history_data = json.load(f)
#         except Exception:
#             history_data = []

#     history_data.append({
#         "timestamp": datetime.now().isoformat(),
#         "query": query,
#         "source": source_input,
#         "answer": answer,
#     })

#     with open(HISTORY_FILE, "w", encoding="utf-8") as f:
#         json.dump(history_data, f, indent=2, ensure_ascii=False)

def save_to_history(query, answer, sources):
    """Save conversation with sources"""
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
        "sources": sources,                    # <-- Now stores list of sources
        "answer": answer,
    })

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(force_reindex=False):
    print("\nInitializing Multimodal Voice RAG System...\n")
    preload_all_models()  # preload all models to reduce first-query latency
    print("\nSystem is fully ready!\n")
    print("Warming up Agent and retrieval models...")
    try:
        _ = run_agent("warmup query for initialization")
    except Exception as e:
        print(f"Warning during warmup: {e}")
    print("System Ready!\n")

    voice_interface = get_voice_interface(run_agent)

    def user_turn(message, history):
        if not message or not message.strip():
            return history, "", gr.update(interactive=False), gr.update(interactive=False)
        history = history or []
        history.append([message.strip(), "Thinking…"])
        return history, "", gr.update(interactive=False), gr.update(interactive=False)

    def bot_turn(history):
        if not history:
            return history, gr.update(interactive=True), gr.update(interactive=True)
        query = history[-1][0]
        # try:
        #     bot_response = run_agent(query)
        #     save_to_history(query, "Agent (SBC/SPD)", bot_response)
        # except Exception as e:
        #     bot_response = f"**Error generating response:**\n\n{str(e)}"
        # history[-1][1] = bot_response
        # aggressive_cleanup()
        # clear_page_cache()
        # return history, gr.update(interactive=True), gr.update(interactive=True)
        try:
            result = run_agent(query)                    # Now returns dict
        
            answer = result.get("answer", "No response generated.")
            sources = result.get("sources", [])
        
            # Create nice source display
            if sources:
                source_text = f"\n\n**Sources:** {', '.join(sources)}"
                display_answer = answer + source_text
            else:
                display_answer = answer
            
            save_to_history(query, answer, sources)     # Save raw for history
        
        except Exception as e:
            display_answer = f"**Error generating response:**\n\n{str(e)}"
            sources = []

        history[-1][1] = display_answer
        aggressive_cleanup()
        clear_page_cache()
        return history, gr.update(interactive=True), gr.update(interactive=True)

    # def streaming_voice_pipeline(audio):
    #     if audio is None:
    #         return None, "No audio received. Please record again.", "**Please record something.**"
    #     try:
    #         vi = get_voice_interface(run_agent)
    #         query = vi.transcribe_audio(audio)
    #         if not query or not query.strip():
    #             return None, "Could not understand audio.", "**Try speaking more clearly.**"
    #         transcription_text = f"**You said:** {query}"
    #         answer = run_agent(query)
    #         final_text = f"**{answer}**"
    #         for chunk in vi.speak_stream(answer):
    #             yield chunk, transcription_text, final_text
    #     except Exception as e:
    #         print(f"Voice Error: {e}")
    #         return None, f"Error: {str(e)}", "**Processing failed.**"

    # def streaming_voice_pipeline(audio):
    #     if audio is None:
    #         return None, "No audio received. Please record again.", "**Please record something.**"
    
    #     try:
    #         vi = get_voice_interface(run_agent)
    #         query = vi.transcribe_audio(audio)
        
    #         if not query or not query.strip():
    #             return None, "Could not understand audio.", "**Try speaking more clearly.**"
        
    #             transcription_text = f"**You said:** {query}"
        
    #         # Call agent
    #         result = run_agent(query)
    #         answer = result.get("answer", "No response generated.")
    #         sources = result.get("sources", [])
        
    #         # Add sources to the displayed text
    #         if sources:
    #             source_text = f"\n\n**Sources:** {', '.join(sources)}"
    #             final_text = f"**{answer}**{source_text}"
    #         else:
    #             final_text = f"**{answer}**"
        
    #         # Stream the spoken response (without sources)
    #         for chunk in vi.speak_stream(answer):
    #             yield chunk, transcription_text, final_text
            
    #     except Exception as e:
    #         print(f"Voice Error: {e}")
    #         return None, f"Error: {str(e)}", "**Processing failed.**"
    def streaming_voice_pipeline(audio):
        if audio is None:
            return None, "No audio received. Please record again.", "**Please record something.**"
    
        transcription_text = "**Transcription failed.**"
        final_text = "**Processing failed.**"
    
        try:
            vi = get_voice_interface(run_agent)
            query = vi.transcribe_audio(audio)
        
            if not query or not query.strip():
                transcription_text = "Could not understand audio."
                return None, transcription_text, "**Try speaking more clearly.**"
        
            transcription_text = f"**You said:** {query}"
        
            # Call the agent
            result = run_agent(query)
            answer = result.get("answer", "No response generated.")
            sources = result.get("sources", [])
        
            if sources:
                source_text = f"\n\n**Sources:** {', '.join(sources)}"
                final_text = f"**{answer}**{source_text}"
            else:
                final_text = f"**{answer}**"
        
            # Stream the spoken response
            for chunk in vi.speak_stream(answer):
                yield chunk, transcription_text, final_text
            
        except Exception as e:
            print(f"Voice Error: {e}")
            error_msg = f"Error: {str(e)}"
            return None, transcription_text, error_msg   # Now safe to use transcription_text


    # ── Premium CSS ────────────────────────────────────────────────────────────
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    /* ── Base ── */
    .gradio-container {
        font-family: 'Inter', system-ui, sans-serif !important;
        background: #F7F6F3 !important;
        color: #1C1C1A !important;
    }
    footer { display: none !important; }
    * { box-sizing: border-box; }

    /* ── App header ── */
    #app-header {
        padding: 40px 0 28px;
        text-align: center;
        border-bottom: 0.5px solid #DDDBD2;
        margin-bottom: 28px;
    }
    #app-wordmark {
        font-family: 'DM Serif Display', serif;
        font-size: 26px;
        font-weight: 400;
        color: #1C1C1A;
        letter-spacing: -0.3px;
        margin: 0 0 6px;
    }
    #app-tagline {
        font-size: 13px;
        color: #7A7869;
        font-weight: 400;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin: 0;
    }

    /* ── Tab strip ── */
    .tab-nav {
        border-bottom: 0.5px solid #DDDBD2 !important;
        background: transparent !important;
        margin-bottom: 0 !important;
    }
    .tab-nav button {
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #7A7869 !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
        padding: 10px 20px !important;
        border: none !important;
        background: transparent !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.15s, border-color 0.15s !important;
    }
    .tab-nav button.selected,
    .tab-nav button:hover {
        color: #1C1C1A !important;
        border-bottom-color: #1C1C1A !important;
        background: transparent !important;
    }

    /* ── Layout shell ── */
    .layout-shell {
        display: flex;
        gap: 0;
        background: #FFFFFF;
        border: 0.5px solid #DDDBD2;
        border-radius: 12px;
        overflow: hidden;
        min-height: 700px;
    }

    /* ── Sidebar ── */
    #sidebar {
        width: 240px;
        min-width: 240px;
        background: #F7F6F3;
        border-right: 0.5px solid #DDDBD2;
        padding: 28px 20px;
        display: flex;
        flex-direction: column;
        gap: 0;
    }
    .sidebar-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #A8A695;
        margin: 0 0 10px;
    }
    .sidebar-doc {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 10px;
        border-radius: 6px;
        background: #EEECD8;
        border: 0.5px solid #DDDBD2;
        margin-bottom: 6px;
    }
    .sidebar-doc-icon {
        width: 28px;
        height: 28px;
        border-radius: 4px;
        background: #1C1C1A;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .sidebar-doc-icon span {
        color: #F7F6F3;
        font-size: 9px;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .sidebar-doc-name {
        font-size: 12px;
        font-weight: 500;
        color: #1C1C1A;
        line-height: 1.3;
    }
    .sidebar-doc-sub {
        font-size: 11px;
        color: #7A7869;
    }
    .sidebar-divider {
        height: 0.5px;
        background: #DDDBD2;
        margin: 20px 0;
    }
    .sidebar-stack-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #A8A695;
        margin: 0 0 8px;
    }
    .sidebar-stack-item {
        font-size: 12px;
        color: #7A7869;
        padding: 3px 0;
    }
    .sidebar-footer {
        margin-top: auto;
        padding-top: 20px;
    }
    .status-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #3B6D11;
        margin-right: 6px;
        vertical-align: middle;
    }
    .status-text {
        font-size: 11px;
        color: #7A7869;
    }

    /* ── Clear button ── */
    #clear-btn {
        width: 100% !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #7A7869 !important;
        background: transparent !important;
        border: 0.5px solid #DDDBD2 !important;
        border-radius: 6px !important;
        padding: 7px 12px !important;
        text-align: left !important;
        cursor: pointer !important;
        transition: background 0.12s, color 0.12s !important;
    }
    #clear-btn:hover {
        background: #EEECD8 !important;
        color: #1C1C1A !important;
    }

    /* ── Chat area ── */
    #chat-area {
        flex: 1;
        display: flex;
        flex-direction: column;
        background: #FFFFFF;
    }

    /* ── Chatbot messages ── */
    #chatbot {
        background: transparent !important;
        border: none !important;
        flex: 1;
        padding: 24px 28px 0 !important;
    }
    #chatbot .message-row {
        margin-bottom: 16px !important;
    }
    #chatbot .user .message,
    #chatbot .bot .message {
        font-size: 14px !important;
        line-height: 1.7 !important;
        border-radius: 8px !important;
    }
    #chatbot .user .message {
        background: #1C1C1A !important;
        color: #F7F6F3 !important;
        max-width: 72% !important;
    }
    #chatbot .bot .message {
        background: #F7F6F3 !important;
        color: #1C1C1A !important;
        border: 0.5px solid #DDDBD2 !important;
    }

    /* ── Input row ── */
    #input-row {
        padding: 20px 28px 24px !important;
        border-top: 0.5px solid #DDDBD2;
        background: #FFFFFF;
    }
    #msg-input textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        color: #1C1C1A !important;
        background: #F7F6F3 !important;
        border: 0.5px solid #DDDBD2 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        resize: none !important;
        transition: border-color 0.15s !important;
    }
    #msg-input textarea:focus {
        border-color: #1C1C1A !important;
        outline: none !important;
        box-shadow: none !important;
    }
    #msg-input textarea::placeholder {
        color: #A8A695 !important;
    }
    #send-btn {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #F7F6F3 !important;
        background: #1C1C1A !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        height: 44px !important;
        align-self: flex-end !important;
        cursor: pointer !important;
        transition: opacity 0.15s !important;
        letter-spacing: 0.01em !important;
    }
    #send-btn:hover { opacity: 0.82 !important; }
    #send-btn:active { opacity: 0.65 !important; }
    #send-btn:disabled { opacity: 0.35 !important; cursor: default !important; }

    /* ── Example pills ── */
    .examples-holder {
        padding: 0 28px 20px !important;
    }
    .example-btn {
        font-size: 12px !important;
        color: #5F5E5A !important;
        background: #F7F6F3 !important;
        border: 0.5px solid #DDDBD2 !important;
        border-radius: 20px !important;
        padding: 5px 14px !important;
        cursor: pointer !important;
        transition: background 0.12s, border-color 0.12s !important;
        white-space: nowrap !important;
    }
    .example-btn:hover {
        background: #EEECD8 !important;
        border-color: #C5C3B8 !important;
    }

    /* ── Voice tab ── */
    #voice-header {
        padding: 32px 40px 0;
    }
    #voice-title {
        font-family: 'DM Serif Display', serif;
        font-size: 22px;
        font-weight: 400;
        color: #1C1C1A;
        margin: 0 0 4px;
    }
    #voice-sub {
        font-size: 13px;
        color: #7A7869;
        margin: 0 0 28px;
    }
    #voice-body {
        padding: 0 40px 40px;
        display: flex;
        flex-direction: column;
        gap: 20px;
    }
    .voice-card {
        background: #FFFFFF;
        border: 0.5px solid #DDDBD2;
        border-radius: 10px;
        padding: 20px 24px;
    }
    .voice-card-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #A8A695;
        margin: 0 0 12px;
    }
    #voice-submit-btn {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #F7F6F3 !important;
        background: #1C1C1A !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 11px 28px !important;
        cursor: pointer !important;
        transition: opacity 0.15s !important;
        letter-spacing: 0.01em !important;
    }
    #voice-submit-btn:hover { opacity: 0.82 !important; }

    /* ── Transcription + response fields ── */
    #transcription-box textarea,
    #response-box {
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        color: #1C1C1A !important;
        background: #F7F6F3 !important;
        border: 0.5px solid #DDDBD2 !important;
        border-radius: 8px !important;
        padding: 14px 16px !important;
        line-height: 1.7 !important;
    }
    """

    # ── Sidebar HTML ──────────────────────────────────────────────────────────
    sidebar_html = """
    <div class="sidebar-label">Knowledge Base</div>

    <div class="sidebar-doc">
        <div class="sidebar-doc-icon"><span>SBC</span></div>
        <div>
            <div class="sidebar-doc-name">Summary of Benefits</div>
            <div class="sidebar-doc-sub">Coverage &amp; Costs</div>
        </div>
    </div>

    <div class="sidebar-doc">
        <div class="sidebar-doc-icon"><span>SPD</span></div>
        <div>
            <div class="sidebar-doc-name">Plan Description</div>
            <div class="sidebar-doc-sub">Eligibility &amp; Rules</div>
        </div>
    </div>

    <div class="sidebar-divider"></div>

    <div class="sidebar-stack-label">Powered By</div>
    <div class="sidebar-stack-item">LangGraph Agent</div>
    <div class="sidebar-stack-item">ColQwen2.5</div>
    <div class="sidebar-stack-item">Qdrant Vector DB</div>

    <div class="sidebar-footer">
        <span class="status-dot"></span>
        <span class="status-text">System ready</span>
    </div>
    """

    with gr.Blocks(
        css=custom_css,
        title="Healthcare Benefits Assistant",
        theme=gr.themes.Base(
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
            primary_hue="stone",
            neutral_hue="stone",
        ),
    ) as demo:

        # ── Global header ─────────────────────────────────────────────────────
        gr.HTML("""
            <div id="app-header">
                <p id="app-wordmark">Benefits Assistant</p>
                <p id="app-tagline">Intelligent Q&amp;A over SBC &amp; SPD Documents</p>
            </div>
        """)

        with gr.Tabs(elem_classes=["tab-nav"]):

            # ═══════════════════ TEXT CHAT TAB ═══════════════════════════════
            with gr.Tab("Text Chat"):
                with gr.Row(elem_classes=["layout-shell"]):

                    # Sidebar
                    with gr.Column(elem_id="sidebar", scale=0, min_width=240):
                        gr.HTML(sidebar_html)
                        clear_btn = gr.Button(
                            "Clear conversation",
                            elem_id="clear-btn",
                            size="sm",
                        )

                    # Main chat column
                    with gr.Column(elem_id="chat-area", scale=1):
                        chatbot = gr.Chatbot(
                            elem_id="chatbot",
                            type="messages",
                            height=540,
                            bubble_full_width=False,
                            render_markdown=True,
                            show_label=False,
                        )

                        with gr.Row(elem_id="input-row"):
                            msg = gr.Textbox(
                                placeholder="Ask about your benefits plan…",
                                scale=8,
                                lines=1,
                                max_lines=4,
                                autofocus=True,
                                show_label=False,
                                container=False,
                                elem_id="msg-input",
                            )
                            submit_btn = gr.Button(
                                "Send",
                                variant="primary",
                                scale=0,
                                min_width=90,
                                elem_id="send-btn",
                            )

                        with gr.Row(elem_classes=["examples-holder"]):
                            gr.Examples(
                                examples=[
                                    ["What is the deductible for this plan?"],
                                    ["What services are covered under preventive care?"],
                                    ["What is the out-of-pocket maximum?"],
                                    ["Tell me about eligibility rules."],
                                ],
                                inputs=[msg],
                                label=None,
                            )

                # Text Chat Events
                msg.submit(
                    user_turn,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg, msg, submit_btn],
                ).then(
                    bot_turn,
                    inputs=[chatbot],
                    outputs=[chatbot, msg, submit_btn],
                )

                submit_btn.click(
                    user_turn,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg, msg, submit_btn],
                ).then(
                    bot_turn,
                    inputs=[chatbot],
                    outputs=[chatbot, msg, submit_btn],
                )

                clear_btn.click(
                    lambda: ([], gr.update(interactive=True), gr.update(interactive=True)),
                    None,
                    [chatbot, msg, submit_btn],
                )

            # ═══════════════════ VOICE TAB ═══════════════════════════════════
            with gr.Tab("Voice Assistant"):
                gr.HTML("""
                    <div id="voice-header">
                        <p id="voice-title">Voice-Enabled Assistant</p>
                        <p id="voice-sub">
                            Record your question — it will be transcribed, answered,
                            and read back to you.
                        </p>
                    </div>
                """)

                with gr.Column(elem_id="voice-body"):

                    with gr.Column(elem_classes=["voice-card"]):
                        gr.HTML('<div class="voice-card-label">Record Question</div>')
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label=None,
                            show_label=False,
                            waveform_options=gr.WaveformOptions(
                                waveform_color="#1C1C1A"
                            ),
                        )
                        voice_submit = gr.Button(
                            "Process voice query",
                            variant="primary",
                            elem_id="voice-submit-btn",
                        )

                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Column(elem_classes=["voice-card"]):
                                gr.HTML('<div class="voice-card-label">Transcription</div>')
                                transcription = gr.Textbox(
                                    label=None,
                                    show_label=False,
                                    placeholder="Your spoken words appear here…",
                                    lines=3,
                                    interactive=False,
                                    elem_id="transcription-box",
                                )

                        with gr.Column(scale=2):
                            with gr.Column(elem_classes=["voice-card"]):
                                gr.HTML('<div class="voice-card-label">Agent Response</div>')
                                voice_output_text = gr.Markdown(
                                    label=None,
                                    elem_id="response-box",
                                )

                    with gr.Column(elem_classes=["voice-card"]):
                        gr.HTML('<div class="voice-card-label">Spoken Response</div>')
                        voice_output_audio = gr.Audio(
                            label=None,
                            show_label=False,
                            streaming=True,
                            autoplay=True,
                            interactive=False,
                            show_download_button=True,
                        )

                voice_submit.click(
                    fn=streaming_voice_pipeline,
                    inputs=[audio_input],
                    outputs=[voice_output_audio, transcription, voice_output_text],
                )

        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
        )


if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
    main(force_reindex)