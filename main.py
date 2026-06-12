

import sys
import gc
import os
import json
from datetime import datetime
import torch
import gradio as gr

from src.utils import clear_page_cache
from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator
from src.agent import run_agent
from src.agent import get_voice_interface

HISTORY_FILE = "chat_history.json"


def save_to_history(query, source_input, answer):
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
        "source": source_input,
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
    
    # indexer = MultimodalIndexer(force_recreate=force_reindex)
    # retriever = MultimodalRetriever(indexer)
    # generator = MultimodalGenerator()

    # print("Warming up retrieval model...")
    # _ = retriever._extract_text_embedding("warmup query")
    # print("System Ready!\n")

    # if force_reindex or indexer.is_collection_empty():
    #     print("Indexing documents...\n")
    #     indexer.index_all_data("data")
    #     print("Indexing completed!\n")
    # else:
    #     print("Existing index found. Skipping indexing.\n")

    # source_map = {
    #     "Both Documents": None,
    #     "SBC": "data/sbc.pdf",
    #     "SPD": "data/spd.pdf",
    # }
    print("Warming up Agent and retrieval models...")
    # Warmup both tools via the agent
    try:
        _ = run_agent("warmup query for initialization")
    except Exception as e:
        print(f"Warning during warmup: {e}")
    print("System Ready!\n")

    # ── User Turn: Show message immediately + thinking indicator ─────────────
    def user_turn(message, history):
        if not message or not message.strip():
            return history, "", gr.update(interactive=False), gr.update(interactive=False)
        
        history = history or []
        history = history + [
            {"role": "user", "content": message.strip()},
            {"role": "assistant", "content": "Thinking..."}
        ]
        
        return history, "", gr.update(interactive=False), gr.update(interactive=False)

    # ── Bot Turn: Generate real answer using LangGraph Agent ─────────────────
    def bot_turn(history):
        if not history:
            return history, gr.update(interactive=True), gr.update(interactive=True)
        
        query = history[-2]["content"]
        
        try:
            bot_response = run_agent(query)
            save_to_history(query, "Agent (SBC/SPD)", bot_response)
        except Exception as e:
            bot_response = f" **Error while generating response:**\n\n{str(e)}"
        
        # Replace thinking message with real answer
        history[-1] = {"role": "assistant", "content": bot_response}
        
        aggressive_cleanup()
        clear_page_cache()
        return history, gr.update(interactive=True), gr.update(interactive=True)

        # ── Voice Functions ─────────────────────────────────────────────────
    def voice_pipeline(audio):
        """Full voice → agent → voice response"""
        if audio is None:
            return None, "No audio received. Please record again."
        
        try:
            audio_path, result_text = voice_iface.voice_pipeline(audio)
            return audio_path, result_text
        except Exception as e:
            return None, f"Error in voice pipeline: {str(e)}"


    # ── Custom CSS ─────────────
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&family=Source+Code+Pro:wght@400;500&display=swap');
    
    .gradio-container {
        font-family: 'Lato', sans-serif !important;
        background: #f5f0e8 !important;
    }
    footer { display: none !important; }

    #main-title {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        color: #2c2a26;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    #main-subtitle {
        text-align: center;
        font-size: 15px;
        color: #8a8070;
        margin-bottom: 20px;
    }

    /* Sidebar & Chat Layout */
    .app-shell {
        display: flex;
        min-height: 85vh;
        overflow: hidden;
        background: #f5f0e8;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    #sidebar { 
        width: 260px; 
        min-width: 260px; 
        background: #ede8de; 
        border-right: 1px solid #d8d0c0; 
        padding: 24px 20px;
        display: flex;
        flex-direction: column;
        gap: 20px;
    }
    #chat-area {
        flex: 1;
        display: flex;
        flex-direction: column;
        background: #faf8f4;
    }
    #chatbot {
        background: transparent !important;
        border: none !important;
        flex: 1;
    }
    """

    with gr.Blocks(
        css=custom_css,
        title="Multimodal RAG Assistant",
        theme=gr.themes.Base(primary_hue="stone", neutral_hue="stone")
    ) as demo:
        
        gr.HTML("""
            <div id="main-title">Multimodal RAG Assistant</div>
            <div id="main-subtitle">Intelligent Document Q&A over SBC & SPD using LangGraph Agent</div>
        """)

        with gr.Row(elem_classes=["app-shell"]):
            # Sidebar
            with gr.Column(elem_id="sidebar", scale=0, min_width=260):
                gr.HTML('<strong>Knowledge Base</strong>')
                gr.HTML('<p><small>SBC + SPD Documents</small></p>')
                
                gr.HTML('<strong>Actions</strong>')
                clear_btn = gr.Button("🗑 Clear Chat", elem_id="clear-btn", size="sm")

                gr.HTML("""
                    <div style="margin-top: auto; font-size: 0.85em; color: #8a8070;">
                        Ready • LangGraph Agent • Qdrant + ColQwen2.5
                    </div>
                """)

            # Main Chat Area
            with gr.Column(elem_id="chat-area", scale=1):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    type="messages",
                    height=620,
                    bubble_full_width=False,
                    show_label=False,
                    render_markdown=True,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question about your benefits plan...",
                        scale=8,
                        container=False,
                        lines=1,
                        max_lines=4,
                        autofocus=True,
                        elem_id="msg-box",
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1, min_width=100)

                gr.Examples(
                    examples=[
                        ["What is the deductible for this plan?"],
                        ["What services are covered under preventive care?"],
                        ["What is the out-of-pocket maximum?"],
                        ["Tell me about eligibility and enrollment rules."],
                    ],
                    inputs=[msg],
                    label="Example Queries",
                    cache_examples=False
                )
        

        # Event Handling
        msg.submit(
            user_turn,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, msg, submit_btn],
            queue=False
        ).then(
            bot_turn,
            inputs=[chatbot],
            outputs=[chatbot, msg, submit_btn]
        )

        submit_btn.click(
            user_turn,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, msg, submit_btn],
            queue=False
        ).then(
            bot_turn,
            inputs=[chatbot],
            outputs=[chatbot, msg, submit_btn]
        )

        clear_btn.click(
            lambda: ([], gr.update(interactive=True), gr.update(interactive=True)),
            None, 
            [chatbot, msg, submit_btn]
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