# import sys
# import gc
# import os
# import json
# from datetime import datetime

# import torch
# import gradio as gr

# from src.utils import clear_page_cache
# from src.indexer import MultimodalIndexer
# from src.retriever import MultimodalRetriever
# from src.generator import MultimodalGenerator


# HISTORY_FILE = "chat_history.json"


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


# def aggressive_cleanup():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# def main(force_reindex=False):

#     print("\nInitializing Multimodal RAG System...\n")

#     indexer = MultimodalIndexer(force_recreate=force_reindex)
#     retriever = MultimodalRetriever(indexer)
#     generator = MultimodalGenerator()

#     print("Warming up retrieval model...")
#     _ = retriever._extract_text_embedding("warmup query")
#     print("System Ready!\n")

#     print("Checking document index...\n")

#     if force_reindex or indexer.is_collection_empty():
#         print("Indexing documents...\n")
#         indexer.index_all_data("data")
#         print("Indexing completed!\n")
#     else:
#         print("Existing index found. Skipping indexing.\n")

#     source_options = {
#         "Both Documents": None,
#         "SBC": "data/sbc.pdf",
#         "SPD": "data/spd.pdf",
#     }

#     def answer_query(message, history, source_choice):

#         if history is None:
#             history = []

#         if not message or not message.strip():
#             return history, ""

#         query = message.strip()

#         source_filter = None

#         if str(source_choice).lower() in ["sbc", "sbc.pdf"]:
#             source_filter = "data/sbc.pdf"
#         elif str(source_choice).lower() in ["spd", "spd.pdf"]:
#             source_filter = "data/spd.pdf"

#         try:

#             hits = retriever.search(
#                 query,
#                 top_k=3,
#                 source_filter=source_filter
#             )

#             if not hits:
#                 bot_response = "No relevant information found in selected documents."

#             else:

#                 context_hits = sorted(hits, key=lambda x: x.score, reverse=True)
#                 best_hit = context_hits[0]

#                 source = best_hit.payload.get("source", "Unknown")
#                 page = best_hit.payload.get("page_number", "N/A")

#                 answer = generator.generate_answer(query, context_hits)

#                 bot_response = f"""
# ### Retrieved Context
# Source: {os.path.basename(source)}
# Page: {page}
# Scope: {source_choice}

# ---

# ### Answer
# {answer}
# """

#                 save_to_history(query, source_choice, answer)

#             history.append({"role": "user", "content": query})
#             history.append({"role": "assistant", "content": bot_response})

#             return history, ""

#         except Exception as e:

#             error_message = f"Error while generating response:\n{str(e)}"

#             history.append({"role": "user", "content": query})
#             history.append({"role": "assistant", "content": error_message})

#             return history, ""

#         finally:
#             clear_page_cache()
#             aggressive_cleanup()

#     # ================= UI (SAFE VERSION) =================
#     with gr.Blocks(title="Multimodal RAG Assistant") as demo:

#         gr.Markdown("# Multimodal RAG Assistant")
#         gr.Markdown("Document QA over SBC & SPD")

#         with gr.Row():

#             with gr.Column(scale=1):

#                 gr.Markdown("### Document Filter")

#                 source_dropdown = gr.Dropdown(
#                     choices=list(source_options.keys()),
#                     value="Both Documents"
#                 )

#                 clear_btn = gr.Button("Clear Chat")

#                 gr.Markdown("### Example Questions")

#             with gr.Column(scale=4):

#                 chatbot = gr.Chatbot(
#                     type="messages",
#                     height=700
#                 )

#                 msg = gr.Textbox(
#                     placeholder="Ask a question...",
#                     lines=2
#                 )

#                 submit_btn = gr.Button("Send")

#                 gr.Examples(
#                     examples=[
#                         ["What is deductible?", "SPD"],
#                         ["What is covered?", "SBC"],
#                         ["Compare both plans", "Both Documents"],
#                     ],
#                     inputs=[msg, source_dropdown]
#                 )

#         # events
#         msg.submit(answer_query, [msg, chatbot, source_dropdown], [chatbot, msg])
#         submit_btn.click(answer_query, [msg, chatbot, source_dropdown], [chatbot, msg])

#         clear_btn.click(lambda: [], None, chatbot)

#     # ================= FIXED LAUNCH =================
#     demo.launch(
#         share=True,
#         server_name="0.0.0.0",
#         server_port=7860,
#         show_error=True
#     )


# if __name__ == "__main__":
#     force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
#     main(force_reindex)

# import sys
# import gc
# import os
# import json
# from datetime import datetime

# import torch
# import gradio as gr

# from src.utils import clear_page_cache
# from src.indexer import MultimodalIndexer
# from src.retriever import MultimodalRetriever
# from src.generator import MultimodalGenerator

# HISTORY_FILE = "chat_history.json"

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

# def aggressive_cleanup():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

# def main(force_reindex=False):
#     print("\nInitializing Multimodal RAG System...\n")

#     indexer = MultimodalIndexer(force_recreate=force_reindex)
#     retriever = MultimodalRetriever(indexer)
#     generator = MultimodalGenerator()

#     print("Warming up retrieval model...")
#     _ = retriever._extract_text_embedding("warmup query")
#     print("System Ready!\n")

#     if force_reindex or indexer.is_collection_empty():
#         print("Indexing documents...\n")
#         indexer.index_all_data("data")
#         print("Indexing completed!\n")
#     else:
#         print("Existing index found. Skipping indexing.\n")

#     # Mapping for robust source filtering
#     source_map = {
#         "Both Documents": None,
#         "SBC": "data/sbc.pdf",
#         "SPD": "data/spd.pdf",
#     }

#     def answer_query(message, history, source_choice):
#         if history is None:
#             history = []

#         if not message or not message.strip():
#             return history, ""

#         query = message.strip()
#         source_filter = source_map.get(source_choice)

#         try:
#             hits = retriever.search(
#                 query,
#                 top_k=3,
#                 source_filter=source_filter
#             )

#             if not hits:
#                 bot_response = "No relevant information found in selected documents."
#             else:
#                 context_hits = sorted(hits, key=lambda x: x.score, reverse=True)
#                 best_hit = context_hits[0]

#                 source = best_hit.payload.get("source", "Unknown")
#                 page = best_hit.payload.get("page_number", "N/A")

#                 answer = generator.generate_answer(query, context_hits)

#                 bot_response = f"""### Retrieved Context
# Source: {os.path.basename(source)}
# Page: {page}
# Scope: {source_choice}

# ---

# ### Answer
# {answer}"""

#                 save_to_history(query, source_choice, answer)

#             history.append({"role": "user", "content": query})
#             history.append({"role": "assistant", "content": bot_response})

#             return history, ""

#         except Exception as e:
#             error_message = f"Error while generating response:\n{str(e)}"
#             history.append({"role": "user", "content": query})
#             history.append({"role": "assistant", "content": error_message})
#             return history, ""

#         finally:
#             clear_page_cache()
#             aggressive_cleanup()

#     # Custom CSS for professional appearance
#     custom_css = """
#     #sidebar {
#         background-color: #f4f4f9;
#         border-right: 1px solid #ddd;
#         padding: 20px;
#     }
#     .main-container {
#         max-width: 1200px;
#         margin: auto;
#     }
#     footer {display: none !important;}
#     """

#     with gr.Blocks(css=custom_css, title="Multimodal RAG Assistant", theme=gr.themes.Soft()) as demo:
        
#         with gr.Row(elem_classes=["main-container"]):
            
#             # Left Sidebar
#             with gr.Column(scale=1, elem_id="sidebar"):
#                 gr.Markdown("## System Control")
#                 gr.Markdown("---")
                
#                 gr.Markdown("### Document Selection")
#                 source_dropdown = gr.Dropdown(
#                     choices=list(source_map.keys()),
#                     value="Both Documents",
#                     label="Knowledge Base",
#                     container=True
#                 )
                
#                 gr.Markdown("### Actions")
#                 clear_btn = gr.Button("Clear Chat History", variant="secondary")
                
#                 gr.Markdown("---")
#                 gr.Markdown("### System Status")
#                 gr.Markdown("Status: Ready")
#                 gr.Markdown("Backend: PyTorch + Qdrant")

#             # Main Chat Column
#             with gr.Column(scale=4):
#                 gr.Markdown("# Multimodal RAG Assistant")
#                 gr.Markdown("Analyze and query document data using vision-language processing.")

#                 chatbot = gr.Chatbot(
#                     type="messages",
#                     height=600,
#                     bubble_full_width=False,
#                     show_label=False
#                 )

#                 with gr.Row():
#                     msg = gr.Textbox(
#                         placeholder="Enter your question here...",
#                         scale=9,
#                         container=False,
#                         autofocus=True
#                     )
#                     submit_btn = gr.Button("Submit", variant="primary", scale=1)

#                 gr.Examples(
#                     examples=[
#                         ["What is the deductible for this plan?", "SPD"],
#                         ["Summarize the coverage details.", "SBC"],
#                         ["Compare the benefits between these documents.", "Both Documents"],
#                     ],
#                     inputs=[msg, source_dropdown],
#                     label="Sample Queries"
#                 )

#         # Event listeners
#         msg.submit(answer_query, [msg, chatbot, source_dropdown], [chatbot, msg])
#         submit_btn.click(answer_query, [msg, chatbot, source_dropdown], [chatbot, msg])
#         clear_btn.click(lambda: [], None, chatbot)

#     demo.launch(
#         share=True,
#         server_name="0.0.0.0",
#         server_port=7860,
#         show_error=True
#     )

# if __name__ == "__main__":
#     force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
#     main(force_reindex)

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
    print("\nInitializing Multimodal RAG System...\n")

    indexer = MultimodalIndexer(force_recreate=force_reindex)
    retriever = MultimodalRetriever(indexer)
    generator = MultimodalGenerator()

    print("Warming up retrieval model...")
    _ = retriever._extract_text_embedding("warmup query")
    print("System Ready!\n")

    if force_reindex or indexer.is_collection_empty():
        print("Indexing documents...\n")
        indexer.index_all_data("data")
        print("Indexing completed!\n")
    else:
        print("Existing index found. Skipping indexing.\n")

    source_map = {
        "Both Documents": None,
        "SBC": "data/sbc.pdf",
        "SPD": "data/spd.pdf",
    }

    # ── Step 1: Immediately push the user message + thinking bubble ──────────
    def user_turn(message, history, source_choice):
        """Append user message and a 'thinking' placeholder instantly."""
        if not message or not message.strip():
            return history, "", gr.update(interactive=False), gr.update(interactive=False)

        history = history or []
        history = history + [
            {"role": "user",      "content": message.strip()},
            {"role": "assistant", "content": "_⏳ Thinking…_"},
        ]
        # Return updated chat + clear textbox + disable controls while processing
        return history, "", gr.update(interactive=False), gr.update(interactive=False)

    # ── Step 2: Replace the placeholder with the real answer ─────────────────
    def bot_turn(history, source_choice):
        """Replace the last 'thinking' message with the actual answer."""
        if not history:
            return history, gr.update(interactive=True), gr.update(interactive=True)

        # Extract the user query from second-to-last message
        query = history[-2]["content"]
        source_filter = source_map.get(source_choice)

        try:
            hits = retriever.search(query, top_k=3, source_filter=source_filter)

            if not hits:
                bot_response = (
                    "ℹ️ No relevant information found in the selected documents.\n\n"
                    "Try rephrasing your question or switching the document filter."
                )
            else:
                context_hits = sorted(hits, key=lambda x: x.score, reverse=True)
                best_hit = context_hits[0]

                source = best_hit.payload.get("source", "Unknown")
                page   = best_hit.payload.get("page_number", "N/A")

                answer = generator.generate_answer(query, context_hits)

                bot_response = (
                    f"📄 **Source:** `{os.path.basename(source)}`  "
                    f"**Page:** {page}  "
                    f"**Scope:** {source_choice}\n\n"
                    f"---\n\n"
                    f"{answer}"
                )

                save_to_history(query, source_choice, answer)

        except Exception as e:
            bot_response = f"⚠️ **Error while generating response:**\n\n```\n{str(e)}\n```"

        finally:
            clear_page_cache()
            aggressive_cleanup()

        # Replace the placeholder
        history[-1] = {"role": "assistant", "content": bot_response}
        return history, gr.update(interactive=True), gr.update(interactive=True)

    # ── UI ────────────────────────────────────────────────────────────────────
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&family=Source+Code+Pro:wght@400;500&display=swap');

    * { box-sizing: border-box; }

    body, .gradio-container {
        font-family: 'Lato', sans-serif !important;
        background: #f5f0e8 !important;
        color: #2c2a26 !important;
    }

    footer { display: none !important; }

    /* ── App shell ── */
    .app-shell {
        display: flex;
        height: 100vh;
        overflow: hidden;
        background: #f5f0e8;
    }

    /* ── Sidebar ── */
    #sidebar {
        width: 260px;
        min-width: 260px;
        background: #ede8de;
        border-right: 1px solid #d8d0c0;
        padding: 28px 20px;
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    .sidebar-title {
        font-size: 17px;
        font-weight: 700;
        letter-spacing: -0.2px;
        color: #2c2a26;
        margin: 0 0 2px 0;
    }

    .sidebar-sub {
        font-size: 11px;
        color: #9a9080;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
        margin-top: 4px;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 12px;
        color: #4a7c59;
        background: #e4f0e8;
        border: 1px solid #b8d8c0;
        border-radius: 20px;
        padding: 5px 12px;
        width: fit-content;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background: #4a9e6a;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.35; }
    }

    /* ── Main chat area ── */
    #chat-area {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        background: #faf8f4;
    }

    /* ── Chat header ── */
    #chat-header {
        padding: 20px 28px 16px;
        border-bottom: 1px solid #e0d8cc;
        background: #faf8f4;
    }

    #chat-header h1 {
        font-size: 21px;
        font-weight: 700;
        margin: 0;
        color: #2c2a26;
        letter-spacing: -0.3px;
    }

    #chat-header p {
        font-size: 13px;
        color: #8a8070;
        margin: 4px 0 0;
    }

    /* ── Chatbot ── */
    #chatbot {
        background: transparent !important;
        border: none !important;
    }

    #chatbot .message-wrap { gap: 14px !important; }

    /* User bubble — warm sand */
    #chatbot .user .message {
        background: #e8e0d0 !important;
        border: 1px solid #d0c8b8 !important;
        color: #2c2a26 !important;
        border-radius: 18px 18px 4px 18px !important;
        font-size: 14px !important;
        max-width: 72% !important;
    }

    /* Bot bubble — clean white */
    #chatbot .bot .message {
        background: #ffffff !important;
        border: 1px solid #e0d8cc !important;
        color: #2c2a26 !important;
        border-radius: 4px 18px 18px 18px !important;
        font-size: 14px !important;
        font-family: 'Lato', sans-serif !important;
        max-width: 84% !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    }

    #chatbot .bot .message code {
        font-family: 'Source Code Pro', monospace !important;
        background: #f0ece4 !important;
        color: #5a4a2a !important;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 12px;
    }

    #chatbot .bot .message hr {
        border-color: #e8e0d0 !important;
        margin: 10px 0 !important;
    }

    #chatbot .bot .message strong { color: #3a3628 !important; }

    /* ── Input row ── */
    #input-row {
        padding: 14px 28px 18px;
        border-top: 1px solid #e0d8cc;
        background: #faf8f4;
        display: flex;
        gap: 10px;
        align-items: flex-end;
    }

    #msg-box textarea {
        background: #ffffff !important;
        border: 1px solid #d0c8b8 !important;
        border-radius: 14px !important;
        color: #2c2a26 !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        resize: none !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    #msg-box textarea:focus {
        border-color: #8a7a5a !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(138,122,90,0.12) !important;
    }

    #msg-box textarea::placeholder { color: #b0a890 !important; }

    #send-btn {
        background: #5a4a2a !important;
        color: #faf8f4 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 22px !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: background 0.15s ease, transform 0.1s ease;
        height: 46px;
        white-space: nowrap;
    }

    #send-btn:hover:not(:disabled)  { background: #6e5c38 !important; }
    #send-btn:active                 { transform: scale(0.97); }
    #send-btn:disabled               { background: #d0c8b8 !important; color: #a09880 !important; cursor: not-allowed !important; }

    /* ── Dropdown override ── */
    select, .gr-dropdown select {
        background: #ffffff !important;
        color: #2c2a26 !important;
        border: 1px solid #d0c8b8 !important;
        border-radius: 10px !important;
        font-size: 13px !important;
        font-family: 'Lato', sans-serif !important;
    }

    /* ── Clear button ── */
    #clear-btn {
        background: transparent !important;
        border: 1px solid #d0c8b8 !important;
        color: #8a8070 !important;
        border-radius: 10px !important;
        font-size: 13px !important;
        font-family: 'Lato', sans-serif !important;
        padding: 8px 14px !important;
        cursor: pointer !important;
        transition: all 0.15s;
    }

    #clear-btn:hover {
        border-color: #c0392b !important;
        color: #c0392b !important;
        background: #fdf0ee !important;
    }

    /* ── Examples ── */
    .gr-examples { margin-top: 10px; }
    .gr-examples button {
        background: #ede8de !important;
        border: 1px solid #d0c8b8 !important;
        color: #6a6050 !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        padding: 6px 12px !important;
        font-family: 'Source Code Pro', monospace !important;
        transition: all 0.15s;
    }

    .gr-examples button:hover {
        background: #e0d8c8 !important;
        border-color: #8a7a5a !important;
        color: #3a3020 !important;
    }

    /* ── Label text (Gradio internals) ── */
    label, .block label span, .gr-form label {
        color: #4a4438 !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 13px !important;
    }
    """

    with gr.Blocks(
        css=custom_css,
        title="RAG Assistant",
        theme=gr.themes.Base(
            primary_hue="stone",
            neutral_hue="stone",
            font=gr.themes.GoogleFont("Lato"),
        ),
    ) as demo:

        with gr.Row(elem_classes=["app-shell"]):

            # ── Sidebar ───────────────────────────────────────────
            with gr.Column(elem_id="sidebar", scale=0, min_width=260):
                gr.HTML("""
                    <div class="sidebar-title">RAG Assistant</div>
                    <div style="font-size:12px;color:#9a9080;margin-top:2px;">
                        Vision-Language Document QA
                    </div>
                """)

                gr.HTML('<div class="sidebar-sub" style="margin-top:8px;">Knowledge Base</div>')
                source_dropdown = gr.Dropdown(
                    choices=list(source_map.keys()),
                    value="Both Documents",
                    label="",
                    container=False,
                )

                gr.HTML('<div class="sidebar-sub">Actions</div>')
                clear_btn = gr.Button("🗑 Clear chat", elem_id="clear-btn", size="sm")

                gr.HTML("""
                    <div style="margin-top: auto; padding-top: 24px;">
                        <div class="sidebar-sub">System</div>
                        <div class="status-badge">
                            <div class="status-dot"></div>
                            Ready · PyTorch + Qdrant
                        </div>
                    </div>
                """)

            # ── Chat area ──────────────────────────────────────────
            with gr.Column(elem_id="chat-area", scale=1):

                gr.HTML("""
                    <div id="chat-header">
                        <h1>Multimodal RAG Assistant</h1>
                        <p>Ask questions about SBC & SPD documents using vision-language retrieval.</p>
                    </div>
                """)

                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    type="messages",
                    height=560,
                    bubble_full_width=False,
                    show_label=False,
                    avatar_images=(None, None),
                    render_markdown=True,
                )

                with gr.Row(elem_id="input-row"):
                    msg = gr.Textbox(
                        placeholder="Ask a question about your documents…",
                        scale=9,
                        container=False,
                        lines=1,
                        max_lines=4,
                        autofocus=True,
                        elem_id="msg-box",
                    )
                    submit_btn = gr.Button(
                        "Send ↑",
                        variant="primary",
                        scale=0,
                        min_width=90,
                        elem_id="send-btn",
                    )

                gr.Examples(
                    examples=[
                        ["What is the deductible for this plan?", "SPD"],
                        ["What services are covered?", "SBC"],
                        ["Compare benefits between both documents.", "Both Documents"],
                        ["What is the out-of-pocket maximum?", "SPD"],
                    ],
                    inputs=[msg, source_dropdown],
                    label="Example queries",
                )

        # ── Event wiring (2-step: user_turn → bot_turn) ──────────────────────
        # On submit, first show the user message immediately, then fetch answer
        (
            msg.submit(
                user_turn,
                inputs=[msg, chatbot, source_dropdown],
                outputs=[chatbot, msg, msg, submit_btn],
                queue=False,
            ).then(
                bot_turn,
                inputs=[chatbot, source_dropdown],
                outputs=[chatbot, msg, submit_btn],
            )
        )

        (
            submit_btn.click(
                user_turn,
                inputs=[msg, chatbot, source_dropdown],
                outputs=[chatbot, msg, msg, submit_btn],
                queue=False,
            ).then(
                bot_turn,
                inputs=[chatbot, source_dropdown],
                outputs=[chatbot, msg, submit_btn],
            )
        )

        clear_btn.click(lambda: ([], gr.update(interactive=True), gr.update(interactive=True)),
                        None, [chatbot, msg, submit_btn])

    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
    main(force_reindex)