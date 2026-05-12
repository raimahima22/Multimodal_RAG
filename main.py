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

    # -- Step 1: Immediately push the user message + placeholder --
    def user_turn(message, history, source_choice):
        if not message or not message.strip():
            return history, "", gr.update(interactive=False), gr.update(interactive=False)

        history = history or []
        history = history + [
            {"role": "user", "content": message.strip()},
            {"role": "assistant", "content": "Thinking..."},
        ]
        return history, "", gr.update(interactive=False), gr.update(interactive=False)

    # -- Step 2: Replace placeholder with real answer --
    def bot_turn(history, source_choice):
        if not history:
            return history, gr.update(interactive=True), gr.update(interactive=True)

        query = history[-2]["content"]
        source_filter = source_map.get(source_choice)

        try:
            hits = retriever.search(query, top_k=3, source_filter=source_filter)

            if not hits:
                bot_response = (
                    "No relevant information found in the selected documents. "
                    "Try rephrasing your question or switching the document filter."
                )
            else:
                context_hits = sorted(hits, key=lambda x: x.score, reverse=True)
                best_hit = context_hits[0]

                source = best_hit.payload.get("source", "Unknown")
                page = best_hit.payload.get("page_number", "N/A")

                answer = generator.generate_answer(query, context_hits)

                bot_response = (
                    f"Source: {os.path.basename(source)} | "
                    f"Page: {page} | "
                    f"Scope: {source_choice}\n\n"
                    f"---\n\n"
                    f"{answer}"
                )

                save_to_history(query, source_choice, answer)

        except Exception as e:
            bot_response = f"Error while generating response: {str(e)}"

        finally:
            clear_page_cache()
            aggressive_cleanup()

        history[-1] = {"role": "assistant", "content": bot_response}
        return history, gr.update(interactive=True), gr.update(interactive=True)

    # -- UI Styling --
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');

    body, .gradio-container {
        font-family: 'Lato', sans-serif !important;
        background: #f5f0e8 !important;
        color: #2c2a26 !important;
    }

    footer { display: none !important; }

    /* Fix for height issue shown in image_9de202.png */
    .app-shell {
        display: flex;
        min-height: 100vh;
        background: #f5f0e8;
    }

    #sidebar {
        width: 260px;
        min-width: 260px;
        background: #ede8de;
        border-right: 1px solid #d8d0c0;
        padding: 28px 20px;
    }

    #chat-area {
        flex: 1;
        padding: 20px;
        background: #faf8f4;
    }

    #chat-header {
        margin-bottom: 20px;
        border-bottom: 1px solid #e0d8cc;
        padding-bottom: 10px;
    }

    .status-badge {
        font-size: 12px;
        color: #4a7c59;
        background: #e4f0e8;
        border: 1px solid #b8d8c0;
        border-radius: 20px;
        padding: 5px 12px;
        display: inline-block;
        margin-top: 10px;
    }

    /* Message Styling */
    #chatbot .user .message { background: #e8e0d0 !important; }
    #chatbot .bot .message { background: #ffffff !important; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
    """

    with gr.Blocks(css=custom_css, title="Multimodal Document RAG", theme=gr.themes.Soft()) as demo:

        with gr.Row(elem_classes=["app-shell"]):
            
            # Sidebar
            with gr.Column(elem_id="sidebar", scale=0, min_width=260):
                gr.Markdown("### RAG System Controls")
                gr.Markdown("---")
                gr.Markdown("Knowledge Base Selection")
                source_dropdown = gr.Dropdown(
                    choices=list(source_map.keys()),
                    value="Both Documents",
                    label="",
                    container=False,
                )
                
                gr.Markdown("---")
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                
                gr.HTML("""
                    <div class="status-badge">
                        Status: Ready | Backend: PyTorch
                    </div>
                """)

            # Main Chat Area
            with gr.Column(elem_id="chat-area", scale=1):
                with gr.Div(elem_id="chat-header"):
                    gr.Markdown("# Multimodal RAG Assistant")
                    gr.Markdown("Analyze insurance documents (SBC and SPD) using vision-language reasoning.")

                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    type="messages",
                    height=550,
                    bubble_full_width=False,
                    show_label=False
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Enter your query...",
                        scale=9,
                        container=False,
                        autofocus=True
                    )
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        ["What is the deductible for this plan?", "SPD"],
                        ["What services are covered?", "SBC"],
                        ["Compare benefits between both documents.", "Both Documents"],
                    ],
                    inputs=[msg, source_dropdown],
                    label="Quick Start Examples"
                )

        # Logic Wiring
        msg.submit(user_turn, [msg, chatbot, source_dropdown], [chatbot, msg, msg, submit_btn], queue=False).then(
            bot_turn, [chatbot, source_dropdown], [chatbot, msg, submit_btn]
        )
        submit_btn.click(user_turn, [msg, chatbot, source_dropdown], [chatbot, msg, msg, submit_btn], queue=False).then(
            bot_turn, [chatbot, source_dropdown], [chatbot, msg, submit_btn]
        )
        clear_btn.click(lambda: ([], gr.update(interactive=True), gr.update(interactive=True)), None, [chatbot, msg, submit_btn])

    demo.launch(share=True, server_name="0.0.0.0", server_port=7860, show_error=True)

if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
    main(force_reindex)