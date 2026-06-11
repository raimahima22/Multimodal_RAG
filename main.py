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
from src.agent import run_agent

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
            bot_response = f"⚠️ **Error while generating response:**\n\n{str(e)}"
        
        # Replace thinking message with real answer
        history[-1] = {"role": "assistant", "content": bot_response}
        
        aggressive_cleanup()
        clear_page_cache()
        return history, gr.update(interactive=True), gr.update(interactive=True)

        
    # ── Custom CSS (Slightly adjusted for better responsiveness) ─────────────
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
            <div id="main-subtitle">Intelligent Document Q&A over SBC & SPD using Vision-Language Retrieval</div>
        """)

        with gr.Row(elem_classes=["app-shell"]):
            # Sidebar
            with gr.Column(elem_id="sidebar", scale=0, min_width=260):
                gr.HTML('<div class="sidebar-sub">Knowledge Base</div>')
                source_dropdown = gr.Dropdown(
                    choices=list(source_map.keys()),
                    value="Both Documents",
                    label="",
                    container=False,
                    scale=1
                )
                
                gr.HTML('<div class="sidebar-sub">Actions</div>')
                clear_btn = gr.Button("🗑 Clear Chat", elem_id="clear-btn", size="sm")

                gr.HTML("""
                    <div style="margin-top: auto;">
                        <div class="status-badge">
                            <div class="status-dot"></div>
                            Ready • Qdrant + PyTorch
                        </div>
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
                        placeholder="Ask a question about your documents...",
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
                        ["What is the deductible for this plan?", "SPD"],
                        ["What services are covered?", "SBC"],
                        ["Compare benefits between both documents.", "Both Documents"],
                        ["What is the out-of-pocket maximum?", "SPD"],
                    ],
                    inputs=[msg, source_dropdown],
                    label="Example Queries",
                    cache_examples=False
                )

        # Event Handling
        msg.submit(
            user_turn,
            inputs=[msg, chatbot, source_dropdown],
            outputs=[chatbot, msg, msg, submit_btn],
            queue=False
        ).then(
            bot_turn,
            inputs=[chatbot, source_dropdown],
            outputs=[chatbot, msg, submit_btn]
        )

        submit_btn.click(
            user_turn,
            inputs=[msg, chatbot, source_dropdown],
            outputs=[chatbot, msg, msg, submit_btn],
            queue=False
        ).then(
            bot_turn,
            inputs=[chatbot, source_dropdown],
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