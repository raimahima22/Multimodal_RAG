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


# =========================================================
# CONFIG
# =========================================================
HISTORY_FILE = "chat_history.json"


# =========================================================
# SAVE CHAT HISTORY
# =========================================================
def save_to_history(query, source_input, answer):

    history_data = []

    if os.path.exists(HISTORY_FILE):

        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history_data = json.load(f)

        except Exception:
            history_data = []

    history_data.append(
        {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "source": source_input,
            "answer": answer,
        }
    )

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            history_data,
            f,
            indent=2,
            ensure_ascii=False
        )


# =========================================================
# CLEANUP
# =========================================================
def aggressive_cleanup():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# MAIN
# =========================================================
def main(force_reindex: bool = False):

    print("\n🧠 Initializing Multimodal RAG System...\n")

    # -----------------------------------------------------
    # INITIALIZE COMPONENTS
    # -----------------------------------------------------
    indexer = MultimodalIndexer(
        force_recreate=force_reindex
    )

    retriever = MultimodalRetriever(indexer)

    generator = MultimodalGenerator()

    # -----------------------------------------------------
    # WARMUP
    # -----------------------------------------------------
    print("🔥 Warming up retrieval model...")

    _ = retriever._extract_text_embedding(
        "warmup query"
    )

    print("✅ System Ready!\n")

    # -----------------------------------------------------
    # INDEXING
    # -----------------------------------------------------
    print("📚 Checking document index...\n")

    if force_reindex or indexer.is_collection_empty():

        print("🔄 Indexing documents...\n")

        indexer.index_all_data("data")

        print("\n✅ Indexing completed!\n")

    else:

        print(
            "✅ Existing index found. "
            "Skipping indexing.\n"
        )

    # -----------------------------------------------------
    # SOURCE OPTIONS
    # -----------------------------------------------------
    source_options = {
        "Both Documents": None,
        "SBC": "data/sbc.pdf",
        "SPD": "data/spd.pdf",
    }

    # =====================================================
    # ANSWER FUNCTION
    # =====================================================
    def answer_query(
        message,
        history,
        source_choice
    ):

        if history is None:
            history = []

        if not message or not message.strip():
            return history, ""

        query = message.strip()

        # -------------------------------------------------
        # SOURCE FILTER
        # -------------------------------------------------
        if source_choice.lower() in [
            "sbc",
            "sbc.pdf"
        ]:

            source_filter = "data/sbc.pdf"

        elif source_choice.lower() in [
            "spd",
            "spd.pdf"
        ]:

            source_filter = "data/spd.pdf"

        else:

            source_filter = None

        try:

            # -------------------------------------------------
            # RETRIEVAL
            # -------------------------------------------------
            hits = retriever.search(
                query,
                top_k=3,
                source_filter=source_filter,
            )

            # -------------------------------------------------
            # NO RESULTS
            # -------------------------------------------------
            if not hits:

                bot_response = (
                    "❌ No relevant information found "
                    "in the selected documents."
                )

            else:

                # ---------------------------------------------
                # SORT RESULTS
                # ---------------------------------------------
                context_hits = sorted(
                    hits,
                    key=lambda x: x.score,
                    reverse=True
                )

                best_hit = context_hits[0]

                source = best_hit.payload.get(
                    "source",
                    "Unknown"
                )

                page = best_hit.payload.get(
                    "page_number",
                    "N/A"
                )

                # ---------------------------------------------
                # GENERATE ANSWER
                # ---------------------------------------------
                answer = generator.generate_answer(
                    query,
                    context_hits
                )

                # ---------------------------------------------
                # FINAL RESPONSE
                # ---------------------------------------------
                bot_response = f"""
## 📄 Retrieved Context

- **Source:** `{os.path.basename(source)}`
- **Page:** `{page}`
- **Search Scope:** `{source_choice}`

---

## 🤖 Answer

{answer}
"""

                # ---------------------------------------------
                # SAVE HISTORY
                # ---------------------------------------------
                save_to_history(
                    query=query,
                    source_input=source_choice,
                    answer=answer
                )

            # -------------------------------------------------
            # APPEND USER MESSAGE
            # -------------------------------------------------
            history.append(
                {
                    "role": "user",
                    "content": query
                }
            )

            # -------------------------------------------------
            # APPEND ASSISTANT MESSAGE
            # -------------------------------------------------
            history.append(
                {
                    "role": "assistant",
                    "content": bot_response
                }
            )

            return history, ""

        # =================================================
        # ERROR HANDLING
        # =================================================
        except Exception as e:

            error_message = (
                "❌ Error while generating response:\n\n"
                f"{str(e)}"
            )

            history.append(
                {
                    "role": "user",
                    "content": query
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )

            return history, ""

        # =================================================
        # CLEANUP
        # =================================================
        finally:

            clear_page_cache()

            aggressive_cleanup()

    # =====================================================
    # UI
    # =====================================================
    with gr.Blocks(
        title="Multimodal RAG Assistant",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="gray"
        ),
        css="""
        .gradio-container {
            max-width: 1500px !important;
            margin: auto !important;
            padding-top: 10px !important;
        }

        footer {
            visibility: hidden;
        }

        .main-title {
            text-align: center;
            margin-bottom: 10px;
        }

        .chatbot {
            border-radius: 20px !important;
            border: 1px solid #e5e7eb !important;
        }

        .left-panel {
            border-radius: 20px;
            padding: 18px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
        }

        .message-wrap {
            font-size: 16px !important;
        }

        .bubble-wrap {
            border-radius: 18px !important;
        }

        .examples {
            margin-top: 10px;
        }

        .input-box textarea {
            font-size: 16px !important;
        }
        """
    ) as demo:

        # -------------------------------------------------
        # HEADER
        # -------------------------------------------------
        gr.HTML(
            """
            <div class="main-title">
                <h1>🧠 Multimodal RAG Assistant</h1>
                <p>
                    Intelligent Question Answering
                    over SBC & SPD Documents
                </p>
            </div>
            """
        )

        # -------------------------------------------------
        # MAIN LAYOUT
        # -------------------------------------------------
        with gr.Row():

            # =================================================
            # CHAT AREA
            # =================================================
            with gr.Column(scale=5):

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=720,
                    type="messages",
                    render_markdown=True,
                    show_copy_button=True,
                    bubble_full_width=False,
                    elem_classes="chatbot"
                )

                with gr.Row():

                    msg = gr.Textbox(
                        placeholder=(
                            "Ask anything about "
                            "SBC or SPD documents..."
                        ),
                        show_label=False,
                        lines=2,
                        max_lines=5,
                        autofocus=True,
                        elem_classes="input-box",
                        scale=8
                    )

                    submit_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1
                    )

            # =================================================
            # SIDEBAR
            # =================================================
            with gr.Column(
                scale=1,
                elem_classes="left-panel"
            ):

                gr.Markdown(
                    "## 📄 Document Filter"
                )

                source_dropdown = gr.Dropdown(
                    choices=list(source_options.keys()),
                    value="Both Documents",
                    label="Search Scope",
                    info=(
                        "Restrict search "
                        "to a specific document"
                    )
                )

                gr.Markdown(
                    "##  Example Questions"
                )

                gr.Examples(
                    examples=[
                        [
                            "What is the deductible "
                            "for the Gold PPO plan?",
                            "SPD"
                        ],
                        [
                            "What services are covered "
                            "before deductible?",
                            "SBC"
                        ],
                        [
                            "Compare SBC and SPD "
                            "requirements",
                            "Both Documents"
                        ],
                        [
                            "Does the plan require "
                            "specialist referrals?",
                            "Both Documents"
                        ],
                        
                    ],
                    inputs=[
                        msg,
                        source_dropdown
                    ],
                    cache_examples=False,
                    # elem_classes="examples"
                )

                clear_btn = gr.Button(
                    "🗑️ Clear Conversation",
                    variant="secondary"
                )

        # =====================================================
        # EVENTS
        # =====================================================
        msg.submit(
            fn=answer_query,
            inputs=[
                msg,
                chatbot,
                source_dropdown
            ],
            outputs=[
                chatbot,
                msg
            ]
        )

        submit_btn.click(
            fn=answer_query,
            inputs=[
                msg,
                chatbot,
                source_dropdown
            ],
            outputs=[
                chatbot,
                msg
            ]
        )

        clear_btn.click(
            fn=lambda: [],
            inputs=None,
            outputs=chatbot,
            queue=False
        )

        # -------------------------------------------------
        # FOOTER
        # -------------------------------------------------
        gr.Markdown(
            """
---
### 🚀 Powered By

- Qdrant
- ColQwen2.5
- OpenRouter
- Gradio
- Multimodal RAG
"""
        )

    # =====================================================
    # LAUNCH
    # =====================================================
    try:

        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )

    finally:

        print(
            "\n🛑 Shutting down Qdrant client...\n"
        )

        try:
            indexer.local_client.close()

        except Exception:
            pass


# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":

    force_reindex = (
        "--reindex" in sys.argv
        or "-r" in sys.argv
    )

    if force_reindex:
        print("🔄 Reindex mode activated\n")

    main(force_reindex=force_reindex)