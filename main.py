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

    print("\n Initializing Multimodal RAG System...\n")

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
    print(" Warming up retrieval model...")

    _ = retriever._extract_text_embedding(
        "warmup query"
    )

    print(" System Ready!\n")

    # -----------------------------------------------------
    # INDEXING
    # -----------------------------------------------------
    print(" Checking document index...\n")

    if force_reindex or indexer.is_collection_empty():

        print(" Indexing documents...\n")

        indexer.index_all_data("data")

        print("\n Indexing completed!\n")

    else:

        print(
            " Existing index found. "
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
                    " No relevant information found "
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
##  Retrieved Context

- **Source:** `{os.path.basename(source)}`
- **Page:** `{page}`
- **Search Scope:** `{source_choice}`

---

##  Answer

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
                " Error while generating response:\n\n"
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
        max-width: 1600px !important;
        margin: auto !important;
        padding-top: 0px !important;
        background: #f5f7fb;
    }

    footer {
        visibility: hidden;
    }

    .main-header {
        padding-top: 20px;
        padding-bottom: 10px;
        text-align: center;
    }

    .main-header h1 {
        font-size: 34px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 5px;
    }

    .main-header p {
        font-size: 16px;
        color: #6b7280;
    }

    .sidebar {
        background: white;
        border-right: 1px solid #e5e7eb;
        padding: 22px;
        min-height: 100vh;
    }

    .sidebar-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 12px;
        color: #111827;
    }

    .chat-area {
        padding: 20px;
    }

    .chatbot {
        border-radius: 18px !important;
        border: 1px solid #dbe1ea !important;
        background: white !important;
    }

    .input-container {
        margin-top: 12px;
    }

    .input-box textarea {
        border-radius: 14px !important;
        font-size: 16px !important;
        padding: 14px !important;
    }

    .message-wrap {
        font-size: 15px !important;
    }

    .bubble-wrap {
        border-radius: 16px !important;
    }

    .send-btn {
        height: 52px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }

    .clear-btn {
        margin-top: 15px;
        border-radius: 12px !important;
    }

    .examples-table {
        margin-top: 8px;
    }
    """
) as demo:

    # =================================================
    # HEADER
    # =================================================
    gr.HTML(
        """
        <div class="main-header">
            <h1>Multimodal RAG Assistant</h1>
            <p>
                Intelligent document question answering over SBC and SPD files
            </p>
        </div>
        """
    )

    # =================================================
    # MAIN LAYOUT
    # =================================================
    with gr.Row():

        # =============================================
        # LEFT SIDEBAR
        # =============================================
        with gr.Column(
            scale=1,
            elem_classes="sidebar",
            min_width=320
        ):

            gr.HTML(
                """
                <div class="sidebar-title">
                    Document Filter
                </div>
                """
            )

            source_dropdown = gr.Dropdown(
                choices=list(source_options.keys()),
                value="Both Documents",
                show_label=False,
                info="Restrict retrieval to a specific document"
            )

            gr.HTML(
                """
                <div class="sidebar-title" style="margin-top:30px;">
                    Example Questions
                </div>
                """
            )

            gr.Examples(
                examples=[
                    [
                        "What is the deductible for the Gold PPO plan?",
                        "SPD"
                    ],
                    [
                        "What services are covered before deductible?",
                        "SBC"
                    ],
                    [
                        "Compare SBC and SPD requirements",
                        "Both Documents"
                    ],
                    [
                        "Does the plan require specialist referrals?",
                        "Both Documents"
                    ],
                    [
                        "What are the out-of-pocket limits?",
                        "Both Documents"
                    ],
                ],
                inputs=[
                    msg,
                    source_dropdown
                ],
                cache_examples=False
            )

            clear_btn = gr.Button(
                "Clear Conversation",
                variant="secondary",
                elem_classes="clear-btn"
            )

        # =============================================
        # CHAT SECTION
        # =============================================
        with gr.Column(
            scale=4,
            elem_classes="chat-area"
        ):

            chatbot = gr.Chatbot(
                label="Conversation",
                height=760,
                type="messages",
                render_markdown=True,
                show_copy_button=True,
                bubble_full_width=False,
                elem_classes="chatbot"
            )

            with gr.Row(
                elem_classes="input-container"
            ):

                msg = gr.Textbox(
                    placeholder=(
                        "Ask a question about the documents..."
                    ),
                    show_label=False,
                    lines=2,
                    max_lines=5,
                    autofocus=True,
                    scale=8,
                    elem_classes="input-box"
                )

                submit_btn = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    elem_classes="send-btn"
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

    # =====================================================
    # FOOTER
    # =====================================================
    gr.Markdown(
        """
---
Qdrant • ColQwen2.5 • OpenRouter • Gradio • Multimodal RAG
"""
    )