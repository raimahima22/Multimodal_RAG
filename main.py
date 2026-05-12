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


def aggressive_cleanup():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(force_reindex=False):

    print("\nInitializing Multimodal RAG System...\n")

    indexer = MultimodalIndexer(
        force_recreate=force_reindex
    )

    retriever = MultimodalRetriever(indexer)

    generator = MultimodalGenerator()

    print("Warming up retrieval model...")

    _ = retriever._extract_text_embedding(
        "warmup query"
    )

    print("System Ready!\n")

    print("Checking document index...\n")

    if force_reindex or indexer.is_collection_empty():

        print("Indexing documents...\n")

        indexer.index_all_data("data")

        print("Indexing completed!\n")

    else:

        print(
            "Existing index found. "
            "Skipping indexing.\n"
        )

    source_options = {
        "Both Documents": None,
        "SBC": "data/sbc.pdf",
        "SPD": "data/spd.pdf",
    }

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

            hits = retriever.search(
                query,
                top_k=3,
                source_filter=source_filter
            )

            if not hits:

                bot_response = (
                    "No relevant information found "
                    "in the selected documents."
                )

            else:

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

                answer = generator.generate_answer(
                    query,
                    context_hits
                )

                bot_response = f"""
## Retrieved Context

- **Source:** `{os.path.basename(source)}`
- **Page:** `{page}`
- **Search Scope:** `{source_choice}`

---

## Answer

{answer}
"""

                save_to_history(
                    query=query,
                    source_input=source_choice,
                    answer=answer
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
                    "content": bot_response
                }
            )

            return history, ""

        except Exception as e:

            error_message = (
                "Error while generating response:\n\n"
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

        finally:

            clear_page_cache()

            aggressive_cleanup()

    custom_css = """
    .gradio-container {
        max-width: 1600px !important;
        margin: auto !important;
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
    """

    with gr.Blocks(
        title="Multimodal RAG Assistant"
    ) as demo:

        gr.HTML(
            """
            <div class="main-header">
                <h1>Multimodal RAG Assistant</h1>
                <p>
                    Intelligent document question answering
                    over SBC and SPD files
                </p>
            </div>
            """
        )

        with gr.Row():

            with gr.Column(
                scale=1,
                min_width=320,
                elem_classes="sidebar"
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
                    <div class="sidebar-title"
                    style="margin-top:30px;">
                        Example Questions
                    </div>
                    """
                )

                clear_btn = gr.Button(
                    "Clear Conversation",
                    variant="secondary",
                    elem_classes="clear-btn"
                )

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

        gr.Markdown(
            """
---
Qdrant • ColQwen2.5 • OpenRouter • Gradio • Multimodal RAG
"""
        )

    try:

        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate",
                neutral_hue="gray"
            ),
            css=custom_css
        )

    finally:

        print(
            "\nShutting down Qdrant client...\n"
        )

        try:
            indexer.local_client.close()

        except Exception:
            pass


if __name__ == "__main__":

    force_reindex = (
        "--reindex" in sys.argv
        or "-r" in sys.argv
    )

    if force_reindex:
        print("Reindex mode activated\n")

    main(force_reindex=force_reindex)