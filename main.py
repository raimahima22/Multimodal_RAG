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

    print("Checking document index...\n")

    if force_reindex or indexer.is_collection_empty():
        print("Indexing documents...\n")
        indexer.index_all_data("data")
        print("Indexing completed!\n")
    else:
        print("Existing index found. Skipping indexing.\n")

    source_options = {
        "Both Documents": None,
        "SBC": "data/sbc.pdf",
        "SPD": "data/spd.pdf",
    }

    def answer_query(message, history, source_choice):

        if history is None:
            history = []

        if not message or not message.strip():
            return history, ""

        query = message.strip()

        source_filter = None

        if str(source_choice).lower() in ["sbc", "sbc.pdf"]:
            source_filter = "data/sbc.pdf"
        elif str(source_choice).lower() in ["spd", "spd.pdf"]:
            source_filter = "data/spd.pdf"

        try:

            hits = retriever.search(
                query,
                top_k=3,
                source_filter=source_filter
            )

            if not hits:
                bot_response = "No relevant information found in selected documents."

            else:

                context_hits = sorted(hits, key=lambda x: x.score, reverse=True)
                best_hit = context_hits[0]

                source = best_hit.payload.get("source", "Unknown")
                page = best_hit.payload.get("page_number", "N/A")

                answer = generator.generate_answer(query, context_hits)

                bot_response = f"""
### Retrieved Context
Source: {os.path.basename(source)}
Page: {page}
Scope: {source_choice}

---

### Answer
{answer}
"""

                save_to_history(query, source_choice, answer)

            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": bot_response})

            return history, ""

        except Exception as e:

            error_message = f"Error while generating response:\n{str(e)}"

            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": error_message})

            return history, ""

        finally:
            clear_page_cache()
            aggressive_cleanup()

    # ================= UI (SAFE VERSION) =================
    with gr.Blocks(title="Multimodal RAG Assistant") as demo:

        gr.Markdown("# Multimodal RAG Assistant")
        gr.Markdown("Document QA over SBC & SPD")

        with gr.Row():

            with gr.Column(scale=1):

                gr.Markdown("### Document Filter")

                source_dropdown = gr.Dropdown(
                    choices=list(source_options.keys()),
                    value="Both Documents"
                )

                clear_btn = gr.Button("Clear Chat")

                gr.Markdown("### Example Questions")

            with gr.Column(scale=4):

                chatbot = gr.Chatbot(
                    type="messages",
                    height=700
                )

                msg = gr.Textbox(
                    placeholder="Ask a question...",
                    lines=2
                )

                submit_btn = gr.Button("Send")

                gr.Examples(
                    examples=[
                        ["What is deductible?", "SPD"],
                        ["What is covered?", "SBC"],
                        ["Compare both plans", "Both Documents"],
                    ],
                    inputs=[msg, source_dropdown]
                )

        # events
        msg.submit(answer_query, [msg, chatbot, source_dropdown], [chatbot, msg])
        submit_btn.click(answer_query, [msg, chatbot, source_dropdown], [chatbot, msg])

        clear_btn.click(lambda: [], None, chatbot)

    # ================= FIXED LAUNCH =================
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
    main(force_reindex)