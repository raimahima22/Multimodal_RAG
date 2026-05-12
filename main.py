import sys
import gc
import torch
import gradio as gr
from datetime import datetime
import os
import json

from src.utils import clear_page_cache
from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator

HISTORY_FILE = "chat_history.json"


def save_to_history(query, source_input, answer):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    
    history.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "source": source_input,
        "answer": answer
    })
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(force_reindex: bool = False):
    print("Initializing Multimodal RAG System...\n")

    indexer = MultimodalIndexer(force_recreate=force_reindex)
    retriever = MultimodalRetriever(indexer)
    generator = MultimodalGenerator()

    print("Warming up model...")
    _ = retriever._extract_text_embedding("warmup query")
    print("✅ Ready!")

    # Indexing Phase
    print("--- Phase 1: Checking Index ---")
    if force_reindex or indexer.is_collection_empty():
        print("Indexing documents...")
        indexer.index_all_data("data")
    else:
        print("✅ Index already exists. Skipping indexing.")

    print("System is ready!\n")

    # Source Options
    source_options = {
        "Both Documents": None,
        "SBC": "data/sbc.pdf",
        "SPD": "data/spd.pdf"
    }

    # ====================== CHAT FUNCTION ======================
    def answer_query(message: str, history: list, source_choice: str):
        if not message or not message.strip():
            return history, ""

        query = message.strip()
        source_input = source_choice if source_choice != "Both Documents" else ""

        # Source filter logic (same as your original)
        if source_input.lower() in ["sbc", "sbc.pdf"]:
            source_filter = "data/sbc.pdf"
        elif source_input.lower() in ["spd", "spd.pdf"]:
            source_filter = "data/spd.pdf"
        else:
            source_filter = None

        try:
            hits = retriever.search(query, top_k=3, source_filter=source_filter)

            if not hits:
                response = "No relevant documents found."
            else:
                context_hits = sorted(hits, key=lambda x: x.score, reverse=True)
                best_hit = context_hits[0]

                source = best_hit.payload.get('source', 'Unknown')
                page = best_hit.payload.get('page_number', 'N/A')

                answer = generator.generate_answer(query, context_hits)

                response = f"""**Source**: {source} (Page {page})
**Filter**: {"Both documents" if source_filter is None else source_filter}

{answer}"""

                save_to_history(query, source_choice, answer)

            # Append bot response
            history.append((query, response))
            return history, ""

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            history.append((query, error_msg))
            return history, ""
        finally:
            clear_page_cache()
            aggressive_cleanup()

    # ====================== BEAUTIFUL GRADIO UI ======================
    with gr.Blocks(
        title="Multimodal RAG Chat",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")
    ) as demo:
        
        gr.Markdown("# 🧠 Multimodal RAG Chatbot")
        gr.Markdown("**Intelligent Q&A** over SBC & SPD documents with source filtering.")

        with gr.Row():
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    height=680,
                    type="tuples",
                    show_copy_button=True,
                    show_share_button=False,
                    allow_tags=False,
                    avatar_images=["👤", "🤖"],
                    bubble_full_width=False,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask your question here... (Press Enter to send)",
                        label=None,
                        scale=8,
                        container=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Filter")
                source_dropdown = gr.Dropdown(
                    choices=list(source_options.keys()),
                    value="Both Documents",
                    label="Search Scope",
                    info="Limit search to one document"
                )

                gr.Markdown("### Quick Examples")
                gr.Examples(
                    examples=[
                        ["What is the main objective of SBC?", "SBC"],
                        ["What amount is required for Embedded deductible for Gold PPO plan?", "SPD"],
                        ["Compare SBC and SPD requirements", "Both Documents"],
                    ],
                    inputs=[msg, source_dropdown],
                    cache_examples=False
                )

                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        # ====================== EVENT LOGIC (Fixed) ======================
        def user_submit(message, history):
            if not message:
                return "", history
            # Add user message with placeholder for bot reply
            history = history + [(message, None)]
            return "", history

        # Submit using button or Enter key
        submit_event = msg.submit(
            fn=user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        ).then(
            fn=answer_query,
            inputs=[msg, chatbot, source_dropdown],  # msg is already cleared
            outputs=[chatbot, msg]
        )

        submit_btn.click(
            fn=user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        ).then(
            fn=answer_query,
            inputs=[msg, chatbot, source_dropdown],
            outputs=[chatbot, msg]
        )

        clear_btn.click(
            fn=lambda: [],
            inputs=None,
            outputs=chatbot,
            queue=False
        )

        gr.Markdown("---\nBuilt with ❤️ using Qdrant • Gradio • Multimodal RAG")

    try:
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    finally:
        print("Shutting down Qdrant client...")
        try:
            indexer.local_client.close()
        except:
            pass


if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
    if force_reindex:
        print(" Reindex mode activated\n")
    main(force_reindex=force_reindex)