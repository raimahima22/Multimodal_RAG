import pandas as pd
import time
from pathlib import Path
from datetime import datetime

from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator


class SPDEvaluator:
    def __init__(self):
        self.indexer = MultimodalIndexer(force_recreate=False)
        self.retriever = MultimodalRetriever(self.indexer)
        self.generator = MultimodalGenerator()

    def evaluate(
        self,
        excel_path="/content/drive/MyDrive/Question_samples.xlsx",
        output_dir="/content/drive/MyDrive/evaluation_results"
    ):
        print(f" Reading questions from: {excel_path}")

        df = pd.read_excel(excel_path)

        print(f" Loaded {len(df)} rows")
        print(f" Columns found: {df.columns.tolist()}")

        # 🔥 Force correct column names (adjust if needed)
        df.columns = ["question", "ground_truth"]

        results = []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in df.iterrows():
            query = str(row["question"]).strip()
            ground_truth = str(row["ground_truth"]).strip()

            print(f"[{idx+1}/{len(df)}] {query[:80]}...")

            start_time = time.time()

            hits = self.retriever.search(
                query_text=query,
                top_k=12,
                source_filter="data/spd.pdf",
                generator=self.generator
            )

            answer = self.generator.generate_answer(query, hits[:3])

            latency = time.time() - start_time

            results.append({
                "id": idx,
                "question": query,
                "ground_truth": ground_truth,
                "generated_answer": answer,
                "latency_seconds": round(latency, 2),
                "retrieved_pages": ", ".join(
                    [f"Page {p.payload.get('page_number')}" for p in hits[:5]]
                ),
                "timestamp": datetime.now().isoformat()
            })

        return self._save_results(results, output_dir)

    def _save_results(self, results, output_dir):
        df_results = pd.DataFrame(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ✅ SAVE DIRECTLY TO DRIVE
        excel_path = output_dir / f"SPD_Evaluation_{timestamp}.xlsx"

        df_results.to_excel(excel_path, index=False)

        print(f"\n Saved to Drive: {excel_path}")

        return df_results


# ================= RUN =================
if __name__ == "__main__":
    evaluator = SPDEvaluator()
    results_df = evaluator.evaluate()