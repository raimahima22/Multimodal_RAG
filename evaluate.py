import pandas as pd
import time
from pathlib import Path
from datetime import datetime

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator


class SPDEvaluator:
    def __init__(self):
        self.indexer = MultimodalIndexer(force_recreate=False)
        self.retriever = MultimodalRetriever(self.indexer)
        self.generator = MultimodalGenerator()

        #  semantic model (fast + good)
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")



    def compute_metrics(self, gt, pred):
        gt = gt.lower().strip()
        pred = pred.lower().strip()

        #  exact match
        exact = int(gt == pred)

        #  fuzzy match
        fuzzy = fuzz.token_set_ratio(gt, pred)

        #  semantic similarity
        emb1 = self.sim_model.encode(gt, convert_to_tensor=True)
        emb2 = self.sim_model.encode(pred, convert_to_tensor=True)
        semantic = float(util.cos_sim(emb1, emb2))

        #  final decision rule (tunable)
        answer_found = int(
            exact == 1 or
            fuzzy > 65 or
            semantic > 0.65
        )

        return exact, fuzzy, semantic, answer_found


    def evaluate(
        self,
        excel_path="/content/drive/MyDrive/Question_samples.xlsx",
        output_dir="/content/drive/MyDrive/evaluation_results2"
    ):
        print(f"Reading questions from: {excel_path}")

        df = pd.read_excel(excel_path)
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

            #compute metrics
            exact, fuzzy, semantic, found = self.compute_metrics(
                ground_truth, answer
            )

            results.append({
                "id": idx,
                "question": query,
                "ground_truth": ground_truth,
                "generated_answer": answer,

                #  metrics
                "exact_match": exact,
                "fuzzy_score": round(fuzzy, 2),
                "semantic_score": round(semantic, 4),
                "answer_found": found,

                # performance
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
        excel_path = output_dir / f"SPD_Evaluation_{timestamp}.xlsx"

        df_results.to_excel(excel_path, index=False)

        print(f"\nSaved to Drive: {excel_path}")

        # summary stats
        print("\n===== SUMMARY =====")
        print(f"Accuracy: {df_results['answer_found'].mean()*100:.2f}%")
        print(f"Avg Latency: {df_results['latency_seconds'].mean():.2f}s")

        return df_results


if __name__ == "__main__":
    evaluator = SPDEvaluator()
    evaluator.evaluate(
        excel_path="/content/drive/MyDrive/Question_samples.xlsx",
        output_dir="/content/drive/MyDrive/evaluation_results"
    )