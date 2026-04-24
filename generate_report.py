import pandas as pd
from datetime import datetime
from pathlib import Path

df = pd.read_excel("/content/drive/MyDrive/evaluation_results/SPD_Evaluation_xxxx.xlsx")

report = f"""
# SPD Multimodal RAG - Evaluation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Total Questions:** {len(df)}

## Performance
- Accuracy: {df['answer_found'].mean()*100:.2f}%
- Avg Latency: {df['latency_seconds'].mean():.2f}s
- Fastest: {df['latency_seconds'].min():.2f}s
- Slowest: {df['latency_seconds'].max():.2f}s

## Quality Metrics
- Avg Fuzzy Score: {df['fuzzy_score'].mean():.2f}
- Avg Semantic Score: {df['semantic_score'].mean():.3f}

## Notes
- Check rows where answer_found = 0
- Improve retrieval if semantic < 0.6
"""

Path("/content/drive/MyDrive/evaluation_results/SPD_FINAL_REPORT.md").write_text(report)

print("Report saved.")