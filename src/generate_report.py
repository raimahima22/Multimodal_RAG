import pandas as pd
from datetime import datetime
from pathlib import Path

# Load latest file manually or specify path
df = pd.read_excel("/content/drive/MyDrive/evaluation_results/SPD_Evaluation_xxxx.xlsx")

report = f"""
# SPD Multimodal RAG - Evaluation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Total Questions:** {len(df)}

## Summary
- Avg Latency: {df['latency_seconds'].mean():.2f}s
- Fastest: {df['latency_seconds'].min():.2f}s
- Slowest: {df['latency_seconds'].max():.2f}s

## Notes
- Review low-quality answers manually
- Add expected page column later
- Improve OCR if needed
"""

report_path = "/content/drive/MyDrive/evaluation_results/SPD_FINAL_REPORT.md"
Path(report_path).write_text(report)

print(f" Report saved: {report_path}")