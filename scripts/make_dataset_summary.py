from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
labels = pd.read_csv(ROOT / "dataset" / "annotations" / "video_labels.csv")
results = pd.read_csv(ROOT / "dataset" / "annotations" / "program_results.csv")
print("Videos by exercise:")
print(labels.groupby(["exercise", "quality_label"]).size())
print("\nProgram results:")
print(results[["filename", "manual_quality", "program_score", "program_status"]])
