from pathlib import Path
import os
import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
DB_URL = os.getenv("POWERLIFT_DB_URL", "postgresql+psycopg2://powerlift_user:powerlift_pass@localhost:5432/powerlift_db")
engine = create_engine(DB_URL)

files = {
    "videos": ROOT / "dataset" / "annotations" / "video_labels.csv",
    "analysis_results": ROOT / "dataset" / "annotations" / "program_results.csv",
    "rep_results": ROOT / "dataset" / "annotations" / "rep_annotations.csv",
}

for table, path in files.items():
    df = pd.read_csv(path)
    df.to_sql(table, engine, if_exists="replace", index=False)
    print(f"loaded {len(df)} rows into {table}")
