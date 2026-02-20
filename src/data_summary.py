# src/data_summary.py
import os
import pandas as pd
from src.data_loader import load_ag_news_csv

def main():
    train_df = load_ag_news_csv("data/raw/train.csv")

    train_df["text_len_words"] = train_df["text"].str.split().str.len()
    summary = {
        "rows": [len(train_df)],
        "avg_words": [train_df["text_len_words"].mean()],
        "median_words": [train_df["text_len_words"].median()],
        "missing_text": [train_df["text"].isna().sum()]
    }

    os.makedirs("reports/metrics", exist_ok=True)
    pd.DataFrame(summary).to_csv("reports/metrics/week5_data_summary.csv", index=False)

    train_df["label"].value_counts().sort_index().to_csv("reports/metrics/week5_class_distribution.csv")

if __name__ == "__main__":
    main()
