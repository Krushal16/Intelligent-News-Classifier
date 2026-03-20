import pandas as pd
import os

os.makedirs("reports/metrics", exist_ok=True)

def main():
    df = pd.read_csv("reports/metrics/week8_misclassified_samples.csv")

    print("Total sampled misclassified rows:", len(df))

    print("\nTrue label distribution in errors:")
    print(df["true_label"].value_counts())

    print("\nPredicted label distribution in errors:")
    print(df["pred_label"].value_counts())

    pair_counts = (
        df.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print("\nTop confusion pairs:")
    print(pair_counts.head(10))

    pair_counts.to_csv("reports/metrics/week8_confusion_pairs.csv", index=False)
    print("\nSaved: reports/metrics/week8_confusion_pairs.csv")

if __name__ == "__main__":
    main()
