import pandas as pd
import os

os.makedirs("reports/metrics", exist_ok=True)

def main():
    df = pd.read_csv("reports/metrics/week6_misclassified_samples.csv")

    pair_counts = (
        df.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print("Top confusion pairs:")
    print(pair_counts.head(10))

    pair_counts.to_csv("reports/metrics/week7_error_patterns.csv", index=False)
    print("\nSaved: reports/metrics/week7_error_patterns.csv")


if __name__ == "__main__":
    main()
