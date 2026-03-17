import pandas as pd


def main():
    df = pd.read_csv("reports/metrics/week6_misclassified_samples.csv")

    print("First 10 misclassified rows:")
    print(df.head(10))

    print("\nTrue label counts:")
    print(df["true_label"].value_counts())

    print("\nPredicted label counts:")
    print(df["pred_label"].value_counts())

    print("\nTop confusion pairs:")
    pair_counts = (
        df.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    print(pair_counts.head(10))


if __name__ == "__main__":
    main()
