import pandas as pd
import os

os.makedirs("reports/metrics", exist_ok=True)

def main():
    metrics_df = pd.read_csv("reports/metrics/week8_best_model_metrics.csv")
    confusion_df = pd.read_csv("reports/metrics/week8_confusion_pairs.csv")

    best_accuracy = metrics_df.loc[0, "accuracy"]
    best_macro_f1 = metrics_df.loc[0, "macro_f1"]

    top_confusions = confusion_df.head(5)

    summary_rows = [
        {"item": "Final model accuracy", "value": best_accuracy},
        {"item": "Final model macro_f1", "value": best_macro_f1},
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("reports/metrics/week9_results_summary.csv", index=False)

    print("=== Final Results Summary ===")
    print(summary_df.to_string(index=False))

    print("\nTop confusion pairs:")
    print(top_confusions.to_string(index=False))

if __name__ == "__main__":
    main()
