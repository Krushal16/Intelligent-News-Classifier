import pandas as pd
import os

os.makedirs("reports/metrics", exist_ok=True)

def main():
    df = pd.read_csv("reports/metrics/week7_experiment_results.csv")
    df = df.sort_values("macro_f1", ascending=False)

    best_row = df.iloc[0]
    summary = pd.DataFrame([{
        "best_experiment": best_row["experiment"],
        "best_model": best_row["model"],
        "best_accuracy": best_row["accuracy"],
        "best_macro_f1": best_row["macro_f1"]
    }])

    summary.to_csv("reports/metrics/week7_best_experiment_summary.csv", index=False)

    print("Best experiment summary:")
    print(summary)
    print("\nSaved: reports/metrics/week7_best_experiment_summary.csv")


if __name__ == "__main__":
    main()