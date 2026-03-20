import os
import pandas as pd

os.makedirs("reports/metrics", exist_ok=True)

def main():
    summary_rows = [
        {
            "week": "Week 5",
            "model": "best_model.pkl",
            "accuracy": 0.9017,
            "macro_f1": 0.9015,
            "notes": "Reproducible evaluation baseline"
        },
        {
            "week": "Week 7",
            "model": "Experiment comparison",
            "accuracy": None,
            "macro_f1": None,
            "notes": "Controlled TF-IDF/model tuning"
        },
        {
            "week": "Week 8",
            "model": "TF-IDF + LinearSVC",
            "accuracy": None,
            "macro_f1": None,
            "notes": "Final selected model (fill values after run)"
        }
    ]

    df = pd.DataFrame(summary_rows)
    df.to_csv("reports/metrics/week8_experiment_summary.csv", index=False)
    print(df.to_string(index=False))
    print("\nSaved: reports/metrics/week8_experiment_summary.csv")

if __name__ == "__main__":
    main()