import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import get_train_val_test
from src.preprocess import apply_cleaning

os.makedirs("reports/metrics", exist_ok=True)

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(
        "data/raw/train.csv",
        "data/raw/test.csv"
    )

    X_train_clean = apply_cleaning(X_train)
    X_val_clean = apply_cleaning(X_val)

    experiments = [
        {
            "name": "logreg_2000",
            "model": LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1),
            "tfidf": {"ngram_range": (1, 2), "max_features": 2000}
        },
        {
            "name": "logreg_3000",
            "model": LogisticRegression(max_iter=1000, C=2.0, n_jobs=-1),
            "tfidf": {"ngram_range": (1, 2), "max_features": 3000}
        },
        {
            "name": "svm_3000",
            "model": LinearSVC(C=1.0),
            "tfidf": {"ngram_range": (1, 2), "max_features": 3000}
        }
    ]

    rows = []

    for exp in experiments:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**exp["tfidf"])),
            ("clf", exp["model"])
        ])

        pipe.fit(X_train_clean, y_train)
        y_val_pred = pipe.predict(X_val_clean)

        acc = accuracy_score(y_val, y_val_pred)
        macro_f1 = f1_score(y_val, y_val_pred, average="macro")

        rows.append({
            "experiment": exp["name"],
            "model": exp["model"].__class__.__name__,
            "tfidf_params": str(exp["tfidf"]),
            "accuracy": acc,
            "macro_f1": macro_f1
        })

    results_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    results_df.to_csv("reports/metrics/week7_experiment_results.csv", index=False)

    print(results_df)
    print("\nSaved: reports/metrics/week7_experiment_results.csv")


if __name__ == "__main__":
    main()