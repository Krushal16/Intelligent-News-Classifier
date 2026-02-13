# src/tune.py
import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

from src.data_loader import get_train_val_test
from src.preprocess import apply_cleaning


def build_logreg(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=-1, C=C))
    ])

def build_svm(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
        ("clf", LinearSVC(C=C))
    ])

def score(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro")
    }

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test()
    X_train = apply_cleaning(X_train)
    X_val = apply_cleaning(X_val)
    X_test = apply_cleaning(X_test)

    C_values = [0.5, 1, 2, 4]

    results = []

    # Tune LogReg
    for C in C_values:
        m = build_logreg(C)
        m.fit(X_train, y_train)
        val_scores = score(m, X_val, y_val)
        results.append({"model":"logreg", "C":C, **val_scores})

    # Tune SVM
    for C in C_values:
        m = build_svm(C)
        m.fit(X_train, y_train)
        val_scores = score(m, X_val, y_val)
        results.append({"model":"svm", "C":C, **val_scores})

    df = pd.DataFrame(results).sort_values(["macro_f1","accuracy"], ascending=False)
    os.makedirs("reports/metrics", exist_ok=True)
    df.to_csv("reports/metrics/week4_val_tuning_results.csv", index=False)
    print(df.head(10))

    # Pick best row
    best = df.iloc[0].to_dict()
    print("Best config:", best)

    # Retrain best on train+val, evaluate on test
    X_train_full = X_train + X_val
    y_train_full = list(y_train) + list(y_val)

    if best["model"] == "logreg":
        best_model = build_logreg(best["C"])
    else:
        best_model = build_svm(best["C"])

    best_model.fit(X_train_full, y_train_full)

    test_scores = score(best_model, X_test, y_test)
    print("Test scores:", test_scores)

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    # Save test metrics
    pd.DataFrame([{
        "best_model": best["model"],
        "best_C": best["C"],
        **test_scores
    }]).to_csv("reports/metrics/week4_best_model_test_metrics.csv", index=False)

if __name__ == "__main__":
    main()