# src/error_analysis.py
import os
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import get_train_val_test
from src.preprocess import apply_cleaning

LABEL_MAP = {1:"World", 2:"Sports", 3:"Business", 4:"Sci/Tech"}

def main():
    # Load data (we use test split for error analysis)
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test()
    X_test_clean = apply_cleaning(X_test)

    # Load best model
    model = joblib.load("models/best_model.pkl")

    # Predict
    y_pred = model.predict(X_test_clean)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # Save metrics
    pd.DataFrame([{
        "accuracy": acc,
        "macro_f1": macro_f1
    }]).to_csv("reports/metrics/week5_best_model_metrics.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Best Model (Week 5)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("reports/figures/week5_best_model_cm.png")
    plt.close()

    # Misclassified samples
    mis_idx = [i for i, (t, p) in enumerate(zip(y_test, y_pred)) if t != p]

    rows = []
    for i in mis_idx[:50]:  # take first 50 misclassifications
        text = X_test.iloc[i]
        rows.append({
            "true_label_id": int(y_test.iloc[i]) if hasattr(y_test, "iloc") else int(y_test[i]),
            "pred_label_id": int(y_pred[i]),
            "true_label": LABEL_MAP.get(int(y_test.iloc[i]) if hasattr(y_test, "iloc") else int(y_test[i]), "NA"),
            "pred_label": LABEL_MAP.get(int(y_pred[i]), "NA"),
            "text_snippet": (str(text)[:200] + "...")
        })

    pd.DataFrame(rows).to_csv("reports/metrics/week5_misclassified_samples.csv", index=False)

    print("Saved:")
    print("- reports/metrics/week5_best_model_metrics.csv")
    print("- reports/figures/week5_best_model_cm.png")
    print("- reports/metrics/week5_misclassified_samples.csv")

if __name__ == "__main__":
    main()
