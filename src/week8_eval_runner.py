import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.data_loader import get_train_val_test
from src.preprocess import apply_cleaning

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

os.makedirs("reports/metrics", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(
        "data/raw/train.csv",
        "data/raw/test.csv"
    )

    X_train_clean = apply_cleaning(X_train)
    X_test_clean = apply_cleaning(X_test)

    best_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            min_df=2,
            max_df=0.95
        )),
        ("clf", LinearSVC(C=1.0))
    ])

    best_pipeline.fit(X_train_clean, y_train)
    y_pred = best_pipeline.predict(X_test_clean)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    metrics_df = pd.DataFrame([{
        "model": "TF-IDF + LinearSVC",
        "accuracy": acc,
        "macro_f1": macro_f1
    }])

    metrics_df.to_csv("reports/metrics/week8_best_model_metrics.csv", index=False)

    print(metrics_df)
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(LABEL_MAP.keys())
    tick_labels = [LABEL_MAP[c] for c in classes]

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(classes)) + 0.5, tick_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(classes)) + 0.5, tick_labels, rotation=0)
    plt.title("Confusion Matrix - Best Model (Week 8)")
    plt.tight_layout()
    plt.savefig("reports/figures/week8_confusion_matrix_best_model.png", bbox_inches="tight")
    plt.show()

    mis_idx = np.where(pd.Series(y_test).to_numpy() != y_pred)[0]
    rows = []
    for i in mis_idx[:50]:
        true_id = int(y_test.iloc[i]) if hasattr(y_test, "iloc") else int(y_test[i])
        pred_id = int(y_pred[i])
        rows.append({
            "true_id": true_id,
            "pred_id": pred_id,
            "true_label": LABEL_MAP.get(true_id, str(true_id)),
            "pred_label": LABEL_MAP.get(pred_id, str(pred_id)),
            "text_snippet": str(X_test.iloc[i])[:250]
        })

    pd.DataFrame(rows).to_csv("reports/metrics/week8_misclassified_samples.csv", index=False)

    joblib.dump(best_pipeline, "models/best_model.pkl")
    print("\nSaved final best_model.pkl and Week 8 outputs.")

if __name__ == "__main__":
    main()