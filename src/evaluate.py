# src/evaluate.py
import os
import json
from typing import Dict

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(
    model_path: str,
    X_test,
    y_test,
    model_name: str,
    metrics_dir: str = "reports/metrics",
    figures_dir: str = "reports/figures"
) -> Dict[str, float]:
    """
    Load a saved model and evaluate it on test data.
    Saves metrics (JSON) and confusion matrix (PNG).
    Returns metrics dict.
    """
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Evaluating model: {model_name} from {model_path}")
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # per-class metrics
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision_per_class": prec.tolist(),
        "recall_per_class": rec.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist()
    }

    # Save metrics JSON
    metrics_path = os.path.join(metrics_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    fig_path = os.path.join(figures_dir, f"{model_name}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {fig_path}")

    # Print classification report to console (optional)
    print(classification_report(y_test, y_pred))

    return metrics
