# src/train.py
import os
import joblib
from typing import Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_logreg_pipeline(max_features: int = 5000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        ))
    ])


def build_svm_pipeline(max_features: int = 5000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features
        )),
        ("clf", LinearSVC())
    ])


def train_models(X_train, y_train, output_dir: str = "models") -> Dict[str, str]:
    """
    Train multiple models and save them to disk.
    Returns a dict: {model_name: model_path}
    """
    os.makedirs(output_dir, exist_ok=True)

    models = {
        "logreg": build_logreg_pipeline(),
        "svm": build_svm_pipeline()
    }

    saved_paths: Dict[str, str] = {}

    for name, pipeline in models.items():
        print(f"Training model: {name}")
        pipeline.fit(X_train, y_train)
        model_path = os.path.join(output_dir, f"{name}_tfidf.pkl")
        joblib.dump(pipeline, model_path)
        print(f"Saved {name} model to {model_path}")
        saved_paths[name] = model_path

    return saved_paths
