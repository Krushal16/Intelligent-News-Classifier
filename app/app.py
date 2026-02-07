# app/app.py
import sys
import streamlit as st
import joblib
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
# Prefer insert at front so imports from project root take precedence
sys.path.insert(0, str(project_root))

from src.preprocess import clean_text

MODEL_PATH = "models/svm_tfidf.pkl"  # or logreg_tfidf.pkl

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None

def main():
    st.title("Intelligent News Classifier (AG News)")
    st.write("Paste a news headline or short article to get its predicted category.")

    model = load_model()
    if model is None:
        st.error("Model not found. Please run `python run_pipeline.py` first.")
        return

    user_text = st.text_area("Enter news text here:", height=200)

    if st.button("Predict Category"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(user_text)
            pred_id = model.predict([cleaned])[0]
            # If your labels are 1-4, use as is; if 0-3, adjust map accordingly
            label_name = LABEL_MAP.get(pred_id, str(pred_id))

            # predict_proba only works for LogisticRegression, not LinearSVC
            proba_text = ""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([cleaned]).max()
                proba_text = f" (confidence: {proba:.2f})"

            st.success(f"Predicted category: {label_name}{proba_text}")

if __name__ == "__main__":
    main()