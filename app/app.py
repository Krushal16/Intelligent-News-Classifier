# app/app.py
import sys
import streamlit as st
import joblib
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
# Prefer insert at front so imports from project root take precedence
sys.path.insert(0, str(project_root))

from src.preprocess import clean_text

# Resolve model path relative to project root so it works regardless of current working directory
MODEL_PATH = project_root / "models" / "tfidf_logreg.pkl"

@st.cache_resource
def load_model():
    try:
        model = joblib.load(str(MODEL_PATH))
        return model
    except FileNotFoundError:
        return None

def main():
    st.title("Intelligent News Classifier (AG News)")
    st.write("Paste a news headline or short article, and the model will predict its category.")

    user_text = st.text_area("Enter news text here:", height=200)

    if st.button("Predict Category"):
        model = load_model()
        if model is None:
            st.error("Model not found. Please train and save the model first.")
        elif not user_text.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(user_text)
            pred = model.predict([cleaned])[0]
            proba = model.predict_proba([cleaned]).max()
            st.success(f"Predicted category: {pred} (confidence: {proba:.2f})")

if __name__ == "__main__":
    main()