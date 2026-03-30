# app/app.py
import sys
import streamlit as st
import joblib
from pathlib import Path

st.markdown("### How to use")
st.markdown("1. Choose an example or enter your own news text.")
st.markdown("2. Click Predict.")
st.markdown("3. The app returns one of four categories: World, Sports, Business, or Sci/Tech.")

project_root = Path(__file__).resolve().parents[1]
# Prefer insert at front so imports from project root take precedence
sys.path.insert(0, str(project_root))

from src.preprocess import clean_text

MODEL_PATH = "models/best_model.pkl"

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

EXAMPLES = [
    "Global leaders gathered in Geneva for international climate talks.",
    "The team won the championship after a dramatic penalty shootout.",
    "Apple reported record quarterly revenue driven by iPhone sales.",
    "Scientists developed a new AI model capable of predicting protein structures."
]

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None

st.set_page_config(page_title="Intelligent News Classifier", layout="centered")
st.title("Intelligent News Classifier")
st.caption("Classifies news articles into: World | Sports | Business | Sci/Tech")

model = load_model()

if model is None:
    st.error("Model not found. Please run the pipeline first.")
else:
    st.caption(f"Model loaded: {MODEL_PATH}")

    selected = st.selectbox("Choose an example (optional):", [""] + EXAMPLES)
    user_input = st.text_area("Enter news text:", value=selected, height=180)

    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(user_input)
            pred_id = model.predict([cleaned])[0]
            pred_name = LABEL_MAP.get(int(pred_id), str(pred_id))

            st.success(f"Predicted category: {pred_name}")

            if hasattr(model, "predict_proba"):
                conf = model.predict_proba([cleaned]).max()
                st.caption(f"Confidence: {conf:.2f}")

    st.markdown("---")
    st.caption("AML-2403 Capstone | Intelligent News Classifier | AG News Dataset")