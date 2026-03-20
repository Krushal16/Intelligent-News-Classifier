# app/app.py
import sys
import streamlit as st
import joblib
from pathlib import Path

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
    "World leaders met in Brussels to discuss international trade and climate policy.",
    "The team secured a dramatic victory after scoring in the final minute.",
    "The company reported strong earnings and higher-than-expected quarterly revenue.",
    "Researchers developed a new AI model for faster language understanding."
]

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None

st.title("Intelligent News Classifier")
st.caption("AG News category prediction demo")

model = load_model()

if model is None:
    st.error("Model not found. Please train or copy best_model.pkl first.")
else:
    st.caption(f"Loaded model: {MODEL_PATH}")

    selected = st.selectbox("Choose an example or enter your own text:", [""] + EXAMPLES)
    user_input = st.text_area("News text", value=selected, height=180)

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