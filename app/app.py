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
LABEL_MAP = {1:"World", 2:"Sports", 3:"Business", 4:"Sci/Tech"}

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None

st.title("Intelligent News Classifier (AG News)")

model = load_model()
if model is None:
    st.error("best_model.pkl not found. Run: python run_pipeline.py")
else:
    st.caption(f"Model loaded: {MODEL_PATH}")

    example1 = "Stock markets rose today after strong earnings from major companies."
    example2 = "The team secured a late victory in the championship match."
    example3 = "Scientists announced a new breakthrough in AI research."

    choice = st.selectbox("Load an example (optional):", ["(none)", "Example 1", "Example 2", "Example 3"])
    if choice == "Example 1":
        text = st.text_area("Enter news text:", value=example1, height=180)
    elif choice == "Example 2":
        text = st.text_area("Enter news text:", value=example2, height=180)
    elif choice == "Example 3":
        text = st.text_area("Enter news text:", value=example3, height=180)
    else:
        text = st.text_area("Enter news text:", height=180)

    if st.button("Predict"):
        if not text.strip():
            st.warning("Enter some text.")
        else:
            pred_id = model.predict([clean_text(text)])[0]
            pred_name = LABEL_MAP.get(pred_id, str(pred_id))

            msg = f"Predicted category: {pred_name}"

            if hasattr(model, "predict_proba"):
                conf = model.predict_proba([clean_text(text)]).max()
                msg += f" (confidence: {conf:.2f})"

            st.success(msg)