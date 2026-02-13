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
    st.error("best_model.pkl not found. Run: python src/tune.py (or python run_pipeline.py) first.")
else:
    text = st.text_area("Enter news text:", height=200)
    if st.button("Predict"):
        if not text.strip():
            st.warning("Enter some text.")
        else:
            pred = model.predict([clean_text(text)])[0]
            st.success(f"Predicted category: {LABEL_MAP.get(pred, pred)}")