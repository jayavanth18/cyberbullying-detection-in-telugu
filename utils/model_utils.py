# utils/model_utils.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Models that require slow tokenizer fallback for your tokenizers version
FORCE_SLOW_TOKENIZER = {
    "ai4bharat/indic-bert",
    "xlm-roberta-base",
    "ai4bharat/IndicBERTv2-SS"
}

@st.cache_resource
def load_tokenizer(model_path: str, model_id_for_config: str = None):
    use_slow = False
    if model_id_for_config and model_id_for_config in FORCE_SLOW_TOKENIZER:
        use_slow = True
    try:
        return AutoTokenizer.from_pretrained(model_path, use_fast=not use_slow, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)

@st.cache_resource
def load_model(model_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    return model
