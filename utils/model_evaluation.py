# model_evaluation.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef
)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.cuda.amp import autocast  # AMP for faster inference on GPU
import multiprocessing

# ---------------------------------------------------------------------------- #
MODEL_TRAINING_CONFIGS = {
    "ai4bharat/indic-bert": {"max_len": 128, "batch_size": 16},
    "ai4bharat/IndicBERTv2-SS": {"max_len": 128, "batch_size": 16},
    "xlm-roberta-base": {"max_len": 128, "batch_size": 8},
    "bert-base-multilingual-cased": {"max_len": 128, "batch_size": 16},
    "prajjwal1/bert-mini": {"max_len": 128, "batch_size": 32},
    "prajjwal1/bert-tiny": {"max_len": 128, "batch_size": 48},
    "distilbert-base-multilingual-cased": {"max_len": 128, "batch_size": 16},
}

BASE_MODEL_DIR = "models"

FORCE_SLOW_TOKENIZER = {
    "ai4bharat/indic-bert",
    "xlm-roberta-base",
}

TARGET_NAMES = ["Non-Toxic", "Toxic"]

# ---------------------------------------------------------------------------- #
class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ---------------------------------------------------------------------------- #
def get_num_workers(max_workers=4):
    # Windows: multiprocessing in Streamlit often causes freezes. Use 0 workers (main process).
    if os.name == "nt":
        return 0
    try:
        return min(max_workers, max(1, multiprocessing.cpu_count() - 1))
    except Exception:
        return 0

# ---------------------------------------------------------------------------- #
def render_model_evaluation_ui():
    st.title("📊 Performance Evaluation")

    st.markdown("Select the model and evaluation CSV (default: data/training/splits/test.csv)")
    default_test = "data/training/splits/test.csv"
    fallback_test = "data/training/splits/val.csv"
    legacy = "data/training/binary/dataset_binary.csv"

    test_path = st.text_input("Evaluation CSV path:",
                              value=default_test if os.path.exists(default_test) else (fallback_test if os.path.exists(fallback_test) else legacy))
    if not os.path.exists(test_path):
        st.error(f"Evaluation CSV not found: {test_path}")
        return

    # Check for trained models
    available_models = []
    for model_name in MODEL_TRAINING_CONFIGS.keys():
        trained_path = os.path.join(BASE_MODEL_DIR, model_name.replace("/", "_"), "trained_model")
        if os.path.exists(trained_path):
            available_models.append((model_name, trained_path))

    if not available_models:
        st.error("❌ No trained models found. Train a model first.")
        return

    model_choice = st.selectbox(
        "🔍 Select Model for Evaluation:",
        available_models,
        format_func=lambda x: f"{x[0]} (trained)"
    )
    model_name, model_path = model_choice
    config = MODEL_TRAINING_CONFIGS[model_name]

    if not st.button("🚀 Start Evaluation"):
        return

    # Load tokenizer & model
    if model_name in FORCE_SLOW_TOKENIZER:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        except Exception as e:
            st.info(f"Fast tokenizer unavailable for {model_name}; using slow tokenizer. Details: {str(e)}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model.to(device)
    model.eval()
    st.info(f"🔎 Loaded `{model_name}` for Binary Classification (Non-Toxic vs Toxic).")

    # Load evaluation CSV
    df = pd.read_csv(test_path)
    if 'Text' not in df.columns or 'Toxic_flag' not in df.columns:
        st.error("Evaluation CSV must contain 'Text' and 'Toxic_flag'.")
        return

    # Show counts and deduplication info
    st.write("🔢 Eval counts (before dedupe):")
    st.write(df['Toxic_flag'].value_counts())
    dup = df.duplicated(subset=['Text']).sum()
    if dup > 0:
        st.warning(f"⚠️ Evaluation CSV contains {dup} duplicate Text rows. Duplicates should be avoided in test set.")

    # Use the test set as-is (faithful evaluation)
    df_eval = df.copy()
    df_eval['label'] = df_eval['Toxic_flag'].apply(lambda x: 1 if x else 0)
    df_eval = df_eval.dropna(subset=['label'])
    texts = df_eval['Text'].tolist()
    labels = df_eval['label'].astype(int).tolist()

    # DataLoader
    val_dataset = ToxicDataset(texts, labels, tokenizer, max_len=config["max_len"])
    num_workers = get_num_workers(max_workers=4)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers
    )

    preds, true_labels_list, logits_all = [], [], []

    # inference (wrapped in try/except to avoid silent crashes)
    with torch.inference_mode():
        progress = st.progress(0)
        total_batches = len(val_loader)
        processed = 0
        try:
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels_batch = batch['labels'].to(device, non_blocking=True)

                if use_amp:
                    with autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs.logits
                logits_all.extend(logits.float().cpu().numpy())
                true_labels_list.extend(labels_batch.cpu().numpy())
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

                processed += 1
                progress.progress(min(processed / total_batches, 1.0))
        except Exception as e:
            st.error(f"⚠️ Inference error during evaluation: {e}")

    # Metrics (only if we have predictions)
    if not preds or not true_labels_list:
        st.error("No predictions were produced. Evaluation halted.")
        return

    st.subheader("📈 Basic Metrics")
    acc = accuracy_score(true_labels_list, preds)
    prec = precision_score(true_labels_list, preds, zero_division=0)
    rec = recall_score(true_labels_list, preds, zero_division=0)
    f1 = f1_score(true_labels_list, preds, zero_division=0)
    mcc = matthews_corrcoef(true_labels_list, preds)

    st.write(f"**Accuracy**: `{acc:.4f}`")
    st.write(f"**Precision**: `{prec:.4f}`")
    st.write(f"**Recall**: `{rec:.4f}`")
    st.write(f"**F1-score**: `{f1:.4f}`")
    st.write(f"**Matthews Correlation Coefficient**: `{mcc:.4f}`")

    try:
        logits_tensor = torch.tensor(logits_all, dtype=torch.float32)
        y_score_bin = torch.softmax(logits_tensor, dim=1).numpy()[:, 1]
        auroc = roc_auc_score(true_labels_list, y_score_bin)
        st.write(f"**AUROC**: `{auroc:.4f}`")
    except Exception as e:
        st.warning(f"AUROC computation failed. Reason: {e}")

    with st.expander("📌 Class-wise Report"):
        report_df = pd.DataFrame(classification_report(
            true_labels_list,
            preds,
            target_names=TARGET_NAMES,
            output_dict=True,
            zero_division=0
        )).transpose()
        st.dataframe(report_df.round(3))
        csv = report_df.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download Report as CSV", csv, "toxicity_evaluation_report.csv", "text/csv")

    with st.expander("📉 Confusion Matrix"):
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(true_labels_list, preds, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🚫 False Positives & Negatives"):
        cm = confusion_matrix(true_labels_list, preds, labels=[0, 1])
        fp_fn_data = []
        for i, label_name in enumerate(TARGET_NAMES):
            fn = cm[i].sum() - cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            tp = cm[i, i]
            fp_fn_data.append({"Class": label_name, "TP": int(tp), "FP": int(fp), "FN": int(fn)})
        st.dataframe(pd.DataFrame(fp_fn_data))
