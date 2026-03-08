# 🛡️ Cyberbullying Detection in Telugu using BERT-based Transformer Models

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Backend-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**An end-to-end NLP system for detecting cyberbullying in Telugu social media comments using Transformer-based language models.**

🎥 **[Watch Demo Video](https://youtu.be/oQMxZZ9ggx8)**

</div>

---

# 📌 Overview

Cyberbullying and abusive language have become a significant problem on social media platforms. While many automated moderation systems exist for English, **regional languages such as Telugu remain underrepresented in NLP research and moderation tools.**

Telugu social media content presents unique challenges:

- Mixed scripts (Telugu + Latin)
- Informal transliterations (Tenglish)
- Slang and creative spellings
- Limited labeled datasets for training models

This project develops a **machine learning pipeline for detecting cyberbullying in Telugu comments**, combining modern **Transformer-based NLP models** with a **custom curated dataset** and an **interactive prediction interface**.

The system aims to assist in **automated moderation and toxicity detection for Telugu digital platforms.**

---

# 🎯 Project Objectives

The primary goals of this project are:

- Build a **curated Telugu cyberbullying dataset**
- Design a **clean preprocessing and dataset preparation pipeline**
- Train and evaluate **multiple BERT-based transformer models**
- Compare model performance across architectures
- Develop a **real-time prediction system using Streamlit**
- Demonstrate practical applications such as **YouTube comment moderation**

---

# 🧠 Methodology

The system follows a modular NLP pipeline:

```
Raw Social Media Comments
        │
        ▼
Data Cleaning & Normalization
        │
        ▼
Dataset Balancing
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Transformer Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Real-Time Prediction Interface
```

## Key steps include:

### Data Preprocessing

The preprocessing pipeline standardizes noisy social media text using:

- Unicode normalization (NFC)
- Removal of URLs, emails, emojis, and noise tokens
- Handling mixed Telugu-English text
- Deduplication to prevent training leakage

### Dataset Balancing

Social media datasets often contain **more non-toxic comments than toxic ones**.  
To prevent bias toward the majority class, the dataset is balanced using **controlled undersampling techniques**.

### Model Training

Multiple Transformer models are fine-tuned for binary classification:

```
Toxic (Cyberbullying)
Non-Toxic
```

Fine-tuning uses:

- Cross-Entropy Loss
- AdamW optimizer
- Tokenized input sequences
- Pretrained contextual embeddings

---

# 🤖 Models Evaluated

The project benchmarks several **BERT-family transformer models**:

| Model | Description |
|------|-------------|
| **IndicBERT** | Pretrained on multiple Indic languages |
| **IndicBERTv2** | Improved Indic multilingual transformer |
| **Multilingual BERT (mBERT)** | Trained on 100+ languages |
| **DistilBERT** | Lightweight distilled version of BERT |
| **XLM-RoBERTa** | Large multilingual transformer trained on massive corpora |
| **BERT-Mini / Tiny** | Compact models for faster inference |

These models allow comparison between:

- multilingual transformers
- Indic-specific models
- lightweight deployment models

---

# 📊 Evaluation Metrics

Model performance is evaluated using several classification metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Matthews Correlation Coefficient (MCC)**
- **AUROC**

These metrics provide a balanced evaluation for **binary toxicity detection tasks**.

---

# 📈 Results & Visualizations

Model performance and training behaviour were analyzed using multiple visualizations.

<div align="center">

### Proposed Pipeline
<img width="1447" height="84" alt="Proposed System" src="https://github.com/user-attachments/assets/f9eea915-1a7e-4e3a-99ce-f70c0173b34d" />


### Training Loss
<img width="2518" height="1920" alt="Training Loss" src="https://github.com/user-attachments/assets/f219cb44-7aab-4d62-a34c-43ab4f486b1e" />


### Confusion Matrices
<img width="4085" height="3084" alt="confusion_matrices" src="https://github.com/user-attachments/assets/833b1e54-b043-4e18-8190-1e83cfb28219" />


</div>

---

# 🖥 Application Interface

A **Streamlit-based application** is provided for interactive predictions.

The interface supports:

- Real-time text classification
- Batch CSV prediction
- Model selection
- Visualization of prediction confidence

This allows researchers and developers to **experiment with cyberbullying detection in real time.**

---

# 📂 Project Structure

```
cyberbullying-detection-in-telugu
│
├── app.py                    # Streamlit application entry point
├── README.md                 # Project documentation
├── LICENSE
│
├── utils                     # Utility modules
│   ├── model_utils.py
│   ├── data_balancing.py
│   ├── data_summary.py
│   ├── model_evaluation.py
│   └── recent_words.txt
│
└── images                    # Documentation images
    ├── proposed_system.png
    ├── training_loss.png
    └── confusion_matrices.png
```

Note:

Dataset and training scripts are **temporarily excluded** from this repository due to an ongoing research publication.

---

# 🚀 Getting Started

## Prerequisites

- Python 3.9+
- PyTorch
- HuggingFace Transformers
- Streamlit

---

## Installation

```bash
# Clone repository
git clone https://github.com/jayavanth18/cyberbullying-detection-in-telugu.git
cd cyberbullying-detection-in-telugu

# Create virtual environment (Windows)
python -m venv venv

# Activate environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

---

# ⚠️ Dataset Availability

The dataset used in this project is **not included in this repository**.

Reason:

- Dataset is part of an **ongoing research publication**
- Data release will follow **paper acceptance**

After publication, the dataset and additional training utilities will be made available.

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Contributors

<div align="center">

### 🔹 A. Jayavanth

[![GitHub](https://img.shields.io/badge/GitHub-jayavanth18-181717?style=for-the-badge&logo=github)](https://github.com/jayavanth18)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jayavanth18-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/jayavanth18)

---

### 🔹 Yellatp

[![GitHub](https://img.shields.io/badge/GitHub-yellatp-181717?style=for-the-badge&logo=github)](https://github.com/yellatp)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-yellatp-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/yellatp)

---

### 🔹 Mohd Rizwan

[![GitHub](https://img.shields.io/badge/GitHub-mohdrizwan11-181717?style=for-the-badge&logo=github)](https://github.com/mohdrizwan11)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-mohdrizwan11-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/mohdrizwan11)

</div>

---

⭐ If you found this repository useful, consider giving it a **star**!
