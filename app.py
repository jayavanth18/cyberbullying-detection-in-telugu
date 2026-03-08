# Torch Patch for Streamlit import issues
import sys
if 'torch.classes' not in sys.modules:
    import types
    sys.modules['torch.classes'] = types.SimpleNamespace()
sys.modules['torch.classes'].__path__ = []

# 🌐 Core Dependencies
import streamlit as st
import os
import pandas as pd

# Application Modules
from utils import (
    data_summary,
    model_training,
    model_evaluation,
    predict,
    data_balancing,
)
# import model utils for potential health checks if needed (not mandatory here)
from utils.model_utils import load_model, load_tokenizer

# ============================================================
# 📂 Configuration Settings
# ============================================================
# 🔒 Primary dataset configuration
CLEAN_DATA_PATH = r"C:\Users\jayav\Downloads\Save\College\Capstone\cyberbullying-telugu\data\toxic_data_cleaned_large.csv"

# Directory initialization
os.makedirs("models", exist_ok=True)

# Set some environment / torch tuning (safe)
import torch
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "4")
_os.environ.setdefault("MKL_NUM_THREADS", "4")
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

# ============================================================
# 🎨 Advanced UI Configuration
# ============================================================
st.set_page_config(
    page_title="Cyberbullying Detection in Telugu",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- REPLACED CSS BLOCK: keeps the look but fixes contrast/readability ---
st.markdown("""
<style>
    /* Page hero */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.25);
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.2rem !important;
        margin: 0 !important;
    }

    /* Feature cards - ensure good contrast and readable text */
    .feature-card {
        background: linear-gradient(180deg, #f6f8fa 0%, #eef1f5 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid rgba(44,62,80,0.08);
        margin: 1rem 0;
        box-shadow: 0 6px 18px rgba(10, 20, 30, 0.07);
        color: #23374b !important;               /* ensure body text is dark */
    }
    .feature-card h3 {
        color: #102a43 !important;               /* strong dark header */
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.75rem !important;
        text-shadow: none !important;
    }
    .feature-card ul {
        color: #2c3e50 !important;
        font-size: 1rem !important;
        margin-left: 1rem !important;
    }
    .feature-card li {
        margin-bottom: 0.5rem !important;
        color: #2c3e50 !important;
    }
    .feature-card strong { color: #123047 !important; }

    /* Metric containers */
    .metric-container {
        background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(10, 20, 30, 0.06);
        text-align: center;
        border: 1px solid rgba(0,0,0,0.04);
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: #1f2d3d !important;
    }
    .metric-container h2 { font-size: 2.2rem !important; margin:0 0 .25rem 0 !important; }
    .metric-container h3 { color:#102a43 !important; font-size:1.05rem !important; margin:0 !important; }
    .metric-container p { color:#5a6b7a !important; font-size:0.9rem !important; margin:0 !important; }

    /* Small responsive tweak to avoid blown-out white cards on dark theme */
    .stApp .block-container { padding-top: 1.5rem; padding-left: 2rem; padding-right: 2rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🎛️ Navigation Panel
# ============================================================
st.sidebar.markdown("### 🎛️ Control Panel")
st.sidebar.markdown("---")

navigation_options = [
    ("🏛️ Dashboard", "dashboard"),
    ("📋 Data Analytics", "analytics"),
    ("⚡ Data Balancing", "processing"),
    ("🤖 Model Training", "training"),
    ("📊 Performance Metrics", "evaluation"),
    ("🔍 Prediction", "prediction"),
    ("🎥 YouTube Comments", "youtube")
]

selected_page = st.sidebar.selectbox(
    "Choose Module:",
    options=[opt[1] for opt in navigation_options],
    format_func=lambda x: next(opt[0] for opt in navigation_options if opt[1] == x),
    index=0
)

# Add sidebar info
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: If you've already built a model, you can easily check its performance or make predictions with it directly.")

# ============================================================
# 🏛️ Main Dashboard
# ============================================================
if selected_page == "dashboard":
    # Hero Section
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Cyberbullying Detection in Telugu</h1>
        <p style='font-size: 1.2em; margin-top: 1rem;'>
            Intelligent content moderation system powered by advanced machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Features Section
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Core Capabilities</h3>
            <ul>
                <li><strong>Real-time Analysis</strong> - Instant content evaluation</li>
                <li><strong>Cyberbullying Detection</strong> - Comprehensive toxicity categorization</li>
                <li><strong>Telugu Language Support</strong> - Native language processing</li>
                <li><strong>Scalable Architecture</strong> - Enterprise-ready deployment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔧 Technical Features</h3>
            <ul>
                <li><strong>Advanced ML Models</strong> - State-of-the-art algorithms</li>
                <li><strong>Performance Analytics</strong> - Comprehensive evaluation metrics</li>
                <li><strong>Data Visualization</strong> - Interactive insights dashboard</li>
                <li><strong>Model Optimization</strong> - Continuous improvement pipeline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 📋 Data Analytics Module
# ============================================================
elif selected_page == "analytics":
    data_summary.render_data_summary_ui(CLEAN_DATA_PATH)

# ============================================================
# ⚡ Data Processing Module
# ============================================================
elif selected_page == "processing":
    data_balancing.render_data_balancing_ui()

# ============================================================
# 🤖 Model Development Center
# ============================================================
elif selected_page == "training":
    st.markdown("# 🤖 Model Training")
    st.markdown("### Machine Learning Model Training and Optimization")
    st.markdown("---")
    model_training.render_model_training_ui()

# ============================================================
# 📊 Performance Metrics Dashboard
# ============================================================
elif selected_page == "evaluation":
    model_evaluation.render_model_evaluation_ui()

# ============================================================
# 🔍 Content Scanner Interface
# ============================================================
elif selected_page == "prediction":
    st.markdown("# 🔍 Content Scanner Interface")
    st.markdown("### 🔮 Real-time content analysis and toxicity detection")
    st.markdown("---")
    predict.render_prediction_ui()

elif selected_page == "youtube":
    from utils.youtube_comment_predictor import render_youtube_ui
    st.markdown("# 🎥 YouTube Comment Analysis")
    st.markdown("### 🔮 Detect cyberbullying in YouTube video comments")
    st.markdown("---")
    render_youtube_ui()

# ============================================================
# 📄 Footer
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>
Cyberbullying Detection in Telugu<br>
Developed for content moderation<br>
© 2025 - Academic Research Project
</small>
""", unsafe_allow_html=True)
