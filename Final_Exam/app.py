import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import plotly.graph_objects as go
import pandas as pd

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1a1a2e;
    }

    .stApp {
        background-color: #eafaf1;
    }

    section[data-testid="stSidebar"] {
        background-color: #1a4731 !important;
        border-right: none;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #2e6b47 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #00ff88 !important;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        color: #666;
        font-size: 0.95rem;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 2rem;
    }

    .prediction-card {
        background: #ffffff;
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(46, 204, 113, 0.15);
        margin-bottom: 1rem;
    }

    .prediction-label {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: -0.5px;
        text-transform: uppercase;
    }

    .confidence-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        color: #666;
        margin-top: 0.3rem;
    }

    .recyclable-card {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }

    .recyclable {
        background: #eafaf1;
        border: 2px solid #2ecc71;
        color: #1e8449;
    }

    .non-recyclable {
        background: #fdf2f2;
        border: 2px solid #e74c3c;
        color: #c0392b;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-label {
        font-size: 0.72rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }

    .upload-zone {
        border: 2px dashed #ccc;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background: #ffffff;
    }

    .stButton > button {
        background-color: #2ecc71;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 0.6rem 2rem;
        font-size: 0.95rem;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #27ae60;
        box-shadow: 0 2px 8px rgba(46,204,113,0.3);
    }

    .analysis-box {
        background: #f0f7ff;
        border: 1px solid #b3d4f5;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #1a3a5c;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    h1, h2, h3, h4 { color: #1a1a2e !important; }
    p, li { color: #333; }

    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_model_and_classes():
    model_path   = 'garbage_classifier.h5'
    classes_path = 'class_names.json'
    if not os.path.exists(model_path):
        return None, None, "Model file 'garbage_classifier.h5' not found."
    if not os.path.exists(classes_path):
        return None, None, "class_names.json not found."
    try:
        model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            class_names = json.load(f)
        return model, class_names, None
    except Exception as e:
        return None, None, str(e)

model, class_names, load_error = load_model_and_classes()

# ── Class Config ───────────────────────────────────────────────
CLASS_COLORS = {
    'cardboard': '#f39c12',
    'glass':     '#3498db',
    'metal':     '#7f8c8d',
    'paper':     '#2ecc71',
    'plastic':   '#e74c3c',
    'trash':     '#9b59b6'
}

RECYCLABLE_MAP = {
    'cardboard': True,
    'glass':     True,
    'metal':     True,
    'paper':     True,
    'plastic':   True,
    'trash':     False
}

DISPOSAL_TIPS = {
    'cardboard': 'Flatten and place in the paper/cardboard recycling bin. Remove any tape or staples.',
    'glass':     'Rinse and place in glass recycling. Do not break — keep intact for safety.',
    'metal':     'Rinse cans and tins. Place in metal recycling. Check if lids are attached.',
    'paper':     'Keep dry. Place in paper recycling. Avoid shredded paper in kerbside bins.',
    'plastic':   'Check the recycling number. Rinse containers before placing in plastic recycling.',
    'trash':     'Non-recyclable waste. Place in general waste bin. Consider reducing single-use items.'
}

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Smart Waste Classifier")
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("""
    - **Architecture:** Custom CNN (4 Conv Blocks)
    - **Input:** 224 x 224 RGB
    - **Output:** 6 waste categories
    - **Framework:** TensorFlow / Keras
    - **Val Accuracy:** 70.58%
    """)
    st.markdown("---")
    st.markdown("**Classes**")
    if class_names:
        for cls in class_names:
            status = "Recyclable" if RECYCLABLE_MAP.get(cls, False) else "Non-Recyclable"
            st.markdown(f"- **{cls.upper()}** — {status}")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Built for NeuralHack 2026 — Deep Learning Hackathon")
    if os.path.exists('training_history.json'):
        with open('training_history.json') as f:
            hist = json.load(f)
        st.markdown("---")
        st.markdown("**Training Results**")
        acc = hist.get('final_val_accuracy', 0) * 100
        st.metric("Validation Accuracy", f"{acc:.1f}%")

# ── Main Header ────────────────────────────────────────────────
st.markdown('<div class="hero-title">Smart Waste Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">CNN-powered garbage classification  |  NeuralHack 2026  </div>', unsafe_allow_html=True)

if load_error:
    st.error(f"Error: {load_error}")
    st.info("Make sure garbage_classifier.h5 and class_names.json are in the same folder as app.py")
    st.stop()

# ── Main Layout ────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### Upload Image")
    uploaded_file = st.file_uploader(
        "Select an image of a waste item",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload a clear image of the waste item"
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded image", use_container_width=True)
        classify_btn = st.button("Classify Waste", use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 2.5rem; color: #ccc;">[ Image ]</div>
            <div style="color: #888; margin-top: 1rem; font-size: 0.95rem;">Upload an image to classify</div>
            <div style="color: #bbb; font-size: 0.82rem; margin-top: 0.4rem;">Supports JPG, PNG, WEBP</div>
        </div>
        """, unsafe_allow_html=True)
        classify_btn = False

with col_result:
    st.markdown("#### Prediction")

    if uploaded_file and classify_btn:
        with st.spinner("Analysing image..."):
            img_array     = np.array(image.resize((224, 224))) / 255.0
            img_array     = np.expand_dims(img_array, axis=0)
            predictions   = model.predict(img_array, verbose=0)[0]
            pred_idx      = np.argmax(predictions)
            pred_class    = class_names[pred_idx]
            confidence    = float(predictions[pred_idx]) * 100
            is_recyclable = RECYCLABLE_MAP.get(pred_class, False)

        # 1. Predicted Class Card
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">{pred_class.upper()}</div>
            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Recyclable / Non-Recyclable Banner
        if is_recyclable:
            st.markdown("""
            <div class="recyclable-card recyclable">
                RECYCLABLE
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="recyclable-card non-recyclable">
                NON-RECYCLABLE
            </div>
            """, unsafe_allow_html=True)

        # 3. Disposal Tip
        tip = DISPOSAL_TIPS.get(pred_class, '')
        st.info(f"Disposal Tip: {tip}")

        # 4. Confidence Bar Chart
        st.markdown("#### Class Probabilities")
        df = pd.DataFrame({
            'Class':          [c.upper() for c in class_names],
            'Confidence (%)': [float(p) * 100 for p in predictions],
            'Color':          [CLASS_COLORS.get(c, '#888') for c in class_names]
        }).sort_values('Confidence (%)', ascending=True)

        fig = go.Figure(go.Bar(
            x=df['Confidence (%)'],
            y=df['Class'],
            orientation='h',
            marker=dict(color=df['Color'], line=dict(color='rgba(0,0,0,0)', width=0)),
            text=[f"{v:.1f}%" for v in df['Confidence (%)']],
            textposition='outside',
            textfont=dict(color='#333', size=11)
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#f8f9fa',
            font=dict(color='#333', family='Inter'),
            xaxis=dict(range=[0, 115], showgrid=True,
                       gridcolor='#e0e0e0',
                       ticksuffix='%', color='#666'),
            yaxis=dict(color='#333'),
            margin=dict(l=10, r=60, t=10, b=10),
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)

    elif not uploaded_file:
        st.markdown("""
        <div style="background:#ffffff; border:1px solid #e0e0e0; border-radius:12px;
                    padding:3rem; text-align:center;">
            <div style="color:#bbb; font-size:1rem; margin-top:0.5rem;">Awaiting image upload</div>
        </div>
        """, unsafe_allow_html=True)

# ── Architecture Section ───────────────────────────────────────
st.markdown("---")
st.markdown("#### CNN Architecture")

a1, a2, a3, a4 = st.columns(4)
for col, val, label in zip(
    [a1, a2, a3, a4],
    ['4', '6', '70.6%', '0.5'],
    ['Conv Blocks', 'Output Classes', 'Val Accuracy', 'Dropout Rate']
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

arch_data = {
    'Layer':          ['Input', 'Conv2D + BN', 'MaxPool', 'Conv2D + BN', 'MaxPool',
                       'Conv2D + BN', 'MaxPool', 'Conv2D + BN', 'MaxPool',
                       'GlobalAvgPool', 'Dense + Dropout', 'Dense + Dropout', 'Output'],
    'Filters/Units':  ['224x224x3', '32', '2x2', '64', '2x2',
                       '128', '2x2', '256', '2x2',
                       '-', '512 + p=0.5', '256 + p=0.3', '6'],
    'Activation':     ['-', 'ReLU', '-', 'ReLU', '-',
                       'ReLU', '-', 'ReLU', '-',
                       '-', 'ReLU', 'ReLU', 'Softmax'],
    'Regularization': ['-', 'L2 = 0.001', '-', 'L2 = 0.001', '-',
                       'L2 = 0.001', '-', 'L2 = 0.001', '-',
                       '-', 'Dropout', 'Dropout', '-']
}
st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

# ── Training History ───────────────────────────────────────────
if os.path.exists('training_history.json'):
    with open('training_history.json') as f:
        hist_data = json.load(f)

    st.markdown("---")
    st.markdown("#### Training History — Custom CNN")

    epochs = list(range(1, len(hist_data['accuracy']) + 1))
    h1, h2 = st.columns(2)

    with h1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs, y=hist_data['accuracy'],
                                     name='Train',
                                     line=dict(color='#2ecc71', width=2)))
        fig_acc.add_trace(go.Scatter(x=epochs, y=hist_data['val_accuracy'],
                                     name='Validation',
                                     line=dict(color='#3498db', width=2, dash='dash')))
        fig_acc.update_layout(
            title='Accuracy',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f8f9fa',
            font=dict(color='#333'), height=260,
            xaxis=dict(gridcolor='#e0e0e0', color='#666', title='Epoch'),
            yaxis=dict(gridcolor='#e0e0e0', color='#666'),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(font=dict(color='#333'))
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with h2:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=hist_data['loss'],
                                      name='Train',
                                      line=dict(color='#e74c3c', width=2)))
        fig_loss.add_trace(go.Scatter(x=epochs, y=hist_data['val_loss'],
                                      name='Validation',
                                      line=dict(color='#f39c12', width=2, dash='dash')))
        fig_loss.update_layout(
            title='Loss',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f8f9fa',
            font=dict(color='#333'), height=260,
            xaxis=dict(gridcolor='#e0e0e0', color='#666', title='Epoch'),
            yaxis=dict(gridcolor='#e0e0e0', color='#666'),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(font=dict(color='#333'))
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown("""
    <div class="analysis-box">
        <b>Analysis:</b>
        Training accuracy reached 81.6% while validation peaked at 70.6% —
        an 11% generalization gap indicating moderate overfitting.
        The zigzag validation loss pattern is caused by high variance from the small
        validation set (~500 images). This can be further improved using
        Transfer Learning (MobileNetV2) or K-Fold Cross Validation.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#aaa; font-size:0.8rem; font-family: 'JetBrains Mono', monospace;">
    NeuralHack 2026 — Deep Learning Hackathon  |  Custom CNN Waste Classification
</div>
""", unsafe_allow_html=True)
