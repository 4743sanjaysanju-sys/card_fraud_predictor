"""
app.py — Credit Card Fraud Detection Web App
---------------------------------------------
Run with:  streamlit run app.py

Place this file inside your credit_card_fraud_predictor/ folder.
"""

import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1424 !important;
    border-right: 1px solid #1e2a45;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #1e2a45;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #e8eaf0;
    letter-spacing: -1px;
    margin: 0;
}
.hero h1 span { color: #4f8ef7; }
.hero p {
    color: #6b7a99;
    font-size: 1rem;
    margin-top: 0.4rem;
}

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1e2a45;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: #4f8ef7;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Result boxes */
.result-fraud {
    background: linear-gradient(135deg, #1f0a0a, #2d0f0f);
    border: 2px solid #e53e3e;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.result-legit {
    background: linear-gradient(135deg, #0a1f12, #0d2b18);
    border: 2px solid #38a169;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.result-fraud .result-title { color: #fc8181; }
.result-legit .result-title { color: #68d391; }
.result-sub {
    font-size: 0.9rem;
    color: #9ca3af;
    margin-top: 0.3rem;
}

/* Metric chips */
.metric-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-top: 1rem;
    justify-content: center;
}
.metric-chip {
    background: #1a2035;
    border: 1px solid #2a3a5c;
    border-radius: 30px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    color: #94a3b8;
}
.metric-chip b { color: #e8eaf0; }

/* Input section */
.input-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e2a45;
    margin: 1.5rem 0;
}

/* Risk badge */
.badge-high   { color: #fc8181; font-weight: 700; }
.badge-medium { color: #f6ad55; font-weight: 700; }
.badge-low    { color: #68d391; font-weight: 700; }

/* Prob bar */
.prob-track {
    background: #1a2035;
    border-radius: 99px;
    height: 10px;
    width: 100%;
    margin: 0.5rem 0;
    overflow: hidden;
}
.prob-fill-fraud { background: linear-gradient(90deg, #e53e3e, #fc8181); border-radius: 99px; height: 10px; }
.prob-fill-legit { background: linear-gradient(90deg, #276749, #38a169); border-radius: 99px; height: 10px; }

/* History table */
.history-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1e2a45;
    font-size: 0.85rem;
}

/* Streamlit overrides */
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}
.stButton > button {
    background: #4f8ef7 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #3a7ae0 !important;
    box-shadow: 0 0 20px rgba(79,142,247,0.35) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model & Scaler ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    scaler_path = ROOT / "models" / "scaler.pkl"
    model_path  = ROOT / "models" / "best_model.pkl"

    if not scaler_path.exists() or not model_path.exists():
        return None, None, None

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return scaler, bundle["model"], bundle["name"]


scaler, model, model_name = load_artifacts()


# ── Feature engineering (mirrors preprocess.py) ───────────────
def build_features(txn: dict) -> np.ndarray:
    log_amount      = np.log1p(txn["amount"])
    suspicious_hour = 1 if txn["hour"] < 6 else 0
    risk_score      = (txn["online"]   * 0.3 +
                       txn["foreign"]  * 0.4 +
                       suspicious_hour * 0.3)
    return np.array([[
        log_amount, txn["hour"],
        txn["v1"], txn["v2"], txn["v3"], txn["v4"], txn["v5"],
        txn["online"], txn["foreign"], txn["freq"],
        suspicious_hour, risk_score
    ]])


def predict(txn: dict):
    feat   = build_features(txn)
    scaled = scaler.transform(feat)
    pred   = model.predict(scaled)[0]
    prob   = model.predict_proba(scaled)[0][1]
    return int(pred), round(float(prob) * 100, 2)


# ── Session state for history ─────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛡️ Fraud<span>Shield</span></h1>
  <p>AI-powered Credit Card Fraud Detection &nbsp;·&nbsp; Real-time Prediction Engine</p>
</div>
""", unsafe_allow_html=True)


# ── Not trained warning ───────────────────────────────────────
if model is None:
    st.error("⚠️ No trained model found. Run `python main.py` first to train the model, then relaunch the app.")
    st.stop()

# ── Layout ────────────────────────────────────────────────────
left, mid, right = st.columns([1.1, 1.3, 0.9])


# ════════════════════════════════════════════════════════════════
# LEFT — Transaction Input
# ════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="card"><div class="card-title">Transaction Details</div>', unsafe_allow_html=True)

    amount = st.number_input("💰 Amount ($)", min_value=0.01, max_value=20000.0,
                              value=120.00, step=10.0, format="%.2f")
    hour   = st.slider("🕐 Hour of Day (0 = midnight)", 0, 23, 14)

    st.markdown("**Behavioral Flags**")
    col1, col2 = st.columns(2)
    with col1:
        online  = st.selectbox("🌐 Online?",  ["No", "Yes"])
    with col2:
        foreign = st.selectbox("✈️ Foreign?", ["No", "Yes"])

    freq = st.slider("🔁 Transactions in Last Hour", 0, 20, 2)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Advanced Features (V1–V5)**")
    st.caption("PCA-anonymized behavioral signals. Leave at 0 for a typical transaction.")

    c1, c2 = st.columns(2)
    with c1:
        v1 = st.number_input("V1", value=0.0, step=0.1, format="%.2f")
        v2 = st.number_input("V2", value=0.0, step=0.1, format="%.2f")
        v3 = st.number_input("V3", value=0.0, step=0.1, format="%.2f")
    with c2:
        v4 = st.number_input("V4", value=0.0, step=0.1, format="%.2f")
        v5 = st.number_input("V5", value=0.0, step=0.1, format="%.2f")

    st.markdown('</div>', unsafe_allow_html=True)

    run = st.button("⚡ ANALYSE TRANSACTION")

    # Quick presets
    st.markdown('<div class="card-title" style="margin-top:1rem;">Quick Presets</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    preset = None
    with p1:
        if st.button("✅ Normal"):    preset = "normal"
    with p2:
        if st.button("⚠️ Suspect"):   preset = "suspect"
    with p3:
        if st.button("🔴 High Risk"): preset = "highrisk"


# ── Apply preset ──────────────────────────────────────────────
# (re-run happens automatically via Streamlit's reactivity)
preset_vals = {
    "normal":  dict(amount=45.0,   hour=14, online=0, foreign=0, freq=1,  v1=0.1,  v2=-0.1, v3=0.2,  v4=-0.1, v5=0.1),
    "suspect": dict(amount=850.0,  hour=22, online=1, foreign=1, freq=4,  v1=-1.5, v2=1.2,  v3=-1.8, v4=1.0,  v5=-0.8),
    "highrisk":dict(amount=3200.0, hour=2,  online=1, foreign=1, freq=12, v1=-4.1, v2=3.5,  v3=-5.2, v4=4.0,  v5=-3.0),
}


# ════════════════════════════════════════════════════════════════
# MID — Result
# ════════════════════════════════════════════════════════════════
with mid:
    # Build transaction dict
    txn = dict(
        amount=amount, hour=hour,
        online=1 if online=="Yes" else 0,
        foreign=1 if foreign=="Yes" else 0,
        freq=freq, v1=v1, v2=v2, v3=v3, v4=v4, v5=v5
    )

    # Override with preset if clicked
    if preset:
        txn = preset_vals[preset]

    if run or preset:
        pred, prob = predict(txn)

        # Save to history
        st.session_state.history.insert(0, {
            "amount": txn["amount"], "hour": txn["hour"],
            "result": "FRAUD" if pred == 1 else "Legit",
            "prob": prob
        })

        # Risk level
        if prob > 70:   risk, risk_cls = "HIGH",   "badge-high"
        elif prob > 40: risk, risk_cls = "MEDIUM",  "badge-medium"
        else:           risk, risk_cls = "LOW",     "badge-low"

        if pred == 1:
            st.markdown(f"""
            <div class="result-fraud">
              <div style="font-size:2.5rem">🚨</div>
              <div class="result-title">FRAUD DETECTED</div>
              <div class="result-sub">This transaction shows strong fraud indicators.</div>
              <div class="metric-row">
                <div class="metric-chip">Fraud Probability <b>{prob}%</b></div>
                <div class="metric-chip">Risk <b class="{risk_cls}">{risk}</b></div>
                <div class="metric-chip">Amount <b>${txn['amount']:.2f}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legit">
              <div style="font-size:2.5rem">✅</div>
              <div class="result-title">LEGITIMATE</div>
              <div class="result-sub">Transaction appears safe. No fraud indicators.</div>
              <div class="metric-row">
                <div class="metric-chip">Fraud Probability <b>{prob}%</b></div>
                <div class="metric-chip">Risk <b class="{risk_cls}">{risk}</b></div>
                <div class="metric-chip">Amount <b>${txn['amount']:.2f}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Probability bar
        st.markdown(f"""
        <div style="margin-top:1.2rem;">
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#6b7a99;">
            <span>0% Fraud</span><span>100% Fraud</span>
          </div>
          <div class="prob-track">
            <div class="{'prob-fill-fraud' if pred==1 else 'prob-fill-legit'}"
                 style="width:{prob}%;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature breakdown chart
        st.markdown('<div class="card-title" style="margin-top:1.4rem;">Feature Risk Breakdown</div>', unsafe_allow_html=True)
        feat_labels = ["Log Amount","Hour","V1","V2","V3","V4","V5",
                       "Online","Foreign","Tx Freq","Susp Hour","Risk Score"]
        raw = build_features(txn)[0]
        scaled = scaler.transform(build_features(txn))[0]

        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
        colors = ["#e53e3e" if abs(v) > 1.5 else "#4f8ef7" if abs(v) > 0.5 else "#2a3a5c"
                  for v in scaled]
        bars = ax.barh(feat_labels, scaled, color=colors, edgecolor="#0a0e1a", linewidth=0.5)
        ax.axvline(0, color="#2a3a5c", linewidth=1)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xlabel("Scaled Value", color="#6b7a99", fontsize=8)
        ax.tick_params(axis='y', labelcolor="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 1rem; color:#2a3a5c;">
          <div style="font-size:4rem;">🛡️</div>
          <div style="font-family:'Space Mono',monospace; font-size:1rem; margin-top:1rem; color:#4f8ef7;">
            AWAITING TRANSACTION
          </div>
          <div style="font-size:0.85rem; margin-top:0.5rem; color:#3a4a6c;">
            Fill in the details and click Analyse
          </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# RIGHT — Info Panel + History
# ════════════════════════════════════════════════════════════════
with right:
    # Model info
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Model Info</div>
      <div style="font-size:0.82rem; color:#94a3b8; line-height:1.7;">
        <div>🤖 <b style="color:#e8eaf0;">{model_name}</b></div>
        <div>📊 Trained on 10,000 transactions</div>
        <div>⚖️ Class-balanced training</div>
        <div>🎯 Features: 12 engineered</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk guide
    st.markdown("""
    <div class="card">
      <div class="card-title">Risk Guide</div>
      <div style="font-size:0.82rem; line-height:2;">
        <div>🔴 <span class="badge-high">HIGH</span> &nbsp;&nbsp;— Prob &gt; 70%</div>
        <div>🟡 <span class="badge-medium">MEDIUM</span> — Prob 40–70%</div>
        <div>🟢 <span class="badge-low">LOW</span> &nbsp;&nbsp;&nbsp;&nbsp;— Prob &lt; 40%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Fraud signals
    st.markdown("""
    <div class="card">
      <div class="card-title">Fraud Signals</div>
      <div style="font-size:0.8rem; color:#94a3b8; line-height:1.9;">
        🕐 Transactions 12AM–5AM<br>
        ✈️ Foreign card activity<br>
        💸 Unusually large amounts<br>
        🔁 High transaction frequency<br>
        🌐 Online-only pattern<br>
        📉 Abnormal V1–V5 scores
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Prediction history
    if st.session_state.history:
        st.markdown('<div class="card-title" style="margin-top:0.5rem;">Recent Predictions</div>', unsafe_allow_html=True)
        for h in st.session_state.history[:6]:
            color = "#fc8181" if h["result"] == "FRAUD" else "#68d391"
            st.markdown(f"""
            <div class="history-row">
              <span style="color:{color}; font-family:'Space Mono',monospace; font-size:0.75rem;">{h['result']}</span>
              <span style="color:#6b7a99;">${h['amount']:.0f} · {h['hour']:02d}h</span>
              <span style="color:{color}; font-size:0.75rem;">{h['prob']}%</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()


# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:#2a3a5c; font-size:0.78rem; border-top:1px solid #1e2a45; margin-top:2rem;">
  FraudShield · AI/ML Credit Card Fraud Detection · Built with Python & Streamlit
</div>
""", unsafe_allow_html=True)
