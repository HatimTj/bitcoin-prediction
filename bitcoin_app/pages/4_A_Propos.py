"""
Page 4 — À Propos du projet et de l'équipe
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="À Propos — BTC", page_icon="👥", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #F7931A33;
    }
    hr { border-color: #F7931A33 !important; }
    .member-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .member-card:hover { border-color: #F7931A66; }
    .avatar {
        width: 72px; height: 72px;
        background: linear-gradient(135deg, #F7931A, #e67e00);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem;
        margin: 0 auto 12px auto;
    }
    .tech-badge {
        display: inline-block;
        background: #0e1117;
        border: 1px solid #F7931A44;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.82rem;
        color: #F7931A;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ── En-tête ───────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 30px 0;'>
    <div style='font-size:3rem;'>₿</div>
    <h1 style='color: #F7931A; margin: 8px 0 4px 0;'>
        Prévision du Prix Bitcoin
    </h1>
    <p style='color: #aaa; font-size: 1rem;'>
        Projet de Fin d'Année (PFA) — EMSI Rabat<br>
        Spécialité Intelligence Artificielle & Data Science — 4IIR<br>
        <span style='color:#F7931A;'>Hatim Tajimi</span>
        &nbsp;|&nbsp; Encadrant : Pr. Idriss BARBARA
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Description du projet ─────────────────────────────────────
col_desc, col_stats = st.columns([3, 2])

with col_desc:
    st.markdown("### Description du Projet")
    st.markdown("""
    Ce projet a pour objectif de **comparer différentes approches de Machine Learning**
    pour la prévision du prix du Bitcoin (BTC/USD), dans le cadre du PFA
    de la filière IA & Data Science à l'EMSI Rabat.

    #### Objectifs
    - Analyser les séries temporelles financières du Bitcoin (2010–2024)
    - Implémenter et comparer 7 modèles de prévision
    - Évaluer les performances via MAE, RMSE et MAPE
    - Déployer une application interactive pour visualiser les prédictions

    #### Données
    - **Source** : Investing.com — BTC/USD Historical Data
    - **Période** : Juillet 2010 → Décembre 2024
    - **Fréquence** : Données journalières (OHLCV)
    - **Volume** : ~5 200 observations
    """)

with col_stats:
    st.markdown("### Résultats Clés")
    for model, mape, color in [
        ("GRU",          "1.72%", "#4CAF50"),
        ("CNN-LSTM",      "1.78%", "#9C27B0"),
        ("Stacked LSTM",  "2.10%", "#FF9800"),
        ("LSTM",          "2.18%", "#2196F3"),
        ("XGBoost",       "8.87%", "#795548"),
        ("ARIMA",        "19.21%", "#9E9E9E"),
        ("Prophet",      "24.73%", "#E91E63"),
    ]:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; align-items:center;
                    background:#1a1a2e; border-radius:8px; padding:8px 14px;
                    margin-bottom:6px; border-left: 3px solid {color};'>
            <span style='color:#ccc;'>{model}</span>
            <span style='color:{color}; font-weight:700;'>{mape}</span>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Équipe ────────────────────────────────────────────────────
st.markdown("### Auteur")
col1, col2 = st.columns(2)

members = [
    ("🧑‍💻", "Hatim Tajimi",       "Développeur ML\nModèles DL, XGBoost & Évaluation"),
    ("👨‍🏫", "Pr. Idriss BARBARA", "Encadrant\nEMSI Rabat"),
]

for col, (icon, name, role) in zip([col1, col2], members):
    with col:
        is_supervisor = "Encadrant" in role
        border_color  = "#F7931A" if is_supervisor else "#2a2a4a"
        st.markdown(f"""
        <div class='member-card' style='border-color: {border_color};'>
            <div class='avatar'>{icon}</div>
            <div style='font-weight: 700; color: #F7931A; font-size: 1rem;'>
                {name}
            </div>
            <div style='color: #aaa; font-size: 0.82rem; margin-top: 6px;
                        white-space: pre-line;'>
                {role}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Stack technologique ───────────────────────────────────────
st.markdown("### Stack Technologique")

col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    st.markdown("**Deep Learning**")
    for tech in ["TensorFlow 2.x", "Keras", "LSTM", "GRU", "CNN-LSTM"]:
        st.markdown(f"<span class='tech-badge'>{tech}</span>", unsafe_allow_html=True)

with col_t2:
    st.markdown("**Machine Learning**")
    for tech in ["XGBoost", "Prophet (Meta)", "ARIMA", "statsmodels", "pmdarima"]:
        st.markdown(f"<span class='tech-badge'>{tech}</span>", unsafe_allow_html=True)

with col_t3:
    st.markdown("**Data & Visualisation**")
    for tech in ["Python 3.11", "Pandas", "NumPy", "Plotly", "Streamlit", "scikit-learn"]:
        st.markdown(f"<span class='tech-badge'>{tech}</span>", unsafe_allow_html=True)

st.divider()

# ── Architecture des modèles ──────────────────────────────────
st.markdown("### Architecture des Modèles DL")

arch_data = {
    "Modèle": ["LSTM", "GRU", "Stacked LSTM", "CNN-LSTM"],
    "Architecture": [
        "Input(60,1) → LSTM(64) → Dropout(0.2) → Dense(1)",
        "Input(60,1) → GRU(64) → Dropout(0.2) → Dense(1)",
        "Input(60,1) → LSTM(128,seq) → Dropout → LSTM(64) → Dropout → Dense(1)",
        "Input(60,1) → Conv1D(64,3) → MaxPool(2) → LSTM(64) → Dropout → Dense(1)",
    ],
    "Paramètres": ["~33K", "~25K", "~115K", "~42K"],
    "Lookback": ["60 jours"] * 4,
}
import pandas as pd
st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

st.divider()

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color: #555; font-size: 0.8rem; padding: 10px 0;'>
    EMSI Rabat — École Marocaine des Sciences de l'Ingénieur<br>
    Filière IA & Data Science — Année 2024/2025<br>
    <span style='color: #F7931A;'>Hatim Tajimi</span>
</div>
""", unsafe_allow_html=True)
