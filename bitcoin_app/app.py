"""
=============================================================
  Bitcoin Price Prediction — Application Streamlit
  PFA EMSI Rabat — IA & Data (4IIR)
  Auteur : Hatim Tajimi
  Encadrant : Pr. Idriss BARBARA
=============================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np

# ── Config page (DOIT être le premier appel Streamlit) ────────
st.set_page_config(
    page_title="Bitcoin Prediction — EMSI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────
st.markdown("""
<style>
    /* Fond global */
    .stApp { background-color: #0e1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #F7931A33;
    }

    /* Cartes métriques */
    [data-testid="metric-container"] {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px 20px;
        transition: border-color 0.2s;
    }
    [data-testid="metric-container"]:hover {
        border-color: #F7931A66;
    }

    /* Titre principal */
    .btc-header {
        text-align: center;
        padding: 10px 0 20px 0;
    }
    .btc-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #F7931A;
        letter-spacing: -1px;
    }
    .btc-subtitle {
        font-size: 1rem;
        color: #aaa;
        margin-top: 4px;
    }

    /* Badge modèle */
    .model-badge {
        display: inline-block;
        background: #F7931A22;
        border: 1px solid #F7931A55;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        color: #F7931A;
        margin: 2px;
    }

    /* Séparateur orange */
    hr { border-color: #F7931A33 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #aaa;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #F7931A !important;
        border-bottom-color: #F7931A !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Imports locaux ────────────────────────────────────────────
from src.data_loader import load_data, load_comparison_results
from src.visualizations import plot_recent_price
from src.predictor import models_are_ready, available_models

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0;'>
        <span style='font-size:2.5rem;'>₿</span>
        <div style='font-size:1.1rem; font-weight:700; color:#F7931A;'>Bitcoin Predictor</div>
        <div style='font-size:0.75rem; color:#aaa;'>EMSI — IA & Data 4IIR</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("**Navigation**")
    st.markdown("""
    - 🏠 **Accueil** ← vous êtes ici
    - 📊 **Exploration** — EDA interactif
    - 🔮 **Prédiction** — Prévision par modèle
    - 📈 **Comparaison** — Benchmark
    - 👥 **À Propos** — Équipe & projet
    """)
    st.divider()
    st.markdown("""
    <div style='font-size:0.8rem; color:#aaa; line-height:1.8;'>
        <b style='color:#F7931A;'>Binôme</b><br>
        Hatim Tajimi<br><br>
        <i>Encadrant :</i><br>
        Pr. Idriss BARBARA
    </div>
    """, unsafe_allow_html=True)

# ── Chargement des données ────────────────────────────────────
df      = load_data()
df_comp = load_comparison_results()

# ── Calculs ───────────────────────────────────────────────────
last_price   = df['Close'].iloc[-1]
prev_price   = df['Close'].iloc[-2]
price_change = last_price - prev_price
price_pct    = (price_change / prev_price) * 100
last_date    = df['Date'].iloc[-1]
date_range   = f"{df['Date'].min().strftime('%Y')} → {df['Date'].max().strftime('%Y')}"
n_days       = len(df)

best_model = "GRU"
best_mape  = "1.72%"
if not df_comp.empty and 'MAPE' in df_comp.columns and 'Model' in df_comp.columns:
    best_idx   = df_comp['MAPE'].idxmin()
    best_model = df_comp.loc[best_idx, 'Model']
    best_mape  = f"{df_comp.loc[best_idx, 'MAPE']:.2f}%"

# Tendance rapide (MA7 vs MA30)
ma7  = df['MA7'].iloc[-1]
ma30 = df['MA30'].iloc[-1]
trend_up    = ma7 > ma30
trend_label = "HAUSSIÈRE" if trend_up else "BAISSIÈRE"
trend_color = "#4CAF50"   if trend_up else "#F44336"
trend_icon  = "📈" if trend_up else "📉"

# Badge tendance dans la sidebar
with st.sidebar:
    st.divider()
    st.markdown(f"""
    <div style='background:#1a1a2e; border:1px solid {trend_color}44;
                border-radius:10px; padding:12px; text-align:center;'>
        <div style='font-size:1.5rem;'>{trend_icon}</div>
        <div style='font-size:0.75rem; color:#aaa;'>Tendance actuelle</div>
        <div style='color:{trend_color}; font-weight:700; font-size:1rem;'>{trend_label}</div>
        <div style='font-size:0.75rem; color:#aaa;'>MA7 {">" if trend_up else "<"} MA30</div>
    </div>
    """, unsafe_allow_html=True)

# ── Ticker animé ──────────────────────────────────────────────
ticker_color = "#4CAF50" if price_change >= 0 else "#F44336"
ticker_arrow = "▲" if price_change >= 0 else "▼"
st.markdown(f"""
<div style='background: linear-gradient(90deg, #1a1a2e, #16213e);
            border: 1px solid #F7931A44; border-radius: 14px;
            padding: 18px 28px; margin-bottom: 16px;
            display: flex; align-items: center; justify-content: space-between;'>
    <div>
        <div style='font-size:2rem; font-weight:800; color:#F7931A; letter-spacing:-1px;'>
            ₿ Bitcoin Price Prediction
        </div>
        <div style='color:#aaa; font-size:0.9rem; margin-top:4px;'>
            PFA EMSI 2024/2025 — Hatim Tajimi
        </div>
    </div>
    <div style='text-align:right;'>
        <div style='font-size:2.2rem; font-weight:800; color:#fff;'>
            ${last_price:,.0f}
        </div>
        <div style='font-size:1.1rem; font-weight:600; color:{ticker_color};'>
            {ticker_arrow} {price_pct:+.2f}% &nbsp;
            <span style='font-size:0.85rem; color:#aaa;'>
                {last_date.strftime('%d/%m/%Y')}
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="Dernier Prix",       value=f"${last_price:,.0f}",
              delta=f"{price_pct:+.2f}%")
with col2:
    st.metric(label="Période des données",value=date_range)
with col3:
    st.metric(label="Jours d'historique", value=f"{n_days:,}")
with col4:
    st.metric(label="Meilleur modèle",    value=best_model)
with col5:
    st.metric(label="Meilleur MAPE",      value=best_mape)

st.divider()

# ── Graphique prix récent ─────────────────────────────────────
col_chart, col_info = st.columns([3, 1])

with col_chart:
    period = st.select_slider(
        "Période d'affichage",
        options=[30, 60, 90, 180, 365],
        value=90,
        format_func=lambda x: f"{x} jours",
    )
    fig = plot_recent_price(df, days=period)
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.markdown("#### Statistiques")
    recent = df.tail(period)
    st.markdown(f"""
    <div style='line-height: 2.2; font-size: 0.9rem;'>
        <b>Prix max</b><br>
        <span style='color:#4CAF50; font-size:1.1rem;'>${recent['Close'].max():,.0f}</span><br>
        <b>Prix min</b><br>
        <span style='color:#F44336; font-size:1.1rem;'>${recent['Close'].min():,.0f}</span><br>
        <b>Variation</b><br>
        <span style='color:#F7931A; font-size:1.1rem;'>
            {((recent['Close'].iloc[-1]/recent['Close'].iloc[0]-1)*100):+.1f}%
        </span><br>
        <b>Volatilité</b><br>
        <span style='color:#9C27B0; font-size:1.1rem;'>
            {recent['daily_return'].std():.2f}% / jour
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.divider()
    st.markdown("#### Modèles")
    for model in ['GRU', 'CNN-LSTM', 'Stacked LSTM', 'LSTM', 'ARIMA', 'XGBoost', 'Prophet']:
        st.markdown(f"<span class='model-badge'>{model}</span>", unsafe_allow_html=True)

# ── Résumé des résultats ──────────────────────────────────────
st.divider()
st.markdown("### Résultats du Benchmark")

if not df_comp.empty:
    col_t, col_r = st.columns([2, 1])
    with col_t:
        display_cols = [c for c in ['Model', 'MAE', 'RMSE', 'MAPE'] if c in df_comp.columns]
        df_display = df_comp[display_cols].copy()
        if 'MAPE' in df_display.columns:
            df_display = df_display.sort_values('MAPE')

        def style_table(df):
            styled = df.style
            if 'MAPE' in df.columns:
                styled = styled.highlight_min(subset=['MAPE'], color='#1a3a1a')
                styled = styled.highlight_max(subset=['MAPE'], color='#3a1a1a')
            if 'MAE' in df.columns:
                styled = styled.highlight_min(subset=['MAE'], color='#1a3a1a')
            return styled

        st.dataframe(
            style_table(df_display),
            use_container_width=True,
            hide_index=True,
        )

    with col_r:
        st.markdown("#### Podium")
        medals = ["🥇", "🥈", "🥉"]
        if 'MAPE' in df_comp.columns:
            top3 = df_comp.nsmallest(3, 'MAPE')
            for i, (_, row) in enumerate(top3.iterrows()):
                st.markdown(f"""
                <div style='background:#1a1a2e; border-radius:10px;
                            padding:10px 15px; margin-bottom:8px;
                            border-left: 3px solid #F7931A;'>
                    <span style='font-size:1.2rem;'>{medals[i]}</span>
                    <b style='color:#F7931A;'>{row['Model']}</b><br>
                    <span style='color:#aaa; font-size:0.85rem;'>
                        MAPE: {row['MAPE']:.2f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("Lancez `python save_models.py` pour générer les résultats du benchmark.")

# ── Statut des modèles ────────────────────────────────────────
st.divider()
col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown("#### Statut des Modèles")
    available = available_models()
    st.success(f"{len(available)} modèle(s) disponible(s) : {', '.join(available)}")

with col_s2:
    st.markdown("#### Données")
    st.success(
        f"Dataset chargé : **{n_days:,} jours** "
        f"({df['Date'].min().strftime('%d/%m/%Y')} → "
        f"{df['Date'].max().strftime('%d/%m/%Y')})"
    )
