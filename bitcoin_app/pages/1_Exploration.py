"""
Page 1 — Exploration des Données (EDA interactif)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Exploration — BTC", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #F7931A33;
    }
    [data-testid="metric-container"] {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px 20px;
    }
    hr { border-color: #F7931A33 !important; }
    .stTabs [aria-selected="true"] {
        color: #F7931A !important;
        border-bottom-color: #F7931A !important;
    }
</style>
""", unsafe_allow_html=True)

from src.data_loader import load_data
from src.visualizations import (plot_candlestick, plot_price_history, plot_volume,
                                  plot_returns_distribution, plot_rsi, plot_macd)

df = load_data()

st.title("📊 Exploration des Données")
st.markdown(f"Dataset Bitcoin — **{len(df):,} jours** | "
            f"{df['Date'].min().strftime('%d/%m/%Y')} → {df['Date'].max().strftime('%d/%m/%Y')}")
st.divider()

# ── Sélecteur de période ──────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
with col_f1:
    start_date = st.date_input("Date de début", value=df['Date'].min().date(),
                                min_value=df['Date'].min().date(),
                                max_value=df['Date'].max().date())
with col_f2:
    end_date = st.date_input("Date de fin", value=df['Date'].max().date(),
                              min_value=df['Date'].min().date(),
                              max_value=df['Date'].max().date())
with col_f3:
    show_ma = st.checkbox("Moyennes Mobiles", value=True)
    show_bb = st.checkbox("Bollinger Bands", value=False)

# Filtrer les données
mask = (df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))
df_filtered = df[mask]

# ── Métriques de la période ───────────────────────────────────
if len(df_filtered) > 1:
    variation = (df_filtered['Close'].iloc[-1] / df_filtered['Close'].iloc[0] - 1) * 100
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Prix Début", f"${df_filtered['Close'].iloc[0]:,.0f}")
    c2.metric("Prix Fin",   f"${df_filtered['Close'].iloc[-1]:,.0f}",
              delta=f"{variation:+.1f}%")
    c3.metric("Maximum",    f"${df_filtered['Close'].max():,.0f}")
    c4.metric("Minimum",    f"${df_filtered['Close'].min():,.0f}")
    c5.metric("Jours",      f"{len(df_filtered):,}")

st.divider()

# ── Onglets ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🕯️ Chandelier OHLC", "📈 Prix Ligne",
                                          "📊 Distribution & Volatilité",
                                          "📉 RSI", "📊 MACD"])

with tab1:
    st.plotly_chart(
        plot_candlestick(df, start_date, end_date, show_ma, show_bb),
        use_container_width=True
    )
    col_l, col_r = st.columns(2)
    with col_l:
        d = df_filtered
        hausse = (d['Change_pct'] > 0).sum()
        baisse = (d['Change_pct'] < 0).sum()
        st.markdown(f"""
        <div style='background:#1a1a2e;border-radius:10px;padding:12px 18px;'>
            <b>Jours haussiers :</b>
            <span style='color:#4CAF50;font-size:1.1rem;'> {hausse}</span>
            &nbsp;|&nbsp;
            <b>Jours baissiers :</b>
            <span style='color:#F44336;font-size:1.1rem;'> {baisse}</span>
            &nbsp;|&nbsp;
            <b>Ratio :</b>
            <span style='color:#F7931A;'> {hausse/(hausse+baisse)*100:.1f}% haussier</span>
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        if len(d) > 1:
            avg_up   = d[d['Change_pct'] > 0]['Change_pct'].mean()
            avg_down = d[d['Change_pct'] < 0]['Change_pct'].mean()
            st.markdown(f"""
            <div style='background:#1a1a2e;border-radius:10px;padding:12px 18px;'>
                <b>Gain moyen / jour haussier :</b>
                <span style='color:#4CAF50;'> +{avg_up:.2f}%</span>
                &nbsp;|&nbsp;
                <b>Perte moyenne / jour baissier :</b>
                <span style='color:#F44336;'> {avg_down:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.plotly_chart(
        plot_price_history(df, start_date, end_date, show_ma, show_bb),
        use_container_width=True
    )
    st.plotly_chart(
        plot_volume(df, start_date, end_date),
        use_container_width=True
    )

with tab3:
    st.plotly_chart(
        plot_returns_distribution(df_filtered),
        use_container_width=True
    )

    # Stats descriptives
    st.markdown("#### Statistiques Descriptives")
    returns = df_filtered['daily_return'].dropna()
    stats_data = {
        "Indicateur": ["Rendement Moyen / Jour", "Volatilité / Jour",
                        "Rendement Max", "Rendement Min",
                        "Skewness", "Kurtosis"],
        "Valeur": [
            f"{returns.mean():.4f}%",
            f"{returns.std():.4f}%",
            f"{returns.max():.2f}%",
            f"{returns.min():.2f}%",
            f"{returns.skew():.4f}",
            f"{returns.kurtosis():.4f}",
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True,
                 hide_index=True)

with tab4:
    if df_filtered['RSI'].notna().sum() > 10:
        st.plotly_chart(
            plot_rsi(df, start_date, end_date),
            use_container_width=True
        )
        st.info(
            "**RSI > 70** : Zone de surachat (possible correction) | "
            "**RSI < 30** : Zone de survente (possible rebond)"
        )
    else:
        st.warning("Période trop courte pour afficher le RSI (minimum 14 jours).")

with tab5:
    if df_filtered['MACD'].notna().sum() > 10:
        st.plotly_chart(
            plot_macd(df, start_date, end_date),
            use_container_width=True
        )
        st.info(
            "**MACD > Signal** : Signal haussier | "
            "**MACD < Signal** : Signal baissier"
        )
    else:
        st.warning("Période trop courte pour afficher le MACD (minimum 26 jours).")
