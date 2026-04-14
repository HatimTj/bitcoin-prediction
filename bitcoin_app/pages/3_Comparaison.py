"""
Page 3 — Comparaison de tous les modèles
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Comparaison — BTC", page_icon="📈", layout="wide")

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

from src.data_loader import load_comparison_results
from src.metrics import get_model_rank
from src.visualizations import plot_metrics_bars, plot_radar_chart
from src.config import MODEL_COLORS

df_comp = load_comparison_results()

st.title("📈 Comparaison des Modèles")
st.markdown("Benchmark complet — MAE, RMSE, MAPE sur le jeu de test (10% des données).")
st.divider()

if df_comp.empty:
    st.warning(
        "Résultats non trouvés. Lancez `python save_models.py` "
        "pour générer le benchmark."
    )
    st.stop()

# ── Enrichissement avec rang ──────────────────────────────────
df_ranked = get_model_rank(df_comp)

# ── KPIs du meilleur modèle ───────────────────────────────────
if 'MAPE' in df_ranked.columns:
    best = df_ranked.nsmallest(1, 'MAPE').iloc[0]
    worst = df_ranked.nlargest(1, 'MAPE').iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meilleur Modèle", best['Model'])
    c2.metric("Meilleur MAPE",   f"{best['MAPE']:.2f}%")
    if 'MAE' in best:
        c3.metric("Meilleur MAE", f"${best['MAE']:,.0f}")
    c4.metric("Modèles Testés",  str(len(df_comp)))

st.divider()

# ── Onglets ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Tableau de Classement", "Graphiques", "Radar Chart"])

with tab1:
    st.markdown("#### Classement par MAPE (du meilleur au moins bon)")

    display_cols = [c for c in ['Rang', 'Model', 'Niveau', 'MAE', 'RMSE', 'MAPE', 'Temps']
                    if c in df_ranked.columns]
    df_show = df_ranked[display_cols].copy()

    # Formatage
    if 'MAE' in df_show.columns:
        df_show['MAE']  = df_show['MAE'].apply(lambda x: f"${x:,.2f}")
    if 'RMSE' in df_show.columns:
        df_show['RMSE'] = df_show['RMSE'].apply(lambda x: f"${x:,.2f}")
    if 'MAPE' in df_show.columns:
        df_show['MAPE'] = df_show['MAPE'].apply(lambda x: f"{x:.4f}%")

    # Ajouter médaille au rang
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    if 'Rang' in df_show.columns:
        df_show['Rang'] = df_show['Rang'].apply(lambda r: f"{medals.get(int(r) if str(r).isdigit() else 0, '')} #{r}")

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Interprétation des Métriques")
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.markdown("""
        **MAE** (Mean Absolute Error)
        - Erreur moyenne absolue en $
        - Facile à interpréter
        - Moins sensible aux outliers
        """)
    with col_i2:
        st.markdown("""
        **RMSE** (Root Mean Square Error)
        - Pénalise les grandes erreurs
        - En dollars (même unité)
        - Sensible aux pics d'erreur
        """)
    with col_i3:
        st.markdown("""
        **MAPE** (Mean Abs. % Error)
        - Erreur relative en %
        - Indépendant du prix
        - **Métrique principale**
        """)

with tab2:
    st.markdown("#### Comparaison des Métriques par Modèle")
    st.plotly_chart(
        plot_metrics_bars(df_comp),
        use_container_width=True
    )

    # Mini bar chart MAPE uniquement (plus lisible)
    st.markdown("#### Classement MAPE")
    if 'MAPE' in df_comp.columns:
        df_mape = df_comp.sort_values('MAPE')
        import plotly.graph_objects as go
        colors = [MODEL_COLORS.get(m, '#F7931A') for m in df_mape['Model']]
        fig_mape = go.Figure(go.Bar(
            x=df_mape['MAPE'],
            y=df_mape['Model'],
            orientation='h',
            marker_color=colors,
            text=df_mape['MAPE'].apply(lambda x: f"{x:.2f}%"),
            textposition='outside',
        ))
        fig_mape.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            xaxis_title="MAPE (%)",
            height=350,
            margin=dict(l=20, r=60, t=20, b=20),
        )
        st.plotly_chart(fig_mape, use_container_width=True)

with tab3:
    st.markdown("#### Performance Relative (plus grand = meilleur)")
    st.plotly_chart(
        plot_radar_chart(df_comp),
        use_container_width=True
    )
    st.caption(
        "Le radar affiche les scores normalisés : "
        "1 = meilleur modèle sur cette métrique, 0 = moins bon."
    )

# ── Conclusion ────────────────────────────────────────────────
st.divider()
st.markdown("### Conclusion")

if 'MAPE' in df_ranked.columns:
    best_model = df_ranked.nsmallest(1, 'MAPE').iloc[0]['Model']
    best_mape  = df_ranked.nsmallest(1, 'MAPE').iloc[0]['MAPE']
    color      = MODEL_COLORS.get(best_model, '#F7931A')

    st.markdown(f"""
    <div style='background: #1a1a2e; border: 1px solid {color};
                border-radius: 12px; padding: 20px;'>
        <h4 style='color: {color}; margin: 0 0 10px 0;'>
            🏆 Meilleur modèle : {best_model}
        </h4>
        <p style='color: #ccc; margin: 0;'>
            Le modèle <b>{best_model}</b> obtient le meilleur MAPE de <b>{best_mape:.2f}%</b>,
            ce qui signifie qu'en moyenne, ses prédictions s'écartent de
            <b>{best_mape:.2f}%</b> du prix réel du Bitcoin.
            Les modèles Deep Learning (LSTM, GRU, CNN-LSTM) surpassent
            significativement les approches statistiques (ARIMA, Prophet)
            sur ce type de données non-linéaires et volatiles.
        </p>
    </div>
    """, unsafe_allow_html=True)
