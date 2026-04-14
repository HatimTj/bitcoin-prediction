"""
Page 5 — Métriques Détaillées : résidus, heatmap d'erreur par période, QQ-plot.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Métriques — BTC", page_icon="📉", layout="wide")

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

from src.data_loader import get_preprocessed_sequences
from src.predictor import available_models, predict_on_test, model_info
from src.metrics import compute_metrics
from src.config import MODEL_COLORS, BTC_COLOR

LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#e0e0e0"),
    margin=dict(l=40, r=20, t=50, b=40),
)

st.title("📉 Analyse Détaillée des Erreurs")
st.markdown("Résidus, distribution des erreurs, heatmap par période et QQ-plot.")
st.divider()

# ── Données ───────────────────────────────────────────────────
with st.spinner("Chargement..."):
    X_test, y_test_real, test_df, scaler, scaled, df_full = get_preprocessed_sequences()

n         = len(scaled)
train_end = int(n * 0.80)
val_end   = int(n * 0.90)
test_dates = pd.to_datetime(test_df['Date'].values)
available  = available_models()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Paramètres")
    selected = st.selectbox(
        "Modèle à analyser",
        options=available,
        format_func=lambda m: f"{m} — {model_info(m)}" if model_info(m) != 'Deep Learning' else m,
    )

# ── Prédictions ───────────────────────────────────────────────
SPINNER = {
    'XGBoost': 'XGBoost (~2s)...', 'Random Forest': 'RF (~3s)...',
    'Régression Linéaire': 'LinReg (~1s)...', 'ARIMA': 'ARIMA (~30s)...',
}
with st.spinner(SPINNER.get(selected, f'{selected}...')):
    y_pred = predict_on_test(
        selected, X_test, scaler,
        scaled=scaled, train_end=train_end,
        val_end=val_end, test_start=val_end,
    )

min_len   = min(len(y_test_real), len(y_pred))
y_true    = y_test_real[:min_len]
y_pred    = y_pred[:min_len]
dates     = test_dates[:min_len]
residuals = y_true - y_pred
color     = MODEL_COLORS.get(selected, '#F7931A')

metrics = compute_metrics(y_true, y_pred, selected)

# ── KPIs ──────────────────────────────────────────────────────
st.markdown(f"### Modèle : **{selected}**")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("MAE",               f"${metrics['MAE']:,.2f}")
c2.metric("RMSE",              f"${metrics['RMSE']:,.2f}")
c3.metric("MAPE",              f"{metrics['MAPE']:.2f}%")
c4.metric("Direction Accuracy",f"{metrics['DA']:.1f}%")
c5.metric("Biais moyen",       f"${residuals.mean():+,.0f}")
st.divider()

# ── Onglets ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Résidus dans le temps",
    "📊 Distribution des erreurs",
    "🗓️ Heatmap par période",
    "📐 QQ-Plot",
])

# ── Tab 1 : Résidus ────────────────────────────────────────────
with tab1:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
                        shared_xaxes=True, vertical_spacing=0.06)

    # Prix réel vs prédit
    fig.add_trace(go.Scatter(
        x=dates, y=y_true,
        line=dict(color=BTC_COLOR, width=2), name='Prix Réel',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=y_pred,
        line=dict(color=color, width=1.5, dash='dash'), name=f'Prédit ({selected})',
    ), row=1, col=1)

    # Résidus
    res_colors = np.where(residuals >= 0, '#4CAF50', '#F44336')
    fig.add_trace(go.Bar(
        x=dates, y=residuals,
        marker_color=res_colors, opacity=0.7, name='Résidu',
    ), row=2, col=1)
    fig.add_hline(y=0, line_color='white', line_width=1, row=2, col=1)
    fig.add_hline(y=residuals.std(),  line_color='orange', line_dash='dot', row=2, col=1)
    fig.add_hline(y=-residuals.std(), line_color='orange', line_dash='dot', row=2, col=1)

    fig.update_layout(**LAYOUT, height=500,
                      title=f'Résidus dans le Temps — {selected}',
                      yaxis_title='Prix ($)', yaxis2_title='Erreur ($)')
    fig.update_yaxes(tickprefix='$', tickformat=',.0f', row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"Les barres **orange pointillées** représentent ±1σ = ±${residuals.std():,.0f} "
            f"— les résidus restent majoritablement dans cette plage.")

# ── Tab 2 : Distribution des erreurs ──────────────────────────
with tab2:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Distribution des Résidus ($)",
                                        "Distribution MAPE par jour (%)"))

    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=60,
        marker_color=color, opacity=0.75, name='Résidus',
    ), row=1, col=1)
    fig.add_vline(x=0, line_color='white', line_width=2, row=1, col=1)
    fig.add_vline(x=residuals.mean(), line_color='orange',
                  line_dash='dash', row=1, col=1,
                  annotation_text=f"μ={residuals.mean():+,.0f}$")

    ape = np.abs(residuals / (y_true + 1e-8)) * 100
    fig.add_trace(go.Histogram(
        x=ape, nbinsx=60,
        marker_color='#9C27B0', opacity=0.75, name='APE (%)',
    ), row=1, col=2)
    fig.add_vline(x=metrics['MAPE'], line_color='orange',
                  line_dash='dash', row=1, col=2,
                  annotation_text=f"MAPE={metrics['MAPE']:.2f}%")

    fig.update_layout(**LAYOUT, height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Stats résidus
    df_stats = pd.DataFrame({
        'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max',
                        'Skewness', 'Kurtosis',
                        '% dans ±$1000', '% dans ±$5000'],
        'Valeur': [
            f"${residuals.mean():+,.0f}",
            f"${np.median(residuals):+,.0f}",
            f"${residuals.std():,.0f}",
            f"${residuals.min():+,.0f}",
            f"${residuals.max():+,.0f}",
            f"{pd.Series(residuals).skew():.4f}",
            f"{pd.Series(residuals).kurtosis():.4f}",
            f"{(np.abs(residuals) <= 1000).mean()*100:.1f}%",
            f"{(np.abs(residuals) <= 5000).mean()*100:.1f}%",
        ]
    })
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

# ── Tab 3 : Heatmap par période ────────────────────────────────
with tab3:
    df_res = pd.DataFrame({
        'Date': dates, 'Residual': residuals,
        'APE': np.abs(residuals / (y_true + 1e-8)) * 100,
    })
    df_res['Année']  = df_res['Date'].dt.year
    df_res['Mois']   = df_res['Date'].dt.month

    pivot = df_res.pivot_table(
        values='APE', index='Année', columns='Mois',
        aggfunc='mean',
    )
    mois_labels = ['Jan','Fév','Mar','Avr','Mai','Jun',
                   'Jul','Aoû','Sep','Oct','Nov','Déc']
    col_labels  = [mois_labels[c-1] for c in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=col_labels,
        y=[str(y) for y in pivot.index],
        colorscale='RdYlGn_r',
        text=np.round(pivot.values, 1),
        texttemplate='%{text:.1f}%',
        textfont=dict(size=10),
        colorbar=dict(title='MAPE (%)'),
    ))
    fig.update_layout(**LAYOUT, height=max(250, len(pivot)*60+80),
                      title=f'MAPE Moyen (%) par Mois/Année — {selected}',
                      xaxis_title='Mois', yaxis_title='Année')
    st.plotly_chart(fig, use_container_width=True)

    # Analyse par année
    year_stats = df_res.groupby('Année').agg(
        MAPE_moy=('APE', 'mean'),
        MAE_moy=('Residual', lambda x: np.abs(x).mean()),
        N=('APE', 'count'),
    ).round(2).reset_index()
    year_stats.columns = ['Année', 'MAPE moyen (%)', 'MAE moyen ($)', 'Nb jours']
    st.dataframe(year_stats, use_container_width=True, hide_index=True)

# ── Tab 4 : QQ-Plot ────────────────────────────────────────────
with tab4:
    import scipy.stats as stats_sp

    sorted_res = np.sort(residuals)
    n_pts      = len(sorted_res)
    quantiles  = stats_sp.norm.ppf(np.linspace(0.01, 0.99, n_pts))
    quantiles  = quantiles * sorted_res.std() + sorted_res.mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=quantiles, y=sorted_res,
        mode='markers',
        marker=dict(color=color, size=3, opacity=0.6),
        name='Résidus observés',
    ))
    # Droite théorique
    lim = max(abs(quantiles.min()), abs(quantiles.max()))
    fig.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim],
        line=dict(color='white', width=1.5, dash='dash'),
        name='Distribution Normale',
    ))
    fig.update_layout(
        **LAYOUT,
        title=f'QQ-Plot des Résidus — {selected}',
        xaxis_title='Quantiles théoriques (Normale)',
        yaxis_title='Quantiles observés ($)',
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Test de normalité
    try:
        stat, p_val = stats_sp.shapiro(residuals[:5000])
        normal = p_val > 0.05
        st.markdown(f"""
        **Test de Shapiro-Wilk** : statistic={stat:.4f}, p-value={p_val:.4e}
        {'✅ Les résidus suivent approximativement une distribution normale.' if normal
         else '⚠️ Les résidus ne suivent pas une distribution normale (queues épaisses — typique en finance).'}
        """)
    except Exception:
        st.caption("Test de normalité non disponible.")
