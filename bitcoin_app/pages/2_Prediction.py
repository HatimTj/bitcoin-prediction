"""
Page 2 — Prédiction : sélection modèle, prévision future, comparaison 2 modèles,
          feature importance, export CSV, intervalle de confiance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Prédiction — BTC", page_icon="🔮", layout="wide")

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
    .trend-up   { background:#1a3a1a; border:1px solid #4CAF50; border-radius:10px; padding:15px; text-align:center; }
    .trend-down { background:#3a1a1a; border:1px solid #F44336; border-radius:10px; padding:15px; text-align:center; }
    .info-box   { background:#1a2a1a; border:1px solid #4CAF5055; border-radius:8px; padding:10px 14px; font-size:0.85rem; color:#ccc; }
</style>
""", unsafe_allow_html=True)

from src.data_loader import load_data, get_preprocessed_sequences
from src.predictor import (available_models, predict_on_test, predict_future,
                            model_info, get_feature_importance)
from src.metrics import compute_metrics
from src.visualizations import (plot_prediction_vs_real, plot_future_only,
                                  plot_future_with_ci, plot_two_models_comparison,
                                  plot_feature_importance)

df = load_data()

st.title("🔮 Prédiction du Prix Bitcoin")
st.divider()

# ── Données prétraitées ───────────────────────────────────────
with st.spinner("Chargement des données..."):
    X_test, y_test_real, test_df, scaler, scaled, df_full = get_preprocessed_sequences()

n         = len(scaled)
train_end = int(n * 0.80)
val_end   = int(n * 0.90)
test_dates = test_df['Date'].values
available  = available_models()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Paramètres")

    mode = st.radio("Mode", ["1 modèle", "Comparer 2 modèles"], horizontal=True)

    if mode == "1 modèle":
        selected_model = st.selectbox(
            "Modèle",
            options=available,
            format_func=lambda m: f"{m}  —  {model_info(m)}" if model_info(m) != 'Deep Learning' else m,
        )
        model2 = None
    else:
        col_a, col_b = st.columns(2)
        selected_model = st.selectbox("Modèle 1", options=available, index=0)
        model2 = st.selectbox(
            "Modèle 2",
            options=[m for m in available if m != selected_model],
            index=0,
        )

    horizon = st.radio(
        "Horizon prévision",
        options=[7, 14, 30],
        format_func=lambda x: f"{x} jours",
        horizontal=True,
    )
    show_future = st.toggle("Prévision future", value=True)
    show_ci     = st.toggle("Intervalle de confiance", value=True)
    show_fi     = st.toggle("Feature Importance", value=False)

    st.divider()
    st.markdown(f"""
    <div class='info-box'>
        <b>Modèles disponibles :</b><br>
        {''.join(f"• {m}<br>" for m in available)}
        <span style='color:#aaa'>+ LSTM/GRU si TensorFlow</span>
    </div>
    """, unsafe_allow_html=True)


# ── Helper : prédire un modèle ────────────────────────────────
SPINNER_LABELS = {
    'XGBoost':             'Entraînement XGBoost (~2s)...',
    'Random Forest':       'Entraînement Random Forest (~3s)...',
    'Régression Linéaire': 'Entraînement Régression Linéaire (~1s)...',
    'ARIMA':               'Calcul ARIMA (première fois ~30s)...',
}

def run_prediction(model_name):
    label = SPINNER_LABELS.get(model_name, f'Prédiction {model_name}...')
    with st.spinner(label):
        try:
            return predict_on_test(
                model_name, X_test, scaler,
                scaled=scaled, train_end=train_end,
                val_end=val_end, test_start=val_end,
            )
        except Exception as e:
            st.error(f"Erreur [{model_name}] : {e}")
            st.stop()

def run_future(model_name, horizon):
    try:
        return predict_future(
            model_name, scaler, df_full['Close'].values, horizon,
            scaled=scaled, train_end=train_end, val_end=val_end,
        )
    except Exception as e:
        st.warning(f"Prévision future [{model_name}] : {e}")
        return None


# ════════════════════════════════════════════════════════════
# MODE 1 MODÈLE
# ════════════════════════════════════════════════════════════
if mode == "1 modèle":

    y_pred = run_prediction(selected_model)
    min_len = min(len(y_test_real), len(y_pred))
    y_true  = y_test_real[:min_len]
    y_pred  = y_pred[:min_len]
    dates   = test_dates[:min_len]

    # ── Métriques ──────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred, selected_model)
    st.markdown(f"### Résultats — **{selected_model}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",               f"${metrics['MAE']:,.2f}")
    c2.metric("RMSE",              f"${metrics['RMSE']:,.2f}")
    c3.metric("MAPE",              f"{metrics['MAPE']:.2f}%")
    c4.metric("Direction Accuracy",f"{metrics['DA']:.1f}%")
    st.divider()

    # ── Prévision future ────────────────────────────────────────
    future_pred  = None
    future_dates = None
    if show_future:
        with st.spinner(f"Prévision {horizon} jours..."):
            future_pred = run_future(selected_model, horizon)
        if future_pred is not None:
            last_date    = pd.Timestamp(test_dates[-1])
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon, freq='D',
            )

    # ── Graphique principal ─────────────────────────────────────
    fig = plot_prediction_vs_real(
        dates, y_true, y_pred, selected_model,
        future_dates=future_dates,
        future_pred=future_pred,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Prévision future détaillée ──────────────────────────────
    if show_future and future_pred is not None and future_dates is not None:
        st.divider()
        col_fut, col_trend = st.columns([2, 1])
        last_p  = float(y_true[-1])
        final_p = float(future_pred[-1])
        var     = (final_p / last_p - 1) * 100

        with col_fut:
            st.markdown(f"#### Prévision des {horizon} prochains jours")
            if show_ci:
                fig_fut = plot_future_with_ci(
                    future_dates, future_pred, last_p, selected_model)
            else:
                fig_fut = plot_future_only(
                    future_dates, future_pred, last_p, selected_model)
            st.plotly_chart(fig_fut, use_container_width=True)

        with col_trend:
            st.markdown("#### Tendance")
            is_up = final_p > last_p
            cls   = 'trend-up' if is_up else 'trend-down'
            icon  = '📈' if is_up else '📉'
            color = '#4CAF50' if is_up else '#F44336'
            label = 'HAUSSE' if is_up else 'BAISSE'
            sign  = '+' if is_up else ''
            st.markdown(f"""
            <div class='{cls}'>
                <div style='font-size:2.5rem;'>{icon}</div>
                <div style='font-size:1.4rem;color:{color};font-weight:700;'>{label}</div>
                <div style='font-size:1.1rem;color:{color};'>{sign}{var:.1f}%</div>
                <div style='color:#aaa;margin-top:6px;font-size:0.9rem;'>
                    ${last_p:,.0f} → <b>${final_p:,.0f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # Tableau des prévisions
            sigma = np.array([last_p * 0.03 * np.sqrt(t+1) for t in range(len(future_pred))])
            fc_df = pd.DataFrame({
                'Date':      future_dates.strftime('%d/%m/%Y'),
                'Prédit ($)': [f"${p:,.0f}" for p in future_pred],
                'Min ($)':   [f"${max(0,p-s):,.0f}" for p, s in zip(future_pred, sigma)],
                'Max ($)':   [f"${p+s:,.0f}" for p, s in zip(future_pred, sigma)],
                'Var.':      [f"{((p/last_p)-1)*100:+.1f}%" for p in future_pred],
            })
            st.dataframe(fc_df, use_container_width=True, hide_index=True)

            # ── Export CSV ──
            st.divider()
            csv_bytes = fc_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Télécharger les prévisions (CSV)",
                data=csv_bytes,
                file_name=f"previsions_{selected_model}_{horizon}j.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── Feature Importance ──────────────────────────────────────
    if show_fi:
        st.divider()
        st.markdown("#### Feature Importance")
        df_imp = get_feature_importance(selected_model, scaler, scaled, train_end, val_end)
        if df_imp is not None:
            col_fi1, col_fi2 = st.columns([2, 1])
            with col_fi1:
                st.plotly_chart(
                    plot_feature_importance(df_imp, selected_model, top_n=15),
                    use_container_width=True,
                )
            with col_fi2:
                st.markdown("**Top 10 features**")
                st.dataframe(
                    df_imp.head(10)[['Feature', 'Importance_%']].rename(
                        columns={'Importance_%': 'Importance (%)'}),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "**lag_1** = prix d'hier | "
                    "**ma_7** = moyenne 7 jours | "
                    "**return_1** = rendement journalier"
                )
        else:
            st.info(f"Feature importance non disponible pour {selected_model}.")

    # ── Export test predictions ─────────────────────────────────
    st.divider()
    test_export = pd.DataFrame({
        'Date':        pd.to_datetime(dates).strftime('%d/%m/%Y'),
        'Réel ($)':    [f"${v:,.0f}" for v in y_true],
        'Prédit ($)':  [f"${v:,.0f}" for v in y_pred],
        'Erreur ($)':  [f"${abs(r-p):,.0f}" for r, p in zip(y_true, y_pred)],
        'Erreur (%)':  [f"{abs(r-p)/r*100:.2f}%" for r, p in zip(y_true, y_pred)],
    })
    st.download_button(
        label="⬇️ Télécharger les prédictions Test (CSV)",
        data=test_export.to_csv(index=False).encode('utf-8'),
        file_name=f"predictions_test_{selected_model}.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════
# MODE COMPARAISON 2 MODÈLES
# ════════════════════════════════════════════════════════════
else:
    st.markdown(f"### Comparaison : **{selected_model}**  vs  **{model2}**")

    y_pred1 = run_prediction(selected_model)
    y_pred2 = run_prediction(model2)

    min_len = min(len(y_test_real), len(y_pred1), len(y_pred2))
    y_true  = y_test_real[:min_len]
    dates   = test_dates[:min_len]
    y_pred1 = y_pred1[:min_len]
    y_pred2 = y_pred2[:min_len]

    # ── Métriques côte à côte ──────────────────────────────────
    m1 = compute_metrics(y_true, y_pred1, selected_model)
    m2 = compute_metrics(y_true, y_pred2, model2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {selected_model}")
        ca, cb, cc, cd = st.columns(4)
        ca.metric("MAE",  f"${m1['MAE']:,.0f}")
        cb.metric("RMSE", f"${m1['RMSE']:,.0f}")
        cc.metric("MAPE", f"{m1['MAPE']:.2f}%")
        cd.metric("DA",   f"{m1['DA']:.1f}%")
    with col2:
        st.markdown(f"#### {model2}")
        ca, cb, cc, cd = st.columns(4)
        ca.metric("MAE",  f"${m2['MAE']:,.0f}",  delta=f"{m2['MAE']-m1['MAE']:+,.0f}$",  delta_color="inverse")
        cb.metric("RMSE", f"${m2['RMSE']:,.0f}", delta=f"{m2['RMSE']-m1['RMSE']:+,.0f}$",delta_color="inverse")
        cc.metric("MAPE", f"{m2['MAPE']:.2f}%",  delta=f"{m2['MAPE']-m1['MAPE']:+.2f}%", delta_color="inverse")
        cd.metric("DA",   f"{m2['DA']:.1f}%",    delta=f"{m2['DA']-m1['DA']:+.1f}%")

    st.divider()

    # ── Graphique superposé ────────────────────────────────────
    fig_cmp = plot_two_models_comparison(
        dates, y_true, y_pred1, selected_model, y_pred2, model2)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Vainqueur ──────────────────────────────────────────────
    winner = selected_model if m1['MAPE'] < m2['MAPE'] else model2
    loser  = model2 if winner == selected_model else selected_model
    gain   = abs(m1['MAPE'] - m2['MAPE'])
    st.success(f"🏆 **{winner}** gagne avec un MAPE de {min(m1['MAPE'],m2['MAPE']):.2f}% "
               f"({gain:.2f}% de mieux que {loser})")

    # ── Export comparaison ─────────────────────────────────────
    comp_df = pd.DataFrame({
        'Date':             pd.to_datetime(dates).strftime('%d/%m/%Y'),
        'Réel ($)':         y_true.astype(int),
        f'{selected_model} ($)': y_pred1.astype(int),
        f'{model2} ($)':         y_pred2.astype(int),
    })
    st.download_button(
        label="⬇️ Télécharger la comparaison (CSV)",
        data=comp_df.to_csv(index=False).encode('utf-8'),
        file_name=f"comparaison_{selected_model}_vs_{model2}.csv",
        mime="text/csv",
    )


st.divider()
st.caption("⚠️ Ces prédictions sont réalisées à des fins académiques uniquement.")
