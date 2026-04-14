"""
Tous les graphiques Plotly de l'application.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from .config import BTC_COLOR, MODEL_COLORS


# ── Thème commun ──────────────────────────────────────────────

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#e0e0e0", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ── Page Accueil ──────────────────────────────────────────────

def plot_recent_price(df: pd.DataFrame, days: int = 90) -> go.Figure:
    """Graphique du prix des N derniers jours avec MA7 et MA30."""
    recent = df.tail(days).copy()

    fig = go.Figure()

    # Zone remplie sous la courbe
    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['Close'],
        fill='tozeroy',
        fillcolor='rgba(247,147,26,0.08)',
        line=dict(color=BTC_COLOR, width=2.5),
        name='Prix BTC',
        hovertemplate='%{x|%d %b %Y}<br><b>$%{y:,.0f}</b><extra></extra>',
    ))

    if 'MA7' in recent.columns:
        fig.add_trace(go.Scatter(
            x=recent['Date'], y=recent['MA7'],
            line=dict(color='#2196F3', width=1.2, dash='dot'),
            name='MA 7j',
            hovertemplate='MA7: $%{y:,.0f}<extra></extra>',
        ))

    if 'MA30' in recent.columns:
        fig.add_trace(go.Scatter(
            x=recent['Date'], y=recent['MA30'],
            line=dict(color='#FF9800', width=1.2, dash='dash'),
            name='MA 30j',
            hovertemplate='MA30: $%{y:,.0f}<extra></extra>',
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Prix Bitcoin — {days} derniers jours",
        yaxis_title="Prix (USD)",
        legend=dict(orientation='h', y=1.02, x=0),
        height=360,
    )
    fig.update_yaxes(tickprefix='$', tickformat=',.0f')
    return fig


# ── Candlestick OHLC ─────────────────────────────────────────

def plot_candlestick(df: pd.DataFrame,
                     start_date=None, end_date=None,
                     show_ma: bool = True,
                     show_bb: bool = False) -> go.Figure:
    """
    Graphique Candlestick OHLC avec volume, MA et Bollinger Bands optionnels.
    """
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= df['Date'] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df['Date'] <= pd.Timestamp(end_date)
    d = df[mask].copy()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    # ── Bollinger Bands ──
    if show_bb and 'BB_upper' in d.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([d['Date'], d['Date'][::-1]]),
            y=pd.concat([d['BB_upper'], d['BB_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(33,150,243,0.07)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Bollinger Bands',
            showlegend=True,
        ), row=1, col=1)

    # ── Candlestick principal ──
    fig.add_trace(go.Candlestick(
        x=d['Date'],
        open=d['Open'], high=d['High'],
        low=d['Low'],   close=d['Close'],
        increasing_line_color='#4CAF50',
        decreasing_line_color='#F44336',
        increasing_fillcolor='#4CAF50',
        decreasing_fillcolor='#F44336',
        name='OHLC',
        line=dict(width=1),
    ), row=1, col=1)

    # ── Moyennes mobiles ──
    if show_ma:
        for ma, color, dash in [
            ('MA7',  '#F7931A', 'dot'),
            ('MA30', '#2196F3', 'dash'),
            ('MA90', '#9C27B0', 'longdash'),
        ]:
            if ma in d.columns:
                fig.add_trace(go.Scatter(
                    x=d['Date'], y=d[ma],
                    line=dict(color=color, width=1.2, dash=dash),
                    name=ma,
                    hovertemplate=f'{ma}: $%{{y:,.0f}}<extra></extra>',
                ), row=1, col=1)

    # ── Volume ──
    if 'Volume' in d.columns and 'Change_pct' in d.columns:
        vol_colors = np.where(
            d['Change_pct'] >= 0,
            'rgba(76,175,80,0.5)',
            'rgba(244,67,54,0.5)'
        )
        fig.add_trace(go.Bar(
            x=d['Date'], y=d['Volume'],
            marker_color=list(vol_colors),
            name='Volume',
            showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title='Bitcoin BTC/USD — Graphique Chandelier (OHLC)',
        yaxis_title='Prix (USD)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.02, x=0),
        height=520,
    )
    fig.update_yaxes(tickprefix='$', tickformat=',.0f', row=1, col=1)
    return fig


# ── Page Exploration ──────────────────────────────────────────

def plot_price_history(df: pd.DataFrame,
                       start_date=None, end_date=None,
                       show_ma: bool = True,
                       show_bb: bool = False) -> go.Figure:
    """Graphique historique complet avec indicateurs optionnels."""
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= df['Date'] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df['Date'] <= pd.Timestamp(end_date)
    d = df[mask]

    fig = go.Figure()

    if show_bb and 'BB_upper' in d.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([d['Date'], d['Date'][::-1]]),
            y=pd.concat([d['BB_upper'], d['BB_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(33,150,243,0.08)',
            line=dict(color='rgba(33,150,243,0)'),
            name='Bollinger Bands',
            showlegend=True,
        ))

    fig.add_trace(go.Scatter(
        x=d['Date'], y=d['Close'],
        line=dict(color=BTC_COLOR, width=1.8),
        name='Prix BTC/USD',
        hovertemplate='%{x|%d %b %Y}<br>$%{y:,.2f}<extra></extra>',
    ))

    if show_ma:
        for ma, color, dash in [('MA7', '#4CAF50', 'dot'),
                                  ('MA30', '#2196F3', 'dash'),
                                  ('MA90', '#FF9800', 'longdash')]:
            if ma in d.columns:
                fig.add_trace(go.Scatter(
                    x=d['Date'], y=d[ma],
                    line=dict(color=color, width=1, dash=dash),
                    name=ma,
                    hovertemplate=f'{ma}: $%{{y:,.0f}}<extra></extra>',
                ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Évolution du Prix Bitcoin (BTC/USD)",
        yaxis_title="Prix (USD)",
        xaxis_title="Date",
        legend=dict(orientation='h', y=1.02, x=0),
        height=420,
    )
    fig.update_yaxes(tickprefix='$', tickformat=',.0f')
    return fig


def plot_volume(df: pd.DataFrame, start_date=None, end_date=None) -> go.Figure:
    """Graphique volume avec coloration hausse/baisse."""
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= df['Date'] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df['Date'] <= pd.Timestamp(end_date)
    d = df[mask].copy()
    d['color'] = np.where(d['Change_pct'] >= 0, '#4CAF50', '#F44336')

    fig = go.Figure(go.Bar(
        x=d['Date'], y=d['Volume'],
        marker_color=d['color'],
        opacity=0.7,
        name='Volume',
        hovertemplate='%{x|%d %b %Y}<br>Volume: %{y:,.0f}<extra></extra>',
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Volume d'Échanges",
        yaxis_title="Volume",
        height=280,
    )
    return fig


def plot_returns_distribution(df: pd.DataFrame) -> go.Figure:
    """Distribution des rendements journaliers."""
    returns = df['daily_return'].dropna()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Distribution des Rendements",
                                        "Volatilité Glissante 30j"))

    fig.add_trace(go.Histogram(
        x=returns, nbinsx=120,
        marker_color=BTC_COLOR, opacity=0.75,
        name='Rendements',
        hovertemplate='Rendement: %{x:.2f}%<extra></extra>',
    ), row=1, col=1)

    fig.add_vline(x=returns.mean(), line_color='red',
                  line_dash='dash', row=1, col=1)
    fig.add_vline(x=0, line_color='white',
                  line_width=1, row=1, col=1)

    if 'volatility_30' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['volatility_30'],
            line=dict(color='#9C27B0', width=1.5),
            fill='tozeroy', fillcolor='rgba(156,39,176,0.1)',
            name='Volatilité 30j',
        ), row=1, col=2)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        showlegend=False,
        height=360,
    )
    return fig


def plot_rsi(df: pd.DataFrame, start_date=None, end_date=None) -> go.Figure:
    """Graphique RSI avec zones surachat/survente."""
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= df['Date'] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df['Date'] <= pd.Timestamp(end_date)
    d = df[mask]

    fig = go.Figure()

    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(244,67,54,0.1)',
                  line_width=0, name='Surachat')
    fig.add_hrect(y0=0, y1=30, fillcolor='rgba(76,175,80,0.1)',
                  line_width=0, name='Survente')

    fig.add_trace(go.Scatter(
        x=d['Date'], y=d['RSI'],
        line=dict(color='#2196F3', width=1.5),
        name='RSI 14',
    ))
    fig.add_hline(y=70, line_color='#F44336', line_dash='dash', line_width=1)
    fig.add_hline(y=30, line_color='#4CAF50', line_dash='dash', line_width=1)
    fig.add_hline(y=50, line_color='gray', line_dash='dot', line_width=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="RSI (14 jours)",
        yaxis=dict(range=[0, 100], title='RSI'),
        height=280,
    )
    return fig


def plot_macd(df: pd.DataFrame, start_date=None, end_date=None) -> go.Figure:
    """Graphique MACD."""
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= df['Date'] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df['Date'] <= pd.Timestamp(end_date)
    d = df[mask].copy()

    fig = make_subplots(rows=2, cols=1, row_heights=[0.4, 0.6],
                        shared_xaxes=True,
                        subplot_titles=("Histogramme MACD", "MACD vs Signal"))

    colors = np.where(d['MACD_hist'] >= 0, '#4CAF50', '#F44336')
    fig.add_trace(go.Bar(
        x=d['Date'], y=d['MACD_hist'],
        marker_color=colors, opacity=0.7, name='Histogramme',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d['Date'], y=d['MACD'],
        line=dict(color='#2196F3', width=1.5), name='MACD',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=d['Date'], y=d['MACD_signal'],
        line=dict(color='#FF9800', width=1.5, dash='dash'), name='Signal',
    ), row=2, col=1)

    fig.update_layout(**LAYOUT_DEFAULTS, height=380, showlegend=True)
    return fig


# ── Page Prédiction ───────────────────────────────────────────

def plot_prediction_vs_real(test_dates, y_true: np.ndarray,
                             y_pred: np.ndarray,
                             model_name: str,
                             future_dates=None,
                             future_pred=None) -> go.Figure:
    """
    Graphique : réel vs prédit sur le jeu de test + prédictions futures optionnelles.
    """
    color = MODEL_COLORS.get(model_name, '#E91E63')

    fig = go.Figure()

    # Zone d'erreur (écart entre réel et prédit)
    fig.add_trace(go.Scatter(
        x=list(test_dates) + list(test_dates)[::-1],
        y=list(y_true) + list(y_pred)[::-1],
        fill='toself',
        fillcolor=f'rgba({_hex_to_rgb(color)},0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False, name='Écart',
    ))

    # Prix réels
    fig.add_trace(go.Scatter(
        x=test_dates, y=y_true,
        line=dict(color=BTC_COLOR, width=2.5),
        name='Prix Réel',
        hovertemplate='%{x|%d %b %Y}<br>Réel: <b>$%{y:,.0f}</b><extra></extra>',
    ))

    # Prédictions test
    fig.add_trace(go.Scatter(
        x=test_dates, y=y_pred,
        line=dict(color=color, width=2, dash='dash'),
        name=f'Prédiction {model_name}',
        hovertemplate='%{x|%d %b %Y}<br>Prédit: <b>$%{y:,.0f}</b><extra></extra>',
    ))

    # Prédictions futures
    if future_dates is not None and future_pred is not None:
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_pred,
            line=dict(color='#00BCD4', width=2.5, dash='dot'),
            name='Prévision Future',
            hovertemplate='%{x|%d %b %Y}<br>Prévision: <b>$%{y:,.0f}</b><extra></extra>',
        ))
        # Ligne verticale séparant test et futur
        fig.add_vline(
            x=str(test_dates[-1]) if hasattr(test_dates[-1], '__str__') else test_dates[-1],
            line_color='gray', line_dash='dot', line_width=1,
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Prédiction {model_name} — BTC/USD",
        yaxis_title="Prix (USD)",
        xaxis_title="Date",
        legend=dict(orientation='h', y=1.02, x=0),
        height=480,
    )
    fig.update_yaxes(tickprefix='$', tickformat=',.0f')
    return fig


def plot_future_with_ci(future_dates, future_pred: np.ndarray,
                        last_price: float, model_name: str,
                        ci_multiplier: float = 0.03) -> go.Figure:
    """
    Graphique de prévision future avec intervalle de confiance ±σ croissant.
    L'incertitude augmente avec l'horizon (effet cône).
    """
    color     = MODEL_COLORS.get(model_name, '#00BCD4')
    n         = len(future_pred)
    # σ croissant : ±3% au jour 1, ±(3% × √t) au jour t
    sigma     = np.array([last_price * ci_multiplier * np.sqrt(t+1) for t in range(n)])
    upper     = future_pred + sigma
    lower     = np.maximum(future_pred - sigma, 0)

    fig = go.Figure()

    # Zone de confiance (cône)
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill='toself',
        fillcolor=f'rgba({_hex_to_rgb(color)},0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Intervalle de confiance 95%',
        showlegend=True,
    ))

    # Ligne haute et basse en pointillés fins
    fig.add_trace(go.Scatter(
        x=future_dates, y=upper,
        line=dict(color=color, width=0.8, dash='dot'),
        showlegend=False, name='Borne haute',
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=lower,
        line=dict(color=color, width=0.8, dash='dot'),
        showlegend=False, name='Borne basse',
    ))

    # Courbe principale
    trend_color = '#4CAF50' if future_pred[-1] > last_price else '#F44336'
    all_prices  = [last_price] + list(future_pred)
    fig.add_trace(go.Scatter(
        x=[future_dates[0]] + list(future_dates),
        y=all_prices,
        line=dict(color=trend_color, width=3),
        name='Prévision centrale',
        hovertemplate='%{x|%d %b %Y}<br><b>$%{y:,.0f}</b><extra></extra>',
    ))

    fig.add_hline(y=last_price, line_color='gray', line_dash='dot', line_width=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f'Prévision {model_name} avec Intervalle de Confiance',
        yaxis_title='Prix (USD)',
        legend=dict(orientation='h', y=1.02, x=0),
        height=320,
    )
    fig.update_yaxes(tickprefix='$', tickformat=',.0f')
    return fig


def plot_two_models_comparison(test_dates, y_true: np.ndarray,
                                preds1: np.ndarray, name1: str,
                                preds2: np.ndarray, name2: str) -> go.Figure:
    """Superpose 2 modèles sur le même graphique pour comparaison directe."""
    color1 = MODEL_COLORS.get(name1, '#2196F3')
    color2 = MODEL_COLORS.get(name2, '#4CAF50')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=test_dates, y=y_true,
        line=dict(color=BTC_COLOR, width=2.5),
        name='Prix Réel',
        hovertemplate='Réel: <b>$%{y:,.0f}</b><extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=test_dates[:len(preds1)], y=preds1,
        line=dict(color=color1, width=2, dash='dash'),
        name=name1,
        hovertemplate=f'{name1}: <b>$%{{y:,.0f}}</b><extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=test_dates[:len(preds2)], y=preds2,
        line=dict(color=color2, width=2, dash='dot'),
        name=name2,
        hovertemplate=f'{name2}: <b>$%{{y:,.0f}}</b><extra></extra>',
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f'Comparaison : {name1}  vs  {name2}',
        yaxis_title='Prix (USD)',
        legend=dict(orientation='h', y=1.02, x=0),
        height=440,
    )
    fig.update_yaxes(tickprefix='$', tickformat=',.0f')
    return fig


def plot_future_only(future_dates, future_pred: np.ndarray,
                     last_price: float, model_name: str) -> go.Figure:
    """Zoom sur la prédiction future uniquement."""
    color = MODEL_COLORS.get(model_name, '#00BCD4')

    fig = go.Figure()

    all_prices = [last_price] + list(future_pred)
    trend_color = '#4CAF50' if future_pred[-1] > last_price else '#F44336'

    fig.add_trace(go.Scatter(
        x=[future_dates[0]] + list(future_dates),
        y=all_prices,
        line=dict(color=trend_color, width=3),
        fill='tozeroy',
        fillcolor=f'rgba({_hex_to_rgb(trend_color)},0.1)',
        name='Prévision',
        hovertemplate='%{x|%d %b %Y}<br><b>$%{y:,.0f}</b><extra></extra>',
    ))

    fig.add_hline(y=last_price, line_color='gray',
                  line_dash='dot', line_width=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Prévision {model_name} — Prochains jours",
        yaxis_title="Prix (USD)",
        height=300,
    )
    fig.update_yaxes(tickprefix='$', tickformat=',.0f')
    return fig


# ── Page Comparaison ──────────────────────────────────────────

def plot_metrics_bars(df_results: pd.DataFrame) -> go.Figure:
    """Bar chart comparatif MAE / RMSE / MAPE pour tous les modèles."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('MAE ($)', 'RMSE ($)', 'MAPE (%)'),
    )

    for col_idx, metric in enumerate(['MAE', 'RMSE', 'MAPE'], start=1):
        if metric not in df_results.columns:
            continue
        colors = [MODEL_COLORS.get(m, BTC_COLOR) for m in df_results['Model']]
        best_idx = df_results[metric].idxmin()

        bar_colors = []
        for i, c in enumerate(colors):
            bar_colors.append(c if i != best_idx else BTC_COLOR)

        fig.add_trace(go.Bar(
            x=df_results['Model'],
            y=df_results[metric],
            marker_color=bar_colors,
            marker_line_color=['gold' if i == best_idx else c
                               for i, c in enumerate(colors)],
            marker_line_width=[3 if i == best_idx else 0.5
                               for i in range(len(colors))],
            text=df_results[metric].apply(
                lambda v: f'${v:,.0f}' if metric != 'MAPE' else f'{v:.2f}%'),
            textposition='outside',
            textfont=dict(size=10),
            name=metric,
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Comparaison des Performances — Tous les Modèles",
        height=430,
    )
    fig.update_xaxes(tickangle=30)
    return fig


def plot_radar_chart(df_results: pd.DataFrame) -> go.Figure:
    """Radar chart multi-métriques — comparaison visuelle des modèles."""
    metrics = ['MAE', 'RMSE', 'MAPE']
    available = [m for m in metrics if m in df_results.columns]
    if not available:
        return go.Figure()

    df = df_results.copy()
    # Inverser les scores : score = 1 - (valeur normalisée) → plus grand = meilleur
    for m in available:
        rng = df[m].max() - df[m].min()
        if rng > 0:
            df[f'{m}_score'] = 1 - (df[m] - df[m].min()) / rng
        else:
            df[f'{m}_score'] = 1.0

    score_cols = [f'{m}_score' for m in available]
    categories = available + [available[0]]  # fermer le polygone

    fig = go.Figure()
    for _, row in df.iterrows():
        values = [row[c] for c in score_cols] + [row[score_cols[0]]]
        color  = MODEL_COLORS.get(row['Model'], BTC_COLOR)
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor=f'rgba({_hex_to_rgb(color)},0.12)',
            line=dict(color=color, width=2),
            name=row['Model'],
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        polar=dict(
            bgcolor='#1a1a2e',
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(size=9)),
        ),
        title="Performance Relative des Modèles (Radar)",
        height=430,
        legend=dict(orientation='h', y=-0.15),
    )
    return fig


def plot_all_predictions(test_dates, y_true: np.ndarray,
                          predictions_dict: dict) -> go.Figure:
    """Superposition de toutes les courbes de prédiction."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=test_dates, y=y_true,
        line=dict(color=BTC_COLOR, width=3),
        name='Prix Réel',
        hovertemplate='%{x|%d %b %Y}<br>Réel: <b>$%{y:,.0f}</b><extra></extra>',
    ))

    for model_name, y_pred in predictions_dict.items():
        min_len = min(len(test_dates), len(y_pred))
        color   = MODEL_COLORS.get(model_name, '#E91E63')
        fig.add_trace(go.Scatter(
            x=test_dates[:min_len], y=y_pred[:min_len],
            line=dict(color=color, width=1.5, dash='dash'),
            name=model_name,
            opacity=0.85,
            hovertemplate=f'{model_name}: <b>$%{{y:,.0f}}</b><extra></extra>',
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Comparaison de Toutes les Prédictions",
        yaxis_title="Prix (USD)",
        xaxis_title="Date",
        legend=dict(orientation='h', y=1.02, x=0),
        height=480,
    )
    fig.update_yaxes(tickprefix='$', tickformat=',.0f')
    return fig


# ── Feature Importance ───────────────────────────────────────

def plot_feature_importance(df_imp: pd.DataFrame, model_name: str,
                             top_n: int = 15) -> go.Figure:
    """Bar chart horizontal des N features les plus importantes."""
    df = df_imp.head(top_n).copy()[::-1]  # inverser pour affichage haut → bas

    # Couleur dégradée selon importance
    max_imp = df['Importance_%'].max()
    colors  = [f'rgba(247,147,26,{0.4 + 0.6*(v/max_imp):.2f})'
               for v in df['Importance_%']]

    fig = go.Figure(go.Bar(
        x=df['Importance_%'],
        y=df['Feature'],
        orientation='h',
        marker_color=colors,
        marker_line_color='rgba(247,147,26,0.5)',
        marker_line_width=1,
        text=df['Importance_%'].apply(lambda v: f'{v:.1f}%'),
        textposition='outside',
        textfont=dict(size=10, color='#e0e0e0'),
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f'Feature Importance — {model_name} (Top {top_n})',
        xaxis_title='Importance (%)',
        height=max(300, top_n * 28),
        margin=dict(l=120, r=60, t=50, b=40),
    )
    return fig


# ── Utilitaire ────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    """Convertit #RRGGBB en 'R,G,B' pour rgba()."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
