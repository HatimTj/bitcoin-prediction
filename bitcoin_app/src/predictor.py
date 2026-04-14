"""
Chargement des modèles et inférence.
Modèles disponibles sans TensorFlow :
  - XGBoost        (~2s)
  - Random Forest  (~3s)
  - ARIMA          (~30s)
  - Linear Reg.    (~1s)
Modèles avec TensorFlow (si installé) :
  - LSTM, GRU, Stacked LSTM, CNN-LSTM
"""
import os
import pickle
import numpy as np
import streamlit as st
from .config import MODELS_DIR, MODEL_FILES, SCALER_FILE, LOOKBACK


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING PARTAGÉ (XGBoost, RF, LinReg)
# ─────────────────────────────────────────────────────────────

def _make_features(prices: np.ndarray, idx: int) -> dict:
    feats = {}
    for lag in [1, 2, 3, 5, 7, 14, 21, 30, 60]:
        feats[f'lag_{lag}'] = prices[idx - lag] if idx >= lag else prices[0]
    for w in [7, 14, 30, 60]:
        s = max(0, idx - w)
        feats[f'ma_{w}'] = prices[s:idx].mean() if idx > s else prices[idx]
    for w in [7, 30]:
        s = max(0, idx - w)
        feats[f'std_{w}'] = prices[s:idx].std() if idx - s > 1 else 0.0
    feats['return_1'] = (prices[idx-1] - prices[max(0,idx-2)]) / (prices[max(0,idx-2)] + 1e-8) if idx > 0 else 0.0
    feats['return_7'] = (prices[idx-1] - prices[max(0,idx-7)]) / (prices[max(0,idx-7)] + 1e-8) if idx >= 7 else 0.0
    return feats


def _build_dataset(prices: np.ndarray, start: int, end: int):
    """Construit X, y à partir de prix pour [start, end)."""
    import pandas as pd
    X, y = [], []
    for i in range(start, end):
        X.append(_make_features(prices, i))
        y.append(prices[i])
    return pd.DataFrame(X), np.array(y)


def _get_prices(scaler, scaled: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(scaled).flatten()


# ─────────────────────────────────────────────────────────────
# MODÈLES KERAS (TensorFlow — optionnel)
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_keras_models() -> dict:
    models = {}
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return models
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            try:
                models[name] = load_model(path, compile=False)
            except Exception:
                pass
    return models


# ─────────────────────────────────────────────────────────────
# XGBOOST
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_xgboost_model(_scaler, _scaled, _train_end, _val_end):
    path = os.path.join(MODELS_DIR, 'xgb_model.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    import xgboost as xgb
    prices = _get_prices(_scaler, _scaled)
    X_tr, y_tr = _build_dataset(prices, LOOKBACK, _train_end)
    X_val, y_val = _build_dataset(prices, _train_end, _val_end)

    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=30, eval_metric='rmse',
        verbosity=0, random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    _save_scaler(_scaler)
    return model


# ─────────────────────────────────────────────────────────────
# RANDOM FOREST
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_rf_model(_scaler, _scaled, _train_end, _val_end):
    path = os.path.join(MODELS_DIR, 'rf_model.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    from sklearn.ensemble import RandomForestRegressor
    prices = _get_prices(_scaler, _scaled)
    X_tr, y_tr = _build_dataset(prices, LOOKBACK, _train_end)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    _save_scaler(_scaler)
    return model


# ─────────────────────────────────────────────────────────────
# LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_linreg_model(_scaler, _scaled, _train_end, _val_end):
    path = os.path.join(MODELS_DIR, 'linreg_model.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler as SS
    prices = _get_prices(_scaler, _scaled)
    X_tr, y_tr = _build_dataset(prices, LOOKBACK, _train_end)

    ss = SS()
    X_scaled = ss.fit_transform(X_tr)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y_tr)

    bundle = {'model': model, 'scaler': ss}
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    _save_scaler(_scaler)
    return bundle


# ─────────────────────────────────────────────────────────────
# ARIMA
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_arima_predictions(_scaler, _scaled, _train_end, _val_end):
    """
    Calcule et cache les prédictions ARIMA sur le jeu de test.
    Walk-forward avec refit tous les 30 jours.
    """
    path = os.path.join(MODELS_DIR, 'arima_preds.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings('ignore')

    prices    = _get_prices(_scaler, _scaled)
    train_ser = prices[:_train_end]
    test_ser  = prices[_val_end:]

    # Chercher le meilleur ordre ARIMA
    order = (2, 1, 2)
    try:
        from pmdarima import auto_arima
        am = auto_arima(train_ser[-300:], d=1, max_p=3, max_q=3,
                        stepwise=True, suppress_warnings=True,
                        error_action='ignore')
        order = am.order
    except Exception:
        pass

    history = list(train_ser)
    preds   = []
    step    = 30
    fit_obj = None

    for t in range(len(test_ser)):
        if t % step == 0 or fit_obj is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit_obj = ARIMA(history, order=order).fit()
            block = fit_obj.forecast(steps=min(step, len(test_ser) - t))
        preds.append(float(block[t % step]))
        history.append(test_ser[t])

    result = {'preds': np.array(preds), 'order': order}
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    return result


# ─────────────────────────────────────────────────────────────
# UTILITAIRES INTERNES
# ─────────────────────────────────────────────────────────────

def get_feature_importance(model_name: str, scaler, scaled, train_end, val_end) -> pd.DataFrame:
    """
    Retourne un DataFrame {feature, importance} trié pour XGBoost ou Random Forest.
    Retourne None pour les autres modèles.
    """
    import pandas as pd

    if model_name == 'XGBoost':
        m = get_xgboost_model(scaler, scaled, train_end, val_end)
        scores = m.feature_importances_
    elif model_name == 'Random Forest':
        m = get_rf_model(scaler, scaled, train_end, val_end)
        scores = m.feature_importances_
    else:
        return None

    # Reconstruire les noms de features
    prices = _get_prices(scaler, scaled)
    import numpy as np
    sample_feats = list(_make_features(prices, LOOKBACK).keys())
    df_imp = pd.DataFrame({'Feature': sample_feats, 'Importance': scores})
    df_imp = df_imp.sort_values('Importance', ascending=False).reset_index(drop=True)
    df_imp['Importance_%'] = (df_imp['Importance'] / df_imp['Importance'].sum() * 100).round(2)
    return df_imp


def _save_scaler(scaler):
    path = os.path.join(MODELS_DIR, SCALER_FILE)
    if not os.path.exists(path):
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)


# ─────────────────────────────────────────────────────────────
# API PUBLIQUE
# ─────────────────────────────────────────────────────────────

# Ordre d'affichage et labels des modèles disponibles sans TF
_SKLEARN_MODELS = {
    'XGBoost':            ('XGBoost',      'Boosting — ~2s'),
    'Random Forest':      ('Random Forest','Ensemble — ~3s'),
    'Régression Linéaire':('Régression Linéaire', 'Baseline — ~1s'),
    'ARIMA':              ('ARIMA',        'Statistique — ~30s'),
}


def available_models() -> list:
    """Retourne tous les modèles sélectionnables."""
    models = []

    # Keras (TF) en premier si disponible
    keras = load_keras_models()
    models.extend(list(keras.keys()))

    # Sklearn / Stats (toujours disponibles)
    for name in _SKLEARN_MODELS:
        if name not in models:
            models.append(name)

    return models


def model_info(name: str) -> str:
    """Retourne la description courte du modèle."""
    info = _SKLEARN_MODELS.get(name)
    return info[1] if info else 'Deep Learning'


def models_are_ready() -> bool:
    return True


def predict_on_test(model_name: str, X_test: np.ndarray, scaler,
                    scaled=None, train_end=None, val_end=None,
                    test_start=None) -> np.ndarray:
    """Prédictions sur le jeu de test pour n'importe quel modèle."""

    # ── Keras ──────────────────────────────────────────────────
    keras_models = load_keras_models()
    if model_name in keras_models:
        pred_scaled = keras_models[model_name].predict(X_test, verbose=0)
        return scaler.inverse_transform(pred_scaled).flatten()

    prices     = _get_prices(scaler, scaled)
    n_total    = len(prices)
    t_start    = test_start if test_start is not None else val_end

    # ── XGBoost ────────────────────────────────────────────────
    if model_name == 'XGBoost':
        import pandas as pd
        m       = get_xgboost_model(scaler, scaled, train_end, val_end)
        X_te, _ = _build_dataset(prices, t_start, n_total)
        return m.predict(X_te)

    # ── Random Forest ──────────────────────────────────────────
    if model_name == 'Random Forest':
        import pandas as pd
        m       = get_rf_model(scaler, scaled, train_end, val_end)
        X_te, _ = _build_dataset(prices, t_start, n_total)
        return m.predict(X_te)

    # ── Régression Linéaire ────────────────────────────────────
    if model_name == 'Régression Linéaire':
        import pandas as pd
        bundle  = get_linreg_model(scaler, scaled, train_end, val_end)
        X_te, _ = _build_dataset(prices, t_start, n_total)
        X_sc    = bundle['scaler'].transform(X_te)
        return bundle['model'].predict(X_sc)

    # ── ARIMA ──────────────────────────────────────────────────
    if model_name == 'ARIMA':
        result = get_arima_predictions(scaler, scaled, train_end, val_end)
        return result['preds']

    raise ValueError(f"Modèle '{model_name}' non reconnu.")


def predict_future(model_name: str, scaler, last_prices: np.ndarray,
                   n_days: int = 30,
                   scaled=None, train_end=None, val_end=None) -> np.ndarray:
    """Prévision future rolling window sur n_days jours."""

    # ── Keras ──────────────────────────────────────────────────
    keras_models = load_keras_models()
    if model_name in keras_models:
        model  = keras_models[model_name]
        window = last_prices[-LOOKBACK:].copy().reshape(-1, 1)
        seq    = scaler.transform(window).flatten()
        preds  = []
        for _ in range(n_days):
            X = seq[-LOOKBACK:].reshape(1, LOOKBACK, 1)
            p = float(scaler.inverse_transform([[model.predict(X, verbose=0)[0, 0]]])[0, 0])
            preds.append(p)
            seq = np.append(seq, scaler.transform([[p]])[0, 0])
        return np.array(preds)

    # ── Sklearn / XGBoost / RF / LinReg ────────────────────────
    if model_name in ('XGBoost', 'Random Forest', 'Régression Linéaire'):
        import pandas as pd

        if model_name == 'XGBoost':
            m = get_xgboost_model(scaler, scaled, train_end, val_end)
            predict_fn = lambda X: m.predict(X)[0]
        elif model_name == 'Random Forest':
            m = get_rf_model(scaler, scaled, train_end, val_end)
            predict_fn = lambda X: m.predict(X)[0]
        else:
            bundle = get_linreg_model(scaler, scaled, train_end, val_end)
            def predict_fn(X):
                return bundle['model'].predict(bundle['scaler'].transform(X))[0]

        history = list(last_prices)
        preds   = []
        for _ in range(n_days):
            idx   = len(history)
            feats = pd.DataFrame([_make_features(np.array(history), idx)])
            p     = float(predict_fn(feats))
            preds.append(p)
            history.append(p)
        return np.array(preds)

    # ── ARIMA (prévision simple avec le dernier fit) ────────────
    if model_name == 'ARIMA':
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings('ignore')

        result = get_arima_predictions(scaler, scaled, train_end, val_end)
        order  = result.get('order', (2, 1, 2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = ARIMA(last_prices[-200:], order=order).fit()
        return fit.forecast(steps=n_days)

    raise ValueError(f"Prévision future non supportée pour '{model_name}'.")
