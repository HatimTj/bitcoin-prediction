"""
=============================================================
  PFA — Prévision de Séries Temporelles | EMSI IA & Data
  Module : models.py
  Auteur : Hatim Tajimi
  Encadrant : Pr. Idriss BARBARA
=============================================================
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────
# MODÈLE 1 : ARIMA (Baseline Statistique)
# ──────────────────────────────────────────────────────────

def build_arima_model(train_series, test_series, order=(5, 1, 0)):
    """
    Construit et entraîne un modèle ARIMA pour la prévision du prix Bitcoin.
    Stratégie : fit unique sur train, puis walk-forward avec refit tous les 50 pas.

    Args:
        train_series : Série d'entraînement (prix réels, non normalisés)
        test_series  : Série de test (pour prévision pas-à-pas)
        order        : (p, d, q) — paramètres ARIMA

    Returns:
        predictions  : Prédictions sur le jeu de test
        best_order   : Ordre ARIMA utilisé
    """
    try:
        from pmdarima import auto_arima
        print("\n🔍 Auto-ARIMA — recherche des paramètres optimaux (p, d, q)...")
        # Utiliser un sous-ensemble pour accélérer la recherche
        sample = train_series[-500:] if len(train_series) > 500 else train_series
        model = auto_arima(
            sample,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            d=1,
            seasonal=False,
            information_criterion='aic',
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        best_order = model.order
        print(f"   ✅ Meilleur ordre ARIMA trouvé : {best_order}")
    except ImportError:
        print("   ⚠️  pmdarima non disponible — utilisation de l'ordre par défaut")
        best_order = order

    from statsmodels.tsa.arima.model import ARIMA
    import warnings as _warnings

    history = list(train_series)
    predictions = []
    refit_every = 50   # Refit complet tous les 50 pas (bien plus rapide)
    model_fit = None

    print(f"\n⏳ Prévision ARIMA sur {len(test_series)} jours (refit ×{refit_every})...")

    for t in range(len(test_series)):
        try:
            # Refit complet à t=0 puis tous les `refit_every` pas
            if t % refit_every == 0 or model_fit is None:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    arima_model = ARIMA(history, order=best_order)
                    model_fit = arima_model.fit()
                steps_ahead = min(refit_every, len(test_series) - t)
                forecast_block = model_fit.forecast(steps=steps_ahead)

            idx_in_block = t % refit_every
            yhat = float(forecast_block[idx_in_block])
        except Exception:
            yhat = history[-1]

        predictions.append(yhat)
        history.append(test_series[t])

        if (t + 1) % 100 == 0 or (t + 1) == len(test_series):
            print(f"   → {t+1}/{len(test_series)} prédictions effectuées")

    predictions = np.array(predictions)
    print(f"   ✅ ARIMA terminé")
    return predictions, best_order


# ──────────────────────────────────────────────────────────
# MODÈLE 2 : LSTM (Deep Learning — Mémoire Longue)
# ──────────────────────────────────────────────────────────

def build_lstm_model(input_shape, units=64, dropout_rate=0.2):
    """
    Construit l'architecture LSTM pour la prévision de séries temporelles.
    
    Architecture :
        Input(lookback, 1) → LSTM(64) → Dropout(0.2) → Dense(1)
    
    Args:
        input_shape   : (timesteps, features) = (60, 1)
        units         : Nombre de neurones LSTM (défaut = 64)
        dropout_rate  : Taux de dropout pour régularisation (défaut = 0.2)
    
    Returns:
        model : Modèle Keras compilé
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        LSTM(units=units,
             input_shape=input_shape,
             return_sequences=False,  # Une seule prédiction (t+1)
             name='lstm_1'),
        Dropout(rate=dropout_rate, name='dropout_1'),
        Dense(units=1, name='output')
    ], name='LSTM_Model')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def build_stacked_lstm_model(input_shape, units=64, dropout_rate=0.2):
    """
    Construit un LSTM empilé (Stacked LSTM) — Niveau 2 intermédiaire.
    
    Architecture :
        Input → LSTM(128, return_sequences=True) → Dropout(0.2)
             → LSTM(64) → Dropout(0.2) → Dense(1)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        LSTM(units=128,
             input_shape=input_shape,
             return_sequences=True,   # Retourner séquence pour la 2e couche
             name='lstm_1'),
        Dropout(rate=dropout_rate, name='dropout_1'),
        LSTM(units=units,
             return_sequences=False,
             name='lstm_2'),
        Dropout(rate=dropout_rate, name='dropout_2'),
        Dense(units=1, name='output')
    ], name='Stacked_LSTM_Model')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


# ──────────────────────────────────────────────────────────
# MODÈLE 3 : GRU (Gated Recurrent Unit)
# ──────────────────────────────────────────────────────────

def build_gru_model(input_shape, units=64, dropout_rate=0.2):
    """
    Construit l'architecture GRU — plus rapide que LSTM, performances similaires.
    
    Architecture :
        Input(lookback, 1) → GRU(64) → Dropout(0.2) → Dense(1)
    
    Args:
        input_shape   : (timesteps, features) = (60, 1)
        units         : Nombre de neurones GRU (défaut = 64)
        dropout_rate  : Taux de dropout (défaut = 0.2)
    
    Returns:
        model : Modèle Keras compilé
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        GRU(units=units,
            input_shape=input_shape,
            return_sequences=False,
            name='gru_1'),
        Dropout(rate=dropout_rate, name='dropout_1'),
        Dense(units=1, name='output')
    ], name='GRU_Model')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


# ──────────────────────────────────────────────────────────
# MODÈLE 4 : CNN-LSTM (Hybride — Niveau 3 Avancé)
# ──────────────────────────────────────────────────────────

def build_cnn_lstm_model(input_shape, filters=64, kernel_size=3,
                          lstm_units=64, dropout_rate=0.2):
    """
    Construit un modèle CNN-LSTM hybride.
    
    Le CNN extrait les features locales de la série temporelle,
    le LSTM les séquence pour capturer les dépendances longues.
    
    Architecture :
        Input → Conv1D(64, 3) → MaxPooling1D(2) → LSTM(64) → Dropout(0.2) → Dense(1)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        Conv1D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               input_shape=input_shape,
               padding='same',
               name='conv1d_1'),
        MaxPooling1D(pool_size=2, name='maxpool_1'),
        LSTM(units=lstm_units,
             return_sequences=False,
             name='lstm_1'),
        Dropout(rate=dropout_rate, name='dropout_1'),
        Dense(units=1, name='output')
    ], name='CNN_LSTM_Model')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


# ──────────────────────────────────────────────────────────
# FONCTION D'ENTRAÎNEMENT GÉNÉRIQUE (DL)
# ──────────────────────────────────────────────────────────

def train_deep_model(model, X_train, y_train, X_val, y_val,
                     epochs=50, batch_size=32, patience=10):
    """
    Entraîne un modèle Deep Learning (LSTM, GRU, CNN-LSTM).
    Utilise EarlyStopping pour éviter le surapprentissage.
    
    Args:
        model      : Modèle Keras compilé
        X_train    : Features d'entraînement
        y_train    : Labels d'entraînement
        X_val      : Features de validation
        y_val      : Labels de validation
        epochs     : Nombre max d'époques (défaut = 50)
        batch_size : Taille de batch (défaut = 32)
        patience   : Patience EarlyStopping (défaut = 10)
    
    Returns:
        history : Historique d'entraînement Keras
    """
    import time
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    print(f"\n⏳ Entraînement {model.name}...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    elapsed = time.time() - start_time
    print(f"   ✅ {model.name} entraîné en {elapsed:.1f}s ({len(history.history['loss'])} époques)")

    return history, elapsed


# ──────────────────────────────────────────────────────────
# MODÈLE 5 : PROPHET (Meta) — Niveau 2
# ──────────────────────────────────────────────────────────

def build_prophet_model(train_df, val_df, test_df):
    """
    Entraîne un modèle Prophet (Meta) sur les données Bitcoin.
    Prophet gère automatiquement tendances, saisonnalité et anomalies.

    Args:
        train_df : DataFrame train avec colonnes Date, Close
        val_df   : DataFrame validation
        test_df  : DataFrame test

    Returns:
        predictions : array des prédictions sur le jeu de test
    """
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        from prophet import Prophet

    import pandas as pd

    # Prophet attend un DataFrame avec colonnes 'ds' (date) et 'y' (valeur)
    train_val = pd.concat([train_df, val_df], ignore_index=True)
    df_prophet = train_val[['Date', 'Close']].rename(
        columns={'Date': 'ds', 'Close': 'y'})

    import numpy as np

    # Utiliser seulement les 500 derniers jours (train+val récents)
    # pour que la tendance reste proche du niveau de prix du test set
    df_prophet = df_prophet.tail(500).reset_index(drop=True)

    print(f"\n⏳ Entraînement Prophet (données récentes : {df_prophet['ds'].min().date()} -> {df_prophet['ds'].max().date()})...")
    model = Prophet(
        changepoint_prior_scale=0.5,    # flexible pour capturer les retournements
        seasonality_prior_scale=0.1,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95
    )
    model.fit(df_prophet)

    # Prédiction sur le jeu de test
    future = pd.DataFrame({'ds': test_df['Date'].values})
    forecast = model.predict(future)

    predictions = np.clip(forecast['yhat'].values, 0, None)
    print(f"   ✅ Prophet — {len(predictions)} prédictions générées")
    return predictions


# ──────────────────────────────────────────────────────────
# MODÈLE 6 : XGBOOST — Niveau 3
# ──────────────────────────────────────────────────────────

def build_xgboost_model(train_df, val_df, test_df, lookback=60):
    """
    Entraîne un modèle XGBoost avec feature engineering temporel.
    Features : lags (1..lookback), moyennes mobiles, rendements.

    Args:
        train_df : DataFrame train
        val_df   : DataFrame validation
        test_df  : DataFrame test
        lookback : nombre de jours de lag

    Returns:
        predictions : array des prédictions sur le jeu de test
    """
    import pandas as pd
    import xgboost as xgb

    # Concatener toutes les données pour créer les features
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df = full_df.sort_values('Date').reset_index(drop=True)

    prices = full_df['Close'].values.astype(float)
    n = len(prices)

    # Feature engineering
    def make_features(prices, idx):
        feats = {}
        # Lags
        for lag in [1, 2, 3, 5, 7, 14, 21, 30, 60]:
            if idx - lag >= 0:
                feats[f'lag_{lag}'] = prices[idx - lag]
            else:
                feats[f'lag_{lag}'] = prices[0]
        # Moyennes mobiles
        for w in [7, 14, 30, 60]:
            start = max(0, idx - w)
            feats[f'ma_{w}'] = prices[start:idx].mean() if idx > start else prices[idx]
        # Ecart-type glissant
        for w in [7, 30]:
            start = max(0, idx - w)
            feats[f'std_{w}'] = prices[start:idx].std() if idx - start > 1 else 0.0
        # Rendement
        if idx > 0:
            feats['return_1'] = (prices[idx-1] - prices[max(0,idx-2)]) / (prices[max(0,idx-2)] + 1e-8)
            feats['return_7'] = (prices[idx-1] - prices[max(0,idx-7)]) / (prices[max(0,idx-7)] + 1e-8)
        else:
            feats['return_1'] = 0.0
            feats['return_7'] = 0.0
        return feats

    train_size = len(train_df)
    val_size   = len(val_df)
    test_size  = len(test_df)

    train_end = train_size
    val_end   = train_size + val_size

    # Construire X, y pour train+val et test
    X_tr, y_tr = [], []
    for i in range(lookback, train_end):
        X_tr.append(make_features(prices, i))
        y_tr.append(prices[i])

    X_val_xgb, y_val_xgb = [], []
    for i in range(train_end, val_end):
        X_val_xgb.append(make_features(prices, i))
        y_val_xgb.append(prices[i])

    X_te, y_te = [], []
    for i in range(val_end, n):
        X_te.append(make_features(prices, i))
        y_te.append(prices[i])

    import pandas as pd
    X_tr_df  = pd.DataFrame(X_tr)
    X_val_df = pd.DataFrame(X_val_xgb)
    X_te_df  = pd.DataFrame(X_te)

    print("\n⏳ Entraînement XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        eval_metric='rmse',
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_tr_df, y_tr,
        eval_set=[(X_val_df, y_val_xgb)],
        verbose=False
    )

    predictions = model.predict(X_te_df)
    print(f"   ✅ XGBoost — {len(predictions)} prédictions ({model.best_iteration} arbres)")
    return predictions
