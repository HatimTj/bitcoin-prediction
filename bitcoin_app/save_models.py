"""
=============================================================
  Script de sauvegarde des modèles entraînés
  À lancer UNE SEULE FOIS depuis le dossier bitcoin_app/

  Usage : python save_models.py

  Ce script :
    1. Charge les données Bitcoin
    2. Entraîne tous les modèles DL (LSTM, GRU, Stacked LSTM, CNN-LSTM)
    3. Sauvegarde les modèles dans models/
    4. Sauvegarde le scaler dans models/scaler.pkl
    5. Sauvegarde les paramètres ARIMA dans models/arima_params.pkl
    6. Génère comparison_table.csv mis à jour
=============================================================
"""
import sys
import os
import pickle
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Ajouter le projet principal au path ──────────────────────
PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..', 'bitcoin_prediction_project')
APP_DIR     = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

from src.data_loader import load_data, get_preprocessed_sequences
from src.metrics import compute_metrics

print("=" * 60)
print("  SAUVEGARDE DES MODÈLES — Bitcoin App")
print("=" * 60)

# ── Chargement des données ────────────────────────────────────
print("\n[1/6] Chargement des données...")
X_test, y_test_real, test_df, scaler, scaled, df = get_preprocessed_sequences()

n         = len(scaled)
train_end = int(n * 0.80)
val_end   = int(n * 0.90)

train_data = scaled[:train_end]
val_data   = scaled[train_end:val_end]
test_data  = scaled[val_end:]
LOOKBACK   = 60

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, LOOKBACK)
X_val,   y_val   = create_sequences(
    np.concatenate([train_data[-LOOKBACK:], val_data]), LOOKBACK)

X_train = X_train.reshape(*X_train.shape, 1)
X_val   = X_val.reshape(*X_val.shape, 1)

input_shape = (LOOKBACK, 1)
MODELS_DIR  = os.path.join(APP_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Sauvegarder le scaler ─────────────────────────────────────
print("\n[2/6] Sauvegarde du scaler...")
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✅ Scaler sauvegardé : {scaler_path}")

# ── Import TensorFlow ─────────────────────────────────────────
print("\n[3/6] Import TensorFlow...")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10,
                  restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-6, verbose=0),
]

all_metrics    = []
training_times = {}


def train_and_save(model, name, filename):
    """Entraîne et sauvegarde un modèle."""
    print(f"\n   ⏳ Entraînement {name}...")
    t0 = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50, batch_size=32,
        callbacks=callbacks, verbose=1,
    )
    elapsed = time.time() - t0
    print(f"   ✅ {name} entraîné en {elapsed:.1f}s")

    # Sauvegarde
    path = os.path.join(MODELS_DIR, filename)
    model.save(path)
    print(f"   ✅ Modèle sauvegardé : {path}")

    # Métriques sur test set
    pred_scaled = model.predict(X_test, verbose=0)
    pred_real   = scaler.inverse_transform(pred_scaled).flatten()
    min_len     = min(len(pred_real), len(y_test_real))
    m           = compute_metrics(y_test_real[:min_len], pred_real[:min_len], name)
    all_metrics.append(m)
    training_times[name] = elapsed
    return pred_real


# ── LSTM ──────────────────────────────────────────────────────
print("\n[4/6] Entraînement des modèles Deep Learning...")

lstm = Sequential([
    LSTM(64, input_shape=input_shape, name='lstm_1'),
    Dropout(0.2),
    Dense(1),
], name='LSTM_Model')
lstm.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
train_and_save(lstm, 'LSTM', 'lstm_model.h5')

# ── GRU ───────────────────────────────────────────────────────
gru = Sequential([
    GRU(64, input_shape=input_shape, name='gru_1'),
    Dropout(0.2),
    Dense(1),
], name='GRU_Model')
gru.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
train_and_save(gru, 'GRU', 'gru_model.h5')

# ── Stacked LSTM ──────────────────────────────────────────────
stacked = Sequential([
    LSTM(128, input_shape=input_shape, return_sequences=True, name='lstm_1'),
    Dropout(0.2),
    LSTM(64, name='lstm_2'),
    Dropout(0.2),
    Dense(1),
], name='Stacked_LSTM_Model')
stacked.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
train_and_save(stacked, 'Stacked LSTM', 'stacked_lstm_model.h5')

# ── CNN-LSTM ──────────────────────────────────────────────────
cnn_lstm = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=input_shape,
           padding='same', name='conv1d_1'),
    MaxPooling1D(2),
    LSTM(64, name='lstm_1'),
    Dropout(0.2),
    Dense(1),
], name='CNN_LSTM_Model')
cnn_lstm.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
train_and_save(cnn_lstm, 'CNN-LSTM', 'cnn_lstm_model.h5')

# ── ARIMA ─────────────────────────────────────────────────────
print("\n[5/6] ARIMA...")
try:
    from pmdarima import auto_arima
    from statsmodels.tsa.arima.model import ARIMA as ARIMAModel

    train_series = df['Close'].values[:train_end]
    test_series  = df['Close'].values[val_end:]
    sample       = train_series[-500:]

    auto_model   = auto_arima(sample, d=1, max_p=3, max_q=3,
                               stepwise=True, suppress_warnings=True,
                               error_action='ignore')
    best_order   = auto_model.order
    print(f"   ✅ Meilleur ordre ARIMA : {best_order}")

    # Sauvegarde des paramètres + historique
    arima_params = {
        'order':        best_order,
        'train_series': train_series,
    }
    arima_path = os.path.join(MODELS_DIR, 'arima_params.pkl')
    with open(arima_path, 'wb') as f:
        pickle.dump(arima_params, f)
    print(f"   ✅ Paramètres ARIMA sauvegardés : {arima_path}")

    # Métriques ARIMA (walk-forward rapide sur 100 points)
    history = list(train_series)
    preds   = []
    sample_test = test_series[:100]
    for obs in sample_test:
        m = ARIMAModel(history[-500:], order=best_order).fit(disp=False)
        preds.append(m.forecast(steps=1)[0])
        history.append(obs)
    arima_m = compute_metrics(sample_test, np.array(preds), 'ARIMA')
    all_metrics.append(arima_m)
    training_times['ARIMA'] = 0.0
except Exception as e:
    print(f"   ⚠️  ARIMA ignoré : {e}")

# ── Tableau comparatif ────────────────────────────────────────
print("\n[6/6] Génération du tableau comparatif...")
import pandas as pd

df_results = pd.DataFrame(all_metrics)
if training_times:
    df_results['Temps'] = df_results['Model'].map(
        lambda m: f"{training_times.get(m, 0):.1f}s")

results_path = os.path.join(APP_DIR, 'data', 'comparison_table.csv')
df_results.to_csv(results_path, index=False)
print(f"   ✅ Tableau sauvegardé : {results_path}")

print("\n" + "=" * 60)
print("  SAUVEGARDE TERMINÉE")
print("=" * 60)
print(df_results.to_string(index=False))
print("\nLancez maintenant : streamlit run app.py")
print("=" * 60)
