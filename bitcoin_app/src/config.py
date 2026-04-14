"""
Configuration globale de l'application Bitcoin Prediction.
"""
import os

# Répertoire racine de l'app
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemins
DATA_PATH    = os.path.join(BASE_DIR, "data", "bitcoin.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_PATH = os.path.join(BASE_DIR, "data", "comparison_table.csv")

# Paramètres modèles
LOOKBACK = 60

# Noms des fichiers modèles
MODEL_FILES = {
    "GRU":          "gru_model.h5",
    "CNN-LSTM":     "cnn_lstm_model.h5",
    "Stacked LSTM": "stacked_lstm_model.h5",
    "LSTM":         "lstm_model.h5",
}
SCALER_FILE  = "scaler.pkl"
ARIMA_FILE   = "arima_params.pkl"

# Couleurs par modèle
MODEL_COLORS = {
    "ARIMA":        "#9E9E9E",
    "LSTM":         "#2196F3",
    "GRU":          "#4CAF50",
    "Stacked LSTM": "#FF9800",
    "CNN-LSTM":     "#9C27B0",
    "Prophet":      "#E91E63",
    "XGBoost":      "#795548",
}

# Couleur principale Bitcoin
BTC_COLOR   = "#F7931A"
BG_COLOR    = "#0e1117"
CARD_COLOR  = "#1a1a2e"
