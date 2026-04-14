"""
Chargement et prétraitement des données Bitcoin.
Réutilise la logique de data_preprocessing.py du projet principal.
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from .config import DATA_PATH, LOOKBACK


@st.cache_data
def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Charge et nettoie le dataset Bitcoin depuis le CSV (format Investing.com).
    Résultat mis en cache par Streamlit.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace('"', '')

    df['Date'] = pd.to_datetime(
        df['Date'].astype(str).str.replace('"', '').str.strip(),
        format='%b %d, %Y'
    )

    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = (df[col].astype(str)
                         .str.replace('"', '')
                         .str.replace(',', '')
                         .str.strip())
        df[col] = pd.to_numeric(df[col], errors='coerce')

    def parse_volume(v):
        v = str(v).replace('"', '').strip()
        if v.endswith('K'):   return float(v[:-1]) * 1_000
        if v.endswith('M'):   return float(v[:-1]) * 1_000_000
        if v.endswith('B'):   return float(v[:-1]) * 1_000_000_000
        try:                  return float(v.replace(',', ''))
        except:               return np.nan

    df['Volume'] = df['Vol.'].apply(parse_volume)
    df['Change_pct'] = (df['Change %'].astype(str)
                                      .str.replace('"', '')
                                      .str.replace('%', '')
                                      .str.strip())
    df['Change_pct'] = pd.to_numeric(df['Change_pct'], errors='coerce')

    df.rename(columns={'Price': 'Close'}, inplace=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(subset=['Close'], inplace=True)

    # Indicateurs techniques
    df = _add_technical_indicators(df)

    return df


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute RSI, MACD, Bollinger Bands et rendements journaliers."""
    df = df.copy()

    # Rendements journaliers
    df['daily_return'] = df['Close'].pct_change() * 100

    # Volatilité glissante (30 jours)
    df['volatility_30'] = df['daily_return'].rolling(30).std()

    # Moyennes mobiles
    df['MA7']  = df['Close'].rolling(7).mean()
    df['MA30'] = df['Close'].rolling(30).mean()
    df['MA90'] = df['Close'].rolling(90).mean()

    # Bollinger Bands (20 jours)
    rolling_20      = df['Close'].rolling(20)
    df['BB_mid']    = rolling_20.mean()
    df['BB_upper']  = df['BB_mid'] + 2 * rolling_20.std()
    df['BB_lower']  = df['BB_mid'] - 2 * rolling_20.std()

    # RSI (14 jours)
    df['RSI'] = _compute_rsi(df['Close'], 14)

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    return df


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


@st.cache_data
def get_preprocessed_sequences(lookback: int = LOOKBACK):
    """
    Retourne les séquences train/val/test normalisées + scaler.
    Utilisé pour reconstruire les prédictions sur le jeu de test.
    """
    df = load_data()
    close_prices = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    n         = len(scaled)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    train_data = scaled[:train_end]
    val_data   = scaled[train_end:val_end]
    test_data  = scaled[val_end:]

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    def create_sequences(data, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i - window:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_test, y_test = create_sequences(
        np.concatenate([val_data[-lookback:], test_data]), lookback)

    X_test = X_test.reshape(*X_test.shape, 1)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    return X_test, y_test_real, test_df, scaler, scaled, df


@st.cache_data
def load_comparison_results(filepath: str = None) -> pd.DataFrame:
    """Charge le tableau comparatif des modèles."""
    import os
    from .config import RESULTS_PATH
    path = filepath or RESULTS_PATH
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df
