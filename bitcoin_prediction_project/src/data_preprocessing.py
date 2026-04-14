"""
=============================================================
  PFA — Prévision de Séries Temporelles | EMSI IA & Data
  Module : data_preprocessing.py
  Auteur : Hatim Tajimi
  Encadrant : Pr. Idriss BARBARA
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge le dataset Bitcoin depuis un fichier CSV (format Investing.com).
    Gère le format avec virgules dans les prix et suffixes K/M/B dans le volume.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace('"', '')

    # Conversion de la date
    df['Date'] = pd.to_datetime(
        df['Date'].astype(str).str.replace('"', '').str.strip(),
        format='%b %d, %Y'
    )

    # Nettoyage des colonnes numériques OHLC
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = (df[col].astype(str)
                         .str.replace('"', '')
                         .str.replace(',', '')
                         .str.strip())
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Nettoyage du volume (ex: "86.85K" → 86850)
    def parse_volume(v):
        v = str(v).replace('"', '').strip()
        if v.endswith('K'):   return float(v[:-1]) * 1_000
        if v.endswith('M'):   return float(v[:-1]) * 1_000_000
        if v.endswith('B'):   return float(v[:-1]) * 1_000_000_000
        try:                  return float(v.replace(',', ''))
        except:               return np.nan

    df['Volume'] = df['Vol.'].apply(parse_volume)

    # Nettoyage de Change %
    df['Change_pct'] = (df['Change %'].astype(str)
                                      .str.replace('"', '')
                                      .str.replace('%', '')
                                      .str.strip())
    df['Change_pct'] = pd.to_numeric(df['Change_pct'], errors='coerce')

    # Renommage et tri
    df.rename(columns={'Price': 'Close'}, inplace=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(subset=['Close'], inplace=True)

    print(f"✅ Données chargées : {len(df)} lignes")
    print(f"   Période : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"   Prix max : ${df['Close'].max():,.2f}  |  Prix min : ${df['Close'].min():.4f}")

    return df


def perform_eda(df: pd.DataFrame, save_path: str = 'results/plots/') -> None:
    """
    Analyse exploratoire complète : visualisations, statistiques, volatilité.
    """
    print("\n" + "=" * 60)
    print("  ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    print("=" * 60)

    print("\n📊 STATISTIQUES DESCRIPTIVES :")
    print(df[['Close', 'Open', 'High', 'Low']].describe().round(2))

    df = df.copy()
    df['daily_return'] = df['Close'].pct_change() * 100

    # ── Figure 1 : Prix + log + rendements ──
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle('Analyse du Prix Bitcoin (BTC/USD) — 2010–2024',
                 fontsize=16, fontweight='bold')

    axes[0].plot(df['Date'], df['Close'], color='#F7931A', linewidth=0.8)
    axes[0].fill_between(df['Date'], df['Close'], alpha=0.1, color='#F7931A')
    axes[0].set_title('Évolution du Prix BTC/USD', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Prix (USD)')
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0].xaxis.set_major_locator(mdates.YearLocator())

    axes[1].semilogy(df['Date'], df['Close'], color='#2196F3', linewidth=0.8)
    axes[1].set_title('Prix BTC/USD — Échelle Logarithmique', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Prix (USD) — log scale')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())

    bar_colors = np.where(df['daily_return'] >= 0, '#4CAF50', '#F44336')
    axes[2].bar(df['Date'], df['daily_return'], color=bar_colors, width=1, alpha=0.7)
    axes[2].set_title('Rendements Journaliers (%)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Rendement (%)')
    axes[2].axhline(0, color='black', linewidth=0.5)
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[2].xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(f'{save_path}01_bitcoin_price_eda.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Figure 1 sauvegardée : {save_path}01_bitcoin_price_eda.png")

    # ── Figure 2 : Distribution + volatilité + corrélation ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution et Analyse de Volatilité — Bitcoin', fontsize=14, fontweight='bold')

    returns_clean = df['daily_return'].dropna()
    axes[0, 0].hist(returns_clean, bins=100, color='#F7931A', alpha=0.7,
                    edgecolor='black', lw=0.3)
    axes[0, 0].set_title('Distribution des Rendements Journaliers')
    axes[0, 0].set_xlabel('Rendement (%)'); axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].axvline(returns_clean.mean(), color='red', linestyle='--',
                       label=f'Moy: {returns_clean.mean():.2f}%')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    df['vol30'] = df['daily_return'].rolling(30).std()
    axes[0, 1].plot(df['Date'], df['vol30'], color='#9C27B0', linewidth=0.8)
    axes[0, 1].set_title('Volatilité Glissante (30 jours)')
    axes[0, 1].set_ylabel('Écart-type (%)'); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0, 1].xaxis.set_major_locator(mdates.YearLocator(2))

    df['Year'] = df['Date'].dt.year
    year_data   = [df[df['Year'] == y]['Close'].values
                   for y in sorted(df['Year'].unique()) if len(df[df['Year'] == y]) > 10]
    year_labels = [y for y in sorted(df['Year'].unique())
                   if len(df[df['Year'] == y]) > 10]
    bp = axes[1, 0].boxplot(year_data, labels=year_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#F7931A'); patch.set_alpha(0.6)
    axes[1, 0].set_title('Distribution du Prix par Année')
    axes[1, 0].set_xlabel('Année'); axes[1, 0].set_ylabel('Prix (USD)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_yscale('log'); axes[1, 0].grid(True, alpha=0.3, axis='y')

    corr = df[['Close', 'Open', 'High', 'Low']].corr()
    sns.heatmap(corr, annot=True, fmt='.4f', cmap='YlOrRd',
                ax=axes[1, 1], square=True)
    axes[1, 1].set_title('Matrice de Corrélation OHLC')

    plt.tight_layout()
    plt.savefig(f'{save_path}02_bitcoin_distribution_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Figure 2 sauvegardée : {save_path}02_bitcoin_distribution_analysis.png")

    print(f"\n📈 STATISTIQUES DE VOLATILITÉ :")
    print(f"   Rendement moyen journalier : {returns_clean.mean():.4f}%")
    print(f"   Volatilité journalière     : {returns_clean.std():.4f}%")
    print(f"   Rendement max              : {returns_clean.max():.2f}%")
    print(f"   Rendement min              : {returns_clean.min():.2f}%")


def preprocess_data(df: pd.DataFrame, lookback: int = 60):
    """
    Prétraitement complet pour les modèles ML/DL.
    Retourne les séquences train/val/test + scaler.
    """
    print("\n" + "=" * 60)
    print("  PRÉTRAITEMENT DES DONNÉES")
    print("=" * 60)

    close_prices = df['Close'].values.reshape(-1, 1)

    # Gestion valeurs manquantes
    missing = np.sum(np.isnan(close_prices))
    if missing > 0:
        print(f"⚠️  {missing} valeur(s) manquante(s) → interpolation linéaire")
        df = df.copy()
        df['Close'] = df['Close'].interpolate(method='linear')
        close_prices = df['Close'].values.reshape(-1, 1)
    else:
        print("✅ Aucune valeur manquante")

    # Normalisation
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)
    print("✅ Normalisation MinMaxScaler [0, 1] appliquée")

    # Split temporel
    n = len(scaled)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    train_data = scaled[:train_end]
    val_data   = scaled[train_end:val_end]
    test_data  = scaled[val_end:]

    print(f"\n📊 DIVISION TEMPORELLE :")
    print(f"   Train      : {len(train_data)} jours (80%)")
    print(f"   Validation : {len(val_data)} jours (10%)")
    print(f"   Test       : {len(test_data)} jours (10%)")

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    # Création des séquences glissantes
    def create_sequences(data, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i - window:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data, lookback)
    X_val,   y_val   = create_sequences(
        np.concatenate([train_data[-lookback:], val_data]), lookback)
    X_test,  y_test  = create_sequences(
        np.concatenate([val_data[-lookback:], test_data]), lookback)

    # Reshape pour LSTM/GRU : (samples, timesteps, 1)
    X_train = X_train.reshape(*X_train.shape, 1)
    X_val   = X_val.reshape(*X_val.shape, 1)
    X_test  = X_test.reshape(*X_test.shape, 1)

    print(f"\n📐 SÉQUENCES (lookback = {lookback} jours) :")
    print(f"   X_train : {X_train.shape}")
    print(f"   X_val   : {X_val.shape}")
    print(f"   X_test  : {X_test.shape}")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            scaler, train_df, val_df, test_df, close_prices)
