"""
=============================================================
  PFA — Prévision de Séries Temporelles | EMSI IA & Data
  Script principal : main.py
  Auteur : Hatim Tajimi
  Encadrant : Pr. Idriss BARBARA
  
  Usage : python main.py
=============================================================

PIPELINE COMPLET :
  1. Chargement et EDA des données Bitcoin (2010–2024)
  2. Prétraitement : normalisation, séquences glissantes
  3. Entraînement : ARIMA (baseline) + LSTM + GRU
  4. Évaluation : MAE, RMSE, MAPE
  5. Visualisation : graphiques + tableau comparatif
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif pour sauvegarde
import warnings
warnings.filterwarnings('ignore')

# ── Ajouter le répertoire src au path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_data, perform_eda, preprocess_data
from models import (build_lstm_model, build_gru_model,
                    build_stacked_lstm_model, build_cnn_lstm_model,
                    build_arima_model, train_deep_model,
                    build_prophet_model, build_xgboost_model)
from evaluate import (compute_metrics, plot_predictions,
                      plot_training_history, plot_residuals,
                      generate_comparison_table)

# ──────────────────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ──────────────────────────────────────────────────────────
CONFIG = {
    'data_path'   : 'data/bitcoin.csv',
    'lookback'    : 60,        # Fenêtre glissante : 60 jours
    'epochs'      : 50,        # Nombre max d'époques DL
    'batch_size'  : 32,        # Taille de batch
    'lstm_units'  : 64,        # Neurones LSTM/GRU
    'dropout'     : 0.2,       # Taux de dropout
    'patience'    : 10,        # EarlyStopping patience
    'plots_path'  : 'results/plots/',
    'metrics_path': 'results/metrics/',
    
    # Modèles à entraîner (True = activer)
    'run_arima'       : True,
    'run_lstm'        : True,
    'run_gru'         : True,
    'run_stacked_lstm': True,    # Niveau 2 — Stacked LSTM
    'run_cnn_lstm'    : True,    # Niveau 3 — CNN-LSTM hybride
    'run_prophet'     : True,    # Niveau 2 — Prophet (Meta)
    'run_xgboost'     : True,    # Niveau 3 — XGBoost
}

# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("\n" + "🟠" * 30)
    print("  PFA — PRÉVISION DU PRIX BITCOIN (BTC/USD)")
    print("  EMSI Rabat — Spécialité IA & Data (4IIR)")
    print("  Auteur : Hatim Tajimi")
    print("  Encadrant : Pr. Idriss BARBARA")
    print("🟠" * 30 + "\n")
    
    # ── Étape 1 : Chargement des données ──────────────────
    print("\n📁 ÉTAPE 1 : CHARGEMENT DES DONNÉES")
    print("-" * 40)
    df = load_data(CONFIG['data_path'])
    
    # ── Étape 2 : EDA ─────────────────────────────────────
    print("\n🔍 ÉTAPE 2 : ANALYSE EXPLORATOIRE (EDA)")
    print("-" * 40)
    perform_eda(df, save_path=CONFIG['plots_path'])
    
    # ── Étape 3 : Prétraitement ───────────────────────────
    print("\n⚙️  ÉTAPE 3 : PRÉTRAITEMENT")
    print("-" * 40)
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, train_df, val_df, test_df, close_prices) = preprocess_data(
        df, lookback=CONFIG['lookback'])
    
    # Dates du jeu de test (pour les graphiques)
    test_dates = test_df['Date'].values
    
    # Prix réels du jeu de test (non normalisés, en USD)
    # Reconstruire y_test en prix réels
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Dictionnaires pour stocker les résultats
    predictions_dict = {}
    histories_dict   = {}
    all_metrics      = []
    training_times   = {}
    
    # ── Étape 4a : ARIMA (Baseline) ───────────────────────
    if CONFIG['run_arima']:
        print("\n📐 ÉTAPE 4a : MODÈLE ARIMA (BASELINE STATISTIQUE)")
        print("-" * 50)
        
        # ARIMA travaille sur prix réels (non normalisés)
        n = len(df)
        train_end = int(n * 0.80)
        val_end   = int(n * 0.90)
        
        train_series = df['Close'].iloc[:train_end].values
        test_series  = df['Close'].iloc[val_end:].values
        
        import time
        t0 = time.time()
        arima_preds, arima_order = build_arima_model(
            train_series, test_series)
        training_times['ARIMA'] = time.time() - t0
        
        # Aligner les longueurs (ARIMA walk-forward = même longueur que test_series)
        min_len = min(len(arima_preds), len(test_series), len(y_test_real))
        predictions_dict['ARIMA'] = arima_preds[:min_len]
        y_test_real_aligned       = y_test_real[:min_len]
        test_dates_aligned        = test_dates[:min_len]
        
        arima_metrics = compute_metrics(
            y_test_real_aligned, predictions_dict['ARIMA'], 'ARIMA')
        all_metrics.append(arima_metrics)
    else:
        y_test_real_aligned  = y_test_real
        test_dates_aligned   = test_dates
    
    # ── Input shape pour les modèles DL ───────────────────
    input_shape = (CONFIG['lookback'], 1)
    
    # ── Étape 4b : LSTM ───────────────────────────────────
    if CONFIG['run_lstm']:
        print("\n🧠 ÉTAPE 4b : MODÈLE LSTM")
        print("-" * 40)
        
        lstm_model = build_lstm_model(
            input_shape,
            units=CONFIG['lstm_units'],
            dropout_rate=CONFIG['dropout']
        )
        lstm_model.summary()
        
        lstm_history, lstm_time = train_deep_model(
            lstm_model, X_train, y_train, X_val, y_val,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            patience=CONFIG['patience']
        )
        
        # Prédictions sur test set
        lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
        lstm_pred_real   = scaler.inverse_transform(lstm_pred_scaled).flatten()
        
        # Aligner sur la longueur minimale
        min_len = min(len(lstm_pred_real), len(y_test_real_aligned))
        predictions_dict['LSTM'] = lstm_pred_real[:min_len]
        histories_dict['LSTM']   = lstm_history
        training_times['LSTM']   = lstm_time
        
        lstm_metrics = compute_metrics(
            y_test_real_aligned[:min_len], predictions_dict['LSTM'], 'LSTM')
        all_metrics.append(lstm_metrics)
    
    # ── Étape 4c : GRU ────────────────────────────────────
    if CONFIG['run_gru']:
        print("\n🧠 ÉTAPE 4c : MODÈLE GRU")
        print("-" * 40)
        
        gru_model = build_gru_model(
            input_shape,
            units=CONFIG['lstm_units'],
            dropout_rate=CONFIG['dropout']
        )
        gru_model.summary()
        
        gru_history, gru_time = train_deep_model(
            gru_model, X_train, y_train, X_val, y_val,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            patience=CONFIG['patience']
        )
        
        # Prédictions
        gru_pred_scaled = gru_model.predict(X_test, verbose=0)
        gru_pred_real   = scaler.inverse_transform(gru_pred_scaled).flatten()
        
        min_len = min(len(gru_pred_real), len(y_test_real_aligned))
        predictions_dict['GRU'] = gru_pred_real[:min_len]
        histories_dict['GRU']   = gru_history
        training_times['GRU']   = gru_time
        
        gru_metrics = compute_metrics(
            y_test_real_aligned[:min_len], predictions_dict['GRU'], 'GRU')
        all_metrics.append(gru_metrics)
    
    # ── Étape 4d : Stacked LSTM (Niveau 2) ────────────────
    if CONFIG['run_stacked_lstm']:
        print("\n🧠 ÉTAPE 4d : MODÈLE STACKED LSTM (NIVEAU 2)")
        print("-" * 40)
        
        stacked_model = build_stacked_lstm_model(
            input_shape,
            units=CONFIG['lstm_units'],
            dropout_rate=CONFIG['dropout']
        )
        
        stacked_history, stacked_time = train_deep_model(
            stacked_model, X_train, y_train, X_val, y_val,
            epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'],
            patience=CONFIG['patience']
        )
        
        stacked_pred_scaled = stacked_model.predict(X_test, verbose=0)
        stacked_pred_real   = scaler.inverse_transform(stacked_pred_scaled).flatten()
        
        min_len = min(len(stacked_pred_real), len(y_test_real_aligned))
        predictions_dict['Stacked LSTM'] = stacked_pred_real[:min_len]
        histories_dict['Stacked LSTM']   = stacked_history
        training_times['Stacked LSTM']   = stacked_time
        
        stacked_metrics = compute_metrics(
            y_test_real_aligned[:min_len], predictions_dict['Stacked LSTM'], 'Stacked LSTM')
        all_metrics.append(stacked_metrics)
    
    # ── Étape 4e : CNN-LSTM (Niveau 3) ────────────────────
    if CONFIG['run_cnn_lstm']:
        print("\n🧠 ÉTAPE 4e : MODÈLE CNN-LSTM (NIVEAU 3)")
        print("-" * 40)
        
        cnn_lstm_model = build_cnn_lstm_model(
            input_shape,
            dropout_rate=CONFIG['dropout']
        )
        
        cnn_lstm_history, cnn_lstm_time = train_deep_model(
            cnn_lstm_model, X_train, y_train, X_val, y_val,
            epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'],
            patience=CONFIG['patience']
        )
        
        cnn_pred_scaled = cnn_lstm_model.predict(X_test, verbose=0)
        cnn_pred_real   = scaler.inverse_transform(cnn_pred_scaled).flatten()
        
        min_len = min(len(cnn_pred_real), len(y_test_real_aligned))
        predictions_dict['CNN-LSTM'] = cnn_pred_real[:min_len]
        histories_dict['CNN-LSTM']   = cnn_lstm_history
        training_times['CNN-LSTM']   = cnn_lstm_time
        
        cnn_metrics = compute_metrics(
            y_test_real_aligned[:min_len], predictions_dict['CNN-LSTM'], 'CNN-LSTM')
        all_metrics.append(cnn_metrics)
    
    # ── Étape 4f : Prophet (Niveau 2) ────────────────────
    if CONFIG['run_prophet']:
        print("\n📈 ÉTAPE 4f : MODÈLE PROPHET (META)")
        print("-" * 40)
        import time
        t0 = time.time()
        prophet_preds = build_prophet_model(
            train_df, val_df, test_df)
        prophet_time = time.time() - t0

        min_len = min(len(prophet_preds), len(y_test_real_aligned))
        predictions_dict['Prophet'] = prophet_preds[:min_len]
        training_times['Prophet'] = prophet_time

        prophet_metrics = compute_metrics(
            y_test_real_aligned[:min_len], predictions_dict['Prophet'], 'Prophet')
        all_metrics.append(prophet_metrics)

    # ── Étape 4g : XGBoost (Niveau 3) ────────────────────
    if CONFIG['run_xgboost']:
        print("\n🌲 ÉTAPE 4g : MODÈLE XGBOOST")
        print("-" * 40)
        import time
        t0 = time.time()
        xgb_preds = build_xgboost_model(
            train_df, val_df, test_df,
            lookback=CONFIG['lookback'])
        xgb_time = time.time() - t0

        min_len = min(len(xgb_preds), len(y_test_real_aligned))
        predictions_dict['XGBoost'] = xgb_preds[:min_len]
        training_times['XGBoost'] = xgb_time
        xgb_metrics = compute_metrics(
            y_test_real_aligned[:min_len], predictions_dict['XGBoost'], 'XGBoost')
        all_metrics.append(xgb_metrics)

    # ── Étape 5 : Visualisations ──────────────────────────
    print("\n📊 ÉTAPE 5 : VISUALISATION DES RÉSULTATS")
    print("-" * 40)
    
    # Aligner toutes les prédictions sur la même longueur
    min_len = min(len(v) for v in predictions_dict.values())
    preds_aligned = {k: v[:min_len] for k, v in predictions_dict.items()}
    y_true_final  = y_test_real_aligned[:min_len]
    dates_final   = test_dates_aligned[:min_len]
    
    plot_predictions(dates_final, y_true_final, preds_aligned,
                     save_path=CONFIG['plots_path'])
    
    plot_training_history(
        {k: v for k, v in histories_dict.items() if v is not None},
        save_path=CONFIG['plots_path']
    )
    
    plot_residuals(dates_final, y_true_final, preds_aligned,
                   save_path=CONFIG['plots_path'])
    
    # ── Étape 6 : Tableau comparatif ──────────────────────
    print("\n📋 ÉTAPE 6 : TABLEAU COMPARATIF FINAL")
    print("-" * 40)
    df_results = generate_comparison_table(
        all_metrics, training_times, save_path=CONFIG['metrics_path'])
    
    # ── Fin ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE TERMINÉ AVEC SUCCÈS !")
    print("=" * 60)
    print(f"  📂 Graphiques : {CONFIG['plots_path']}")
    print(f"  📂 Métriques  : {CONFIG['metrics_path']}")
    print("\n  Résultats générés :")
    print("   → 01_bitcoin_price_eda.png")
    print("   → 02_bitcoin_distribution_analysis.png")
    print("   → 03_predictions_vs_real.png")
    print("   → 04_all_models_comparison.png")
    print("   → 05_training_history.png")
    print("   → 06_residuals_analysis.png")
    print("   → 07_metrics_comparison.png")
    print("   → comparison_table.csv")
    print("=" * 60 + "\n")
    
    return df_results, predictions_dict


if __name__ == '__main__':
    results = main()
