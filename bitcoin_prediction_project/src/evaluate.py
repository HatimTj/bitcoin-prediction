"""
=============================================================
  PFA — Prévision de Séries Temporelles | EMSI IA & Data
  Module : evaluate.py
  Auteur : Hatim Tajimi
  Encadrant : Pr. Idriss BARBARA
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────
# 1. MÉTRIQUES D'ÉVALUATION
# ──────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    model_name: str = '') -> dict:
    """
    Calcule les métriques d'évaluation : MAE, RMSE, MAPE.
    
    Args:
        y_true     : Valeurs réelles (prix en USD)
        y_pred     : Valeurs prédites (prix en USD)
        model_name : Nom du modèle (pour affichage)
    
    Returns:
        dict avec les métriques calculées
    """
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE — éviter division par zéro
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    metrics = {
        'Model': model_name,
        'MAE':   round(mae, 2),
        'RMSE':  round(rmse, 2),
        'MAPE':  round(mape, 4)
    }
    
    if model_name:
        print(f"\n📊 MÉTRIQUES — {model_name} :")
        print(f"   MAE  : ${mae:>10,.2f}")
        print(f"   RMSE : ${rmse:>10,.2f}")
        print(f"   MAPE :  {mape:>10.4f}%")
    
    return metrics


# ──────────────────────────────────────────────────────────
# 2. VISUALISATION DES PRÉDICTIONS
# ──────────────────────────────────────────────────────────

def plot_predictions(test_dates, y_true, predictions_dict,
                     save_path: str = 'results/plots/') -> None:
    """
    Génère les graphiques de comparaison des prédictions vs réalité.
    
    Args:
        test_dates        : Dates du jeu de test
        y_true            : Prix réels
        predictions_dict  : Dict {nom_modèle: array_prédictions}
        save_path         : Chemin de sauvegarde
    """
    colors = {
        'ARIMA':        '#9E9E9E',
        'LSTM':         '#2196F3',
        'GRU':          '#4CAF50',
        'Stacked LSTM': '#FF9800',
        'CNN-LSTM':     '#9C27B0',
        'Prophet':      '#E91E63',
        'XGBoost':      '#795548',
    }
    
    # ── Figure 3 : Prédictions vs Réalité (un graphique par modèle) ──
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 5 * n_models))
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Prédictions vs Valeurs Réelles — Bitcoin BTC/USD',
                 fontsize=15, fontweight='bold', y=1.01)
    
    for ax, (model_name, y_pred) in zip(axes, predictions_dict.items()):
        color = colors.get(model_name, '#E91E63')
        
        ax.plot(test_dates, y_true, color='#F7931A', linewidth=1.5,
                label='Prix Réel', zorder=3)
        ax.plot(test_dates, y_pred, color=color, linewidth=1.2,
                linestyle='--', label=f'Prédiction {model_name}', zorder=2)
        ax.fill_between(test_dates, y_true, y_pred, alpha=0.15, color=color)
        
        ax.set_title(f'{model_name} — Prédiction vs Réalité', fontsize=12, fontweight='bold')
        ax.set_ylabel('Prix BTC/USD ($)')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig(f'{save_path}03_predictions_vs_real.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique prédictions sauvegardé : {save_path}03_predictions_vs_real.png")
    
    # ── Figure 4 : Comparaison de tous les modèles (superposés) ──
    fig, ax = plt.subplots(figsize=(16, 7))
    
    ax.plot(test_dates, y_true, color='#F7931A', linewidth=2.5,
            label='Prix Réel', zorder=5)
    
    for model_name, y_pred in predictions_dict.items():
        color = colors.get(model_name, '#E91E63')
        ax.plot(test_dates, y_pred, color=color, linewidth=1.2,
                linestyle='--', label=model_name, alpha=0.85, zorder=4)
    
    ax.set_title('Comparaison de Tous les Modèles — Bitcoin BTC/USD',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prix BTC/USD ($)')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(f'{save_path}04_all_models_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique comparaison sauvegardé : {save_path}04_all_models_comparison.png")


def plot_training_history(histories_dict, save_path: str = 'results/plots/') -> None:
    """
    Affiche les courbes de loss pendant l'entraînement des modèles DL.
    
    Args:
        histories_dict : Dict {nom_modèle: keras_history}
        save_path      : Chemin de sauvegarde
    """
    dl_models = {k: v for k, v in histories_dict.items() if v is not None}
    if not dl_models:
        return
    
    n = len(dl_models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]
    
    fig.suptitle("Évolution de la Loss pendant l'Entraînement",
                 fontsize=14, fontweight='bold')
    
    colors = {'LSTM': '#2196F3', 'GRU': '#4CAF50',
              'Stacked LSTM': '#FF9800', 'CNN-LSTM': '#9C27B0'}
    
    for ax, (model_name, history) in zip(axes, dl_models.items()):
        color = colors.get(model_name, '#E91E63')
        epochs = range(1, len(history.history['loss']) + 1)
        
        ax.plot(epochs, history.history['loss'], color=color,
                linewidth=1.5, label='Train Loss')
        ax.plot(epochs, history.history['val_loss'], color=color,
                linewidth=1.5, linestyle='--', label='Val Loss', alpha=0.7)
        
        ax.set_title(f'{model_name} — Loss (MSE)', fontsize=12)
        ax.set_xlabel('Époque')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Marquer l'epoch optimal
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_loss  = min(history.history['val_loss'])
        ax.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.7,
                   label=f'Best epoch: {best_epoch}')
        ax.annotate(f'Best: {best_loss:.5f}',
                    xy=(best_epoch, best_loss),
                    xytext=(best_epoch + 2, best_loss * 1.1),
                    fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}05_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Courbes d'entraînement sauvegardées : {save_path}05_training_history.png")


def plot_residuals(test_dates, y_true, predictions_dict,
                   save_path: str = 'results/plots/') -> None:
    """
    Analyse les résidus (erreurs) de chaque modèle.
    """
    colors = {
        'ARIMA': '#9E9E9E', 'LSTM': '#2196F3', 'GRU': '#4CAF50',
        'Stacked LSTM': '#FF9800', 'CNN-LSTM': '#9C27B0',
        'Prophet': '#E91E63', 'XGBoost': '#795548',
    }

    n_models = len(predictions_dict)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Analyse des Résidus par Modèle', fontsize=14, fontweight='bold')
    
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        color = colors.get(model_name, '#E91E63')
        residuals = y_true - y_pred
        
        # Résidus dans le temps
        axes[i, 0].plot(test_dates, residuals, color=color, linewidth=0.7, alpha=0.8)
        axes[i, 0].axhline(y=0, color='black', linewidth=1)
        axes[i, 0].fill_between(test_dates, residuals, 0, alpha=0.2, color=color)
        axes[i, 0].set_title(f'{model_name} — Résidus', fontsize=11)
        axes[i, 0].set_ylabel('Erreur ($)')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Distribution des résidus
        axes[i, 1].hist(residuals, bins=50, color=color, alpha=0.7, edgecolor='black', lw=0.3)
        axes[i, 1].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        axes[i, 1].axvline(x=residuals.mean(), color='blue', linestyle='-.',
                           linewidth=1.2, label=f'Moy: ${residuals.mean():,.0f}')
        axes[i, 1].set_title(f'{model_name} — Distribution des Résidus', fontsize=11)
        axes[i, 1].set_xlabel('Erreur ($)')
        axes[i, 1].set_ylabel('Fréquence')
        axes[i, 1].legend(fontsize=9)
        axes[i, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig(f'{save_path}06_residuals_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Analyse des résidus sauvegardée : {save_path}06_residuals_analysis.png")


# ──────────────────────────────────────────────────────────
# 3. TABLEAU COMPARATIF FINAL
# ──────────────────────────────────────────────────────────

def generate_comparison_table(all_metrics: list,
                               training_times: dict = None,
                               save_path: str = 'results/metrics/') -> pd.DataFrame:
    """
    Génère le tableau comparatif final de tous les modèles.
    
    Args:
        all_metrics    : Liste de dicts de métriques
        training_times : Dict {nom_modèle: temps en secondes}
        save_path      : Chemin de sauvegarde CSV
    
    Returns:
        df_results : DataFrame du tableau comparatif
    """
    df_results = pd.DataFrame(all_metrics)
    
    if training_times:
        df_results['Temps (s)'] = df_results['Model'].map(
            lambda m: f"{training_times.get(m, 0):.1f}s")
    
    # Ajouter le type de modèle
    model_types = {
        'ARIMA':        'Statistique',
        'LSTM':         'Deep Learning',
        'GRU':          'Deep Learning',
        'Stacked LSTM': 'Deep Learning',
        'CNN-LSTM':     'Hybride DL',
        'Prophet':      'Statistique ML',
        'XGBoost':      'ML Boosting',
    }
    df_results['Type'] = df_results['Model'].map(model_types).fillna('Deep Learning')
    
    # Réorganiser les colonnes
    cols = ['Model', 'Type', 'MAE', 'RMSE', 'MAPE']
    if 'Temps (s)' in df_results.columns:
        cols.append('Temps (s)')
    df_results = df_results[cols]
    
    # Trouver le meilleur modèle
    best_mae_idx  = df_results['MAE'].idxmin()
    best_rmse_idx = df_results['RMSE'].idxmin()
    best_mape_idx = df_results['MAPE'].idxmin()
    
    print("\n" + "=" * 70)
    print("  TABLEAU COMPARATIF FINAL — PFA EMSI IA & DATA")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)
    print(f"\n🏆 Meilleur MAE  : {df_results.loc[best_mae_idx, 'Model']} (${df_results.loc[best_mae_idx, 'MAE']:,.2f})")
    print(f"🏆 Meilleur RMSE : {df_results.loc[best_rmse_idx, 'Model']} (${df_results.loc[best_rmse_idx, 'RMSE']:,.2f})")
    print(f"🏆 Meilleur MAPE : {df_results.loc[best_mape_idx, 'Model']} ({df_results.loc[best_mape_idx, 'MAPE']:.4f}%)")
    
    # Sauvegarde CSV
    df_results.to_csv(f'{save_path}comparison_table.csv', index=False)
    print(f"\n✅ Tableau sauvegardé : {save_path}comparison_table.csv")
    
    # Visualisation du tableau comparatif
    plot_metrics_comparison(df_results, save_path)
    
    return df_results


def plot_metrics_comparison(df_results: pd.DataFrame,
                             save_path: str = 'results/plots/') -> None:
    """
    Génère un graphique de comparaison des métriques par modèle.
    """
    model_colors = {
        'ARIMA':        '#9E9E9E',
        'LSTM':         '#2196F3',
        'GRU':          '#4CAF50',
        'Stacked LSTM': '#FF9800',
        'CNN-LSTM':     '#9C27B0',
        'Prophet':      '#E91E63',
        'XGBoost':      '#795548',
    }
    
    colors_list = [model_colors.get(m, '#E91E63') for m in df_results['Model']]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Comparaison des Performances — Tous les Modèles',
                 fontsize=14, fontweight='bold')
    
    metrics = [
        ('MAE',  'Mean Absolute Error ($)',          'MAE ($)'),
        ('RMSE', 'Root Mean Square Error ($)',       'RMSE ($)'),
        ('MAPE', 'Mean Absolute Percentage Error', 'MAPE (%)'),
    ]
    
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        bars = ax.bar(df_results['Model'], df_results[metric],
                      color=colors_list, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Annoter les valeurs sur les barres
        for bar, val in zip(bars, df_results[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_height() * 0.02,
                    f'{val:,.2f}' if metric != 'MAPE' else f'{val:.4f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Mettre en évidence le meilleur
        best_idx = df_results[metric].idxmin()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}07_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique métriques sauvegardé : {save_path}07_metrics_comparison.png")
