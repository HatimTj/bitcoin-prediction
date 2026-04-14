"""
Métriques d'évaluation pour les modèles de prévision.
"""
import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = '') -> dict:
    """Calcule MAE, RMSE, MAPE et Direction Accuracy."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    da = directional_accuracy(y_true, y_pred)

    return {
        'Model': model_name,
        'MAE':   round(mae, 2),
        'RMSE':  round(rmse, 2),
        'MAPE':  round(mape, 4),
        'DA':    round(da, 2),
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pourcentage de jours où la direction (hausse/baisse) est correctement prédite.
    Métrique clé pour l'usage financier.
    """
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir) * 100)


def get_model_rank(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute un rang composite aux modèles basé sur MAPE + MAE normalisés.
    """
    df = df_results.copy()

    # S'assurer que les colonnes numériques sont bien numériques
    for col in ['MAE', 'RMSE', 'MAPE']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['MAE', 'RMSE', 'MAPE']:
        if col in df.columns:
            col_range = df[col].max() - df[col].min()
            if col_range > 0:
                df[f'{col}_norm'] = (df[col] - df[col].min()) / col_range
            else:
                df[f'{col}_norm'] = 0.0

    norm_cols = [c for c in ['MAE_norm', 'RMSE_norm', 'MAPE_norm'] if c in df.columns]
    if norm_cols:
        df['Score'] = df[norm_cols].mean(axis=1)
        df['Rang']  = df['Score'].rank().astype(int)
        df = df.sort_values('Rang')

    return df
