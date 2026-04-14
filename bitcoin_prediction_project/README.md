# 🟠 PFA — Prévision du Prix Bitcoin (BTC/USD)
## Séries Temporelles avec Deep Learning

**EMSI Rabat — Spécialité IA & Data (4IIR)**  
**Encadrant : Pr. Idriss BARBARA**  
**Auteur : Hatim Tajimi**

---

## 📋 Description du Projet

Ce projet de fin d'année (PFA) implémente et compare plusieurs approches de Machine Learning et Deep Learning pour la **prévision du prix du Bitcoin (BTC/USD)** à partir de données historiques (2010–2024).

### Problématique
> Comment apprendre la dynamique temporelle du cours Bitcoin et prédire son évolution future avec précision ?

---

## 🏗️ Structure du Projet

```
bitcoin_prediction_project/
├── data/
│   └── bitcoin.csv              # Dataset historique BTC/USD (2010–2024)
│
├── src/
│   ├── data_preprocessing.py    # Chargement, EDA, prétraitement
│   ├── models.py                # Architectures ARIMA, LSTM, GRU, CNN-LSTM
│   └── evaluate.py              # Métriques et visualisations
│
├── results/
│   ├── plots/                   # Graphiques générés (7 figures)
│   └── metrics/                 # Tableau comparatif CSV
│
├── main.py                      # Script principal — pipeline complet
├── requirements.txt             # Dépendances Python
└── README.md                    # Ce fichier
```

---

## 🤖 Modèles Implémentés

| Modèle | Type | Niveau | Librairie |
|--------|------|--------|-----------|
| ARIMA | Statistique | 1 — Baseline obligatoire | statsmodels / pmdarima |
| LSTM | Deep Learning | 1 — Modèle IA principal | TensorFlow / Keras |
| GRU | Deep Learning | 2 — Recommandé | TensorFlow / Keras |
| Stacked LSTM | Deep Learning | 2 — Intermédiaire | TensorFlow / Keras |
| CNN-LSTM | Hybride | 3 — Avancé (bonus) | TensorFlow / Keras |

---

## ⚙️ Pipeline Méthodologique

```
Étape 1  → Chargement données (CSV Kaggle : 4954 jours)
Étape 2  → EDA : visualisation, volatilité, corrélations
Étape 3  → Prétraitement :
            • Conversion dates, nettoyage volumes
            • Normalisation MinMaxScaler [0, 1]
            • Fenêtres glissantes (lookback = 60 jours)
            • Split temporel : Train 80% / Val 10% / Test 10%
Étape 4  → Entraînement :
            • ARIMA walk-forward (auto sélection p,d,q)
            • LSTM : Input→LSTM(64)→Dropout(0.2)→Dense(1)
            • GRU  : Input→GRU(64)→Dropout(0.2)→Dense(1)
Étape 5  → Évaluation : MAE, RMSE, MAPE
Étape 6  → Visualisation : 7 graphiques
Étape 7  → Tableau comparatif final
```

---

## 🚀 Lancement

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Exécution du pipeline complet
```bash
python main.py
```

### 3. Activer les modèles avancés (optionnel)
Dans `main.py`, modifier le dictionnaire `CONFIG` :
```python
CONFIG = {
    'run_stacked_lstm': True,  # Activer Stacked LSTM
    'run_cnn_lstm'    : True,  # Activer CNN-LSTM (bonus)
    ...
}
```

---

## 📊 Métriques d'Évaluation

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **MAE** | mean(|y - ŷ|) | Erreur absolue moyenne en USD |
| **RMSE** | √mean((y - ŷ)²) | Sensible aux grandes erreurs |
| **MAPE** | mean(|y - ŷ| / y) × 100 | Erreur relative en % |

---

## 📈 Résultats Attendus

Les graphiques générés dans `results/plots/` :
1. `01_bitcoin_price_eda.png` — Évolution du prix + rendements
2. `02_bitcoin_distribution_analysis.png` — Distribution + volatilité
3. `03_predictions_vs_real.png` — Prédictions vs réalité (par modèle)
4. `04_all_models_comparison.png` — Comparaison superposée
5. `05_training_history.png` — Courbes de loss DL
6. `06_residuals_analysis.png` — Analyse des résidus
7. `07_metrics_comparison.png` — Comparaison MAE/RMSE/MAPE

---

## 🔬 Technologies Utilisées

- **Python 3.10+**
- **Pandas / NumPy** — manipulation de données
- **Matplotlib / Seaborn** — visualisation
- **Scikit-learn** — preprocessing (MinMaxScaler)
- **TensorFlow / Keras** — LSTM, GRU, CNN-LSTM
- **statsmodels / pmdarima** — ARIMA

---

## 📚 Références

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation.
- Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Dataset : Bitcoin Historical Data — Kaggle (2010–2024)

---

*Projet réalisé dans le cadre du PFA — EMSI Rabat, Spécialité IA & Data (4IIR)*
