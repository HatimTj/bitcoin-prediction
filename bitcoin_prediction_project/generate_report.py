"""
=============================================================
  PFA — Rapport PDF Final
  EMSI Rabat — IA & Data (4IIR)
  Binome : Hatim Tajimi
  Génère : results/rapport_final_pfa.pdf
=============================================================
"""

import os
import sys
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import pandas as pd
from datetime import datetime
import unicodedata

# Remplace tous les caractères hors latin-1 par des équivalents ASCII
def s(text):
    _MAP = {
        '\u2014': '-', '\u2013': '-', '\u2012': '-',  # em/en dash
        '\u2022': '*', '\u2023': '>', '\u25cf': '*',   # bullets
        '\u2500': '-', '\u2502': '|', '\u250c': '+',   # box drawing
        '\u2190': '<', '\u2192': '>', '\u2193': 'v',   # arrows
        '\u2260': '!=', '\u2265': '>=', '\u2264': '<=',
        '\u00b7': '.', '\u2026': '...',                 # middle dot, ellipsis
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u00e9': 'e', '\u00e8': 'e', '\u00ea': 'e', '\u00eb': 'e',
        '\u00e0': 'a', '\u00e2': 'a', '\u00e4': 'a',
        '\u00f4': 'o', '\u00f6': 'o', '\u00f9': 'u', '\u00fb': 'u', '\u00fc': 'u',
        '\u00ee': 'i', '\u00ef': 'i', '\u00e7': 'c',
        '\u00c9': 'E', '\u00c8': 'E', '\u00ca': 'E',
        '\u00c0': 'A', '\u00c2': 'A', '\u00d4': 'O',
        '\u00ce': 'I', '\u00c7': 'C', '\u00d9': 'U', '\u00db': 'U',
        '\u20bf': 'BTC',  # Bitcoin symbol
        '\u2605': '*', '\u2606': '*',  # stars
        '\u00ab': '<<', '\u00bb': '>>',
    }
    result = []
    for c in str(text):
        if ord(c) < 256:
            result.append(c)
        elif c in _MAP:
            result.append(_MAP[c])
        else:
            try:
                normalized = unicodedata.normalize('NFKD', c)
                ascii_char = normalized.encode('ascii', 'ignore').decode('ascii')
                result.append(ascii_char if ascii_char else '?')
            except Exception:
                result.append('?')
    return ''.join(result)

# ── Chemins ──────────────────────────────────────────────
PLOTS   = "results/plots/"
METRICS = "results/metrics/"
OUT     = "results/rapport_final_pfa.pdf"

# ── Métriques réelles issues de l'exécution ───────────────
RESULTS = {
    "ARIMA":        {"type": "Statistique",   "MAE": 4496.58, "RMSE": 7149.73, "MAPE": 19.2079, "time": "13.2 s",  "epochs": "—",           "niveau": "Niveau 1"},
    "LSTM":         {"type": "Deep Learning", "MAE":  603.03, "RMSE":  880.19, "MAPE":  2.1833, "time": "178.6 s", "epochs": "50",          "niveau": "Niveau 1"},
    "GRU":          {"type": "Deep Learning", "MAE":  482.38, "RMSE":  740.71, "MAPE":  1.7239, "time": "137.3 s", "epochs": "50",          "niveau": "Niveau 2"},
    "Stacked LSTM": {"type": "Deep Learning", "MAE":  578.57, "RMSE":  848.51, "MAPE":  2.1033, "time": "596.8 s", "epochs": "40",          "niveau": "Niveau 2"},
    "Prophet":      {"type": "Stat. ML",      "MAE": 6263.97, "RMSE": 8014.17, "MAPE": 24.7315, "time": "1.4 s",   "epochs": "—",           "niveau": "Niveau 2"},
    "CNN-LSTM":     {"type": "Hybride DL",    "MAE":  493.01, "RMSE":  734.20, "MAPE":  1.7765, "time": "106.7 s", "epochs": "50",          "niveau": "Niveau 3"},
    "XGBoost":      {"type": "ML Boosting",   "MAE": 2347.82, "RMSE": 2981.00, "MAPE":  8.8650, "time": "1.7 s",   "epochs": "88 arbres",   "niveau": "Niveau 3"},
}

# ── Couleurs ──────────────────────────────────────────────
ORANGE  = (247, 147, 26)   # Bitcoin orange
DARK    = (30,  30,  30)
GREY    = (80,  80,  80)
LGREY   = (245, 245, 245)
WHITE   = (255, 255, 255)
GREEN   = (76,  175, 80)
BLUE    = (33,  150, 243)
RED     = (244,  67, 54)


class PFAReport(FPDF):

    # Auto-sanitize all text output
    def cell(self, *args, **kwargs):
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], str):
                args[i] = s(args[i])
        if 'text' in kwargs:
            kwargs['text'] = s(kwargs['text'])
        return super().cell(*args, **kwargs)

    def multi_cell(self, *args, **kwargs):
        # multi_cell(w, h, text, ...) → text is at index 2
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], str):
                args[i] = s(args[i])
        if 'text' in kwargs:
            kwargs['text'] = s(kwargs['text'])
        return super().multi_cell(*args, **kwargs)

    # ── En-tête de page ───────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*ORANGE)
        self.rect(0, 0, 210, 12, "F")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*WHITE)
        self.set_xy(10, 2)
        self.cell(0, 8, "PFA — Prevision du Prix Bitcoin | EMSI Rabat | Hatim Tajimi", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)
        self.ln(4)

    # ── Pied de page ──────────────────────────────────────
    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.set_fill_color(*LGREY)
        self.rect(0, self.get_y(), 210, 13, "F")
        self.cell(0, 10, f"Page {self.page_no()} | Rapport généré le {datetime.now().strftime('%d/%m/%Y %H:%M')} | EMSI Rabat 2024-2025",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Titre de section ──────────────────────────────────
    def section_title(self, num, title):
        self.ln(4)
        self.set_fill_color(*ORANGE)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 9, f"  {num}. {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(*DARK)
        self.ln(3)

    # ── Sous-titre ────────────────────────────────────────
    def sub_title(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*ORANGE)
        self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)

    # ── Paragraphe ────────────────────────────────────────
    def para(self, text, size=9.5):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*GREY)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    # ── Ligne d'info (label : valeur) ─────────────────────
    def info_row(self, label, value, bold_value=False):
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*DARK)
        self.cell(55, 6, label, new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font("Helvetica", "B" if bold_value else "", 9.5)
        self.set_text_color(*GREY if not bold_value else ORANGE)
        self.cell(0, 6, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)

    # ── Insérer une image centrée ─────────────────────────
    def insert_image(self, path, w=170, caption=""):
        if not os.path.exists(path):
            self.para(f"[Image non trouvée : {path}]")
            return
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*GREY)
            self.cell(0, 5, caption, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(*DARK)
        self.ln(3)

    # ── Tableau des métriques ─────────────────────────────
    def metrics_table(self):
        headers = ["Modele", "Niveau", "Type", "MAE ($)", "RMSE ($)", "MAPE (%)", "Temps"]
        col_w   = [24, 18, 26, 26, 26, 20, 20]

        # En-tête
        self.set_fill_color(*DARK)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 8.5)
        for h, w in zip(headers, col_w):
            self.cell(w, 8, h, border=1, align="C", fill=True)
        self.ln()

        niveau_colors = {
            "Niveau 1": (33, 150, 243),
            "Niveau 2": (76, 175, 80),
            "Niveau 3": (156, 39, 176),
        }
        for i, (model, r) in enumerate(RESULTS.items()):
            fill_color = LGREY if i % 2 == 0 else WHITE
            is_best = (model == "GRU")
            is_skip = (r["MAE"] is None)

            self.set_fill_color(*fill_color)
            self.set_text_color(*DARK)
            self.set_font("Helvetica", "B" if is_best else "", 8.5)

            mae_s  = "N/A" if is_skip else f"{r['MAE']:,.2f}"
            rmse_s = "N/A" if is_skip else f"{r['RMSE']:,.2f}"
            mape_s = "N/A (*)" if is_skip else f"{r['MAPE']:.4f}"
            vals = [model, r["niveau"], r["type"], mae_s, rmse_s, mape_s, r["time"]]

            for v, w in zip(vals, col_w):
                self.cell(w, 7, v, border=1, align="C", fill=True)
            self.ln()

        # Ligne "Meilleur modèle"
        self.set_fill_color(*GREEN)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        self.cell(sum(col_w), 7,
                  ">> CHAMPION : GRU  (MAE $482 - RMSE $740 - MAPE 1.72%)  |  Meilleur RMSE : CNN-LSTM ($734)",
                  border=1, align="C", fill=True)
        self.ln(6)


# ═══════════════════════════════════════════════════════════
# CONSTRUCTION DU PDF
# ═══════════════════════════════════════════════════════════

def build_pdf():
    pdf = PFAReport(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 15, 15)

    # ── PAGE DE GARDE ─────────────────────────────────────
    pdf.add_page()

    # Fond orange dégradé (rectangle)
    pdf.set_fill_color(*ORANGE)
    pdf.rect(0, 0, 210, 80, "F")

    # Logo Bitcoin (texte stylisé)
    pdf.set_font("Helvetica", "B", 52)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(0, 12)
    pdf.cell(210, 25, "BTC", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(0, 40)
    pdf.cell(210, 10, "Prévision du Prix Bitcoin", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_xy(0, 52)
    pdf.cell(210, 8, "ARIMA  |  LSTM  |  GRU  |  Stacked LSTM  |  CNN-LSTM  |  Prophet  |  XGBoost", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Bloc infos
    pdf.set_text_color(*DARK)
    pdf.set_xy(30, 92)
    pdf.set_fill_color(*LGREY)
    pdf.rect(25, 88, 160, 90, "F")

    infos = [
        ("Établissement", "EMSI Rabat — École Marocaine des Sciences de l'Ingénieur"),
        ("Filière",        "Ingénierie Informatique — IA & Data (4IIR)"),
        ("Projet",         "Projet de Fin d'Année (PFA) — 2024-2025"),
        ("Binome",          "Hatim Tajimi"),
        ("Encadrant",      "Pr. Idriss BARBARA"),
        ("Données",        "Bitcoin BTC/USD — 4 955 jours (2010–2024)"),
        ("Methodes",        "ARIMA + LSTM + GRU + Stacked LSTM + CNN-LSTM + Prophet + XGBoost (7 modeles)"),
        ("Meilleur modele", "GRU — MAPE 1.72%  |  MAE $482  |  RMSE $740"),
        ("Date",           datetime.now().strftime("%d %B %Y")),
    ]
    pdf.set_xy(35, 94)
    for label, val in infos:
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(*ORANGE)
        pdf.set_x(35)
        pdf.cell(45, 7, label + " :", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*DARK)
        pdf.multi_cell(120, 7, val)

    # Bande basse
    pdf.set_fill_color(*DARK)
    pdf.rect(0, 186, 210, 12, "F")
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(0, 188)
    pdf.cell(210, 8, "Rapport généré automatiquement — Pipeline Python (ARIMA · TensorFlow/Keras · LSTM · GRU)",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── PAGE 2 : INTRODUCTION & CONTEXTE ─────────────────
    pdf.add_page()
    pdf.section_title("1", "Introduction et Contexte")

    pdf.sub_title("1.1 Problematique et Motivations")
    pdf.para(
        "Le Bitcoin (BTC/USD) est la cryptomonnaie la plus capitalisee au monde avec une capitalisation "
        "boursiere depassant 1 000 milliards de dollars. Son prix est caracterise par une volatilite "
        "extreme : des variations de +/-50% en quelques semaines sont courantes, rendant sa prediction "
        "particulierement difficile et scientifiquement interessante.\n\n"
        "Contrairement aux actifs financiers traditionnels, le Bitcoin est influence par :\n"
        "  - Des evenements macroeconomiques (taux Fed, inflation, crise bancaire)\n"
        "  - Des evenements specifiques crypto (halvings, regulations, faillites d'exchanges)\n"
        "  - Des facteurs psychologiques (sentiment de marche, Fear & Greed Index)\n"
        "  - Des decisions d'acteurs influents (tweets d'Elon Musk, decisions de Blackrock)\n\n"
        "Ce projet repond a la problematique centrale du sujet PFA :\n"
        "\"Comment les modeles sequentiels (LSTM, GRU) capturent-ils mieux les dependances\n"
        " temporelles du Bitcoin que les methodes statistiques classiques (ARIMA) ?\""
    )

    pdf.sub_title("1.2 Donnees utilisees")
    pdf.info_row("Source :", "Yahoo Finance (BTC-USD) — Historique complet")
    pdf.info_row("Periode :", "18 juillet 2010  ->  9 fevrier 2024")
    pdf.info_row("Observations :", "4 955 jours de donnees journalieres")
    pdf.info_row("Prix maximum :", "$67 527.90  (8 novembre 2021 — ATH historique)")
    pdf.info_row("Prix minimum :", "$0.10  (17 juillet 2010 — debut de l'historique)")
    pdf.info_row("Prix moyen test :", "~$28 000  (periode 2023-2024)")
    pdf.info_row("Volatilite :", "7.68% / jour  |  Rendement moyen : +0.47% / jour")
    pdf.info_row("Variables :", "Date, Open, High, Low, Close, Volume")
    pdf.info_row("Variable cible :", "Close (prix de cloture journalier en USD)")
    pdf.ln(2)

    pdf.sub_title("1.3 Pipeline Methodologique — 8 Etapes")
    pdf.para(
        "Etape 1 — Collecte des donnees : Chargement de bitcoin.csv, gestion des formats numeriques\n"
        "          (virgules comme separateurs de milliers, suffixes K/M/B pour les volumes).\n\n"
        "Etape 2 — Analyse Exploratoire (EDA) : Visualisation prix lineaire et logarithmique,\n"
        "          calcul des rendements journaliers, analyse de la volatilite glissante 30j,\n"
        "          distribution des rendements, boxplots annuels, matrice de correlation OHLC.\n\n"
        "Etape 3 — Preprocessing : Normalisation MinMaxScaler [0,1], creation des sequences\n"
        "          glissantes (lookback=60 jours), split temporel 80/10/10 sans melange aleatoire.\n\n"
        "Etape 4 — ARIMA : Selection automatique de l'ordre (p,d,q) via critere AIC (auto_arima).\n"
        "          Prevision walk-forward : refit tous les 50 pas sur la fenetre historique croissante.\n\n"
        "Etape 5 — Deep Learning : Entrainement LSTM, GRU, Stacked LSTM, CNN-LSTM avec\n"
        "          EarlyStopping (patience=10), ReduceLROnPlateau, batch_size=32, epochs=50.\n\n"
        "Etape 6 — ML & Stat : Prophet (Meta) entraine sur les 500 derniers jours.\n"
        "          XGBoost avec feature engineering (20 features : lags, MA, volatilite).\n\n"
        "Etape 7 — Evaluation : MAE, RMSE, MAPE calcules sur le jeu de test (496 jours).\n"
        "          Visualisation : predictions vs reel, residus, courbes d'entrainement.\n\n"
        "Etape 8 — Analyse critique : Discussion des resultats, limites, recommandations."
    )

    pdf.sub_title("1.4 Division Temporelle des Donnees")
    pdf.info_row("Train (80%) :", "3 964 jours  —  juillet 2010 -> decembre 2021")
    pdf.info_row("Validation (10%) :", "  495 jours  —  janvier 2022 -> octobre 2022")
    pdf.info_row("Test (10%) :", "  496 jours  —  novembre 2022 -> fevrier 2024")
    pdf.info_row("Fenetre glissante :", "lookback = 60 jours (environ 2 mois de trading)")
    pdf.info_row("Input shape :", "(batch, 60, 1)  — 60 pas de temps, 1 feature (Close normalise)")
    pdf.info_row("Output shape :", "(batch, 1)  — prix du lendemain (normalise, puis inverse_transform)")
    pdf.ln(2)

    pdf.sub_title("1.5 Preprocessing Detail — Normalisation et Sequences Glissantes")
    pdf.para(
        "Normalisation MinMaxScaler :\n"
        "  x_norm = (x - x_min) / (x_max - x_min)  =>  x_norm appartient a [0, 1]\n"
        "  Necessaire car le prix varie de $0.10 a $67 527 — une echelle trop large pour les reseaux.\n"
        "  Apres prediction : x_reel = x_norm * (x_max - x_min) + x_min  (inverse_transform)\n\n"
        "Sequences glissantes (lookback = 60 jours) :\n"
        "  Pour chaque jour J : entree = [Close_{J-60}, ..., Close_{J-1}], cible = Close_J\n"
        "  Exemple : pour predire le prix du 01/03/2023, on utilise les prix du 01/01 au 28/02/2023.\n"
        "  Choix de 60 jours : capture les tendances moyen-terme (cycles de marche ~2 mois)\n"
        "  sans inclure trop de bruit historique lointain."
    )

    # ── PAGE 3 : EDA ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("2", "Analyse Exploratoire des Données (EDA)")

    pdf.sub_title("2.1 Évolution du prix BTC/USD (2010–2024)")
    pdf.para(
        "Le graphique ci-dessous montre l'évolution du prix Bitcoin sur 14 ans. On observe :\n"
        "  • Une croissance exponentielle de $0.10 en 2010 à $67 527 en 2021\n"
        "  • Trois grands cycles haussiers : 2013, 2017-2018, 2020-2021\n"
        "  • Une forte correction en 2022 (-75%) suivie d'une reprise en 2023-2024\n"
        "  • Des rendements journaliers très volatils (σ = 7.68%), avec des pics à ±50%"
    )
    pdf.insert_image(f"{PLOTS}01_bitcoin_price_eda.png", w=175,
                     caption="Figure 1 : Prix BTC/USD — linéaire, logarithmique et rendements journaliers")

    # ── PAGE 4 : EDA suite — Distribution ────────────────
    pdf.add_page()
    pdf.section_title("2", "Analyse Exploratoire (suite)")

    pdf.sub_title("Figure 2 : Distribution, Volatilite et Correlations — Bitcoin BTC/USD")
    pdf.ln(3)
    pdf.para(
        "Distribution des rendements : leptokurticite marquee (queues epaisses) — signature des actifs a risque.\n"
        "Volatilite glissante 30j : pics lors des retournements majeurs (2013, 2018, 2020, 2022).\n"
        "Boxplot annuel : croissance explosive du prix median par annee.\n"
        "Correlation OHLC : quasi-colinearite (r > 0.999) — justifie l'utilisation du seul Close."
    )
    pdf.ln(6)
    pdf.insert_image(f"{PLOTS}02_bitcoin_distribution_analysis.png", w=165,
                     caption="Figure 2 : Distribution des rendements / Volatilite glissante 30j / Boxplot annuel / Correlation OHLC")

    # ── PAGE 5 : MODÈLES ──────────────────────────────────
    pdf.add_page()
    pdf.section_title("3", "Description des Modèles")

    pdf.sub_title("3.1 ARIMA — Baseline Statistique")
    pdf.para(
        "ARIMA (AutoRegressive Integrated Moving Average) est le modèle de référence en économétrie "
        "des séries temporelles. Il modélise les dépendances linéaires à court terme.\n\n"
        "Paramètres sélectionnés par Auto-ARIMA (AIC) : ordre (1, 1, 2)\n"
        "  • p=1 : composante autorégressive (AR) d'ordre 1\n"
        "  • d=1 : différenciation d'ordre 1 pour stationnariser la série\n"
        "  • q=2 : composante moyenne mobile (MA) d'ordre 2\n\n"
        "Stratégie de prévision : refit du modèle tous les 50 pas sur la fenêtre historique "
        "croissante (walk-forward), avec forecast multi-steps entre chaque refit."
    )

    pdf.sub_title("3.2 LSTM — Long Short-Term Memory")
    pdf.para(
        "Le LSTM est un réseau de neurones récurrent conçu pour capturer les dépendances "
        "à long terme dans les séquences temporelles, grâce à ses portes (forget, input, output).\n\n"
        "Architecture utilisée :\n"
        "  Input(60, 1)  →  LSTM(64 neurones)  →  Dropout(0.2)  →  Dense(1)\n\n"
        "Hyperparamètres : lr=0.001 (Adam), MSE loss, EarlyStopping(patience=10), "
        "ReduceLROnPlateau(factor=0.5), batch_size=32, epochs=50"
    )

    pdf.sub_title("3.3 GRU — Gated Recurrent Unit")
    pdf.para(
        "Le GRU est une simplification du LSTM avec seulement deux portes (reset, update), "
        "ce qui le rend plus rapide a entrainer tout en conservant des performances comparables.\n\n"
        "Architecture utilisee :\n"
        "  Input(60, 1)  ->  GRU(64 neurones)  ->  Dropout(0.2)  ->  Dense(1)\n\n"
        "Memes hyperparametres que le LSTM. Le GRU converge en 50 epochs (137s).\n"
        "Resultat : MAE $482  |  RMSE $740  |  MAPE 1.72%  =>  MEILLEUR MODELE du projet."
    )

    pdf.sub_title("3.4 Stacked LSTM — Architecture Profonde")
    pdf.para(
        "Le Stacked LSTM empile plusieurs couches LSTM pour apprendre des representations\n"
        "hierarchiques plus complexes. Chaque couche extrait des abstractions de plus haut niveau.\n\n"
        "Architecture utilisee :\n"
        "  Input(60, 1)\n"
        "  ->  LSTM(128 neurones, return_sequences=True)  ->  Dropout(0.2)\n"
        "  ->  LSTM(64 neurones)  ->  Dropout(0.2)\n"
        "  ->  Dense(32)  ->  Dense(1)\n\n"
        "Hyperparametres : Adam lr=0.001, MSE loss, EarlyStopping(patience=10),\n"
        "ReduceLROnPlateau(factor=0.5, patience=5), batch_size=32, epochs=40.\n\n"
        "Resultat : MAE $578  |  RMSE $848  |  MAPE 2.10%  |  Temps : 596s\n"
        "La profondeur supplementaire n'ameliore pas le GRU simple — trade-off complexite/gain."
    )

    pdf.sub_title("3.5 CNN-LSTM — Architecture Hybride Convolution + Recurrence")
    pdf.para(
        "Le CNN-LSTM combine des couches convolutives 1D (extraire des patterns locaux dans la\n"
        "sequence) avec une couche LSTM (modeliser les dependances temporelles globales).\n\n"
        "Architecture utilisee :\n"
        "  Input(60, 1)\n"
        "  ->  Conv1D(64 filtres, kernel=3, activation='relu')  ->  MaxPooling1D(pool_size=2)\n"
        "  ->  LSTM(64 neurones)  ->  Dropout(0.2)\n"
        "  ->  Dense(1)\n\n"
        "Principe : la Conv1D detecte des motifs locaux sur 3 jours consecutifs (ex: rebonds,\n"
        "cassures de support). Le MaxPooling1D reduit la dimensionalite avant le LSTM.\n\n"
        "Hyperparametres : Adam lr=0.001, MSE loss, EarlyStopping(patience=10), batch_size=32, epochs=50.\n\n"
        "Resultat : MAE $493  |  RMSE $734  |  MAPE 1.78%  |  Temps : 106s\n"
        "2e meilleur modele — meilleur RMSE ($734 < GRU $740), tres bon rapport qualite/temps."
    )

    pdf.sub_title("3.6 Prophet — Decomposition de Series Temporelles (Meta)")
    pdf.para(
        "Prophet (Meta/Facebook) decompose la serie temporelle en tendance, saisonnalite et\n"
        "effets de jours feries. C'est un modele additif robuste au bruit et aux valeurs aberrantes.\n\n"
        "Modele additif : y(t) = trend(t) + seasonality(t) + holidays(t) + epsilon(t)\n\n"
        "Configuration utilisee :\n"
        "  - Entrainement sur les 500 derniers jours (periode la plus recente)\n"
        "  - changepoint_prior_scale = 0.5  (flexibilite de la tendance)\n"
        "  - yearly_seasonality = True  |  weekly_seasonality = True\n"
        "  - Prediction : horizon = 496 jours (jeu de test complet)\n\n"
        "Resultat : MAE $6263  |  RMSE $8014  |  MAPE 24.73%  |  Temps : 1.4s\n"
        "Prophet echoue car Bitcoin n'a pas de saisonnalite reguliere exploitable.\n"
        "Les halvings (~4 ans) et les cycles macro ne correspondent pas aux patterns annuels/hebdo."
    )

    pdf.sub_title("3.7 XGBoost — Gradient Boosting avec Feature Engineering")
    pdf.para(
        "XGBoost (Extreme Gradient Boosting) est un ensemble d'arbres de decision entraines\n"
        "en serie, chaque arbre corrigeant les erreurs du precedent. Il necessite un feature\n"
        "engineering manuel car il ne traite pas les sequences temporelles nativement.\n\n"
        "Features construites (20 au total) :\n"
        "  - Lags du Close : lag_1, lag_2, lag_3, lag_5, lag_7, lag_14, lag_21, lag_30\n"
        "    (prix des 1, 2, 3, 5, 7, 14, 21, 30 jours precedents)\n"
        "  - Moyennes mobiles : MA_7, MA_14, MA_21, MA_60\n"
        "  - Ecart-types mobiles : STD_7, STD_14 (mesure de volatilite recente)\n"
        "  - Retour sur MA : Close/MA_7 - 1  (eloignement relatif de la moyenne)\n"
        "  - Rendements lags : return_1, return_7 (rendements 1j et 7j precedents)\n\n"
        "Hyperparametres (GridSearch) : n_estimators=119, max_depth=5, learning_rate=0.05,\n"
        "subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=10.\n\n"
        "Resultat : MAE $2347  |  RMSE $2981  |  MAPE 8.87%  |  Temps : 1.7s\n"
        "Moins performant que les DL car les features statiques ne capturent pas les dependances\n"
        "complexes a longue portee — mais 100x plus rapide a entrainer."
    )

    # ── PAGE 6 : RÉSULTATS ────────────────────────────────
    pdf.add_page()
    pdf.section_title("4", "Résultats et Comparaison des Performances")

    pdf.sub_title("4.1 Tableau comparatif final")
    pdf.para("Métriques calculées sur le jeu de test (496 jours, prix réels en USD) :")
    pdf.ln(2)
    pdf.metrics_table()

    pdf.sub_title("4.2 Analyse detaillee par modele")
    pdf.para(
        "Niveau 1 — Baseline : ARIMA + LSTM\n\n"
        "ARIMA (1,1,2) : MAPE 19.21%  |  MAE $4 496  |  RMSE $7 149  |  Temps 13s\n"
        "  Le modele ARIMA est un modele lineaire : il predit le suivant a partir d'une\n"
        "  combinaison lineaire des valeurs passees. Le Bitcoin est fondamentalement non-lineaire\n"
        "  et non-stationnaire malgre la differenciation d'ordre 1. L'ordre (1,1,2) selectionne\n"
        "  par AIC (auto_arima) signifie : 1 retard AR, 1 differenciation, 2 termes MA.\n"
        "  Avec un MAE de $4 496, le modele se trompe en moyenne de $4 496 par prediction,\n"
        "  soit ~15% du prix moyen sur la periode de test (~$29 000).\n\n"
        "LSTM : MAPE 2.18%  |  MAE $603  |  RMSE $880  |  50 epochs  |  Temps 178s\n"
        "  Le LSTM capture les dependances non-lineaires jusqu'a 60 jours en arriere grace\n"
        "  a ses 3 portes (forget, input, output). Amelioration de +89% vs ARIMA sur le MAPE.\n"
        "  Convergence stable — validation loss suit training loss sans sur-apprentissage notable.\n\n"
        "Niveau 2 — Intermediaire : GRU + Stacked LSTM + Prophet\n\n"
        "GRU : MAPE 1.72%  |  MAE $482  |  RMSE $740  |  50 epochs  |  Temps 137s  =>  MEILLEUR\n"
        "  Le GRU simplifie le LSTM (2 portes vs 3 : reset gate + update gate). Moins de\n"
        "  parametres => moins de risque de sur-apprentissage et convergence plus rapide.\n"
        "  C'est le meilleur modele sur le MAPE et le MAE. Son RMSE ($740) est legerement\n"
        "  superieur au CNN-LSTM ($734) mais sa robustesse globale est la meilleure.\n\n"
        "Stacked LSTM : MAPE 2.10%  |  MAE $578  |  RMSE $848  |  40 epochs  |  Temps 596s\n"
        "  Deux couches LSTM (128+64 neurones). La profondeur supplementaire devait permettre\n"
        "  de capter des representations plus abstraites des tendances. En pratique, le gain\n"
        "  est marginal (+5% MAPE vs LSTM simple) pour un cout en temps x3. Cela illustre le\n"
        "  principe de rendement decroissant de la profondeur sur des series univariees.\n\n"
        "Prophet : MAPE 24.73%  |  MAE $6 263  |  RMSE $8 014  |  Temps 1.4s\n"
        "  Prophet est concu pour les series avec des saisonnalites regulieres (ventes retail,\n"
        "  trafic web). Le Bitcoin n'a pas de saisonnalite hebdomadaire ou annuelle stable.\n"
        "  Le modele additif y(t) = trend + seasonality + holidays ne s'adapte pas aux\n"
        "  retournements brutaux (+/-30% en quelques jours) caracteristiques du crypto.\n\n"
        "Niveau 3 — Avance : CNN-LSTM + XGBoost\n\n"
        "CNN-LSTM : MAPE 1.78%  |  MAE $493  |  RMSE $734  |  50 epochs  |  Temps 106s\n"
        "  Architecture hybride : Conv1D(64, kernel=3) detecte les patterns locaux sur 3 jours\n"
        "  (ex: trois chandelles baissières, rebond sur support). MaxPooling1D(2) reduit la\n"
        "  dimensionalite de 60 a 29 avant le LSTM. Meilleur RMSE du projet ($734).\n"
        "  Rapport qualite/temps excellent : 2e meilleur modele en seulement 106s.\n\n"
        "XGBoost : MAPE 8.87%  |  MAE $2 347  |  RMSE $2 981  |  88 arbres  |  Temps 1.7s\n"
        "  Le boosting d'arbres necessite des features manuelles. Avec 20 features (lags,\n"
        "  moyennes mobiles, volatilite), XGBoost capture les tendances court-terme mais\n"
        "  pas les dependances sequentielles complexes. 5x moins performant que GRU mais\n"
        "  100x plus rapide — adapte a un prototype ou a des ressources limitees."
    )

    pdf.sub_title("4.3 Interpretation des metriques — Pourquoi MAPE est la metrique cle ?")
    pdf.para(
        "Trois metriques d'erreur ont ete calculees sur le jeu de test (496 jours) :\n\n"
        "MAE (Mean Absolute Error) — Erreur absolue moyenne en dollars :\n"
        "  MAE = (1/n) * somme |y_reel - y_predit|  =>  unite : USD\n"
        "  Interprete directement : GRU se trompe en moyenne de $482 par jour.\n"
        "  Limite : sensible a l'echelle — un MAE de $482 en 2023 (prix ~$29k) est bon,\n"
        "  mais $482 en 2013 (prix ~$100) serait catastrophique. D'ou l'interet du MAPE.\n\n"
        "RMSE (Root Mean Square Error) — Penalise les grandes erreurs :\n"
        "  RMSE = sqrt[(1/n) * somme (y_reel - y_predit)^2]  =>  unite : USD\n"
        "  Penalise fortement les predictions aberrantes (erreur^2). CNN-LSTM a le meilleur\n"
        "  RMSE ($734) — ses grandes erreurs sont moins frequentes que le GRU ($740).\n\n"
        "MAPE (Mean Absolute Percentage Error) — Erreur relative en % :\n"
        "  MAPE = (100/n) * somme |y_reel - y_predit| / y_reel  =>  unite : %\n"
        "  METRIQUE PRINCIPALE du projet — independante de l'echelle du prix.\n"
        "  GRU : 1.72% => erreur relative de moins de 2% sur le prix reel. Excellent.\n"
        "  ARIMA : 19.21% => erreur relative de ~1/5 du prix. Inutilisable en pratique.\n\n"
        "Seuils d'interpretation :\n"
        "  MAPE < 5%  : tres bonne prediction (DL : GRU, CNN-LSTM, Stacked LSTM, LSTM)\n"
        "  MAPE 5-15% : prediction acceptable avec reserves (XGBoost : 8.87%)\n"
        "  MAPE > 15% : prediction insuffisante (ARIMA : 19.21%, Prophet : 24.73%)"
    )

    if os.path.exists(f"{PLOTS}07_metrics_comparison.png"):
        pdf.insert_image(f"{PLOTS}07_metrics_comparison.png", w=170,
                         caption="Figure 4 : Comparaison MAE / RMSE / MAPE — Tous les modeles")

    # ── PAGE 7 : PRÉDICTIONS GROUPE 1 ────────────────────
    pdf.add_page()
    pdf.section_title("5", "Visualisation des Predictions")

    pdf.sub_title("Figure 3a : Predictions vs Valeurs Reelles — Groupe 1")
    pdf.ln(3)
    pdf.para(
        "ARIMA : ecart important — modele lineaire incapable de suivre les tendances non-lineaires.\n"
        "LSTM : MAPE 2.18% — tres bon, suit fidelement les hausses et baisses.\n"
        "GRU : MAPE 1.72% — MEILLEUR, plus stable et precis que LSTM.\n"
        "Stacked LSTM : MAPE 2.10% — bonnes performances, 2 couches profondes."
    )
    pdf.ln(5)
    if os.path.exists(f"{PLOTS}03a_predictions_groupe1.png"):
        pdf.insert_image(f"{PLOTS}03a_predictions_groupe1.png", w=172,
                         caption="Figure 3a : Predictions vs Reel — ARIMA  /  LSTM  /  GRU  /  Stacked LSTM")
    else:
        pdf.insert_image(f"{PLOTS}03_predictions_vs_real.png", w=172,
                         caption="Figure 3 : Predictions vs Valeurs Reelles — Tous les modeles")

    # ── PAGE 8 : PRÉDICTIONS GROUPE 2 ────────────────────
    pdf.add_page()
    pdf.section_title("5", "Visualisation des Predictions (suite)")

    pdf.sub_title("Figure 3b : Predictions vs Valeurs Reelles — Groupe 2")
    pdf.ln(3)
    pdf.para(
        "CNN-LSTM : MAPE 1.78% — hybride convolution + LSTM, 2e meilleur modele.\n"
        "Prophet : MAPE 24.73% — inadapte au Bitcoin, pas de saisonnalite stable.\n"
        "XGBoost : MAPE 8.87% — correct mais loin des reseaux neuronaux recurrents."
    )
    pdf.ln(5)
    if os.path.exists(f"{PLOTS}03b_predictions_groupe2.png"):
        pdf.insert_image(f"{PLOTS}03b_predictions_groupe2.png", w=172,
                         caption="Figure 3b : Predictions vs Reel — CNN-LSTM  /  Prophet  /  XGBoost")

    # ── PAGE 8 : COMPARAISON SUPERPOSÉE ──────────────────
    pdf.add_page()
    pdf.section_title("5", "Visualisation (suite)")

    pdf.sub_title("Figure 5 : Comparaison Superposee — Tous les Modeles vs Prix Reel")
    pdf.ln(2)
    pdf.para(
        "Ce graphique superpose toutes les predictions sur un seul plan. "
        "GRU (vert) et CNN-LSTM (violet) collent le mieux a la courbe reelle (orange).\n"
        "ARIMA (gris) et Prophet (rose) s'ecartent significativement lors des hausses rapides."
    )
    pdf.ln(4)
    pdf.insert_image(f"{PLOTS}04_all_models_comparison.png", w=178,
                     caption="Figure 5 : Comparaison superposee — ARIMA / LSTM / GRU / Stacked LSTM / CNN-LSTM / Prophet / XGBoost vs Prix Reel")

    # ── PAGE 9 : COURBES D'ENTRAÎNEMENT ──────────────────
    pdf.add_page()
    pdf.section_title("5", "Visualisation (suite)")

    pdf.sub_title("Figure 6 : Courbes d'Entrainement des Modeles Deep Learning")
    pdf.ln(2)
    pdf.para(
        "Les courbes de loss (MSE) montrent la convergence de chaque modele DL.\n"
        "La validation loss (orange) suit la training loss (bleu) sans sur-apprentissage,\n"
        "grace au Dropout(0.2) et a l'EarlyStopping (patience=10 epochs).\n"
        "GRU converge le plus rapidement avec la meilleure generalisation."
    )
    pdf.ln(4)
    pdf.insert_image(f"{PLOTS}05_training_history.png", w=175,
                     caption="Figure 6 : Loss MSE train / validation — LSTM / GRU / Stacked LSTM / CNN-LSTM sur 50 epochs")

    # ── PAGE 10 : RÉSIDUS (partie 1) ─────────────────────
    pdf.add_page()
    pdf.section_title("5", "Analyse des Residus par Modele")

    pdf.sub_title("Figure 7 : Residus Temporels et Distributions d'Erreurs")
    pdf.ln(3)
    pdf.para(
        "Residus = Prix reel - Prix predit.  Des residus centres en zero = bon modele.\n"
        "Colonne gauche : residus dans le temps.  Colonne droite : distribution des erreurs.\n\n"
        "  GRU / CNN-LSTM : residus tres centres, distribution gaussienne etroite  [MEILLEUR]\n"
        "  LSTM / Stacked LSTM : residus stables, legerement plus disperses\n"
        "  XGBoost : residus moderes, heteroscedasticite visible sur les pics\n"
        "  ARIMA : derive systematique lors des hausses rapides\n"
        "  Prophet : residus tres disperses — inadapte aux series crypto"
    )
    pdf.ln(5)
    pdf.insert_image(f"{PLOTS}06_residuals_analysis.png", w=155,
                     caption="Figure 7 : Residus temporels et distributions — ARIMA / LSTM / GRU / Stacked LSTM / CNN-LSTM / Prophet / XGBoost")

    # ── PAGE 10 : CONCLUSION ──────────────────────────────
    pdf.add_page()
    pdf.section_title("6", "Conclusion et Perspectives")

    pdf.sub_title("6.1 Conclusions scientifiques")
    pdf.para(
        "Ce projet a implemente et compare 7 modeles de prevision sur 4 955 jours de donnees\n"
        "Bitcoin (BTC/USD), couvrant les 3 niveaux de complexite du sujet PFA.\n\n"
        "Conclusion 1 — Superiorite des modeles sequentiels :\n"
        "Les reseaux recurrents (LSTM, GRU, Stacked LSTM, CNN-LSTM) surpassent massivement\n"
        "les approches lineaires (ARIMA) et decomposables (Prophet) sur des series crypto.\n"
        "Le MAPE du meilleur DL (1.72%) est 11x inferieur a celui d'ARIMA (19.21%).\n\n"
        "Conclusion 2 — GRU, le meilleur rapport qualite/complexite :\n"
        "GRU est le champion avec MAPE 1.72% et MAE $482. Sa simplicite architecturale\n"
        "(2 portes vs 3 pour LSTM) le rend moins prone au sur-apprentissage sur les series\n"
        "univariees. Il converge en 137s contre 596s pour le Stacked LSTM, pour un gain\n"
        "de MAPE de +0.26 points. La complexite supplementaire ne se justifie pas ici.\n\n"
        "Conclusion 3 — CNN-LSTM, innovant et efficace :\n"
        "CNN-LSTM (MAPE 1.78%, RMSE $734) est le meilleur modele sur le RMSE. La couche\n"
        "Conv1D extrait des patterns locaux (3 jours) que le LSTM globalize sur 60 jours.\n"
        "C'est le modele le plus prometteur pour une extension multi-features.\n\n"
        "Conclusion 4 — Limites des modeles statistiques sur le crypto :\n"
        "ARIMA suppose la linearite et la stationnarite faible (meme apres differenciation).\n"
        "Prophet suppose des saisonnalites repetables. Ces deux hypotheses sont violees\n"
        "par le Bitcoin dont la dynamique est pilotee par des chocs exogenes impredictibles.\n\n"
        "Conclusion 5 — XGBoost : un bon compromis vitesse/precision :\n"
        "Avec MAPE 8.87% et un entrainement en 1.7s, XGBoost est une bonne option pour\n"
        "les contextes a ressources limitees. Son principal defaut : necessite 20 features\n"
        "manuelles et ne capture pas les dependances sequentielles complexes.\n\n"
        "Classement final des 7 modeles :\n"
        "  1er  GRU          : MAPE 1.72%  MAE $482   RMSE $740  (MEILLEUR GLOBAL)\n"
        "  2e   CNN-LSTM     : MAPE 1.78%  MAE $493   RMSE $734  (MEILLEUR RMSE)\n"
        "  3e   Stacked LSTM : MAPE 2.10%  MAE $578   RMSE $848\n"
        "  4e   LSTM         : MAPE 2.18%  MAE $603   RMSE $880\n"
        "  5e   XGBoost      : MAPE 8.87%  MAE $2347  RMSE $2981\n"
        "  6e   ARIMA        : MAPE 19.21% MAE $4496  RMSE $7149\n"
        "  7e   Prophet      : MAPE 24.73% MAE $6263  RMSE $8014"
    )

    pdf.sub_title("6.2 Perspectives d'amélioration")
    pdf.para(
        "• Stacked LSTM / GRU : empiler 2-3 couches récurrentes pour capter des patterns plus complexes\n"
        "• CNN-LSTM hybride : extraire les features locales (CNN) avant le LSTM pour les tendances\n"
        "• Transformer / Attention : architecture state-of-the-art pour les séries temporelles longues\n"
        "• Features enrichies : sentiment Twitter/Reddit, volume on-chain, dominance BTC, indice Fear&Greed\n"
        "• Prévision multi-horizons : prédire à 7, 30, 90 jours au lieu du simple pas suivant\n"
        "• Intervalles de confiance : quantifier l'incertitude des prédictions (Bayesian LSTM)\n"
        "• Données temps-réel : pipeline de prédiction en continu via API (Binance, CoinGecko)"
    )

    pdf.sub_title("6.3 Résumé exécutif")
    pdf.set_fill_color(*LGREY)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*DARK)
    summary_data = [
        ("Donnees",         "4 955 jours  BTC/USD  2010-2024  |  496 jours de test"),
        ("Split",           "80% train / 10% val / 10% test  |  Lookback=60j  MinMax[0,1]"),
        ("ARIMA [N1]",      "Ordre (1,1,2)     ->  MAE $4496   RMSE $7149   MAPE 19.21%"),
        ("LSTM [N1]",       "50 ep 64u Drop0.2 ->  MAE $603    RMSE $880    MAPE 2.18%"),
        ("GRU [N2] BEST",   "50 ep 64u Drop0.2 ->  MAE $482    RMSE $740    MAPE 1.72%"),
        ("Stacked LSTM[N2]","40 ep 128+64u     ->  MAE $578    RMSE $848    MAPE 2.10%"),
        ("Prophet [N2]",    "500j  cp=0.5      ->  MAE $6264   RMSE $8014   MAPE 24.73%"),
        ("CNN-LSTM [N3]",   "50 ep Conv1D+64u  ->  MAE $493    RMSE $734    MAPE 1.78%"),
        ("XGBoost [N3]",    "88 arbres  lr=0.05  ->  MAE $2347  RMSE $2981  MAPE 8.87%"),
        ("CLASSEMENT",      "GRU > CNN-LSTM > Stacked LSTM > LSTM > XGBoost > ARIMA > Prophet"),
    ]
    for label, val in summary_data:
        pdf.rect(15, pdf.get_y(), 180, 7, "F")
        pdf.set_x(18)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*ORANGE)
        pdf.cell(35, 7, label, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 7, val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    pdf.ln(6)
    pdf.set_fill_color(*ORANGE)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 10, "  Hatim Tajimi  |  Encadrant : Pr. Idriss BARBARA  |  EMSI Rabat 2024-2025",
             fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── SAUVEGARDE ────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    pdf.output(OUT)
    print(f"✅ Rapport PDF généré : {OUT}")
    print(f"   Taille : {os.path.getsize(OUT) / 1024:.1f} KB")


if __name__ == "__main__":
    build_pdf()
