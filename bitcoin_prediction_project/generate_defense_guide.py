"""
=============================================================
  PFA - Guide de Soutenance & Comprehension du Projet
  EMSI Rabat - IA & Data (4IIR)
  Binome : Hatim Tajimi | Encadrant : Pr. Idriss BARBARA
  Genere : results/guide_soutenance_pfa.pdf
=============================================================
"""

import os
import sys
import unicodedata
from fpdf import FPDF
from fpdf.enums import XPos, YPos

def s(text):
    _MAP = {
        '\u2014': '-', '\u2013': '-', '\u2012': '-',
        '\u2022': '*', '\u2023': '>', '\u25cf': '*',
        '\u2500': '-', '\u2502': '|',
        '\u2190': '<', '\u2192': '->', '\u2193': 'v',
        '\u2260': '!=', '\u2265': '>=', '\u2264': '<=',
        '\u00b7': '.', '\u2026': '...',
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u00e9': 'e', '\u00e8': 'e', '\u00ea': 'e', '\u00eb': 'e',
        '\u00e0': 'a', '\u00e2': 'a', '\u00e4': 'a',
        '\u00f4': 'o', '\u00f9': 'u', '\u00fb': 'u', '\u00fc': 'u',
        '\u00ee': 'i', '\u00ef': 'i', '\u00e7': 'c',
        '\u00c9': 'E', '\u00c8': 'E', '\u00ca': 'E',
        '\u00c0': 'A', '\u00c2': 'A', '\u00d4': 'O',
        '\u00ce': 'I', '\u00c7': 'C', '\u00d9': 'U', '\u00db': 'U',
        '\u20bf': 'BTC', '\u2605': '*', '\u00ab': '<<', '\u00bb': '>>',
        '\u00e1': 'a', '\u00ed': 'i', '\u00f3': 'o', '\u00fa': 'u',
        '\u00f1': 'n', '\u00fc': 'u',
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

# Couleurs
ORANGE = (247, 147, 26)
DARK   = (30, 30, 30)
GREY   = (80, 80, 80)
LGREY  = (240, 240, 240)
BLUE   = (33, 150, 243)
GREEN  = (76, 175, 80)
RED    = (244, 67, 54)
PURPLE = (156, 39, 176)
WHITE  = (255, 255, 255)
YELLOW = (255, 193, 7)

OUT = "results/guide_soutenance_pfa.pdf"

class PDF(FPDF):
    def header(self):
        self.set_fill_color(*ORANGE)
        self.rect(0, 0, 210, 12, 'F')
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(*WHITE)
        self.set_xy(0, 2)
        self.cell(0, 8, s('PFA - Prevision Bitcoin BTC/USD | EMSI 4IIR | Hatim Tajimi'), align='C')
        self.set_text_color(*DARK)

    def footer(self):
        self.set_y(-12)
        self.set_fill_color(*LGREY)
        self.rect(0, self.get_y(), 210, 12, 'F')
        self.set_font('Helvetica', '', 8)
        self.set_text_color(*GREY)
        self.cell(0, 10, s(f'Guide de Soutenance - Page {self.page_no()} | Encadrant : Pr. Idriss BARBARA'), align='C')
        self.set_text_color(*DARK)

    def section_title(self, num, title, color=ORANGE):
        self.ln(6)
        self.set_fill_color(*color)
        self.set_draw_color(*color)
        self.rect(10, self.get_y(), 190, 10, 'F')
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(*WHITE)
        self.set_x(12)
        self.cell(186, 10, s(f'  {num}.  {title}'), align='L')
        self.ln(12)
        self.set_text_color(*DARK)

    def sub_title(self, title, color=BLUE):
        self.ln(3)
        self.set_fill_color(*color)
        self.rect(10, self.get_y(), 4, 7, 'F')
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(*color)
        self.set_x(16)
        self.cell(0, 7, s(title))
        self.ln(9)
        self.set_text_color(*DARK)

    def body(self, text, size=10):
        self.set_font('Helvetica', '', size)
        self.set_text_color(*DARK)
        self.set_x(12)
        self.multi_cell(186, 5.5, s(text))
        self.ln(2)

    def bullet(self, text, color=ORANGE):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(*DARK)
        self.set_x(14)
        self.set_fill_color(*color)
        self.circle = True
        # bullet point
        y = self.get_y() + 2
        self.set_fill_color(*color)
        self.ellipse(14, y, 2.5, 2.5, 'F')
        self.set_xy(18, self.get_y())
        self.multi_cell(180, 5.5, s(text))
        self.ln(1)

    def qa_box(self, question, answer, q_color=ORANGE, a_color=BLUE):
        # Question box
        self.ln(3)
        self.set_fill_color(*q_color)
        self.set_draw_color(*q_color)
        self.rect(10, self.get_y(), 190, 9, 'F')
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(*WHITE)
        self.set_x(13)
        self.cell(184, 9, s('Q : ' + question))
        self.ln(11)
        # Answer box
        self.set_fill_color(235, 245, 255)
        self.set_draw_color(*a_color)
        start_y = self.get_y()
        self.set_x(12)
        # Draw left border
        self.set_font('Helvetica', '', 10)
        self.set_text_color(*DARK)
        self.set_x(15)
        self.multi_cell(182, 5.5, s('R : ' + answer))
        end_y = self.get_y()
        self.set_fill_color(*a_color)
        self.rect(10, start_y, 3, end_y - start_y, 'F')
        self.ln(4)

    def metric_row(self, model, model_type, mae, rmse, mape, time_, niveau, best=False):
        y = self.get_y()
        if best:
            self.set_fill_color(220, 255, 220)
        else:
            self.set_fill_color(*LGREY if self.page_no() % 2 == 0 else WHITE)
        self.rect(10, y, 190, 8, 'F')
        self.set_font('Helvetica', 'BI' if best else '', 9)
        self.set_text_color(*GREEN if best else DARK)
        self.set_x(10)
        self.cell(32, 8, s(model), border=0)
        self.cell(28, 8, s(model_type), border=0)
        self.cell(25, 8, s(f'${mae:,.0f}'), border=0)
        self.cell(25, 8, s(f'${rmse:,.0f}'), border=0)
        self.cell(22, 8, s(f'{mape:.2f}%'), border=0)
        self.cell(28, 8, s(time_), border=0)
        self.cell(30, 8, s(niveau), border=0)
        self.ln(8)
        self.set_text_color(*DARK)

    def info_box(self, title, content, color=BLUE):
        self.ln(3)
        self.set_fill_color(*color)
        self.rect(10, self.get_y(), 190, 8, 'F')
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(*WHITE)
        self.set_x(12)
        self.cell(186, 8, s(title))
        self.ln(10)
        self.set_fill_color(240, 248, 255)
        start_y = self.get_y()
        self.set_font('Helvetica', '', 9.5)
        self.set_text_color(*DARK)
        self.set_x(14)
        self.multi_cell(182, 5.5, s(content))
        end_y = self.get_y()
        self.set_fill_color(*color)
        self.rect(10, start_y, 3, end_y - start_y, 'F')
        self.ln(3)


def generate():
    os.makedirs('results', exist_ok=True)
    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(10, 15, 10)

    # ================================================================
    # PAGE DE GARDE
    # ================================================================
    pdf.add_page()

    # Bandeau titre
    pdf.set_fill_color(*ORANGE)
    pdf.rect(0, 12, 210, 55, 'F')
    pdf.set_font('Helvetica', 'B', 26)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(0, 20)
    pdf.cell(210, 12, s('GUIDE DE SOUTENANCE PFA'), align='C')
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_xy(0, 34)
    pdf.cell(210, 10, s('Prevision du Prix Bitcoin (BTC/USD)'), align='C')
    pdf.set_font('Helvetica', '', 12)
    pdf.set_xy(0, 47)
    pdf.cell(210, 8, s('Comprendre le projet - Repondre aux questions - Defendre les choix'), align='C')

    pdf.set_text_color(*DARK)
    pdf.set_xy(0, 72)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(210, 8, s('Hatim Tajimi'), align='C')
    pdf.set_font('Helvetica', '', 11)
    pdf.set_xy(0, 82)
    pdf.cell(210, 7, s('Specialite IA & Data (4IIR) - EMSI Rabat'), align='C')
    pdf.set_xy(0, 90)
    pdf.cell(210, 7, s('Encadrant : Pr. Idriss BARBARA'), align='C')

    # Sommaire visuel
    pdf.ln(10)
    pdf.set_fill_color(*LGREY)
    pdf.rect(15, pdf.get_y(), 180, 100, 'F')
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(*ORANGE)
    pdf.set_x(0)
    pdf.cell(210, 10, s('SOMMAIRE'), align='C')
    pdf.ln(2)

    sections = [
        ('1', 'Contexte et Objectif du Projet', '- Pourquoi Bitcoin ? Pourquoi ce sujet ?'),
        ('2', 'Les Donnees : Bitcoin 2010-2024', '- D\'ou viennent les donnees ? Comment on les a preparees ?'),
        ('3', 'Methodologie Complete (8 etapes)', '- Pipeline detaille de A a Z'),
        ('4', 'Les 7 Modeles : Pourquoi ces choix ?', '- Explication simple de chaque modele'),
        ('5', 'Resultats et Analyse Critique', '- Pourquoi GRU gagne ? Pourquoi Prophet echoue ?'),
        ('6', 'Questions Probables du Professeur', '- 20+ questions avec reponses preparees'),
        ('7', 'Points Forts a Mettre en Avant', '- Ce qui impressionne lors de la soutenance'),
    ]

    for num, title, desc in sections:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*ORANGE)
        pdf.set_x(22)
        pdf.cell(10, 7, s(num + '.'))
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*DARK)
        pdf.cell(80, 7, s(title))
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(*GREY)
        pdf.cell(90, 7, s(desc))
        pdf.ln(7)

    # ================================================================
    # SECTION 1 : CONTEXTE ET OBJECTIF
    # ================================================================
    pdf.add_page()
    pdf.section_title('1', 'CONTEXTE ET OBJECTIF DU PROJET', ORANGE)

    pdf.sub_title('Pourquoi ce sujet ?', ORANGE)
    pdf.body(
        'Le sujet du PFA demande de predire l\'evolution future d\'une variable temporelle '
        'et de comparer ARIMA (methode statistique classique) avec des modeles d\'IA '
        '(LSTM, GRU). Nous avons choisi Bitcoin car :'
    )
    pdf.bullet('Donnees publiques et gratuites (4955 jours, 2010-2024)')
    pdf.bullet('Serie temporelle tres connue avec forte volatilite - ideal pour tester les modeles')
    pdf.bullet('Tres grande quantite de donnees = meilleur apprentissage pour les reseaux neuronaux')
    pdf.bullet('Sujet d\'actualite qui valorise le travail academique')
    pdf.bullet('Recommande explicitement dans la strategie de Pr. BARBARA')

    pdf.sub_title('Quelle est la problematique ?', BLUE)
    pdf.info_box(
        'Problematique centrale',
        'Comment predire le prix futur du Bitcoin (BTC/USD) a partir de son historique ?\n'
        'Les modeles sequentiels (LSTM, GRU) capturent-ils mieux les dependances temporelles '
        'que les methodes statistiques classiques comme ARIMA ?\n\n'
        'Reponse apportee par notre projet : OUI, avec une erreur 10x plus faible '
        '(MAPE 1.84% pour GRU vs 19.21% pour ARIMA).',
        GREEN
    )

    pdf.sub_title('Qu\'est-ce qu\'une serie temporelle ?', BLUE)
    pdf.body(
        'Une serie temporelle est une suite de valeurs mesurees a intervalles reguliers dans le temps. '
        'Exemples : temperature quotidienne, chiffre d\'affaires mensuel, prix d\'une action chaque jour. '
        'Dans notre cas : le prix de cloture du Bitcoin chaque jour depuis 2010.'
    )

    pdf.sub_title('Quel est l\'objectif concret ?', BLUE)
    pdf.body(
        'A partir des 60 derniers jours de prix (fenetre glissante), predire le prix du lendemain. '
        'On entraine des modeles sur 80% des donnees, on valide sur 10%, on teste sur les 10% restants. '
        'On compare 7 modeles avec 3 metriques : MAE, RMSE et MAPE.'
    )

    # ================================================================
    # SECTION 2 : LES DONNEES
    # ================================================================
    pdf.add_page()
    pdf.section_title('2', 'LES DONNEES : BITCOIN 2010-2024', BLUE)

    pdf.sub_title('Description des donnees', BLUE)
    pdf.body('Fichier : data/bitcoin.csv | Source : Yahoo Finance (BTC-USD) | 4955 lignes')
    pdf.ln(2)

    # Tableau colonnes
    headers = ['Colonne', 'Description', 'Exemple']
    rows = [
        ('Date', 'Date de la journee boursiere', '2024-01-15'),
        ('Open', 'Prix d\'ouverture du jour (USD)', '$42,350'),
        ('High', 'Prix le plus haut du jour (USD)', '$43,100'),
        ('Low', 'Prix le plus bas du jour (USD)', '$41,800'),
        ('Close', 'Prix de cloture - NOTRE CIBLE', '$42,900'),
        ('Volume', 'Volume echange dans la journee', '28,450,000'),
    ]
    col_w = [35, 100, 55]
    pdf.set_fill_color(*ORANGE)
    pdf.set_text_color(*WHITE)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_x(10)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, s(h), fill=True, border=1)
    pdf.ln(8)
    pdf.set_text_color(*DARK)
    for j, row in enumerate(rows):
        pdf.set_fill_color(*LGREY if j % 2 == 0 else WHITE)
        pdf.set_font('Helvetica', 'BI' if row[0] == 'Close' else '', 9)
        pdf.set_text_color(*GREEN if row[0] == 'Close' else DARK)
        pdf.set_x(10)
        for i, val in enumerate(row):
            pdf.cell(col_w[i], 7, s(val), fill=True, border=1)
        pdf.ln(7)
    pdf.set_text_color(*DARK)

    pdf.ln(4)
    pdf.sub_title('Pourquoi on utilise seulement "Close" ?', BLUE)
    pdf.body(
        'On predit le prix de cloture (Close) car c\'est le prix de reference officiel de la journee. '
        'C\'est aussi la valeur la plus utilisee en finance pour l\'analyse technique. '
        'Open, High, Low et Volume ne sont pas utilises comme entrees dans nos modeles '
        '(sauf XGBoost qui utilise des features derives).'
    )

    pdf.sub_title('Comment on a prepare les donnees ? (Preprocessing)', BLUE)
    pdf.body('Etape 1 - Normalisation MinMaxScaler :')
    pdf.bullet(
        'Le prix varie de 0.05$ (2010) a 73,000$ (2024). Cette grande echelle perturbe '
        'les reseaux neuronaux. On ramene tout entre 0 et 1 avec MinMaxScaler.'
    )
    pdf.bullet('Formule : x_norm = (x - min) / (max - min)')
    pdf.bullet('Apres prediction, on fait l\'inverse (inverse_transform) pour retrouver les USD reels.')

    pdf.body('Etape 2 - Fenetres glissantes (lookback = 60 jours) :')
    pdf.bullet(
        'Pour chaque jour J, on cree une sequence d\'entree = les 60 jours precedents [J-60, ..., J-1], '
        'et la cible = le prix du jour J.'
    )
    pdf.bullet('Exemple : entree = [prix_j1, prix_j2, ..., prix_j60], sortie = prix_j61')
    pdf.bullet(
        'Pourquoi 60 jours ? C\'est environ 2 mois de trading - assez pour capturer '
        'les tendances moyen terme sans trop de bruit.'
    )

    pdf.body('Etape 3 - Split temporel 80/10/10 :')
    pdf.bullet('Train : 3964 jours (80%) - apprentissage des modeles')
    pdf.bullet('Validation : 495 jours (10%) - reglage des hyperparametres (early stopping)')
    pdf.bullet('Test : 496 jours (10%) - evaluation finale jamais vue par les modeles')
    pdf.bullet('IMPORTANT : pas de melange aleatoire - on respecte l\'ordre chronologique !')

    # ================================================================
    # SECTION 3 : METHODOLOGIE
    # ================================================================
    pdf.add_page()
    pdf.section_title('3', 'METHODOLOGIE COMPLETE - LES 8 ETAPES', GREEN)

    etapes = [
        ('Etape 1', 'Collecte des donnees',
         'Telechargement de bitcoin.csv depuis Yahoo Finance. 4955 jours de donnees de 2010 a 2024. '
         'Variables : Date, Open, High, Low, Close, Volume. On garde uniquement Close pour la prediction.'),
        ('Etape 2', 'Analyse Exploratoire (EDA)',
         'Visualisation du prix en echelle lineaire et logarithmique. Calcul des rendements journaliers. '
         'Analyse de la volatilite sur 30 jours. Distribution des rendements (histogramme). '
         'Boxplot par annee pour voir l\'evolution. Matrice de correlation OHLC.'),
        ('Etape 3', 'Preprocessing',
         'Normalisation MinMaxScaler [0,1]. Creation des sequences d\'entree (lookback=60). '
         'Split temporel 80%/10%/10% sans melange. X_train shape: (3904, 60, 1), y_train shape: (3904,).'),
        ('Etape 4', 'ARIMA - Baseline statistique',
         'auto_arima trouve le meilleur ordre (p,d,q) via critere AIC. Ordre optimal : (1,1,2). '
         'Walk-forward prediction : on predit par blocs de 50 jours en refittant. '
         'ARIMA est le modele de reference - si l\'IA fait moins bien, ca ne sert a rien.'),
        ('Etape 5', 'Entrainement modeles Deep Learning',
         'LSTM, GRU, Stacked LSTM, CNN-LSTM entraines avec : epochs=50, batch_size=32, '
         'EarlyStopping(patience=10), ReduceLROnPlateau. '
         'Normalisation obligatoire avant entrainement. Predictions reconverties en USD apres.'),
        ('Etape 6', 'Prophet et XGBoost',
         'Prophet (Meta) : modele additif tendance + saisonnalite. Entraine sur les 500 derniers jours. '
         'XGBoost : features engineering manuel (lags 1,2,3,5,7,14,21,30,60 jours, '
         'moyennes mobiles 7,14,30,60 jours, volatilite, rendements).'),
        ('Etape 7', 'Evaluation et Visualisation',
         'MAE, RMSE, MAPE calcules sur le jeu de test pour chaque modele. '
         'Graphiques : predictions vs reel, comparaison superposee, courbes d\'entrainement, residus. '
         'Tableau comparatif final avec classement.'),
        ('Etape 8', 'Analyse critique',
         'Discussion des resultats : pourquoi GRU domine, pourquoi Prophet echoue sur Bitcoin, '
         'limites des modeles (pas de news, pas de sentiment), conclusion et recommandations.'),
    ]

    for etape, titre, desc in etapes:
        y = pdf.get_y()
        pdf.set_fill_color(*ORANGE)
        pdf.rect(10, y, 30, 14, 'F')
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(10, y + 1)
        pdf.cell(30, 6, s(etape), align='C')
        pdf.set_xy(10, y + 7)
        pdf.cell(30, 6, s('----------'), align='C')

        pdf.set_fill_color(*LGREY)
        pdf.rect(40, y, 160, 14, 'F')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*DARK)
        pdf.set_xy(43, y + 1)
        pdf.cell(154, 6, s(titre))
        pdf.set_font('Helvetica', '', 8.5)
        pdf.set_text_color(*GREY)
        pdf.set_xy(43, y + 8)
        pdf.cell(154, 5, s(desc[:110] + ('...' if len(desc) > 110 else '')))
        pdf.ln(17)
        pdf.set_text_color(*DARK)

    # ================================================================
    # SECTION 4 : LES 7 MODELES
    # ================================================================
    pdf.add_page()
    pdf.section_title('4', 'LES 7 MODELES : POURQUOI CES CHOIX ?', PURPLE)

    modeles = [
        ('ARIMA', 'Niveau 1 - Baseline', GREY,
         'ARIMA = AutoRegressive Integrated Moving Average. C\'est le modele statistique classique '
         'pour les series temporelles. Il utilise les valeurs passees et les erreurs passees pour predire. '
         '"Integrated" signifie qu\'on rend la serie stationnaire en la differenciant.',
         'Pourquoi l\'utiliser : C\'est la reference obligatoire du sujet. Sans ARIMA, on ne peut pas '
         'prouver que l\'IA fait mieux. C\'est le point de depart de toute analyse.',
         'Ordre (1,1,2) : p=1 (1 valeur passee), d=1 (1 difference), q=2 (2 erreurs passees).',
         'MAPE 19.21% - Tres mauvais car Bitcoin est non-lineaire et non-stationnaire.'),

        ('LSTM', 'Niveau 1 - Deep Learning', BLUE,
         'LSTM = Long Short-Term Memory. Reseau neuronal recurrent avec des "portes" (gates) qui '
         'controlent quelles informations garder ou oublier. Ideal pour les longues dependances temporelles.',
         'Architecture : Input(60,1) -> LSTM(64 neurones) -> Dropout(0.2) -> Dense(1)',
         'Pourquoi 64 neurones : compromis entre capacite d\'apprentissage et risque de surapprentissage. '
         'Dropout 0.2 : desactive 20% des neurones aleatoirement pendant l\'entrainement pour eviter l\'overfitting.',
         'MAPE 2.20% - Bon. 10x mieux qu\'ARIMA. Prouve la superiorite de l\'IA.'),

        ('GRU', 'Niveau 2 - Deep Learning', GREEN,
         'GRU = Gated Recurrent Unit. Version simplifiee du LSTM avec moins de parametres. '
         '2 portes (reset, update) au lieu de 3 pour LSTM (input, forget, output). '
         'Plus rapide a entrainer, souvent aussi bon voire meilleur que LSTM.',
         'Architecture : Input(60,1) -> GRU(64 neurones) -> Dropout(0.2) -> Dense(1)',
         'Pourquoi GRU est meilleur que LSTM ici : Moins de parametres = moins de surapprentissage. '
         'Bitcoin suit des patterns relativement simples que GRU capture suffisamment bien.',
         'MAPE 1.84% - MEILLEUR modele ! Plus simple mais plus efficace.'),

        ('Stacked LSTM', 'Niveau 2 - Deep Learning', (255, 152, 0),
         'Stacked LSTM = plusieurs couches LSTM empilees. La premiere couche extrait des features '
         'de bas niveau, la seconde des features de plus haut niveau (patterns complexes).',
         'Architecture : LSTM(128, return_sequences=True) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2) -> Dense(1)',
         'Pourquoi moins bon que GRU : Plus de parametres = plus de risque d\'overfitting sur Bitcoin. '
         'La complexite ajoutee n\'apporte pas de gain pour cette serie.',
         'MAPE 2.07% - Bon mais inferieur a GRU. Complexite inutile ici.'),

        ('CNN-LSTM', 'Niveau 3 - Hybride', PURPLE,
         'CNN-LSTM = Convolutional Neural Network + LSTM. Le CNN extrait des patterns locaux '
         '(tendances courtes, motifs recurrents) puis le LSTM capture les dependances sequentielles. '
         'Approche hybride tres puissante.',
         'Architecture : Conv1D(64, kernel=3) -> MaxPool(2) -> LSTM(64) -> Dropout(0.2) -> Dense(1)',
         'Pourquoi presque aussi bon que GRU : Le CNN pre-traite le signal et facilite la tache du LSTM. '
         'Mais la convolution introduit un biais sur les patterns locaux.',
         'MAPE 1.91% - Tres bon, 2eme meilleur modele.'),

        ('Prophet', 'Niveau 2 - Meta', RED,
         'Prophet = modele additif developpe par Meta (Facebook). Decompose la serie en : '
         'tendance + saisonnalite annuelle + saisonnalite hebdomadaire + effets vacances.',
         'Pourquoi mauvais sur Bitcoin : Bitcoin n\'a PAS de saisonnalite reguliere comme les ventes '
         'de Noel ou la meteo. Prophet est fait pour des series avec des cycles previsibles. '
         'Bitcoin est domine par des evenements impredictibles (halvings, regulations, tweets...). '
         'Entraine sur 500 derniers jours seulement pour eviter les extrapolations extremes.',
         '',
         'MAPE 24.73% - Pire que ARIMA. Normal : mauvais outil pour Bitcoin.'),

        ('XGBoost', 'Niveau 3 - ML Boosting', (121, 85, 72),
         'XGBoost = Extreme Gradient Boosting. Ensemble d\'arbres de decision entraines '
         'sequentiellement ou chacun corrige les erreurs du precedent. '
         'Necessite un feature engineering manuel.',
         'Features utilisees : lags (prix j-1, j-2, j-3, j-5, j-7, j-14, j-21, j-30, j-60), '
         'moyennes mobiles (7, 14, 30, 60 jours), ecart-type (volatilite), rendements. '
         'Total : ~20 features par observation.',
         'Pourquoi moins bon que LSTM/GRU : XGBoost ne comprend pas nativement la sequence temporelle. '
         'Il traite les features independamment, sans notion d\'ordre. Les LSTM/GRU ont cet avantage naturel.',
         'MAPE 7.42% - Moyen. Beaucoup mieux qu\'ARIMA mais loin des reseaux neuronaux.'),
    ]

    for nom, niveau, color, desc, archi, detail, result in modeles:
        if pdf.get_y() > 240:
            pdf.add_page()

        y = pdf.get_y()
        # Header modele
        pdf.set_fill_color(*color)
        pdf.rect(10, y, 190, 9, 'F')
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(12, y + 1)
        pdf.cell(100, 7, s(nom))
        pdf.set_font('Helvetica', '', 9)
        pdf.set_xy(120, y + 1)
        pdf.cell(78, 7, s(niveau), align='R')
        pdf.ln(11)
        pdf.set_text_color(*DARK)

        pdf.set_font('Helvetica', '', 9.5)
        pdf.set_x(14)
        pdf.multi_cell(182, 5, s(desc))

        if archi:
            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_text_color(*color)
            pdf.set_x(14)
            pdf.multi_cell(182, 5, s(archi))
            pdf.set_text_color(*DARK)

        if detail:
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(*GREY)
            pdf.set_x(14)
            pdf.multi_cell(182, 5, s(detail))
            pdf.set_text_color(*DARK)

        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(*GREEN if '1.8' in result or '1.9' in result else (RED if '24' in result or '19' in result else ORANGE))
        pdf.set_x(14)
        pdf.cell(182, 5, s('  Resultat : ' + result))
        pdf.ln(7)
        pdf.set_text_color(*DARK)

    # ================================================================
    # SECTION 5 : RESULTATS ET ANALYSE
    # ================================================================
    pdf.add_page()
    pdf.section_title('5', 'RESULTATS ET ANALYSE CRITIQUE', GREEN)

    pdf.sub_title('Tableau comparatif final', GREEN)

    # Header tableau
    pdf.set_fill_color(*DARK)
    pdf.set_text_color(*WHITE)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_x(10)
    for h, w in [('Modele',32),('Type',28),('MAE ($)',25),('RMSE ($)',25),('MAPE',22),('Temps',28),('Niveau',30)]:
        pdf.cell(w, 8, s(h), fill=True, border=1)
    pdf.ln(8)

    data = [
        ('GRU',          'Deep Learning', 514.26,   782.37,   1.8378, '613.4 s', 'Niveau 2', True),
        ('CNN-LSTM',     'Hybride DL',    528.53,   772.99,   1.9095, '76.4 s',  'Niveau 3', False),
        ('Stacked LSTM', 'Deep Learning', 576.58,   874.43,   2.0717, '385.8 s', 'Niveau 2', False),
        ('LSTM',         'Deep Learning', 609.87,   898.08,   2.2034, '142.9 s', 'Niveau 1', False),
        ('XGBoost',      'ML Boosting',   2012.01, 2421.91,   7.4230, '1.2 s',   'Niveau 3', False),
        ('ARIMA',        'Statistique',   4496.58, 7149.73,  19.2079, '15.5 s',  'Niveau 1', False),
        ('Prophet',      'Stat. ML',      6263.97, 8014.17,  24.7315, '1.1 s',   'Niveau 2', False),
    ]
    for row in data:
        pdf.metric_row(*row)

    pdf.ln(5)
    pdf.sub_title('Analyse : Pourquoi GRU est le meilleur ?', GREEN)
    pdf.body(
        'Le GRU obtient MAPE = 1.84%, ce qui signifie que ses predictions sont en moyenne a seulement '
        '1.84% du prix reel. Sur un prix de 40,000$, c\'est une erreur de ~736$. '
        'Il bat le LSTM car ses 2 portes suffisent pour Bitcoin, avec moins de risque d\'overfitting. '
        'Il est plus rapide a converger et generalise mieux sur les donnees de test.'
    )

    pdf.sub_title('Pourquoi Prophet echoue sur Bitcoin ?', RED)
    pdf.body(
        'Prophet est concu pour des series avec saisonnalite reguliere (ventes de Noel, trafic web...). '
        'Bitcoin n\'a pas de cycle previsible : son prix depend des halvings (tous les 4 ans), '
        'des reglementations imprevues, des tweets d\'Elon Musk, des faillites d\'exchanges (FTX...). '
        'Prophet extrapole une tendance lineaire qui ne correspond pas a la realite volatile de Bitcoin. '
        'C\'est un tres bon modele - mais pas pour ce type de serie.'
    )

    pdf.sub_title('Limites de nos modeles', (255, 152, 0))
    pdf.bullet('Pas de donnees externes : news, sentiment Twitter, volume on-chain, dominance BTC')
    pdf.bullet('Prediction du lendemain uniquement (horizon 1 jour), pas sur plusieurs semaines')
    pdf.bullet('Modeles entraines sur donnees passees - pas certains de tenir en conditions futures')
    pdf.bullet('Le prix Bitcoin peut faire +20% ou -30% en 1 jour (black swan events)')
    pdf.bullet('Pas de gestion du risque financier - ce sont des modeles de prediction, pas de trading')

    # ================================================================
    # SECTION 6 : QUESTIONS DU PROFESSEUR
    # ================================================================
    pdf.add_page()
    pdf.section_title('6', 'QUESTIONS PROBABLES DU PROFESSEUR', RED)

    pdf.sub_title('Questions sur les donnees et le preprocessing', BLUE)

    qas_data = [
        ('Pourquoi avoir choisi Bitcoin ?',
         'Bitcoin est une serie temporelle financiere ideale : donnees publiques, historique long (2010-2024), '
         'forte volatilite qui permet de tester les capacites des modeles. C\'est aussi recommande dans '
         'la strategie du sujet.'),
        ('Pourquoi lookback = 60 jours ?',
         '60 jours represente environ 2 mois de trading. Assez pour capturer les tendances moyen terme '
         '(cycles de marche, support/resistance). Trop court (7 jours) = pas assez de contexte. '
         'Trop long (120 jours) = bruit et information perimee. 60 est le standard dans la litterature.'),
        ('Pourquoi MinMaxScaler et pas StandardScaler ?',
         'MinMaxScaler ramene les donnees entre [0,1], ce qui est optimal pour les fonctions d\'activation '
         'sigmoid et tanh utilisees dans LSTM/GRU. StandardScaler (moyenne=0, std=1) peut produire '
         'des valeurs negatives qui perturbent ces activations.'),
        ('Pourquoi un split 80/10/10 et pas 70/15/15 ?',
         'Plus on donne de donnees au train, mieux le modele apprend. Avec 4955 jours, 10% de test = 496 jours '
         '(plus d\'un an) - suffisant pour une evaluation robuste. Le split est TEMPOREL : jamais de donnees '
         'futures dans le train (sinon data leakage).'),
    ]

    for q, a in qas_data:
        pdf.qa_box(q, a)

    pdf.add_page()
    pdf.sub_title('Questions sur les modeles', ORANGE)

    qas_modeles = [
        ('C\'est quoi ARIMA ?',
         'ARIMA (AutoRegressive Integrated Moving Average) est un modele statistique classique. '
         'AR(p) = regression sur les p valeurs passees. I(d) = differentiation d fois pour rendre '
         'la serie stationnaire. MA(q) = regression sur les q erreurs passees. '
         'Notre ordre optimal : (1,1,2) trouve automatiquement par auto_arima via critere AIC.'),
        ('C\'est quoi le LSTM ?',
         'Long Short-Term Memory : reseau neuronal recurrent avec des portes (gates) qui controlent '
         'la memoire. 3 portes : input gate (quoi apprendre), forget gate (quoi oublier), '
         'output gate (quoi sortir). Resout le probleme du gradient qui disparait dans les RNN classiques. '
         'Ideal pour les sequences longues.'),
        ('Pourquoi GRU bat LSTM ici ?',
         'GRU a moins de parametres (2 portes vs 3). Sur Bitcoin, les dependances temporelles '
         'importantes sont a court-moyen terme. GRU generalise mieux car moins prone a l\'overfitting. '
         'De plus, GRU entraine plus vite. La complexite supplementaire du LSTM n\'apporte pas de gain.'),
        ('C\'est quoi le CNN dans CNN-LSTM ?',
         'Conv1D (convolution 1D) applique un filtre glissant sur la serie temporelle pour extraire '
         'des patterns locaux (comme des motifs de forme recurrents). MaxPooling reduit la dimension. '
         'Ensuite le LSTM traite ces features extraites. Hybride tres efficace : CNN = feature extractor, '
         'LSTM = sequence learner.'),
        ('Pourquoi XGBoost necessite du feature engineering ?',
         'Contrairement a LSTM/GRU qui apprennent automatiquement les dependances temporelles, '
         'XGBoost traite chaque ligne independamment. Il ne "voit" pas la sequence. '
         'On doit donc lui creer manuellement des features temporelles : prix j-1, j-7, j-30, '
         'moyenne mobile 14 jours, volatilite 30 jours, etc. sans quoi il n\'a aucune information temporelle.'),
        ('C\'est quoi le Dropout ?',
         'Dropout = technique de regularisation. Pendant l\'entrainement, on desactive aleatoirement '
         '20% (dropout=0.2) des neurones a chaque etape. Ca oblige le reseau a ne pas dependre '
         'de neurones specifiques et a apprendre des representations plus robustes. '
         'Evite l\'overfitting (memorisation des donnees d\'entrainement).'),
    ]

    for q, a in qas_modeles:
        pdf.qa_box(q, a)

    pdf.add_page()
    pdf.sub_title('Questions sur les resultats et metriques', GREEN)

    qas_results = [
        ('C\'est quoi MAE, RMSE, MAPE ?',
         'MAE (Mean Absolute Error) = moyenne des erreurs absolues en USD. Facile a interpreter : '
         '"en moyenne, on se trompe de X dollars". '
         'RMSE (Root Mean Squared Error) = penalise plus les grandes erreurs (racine de la moyenne des carres). '
         'MAPE (Mean Absolute Percentage Error) = erreur en pourcentage, independante de l\'echelle du prix. '
         'MAPE est la metrique principale car elle est comparable entre periodes (prix 5000$ et 50000$).'),
        ('Pourquoi GRU a un MAPE de 1.84% ?',
         '1.84% signifie que la prediction est a 1.84% du prix reel en moyenne. '
         'Sur un prix de 40,000$, l\'erreur moyenne est d\'environ 736$. '
         'C\'est excellent pour une serie aussi volatile que Bitcoin. '
         'Le modele capture bien les tendances mais pas les micro-variations extremes.'),
        ('Pourquoi Prophet a le pire MAPE (24.73%) ?',
         'Prophet est concu pour des series avec saisonnalite reguliere. Bitcoin n\'a pas de '
         'cycles previsibles annuels ou hebdomadaires stables. Prophet extrapole une tendance '
         'qui peut diverger fortement. On a limite a 500 jours de training pour reduire l\'erreur '
         '(de 186% initial a 24.73%), mais ce modele reste inadapte a Bitcoin.'),
        ('Qu\'est-ce que l\'overfitting ? Comment vous l\'avez evite ?',
         'Overfitting = le modele memorise les donnees d\'entrainement mais generalise mal '
         'sur de nouvelles donnees. On l\'a evite avec : (1) Dropout(0.2), (2) EarlyStopping '
         '(arret si validation loss ne s\'ameliore pas pendant 10 epochs), (3) '
         'ReduceLROnPlateau (reduit le learning rate si stagnation), (4) jeu de validation '
         'separe pour monitorer la generalisation.'),
        ('Pourquoi les modeles DL font mieux qu\'ARIMA ?',
         'ARIMA est lineaire et stationnaire : il suppose que les relations futures ressemblent '
         'au passe de facon lineaire. Bitcoin est hautement non-lineaire. '
         'LSTM/GRU capturent des relations non-lineaires complexes grace aux fonctions d\'activation. '
         'Ils ont aussi plus de "memoire" grace aux gates qui selectionne les informations importantes.'),
    ]

    for q, a in qas_results:
        pdf.qa_box(q, a)

    pdf.add_page()
    pdf.sub_title('Questions sur les choix methodologiques', PURPLE)

    qas_method = [
        ('Pourquoi 50 epochs ?',
         'Une epoch = un passage complet sur toutes les donnees d\'entrainement. '
         '50 epochs est le maximum fixe, mais EarlyStopping arrete avant si necessaire. '
         'En pratique, LSTM s\'est arrete a 50, GRU a 50, Stacked LSTM a 40, CNN-LSTM a 50. '
         'Augmenter les epochs sans EarlyStopping causerait de l\'overfitting.'),
        ('Pourquoi batch_size = 32 ?',
         'Le batch size determine combien d\'exemples sont traites avant une mise a jour des poids. '
         '32 est le standard en deep learning : assez grand pour un gradient stable, '
         'assez petit pour une bonne regularisation. Valeurs classiques : 16, 32, 64, 128.'),
        ('Avez-vous fait de la validation croisee ?',
         'Non - en series temporelles, la validation croisee classique (k-fold) n\'est pas applicable '
         'car elle melange les temps. On utilise la validation temporelle (walk-forward) : '
         'toujours entrainer sur le passe, tester sur le futur. Notre split 80/10/10 '
         'respecte ce principe fondamental.'),
        ('Pourquoi ne pas utiliser un Transformer (TFT) ?',
         'Le Temporal Fusion Transformer est en Niveau 3 "avance" de la strategie - optionnel. '
         'Il necessite PyTorch et pytorch-forecasting, une configuration plus complexe, '
         'et un temps d\'entrainement beaucoup plus long. Nos 6 modeles couvrent deja '
         'les 3 niveaux de complexite demandes et donnent d\'excellents resultats.'),
    ]

    for q, a in qas_method:
        pdf.qa_box(q, a)

    # ================================================================
    # SECTION 7 : POINTS FORTS
    # ================================================================
    pdf.add_page()
    pdf.section_title('7', 'POINTS FORTS A METTRE EN AVANT', ORANGE)

    pdf.sub_title('Ce qui distingue votre projet', ORANGE)

    points = [
        ('7 modeles compares sur 3 niveaux',
         'Le sujet demande ARIMA + 1 modele IA. Vous avez fait 7 modeles couvrant '
         'Niveau 1 (ARIMA, LSTM), Niveau 2 (GRU, Stacked LSTM, Prophet) et '
         'Niveau 3 (CNN-LSTM, XGBoost). Cela montre une comprehension profonde du domaine.'),
        ('8 etapes methodologiques respectees',
         'Vous avez suivi exactement le pipeline recommande par Pr. BARBARA : '
         'collecte -> EDA -> preprocessing -> ARIMA -> LSTM/GRU -> evaluation -> '
         'visualisation -> analyse critique.'),
        ('Metriques reelles mesurees',
         'Tous les chiffres (MAE, RMSE, MAPE, temps d\'entrainement) sont reels, '
         'mesures sur le vrai jeu de test. Pas de valeurs inventees.'),
        ('Analyse critique pertinente',
         'Vous expliquez POURQUOI Prophet echoue sur Bitcoin, POURQUOI GRU bat LSTM, '
         'POURQUOI XGBoost necessite du feature engineering. Ce n\'est pas juste '
         'des chiffres - c\'est une vraie comprehension.'),
        ('GRU MAPE 1.84% = resultat excellent',
         'Une erreur de seulement 1.84% sur une serie aussi volatile que Bitcoin '
         'est un resultat remarquable. 10x meilleur qu\'ARIMA (19.21%).'),
        ('Notebook Jupyter interactif',
         'Tout le projet est dans PFA_Bitcoin_Prediction.ipynb avec graphiques inline, '
         'code propre, explications en markdown. Professionnel et reproductible.'),
        ('Rapport PDF genere automatiquement',
         'generate_report.py produit un PDF complet avec tous les resultats. '
         'Automatisation = bonne pratique de genie logiciel.'),
    ]

    for titre, desc in points:
        pdf.ln(2)
        pdf.set_fill_color(*ORANGE)
        pdf.rect(10, pdf.get_y(), 4, 6, 'F')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*DARK)
        pdf.set_x(16)
        pdf.cell(0, 6, s(titre))
        pdf.ln(7)
        pdf.set_font('Helvetica', '', 9.5)
        pdf.set_text_color(*GREY)
        pdf.set_x(16)
        pdf.multi_cell(180, 5, s(desc))
        pdf.ln(2)
        pdf.set_text_color(*DARK)

    pdf.sub_title('Phrase cle a retenir pour la soutenance', RED)
    pdf.info_box(
        'Conclusion a presenter au professeur',
        '"Notre projet demontre que les modeles sequentiels (GRU, LSTM) surpassent largement '
        'les methodes statistiques classiques (ARIMA) pour la prediction du Bitcoin. '
        'GRU obtient un MAPE de 1.84% contre 19.21% pour ARIMA - soit une amelioration de 10x. '
        'La comparaison sur 3 niveaux (7 modeles) nous permet de conclure que la complexite '
        'du modele n\'est pas toujours synonyme de performance : GRU, plus simple que Stacked LSTM '
        'et CNN-LSTM, donne les meilleurs resultats sur cette serie temporelle financiere."',
        GREEN
    )

    pdf.ln(5)
    pdf.sub_title('Limites a mentionner (montre la maturite academique)', BLUE)
    pdf.bullet('Nos modeles ne prennent pas en compte les donnees externes (actualites, reseaux sociaux)')
    pdf.bullet('Prediction a horizon 1 jour uniquement - difficilement extensible a plusieurs semaines')
    pdf.bullet('Le marche crypto est influence par des black swans impredictibles (FTX, Covid...)')
    pdf.bullet('Pas de backtesting financier (on ne teste pas si ca serait profitable en trading reel)')
    pdf.bullet('Amelioration possible avec le Temporal Fusion Transformer (TFT) - Niveau 3 avance')

    # Page finale
    pdf.add_page()
    pdf.set_fill_color(*ORANGE)
    pdf.rect(0, 80, 210, 80, 'F')
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(0, 100)
    pdf.cell(210, 12, s('Bonne Soutenance !'), align='C')
    pdf.set_font('Helvetica', '', 13)
    pdf.set_xy(0, 118)
    pdf.cell(210, 8, s('Hatim Tajimi'), align='C')
    pdf.set_xy(0, 128)
    pdf.cell(210, 8, s('Encadrant : Pr. Idriss BARBARA | EMSI Rabat 4IIR'), align='C')
    pdf.set_text_color(*DARK)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_xy(0, 170)
    pdf.cell(210, 8, s('Rappel des metriques cles :'), align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_xy(0, 180)
    pdf.cell(210, 7, s('GRU : MAPE 1.84%  |  CNN-LSTM : 1.91%  |  LSTM : 2.20%  |  ARIMA : 19.21%'), align='C')

    pdf.output(OUT)
    print(f'Guide de soutenance genere : {OUT}')
    size = os.path.getsize(OUT)
    print(f'Taille : {size/1024:.0f} KB')


if __name__ == '__main__':
    generate()
