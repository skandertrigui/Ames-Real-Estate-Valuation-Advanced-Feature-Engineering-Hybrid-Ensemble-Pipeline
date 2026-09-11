# 🏆 Kaggle Ames Housing: Top Ranking via EDA, Hybrid Blending & Feature Engineering
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

[![Kaggle Rank](https://img.shields.io/badge/Kaggle_Rank-128th-blue?style=flat&logo=kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![Leaderboard Score](https://img.shields.io/badge/Public_LB-0.11842-brightgreen?style=flat&logo=kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Auteur :** Skander Trigui  
**Compétition :** [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
**Rang Kaggle :** 🏅 **#128** sur le Leaderboard (Score RMSLE : **0.11842**)

---

## 🏆 Classement Kaggle Leaderboard

![Kaggle Leaderboard Ranking](image.png)

> **Score final :** `0.11842` | **Rang :** `128` | **Soumissions :** `33`

---

## 📌 Présentation du Projet

Ce dépôt présente la solution développée pour la compétition officielle Kaggle **House Prices - Advanced Regression Techniques**. L'objectif est de prédire le prix de vente final (`SalePrice`) de résidences à Ames, Iowa, à partir de 79 variables hétérogènes.

La force de cette approche réside dans une **Analyse Exploratoire des Données (EDA) exhaustive**, un **pipeline de pré-traitement garanti sans fuite (Zero-Leakage)**, et un **ensemble hybride multi-modèles** hautement performant en généralisation.

---

## 🛠️ Méthodologie & Pipeline Technique

### 1. Analyse Exploratoire des Données (EDA Complète)
Une phase d'exploration approfondie a été menée en amont afin de comprendre la structure sous-jacente des données :
* **Distribution de la Cible :** Analyse de l'asymétrie (skewness) de `SalePrice` et validation de la transformation logarithmique `log1p` pour restaurer une distribution normale.
* **Détection d'Outliers :** Analyse bivariée (`GrLivArea` vs `SalePrice`) identifiant les 2 anomalies majeures recommandées par De Cock (2011).
* **Cartographie des Valeurs Manquantes :** Typologie fine des données manquantes pour distinguer les réelles absences de données des codes sémantiques (ex: l'absence de sous-sol ou de garage).
* **Analyse de Corrélation & Multicolinéarité :** Étude des matrices de corrélation (Heatmaps) et évaluation de la colinéarité entre variables physiques (`TotalBsmtSF`, `1stFlrSF`, `GrLivArea`).

### 2. Preprocessing & Feature Engineering Anti-Leakage
* **Suppression des Outliers :** Retrait des 2 ventes non résidentielles atypiques (>4000 sqft avec prix anormalement bas), améliorant le RMSE de **+0.011**.
* **Target Encoding Strict :** Normalisation du quartier (`Neighborhood`) basée sur la médiane, calculée uniquement sur les folds d'entraînement pour éliminer tout risque de fuite d'information.
* **Variables Synthétiques Stratégiques :**
  * `QualSurface` = `OverallQual` × `TotalSF` *(Variable #1 en importance)*[cite: 1]
  * `TotalSF` = Surface sous-sol + RDC + Étage[cite: 1]
  * `TotalBath` = Indicateur composite des salles de bain et points d'eau[cite: 1]
* **Imputation Métier & Transformations :** Imputation contextualisée par groupe (ex: `LotFrontage` par quartier), winsorisation des extrêmes (1%-99%), et traitement de l'asymétrie des features continues ($|skew| > 0.75$).

### 3. Validation Croisée & Stratégie Multi-Modèles
* **Scheme CV :** 10-Fold CV (`KFold(n_splits=10, shuffle=True, SEED=42)`).
* **7 Modèles de Base :**
  * **Linéaires Régularisés :** Lasso, Ridge, ElasticNet
  * **Gradient Boosting :** Sklearn GBM (Huber Loss), XGBoost, LightGBM, CatBoost

### 4. Blend Hybride (Performance + Diversité)
Afin d'optimiser le score généralisé sur le test set de Kaggle, le blend applique une matrice de poids composite équilibrant précision OOF et indépendance des algorithmes :
$$\text{Poids} = 0.60 \times \text{Performance (1/RMSE}^2) + 0.40 \times \text{Diversité Algorithmique}$$

---

## 📊 Répartition du Blend Hybride

| Modèle | Type | Rôle dans l'Ensemble | Poids Composite |
| :--- | :--- | :--- | :---: |
| **Lasso** | Linéaire (L1) | Sélection de features & Régularisation forte | ~20% |
| **Ridge** | Linéaire (L2) | Stabilité face à la multicolinéarité | ~15% |
| **ElasticNet** | Linéaire (L1 + L2) | Compromis sélection / stabilité | ~15% |
| **GBM (Huber)** | Gradient Boosting | Isolation des valeurs résiduelles aberrantes | ~15% |
| **XGBoost** | Tree Boosting | Captation des non-linéarités complexes | ~15% |
| **LightGBM** | Tree Boosting | Vitesse & partitionnement par feuille | ~10% |
| **CatBoost** | Tree Boosting | Diversité sur les variables catégorielles | ~5% |

---

## 📁 Structure du Dépôt

```bash
.
├── notebook.ipynb             # Notebook complet (EDA, Preprocessing, CV & Blending)
├── image.png                  # Preuve du classement Leaderboard Kaggle
├── README.md                  # Documentation du projet
└── data/                      # Données de la compétition
    ├── train.csv
    └── test.csv
