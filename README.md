# 🏆 Kaggle Ames Housing: Top Ranking via Hybrid Blending & Feature Engineering

[![Kaggle Rank](https://img.shields.io/badge/Kaggle_Rank-128th-blue?style=flat&logo=kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![Leaderboard Score](https://img.shields.io/badge/Public_LB-0.11842-brightgreen?style=flat&logo=kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Auteur :** Skander Trigui  
**Compétition :** [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
**Rang Kaggle :** 🏅 **#128** sur le Leaderboard (Score RMSLE : **0.11842**)

---

## 🏆 Kaggle Leaderboard Result

![Kaggle Leaderboard Ranking](image.png)

> **Score final :** `0.11842` | **Rang :** `128` | **Nombre de soumissions :** `33`

---

## 📌 Présentation du Projet

Ce dépôt contient la solution développée pour la compétition officielle Kaggle **House Prices - Advanced Regression Techniques**. L'objectif est de prédire le prix de vente final (`SalePrice`) de résidences à Ames, Iowa, à partir de 79 variables hétérogènes.

La force de cette approche réside dans un **pipeline de données strict garanti sans fuite (Zero-Leakage)**, combiné à un **ensemble hybride multi-modèles** optimisé pour généraliser sur le test set de Kaggle.

---

## 🛠️ Pipeline & Choix Techniques

### 1. Preprocessing & Feature Engineering Anti-Leakage
* **Suppression des Outliers :** Retrait ciblée de 2 ventes anormales (>4000 sqft avec prix anormalement bas) basées sur la documentation de De Cock (2011).
* **Target Encoding Strict :** Normalisation du quartier (`Neighborhood`) calculée uniquement sur les folds d'entraînement pour éviter toute fuite vers le test set.
* **Variables Synthétiques Stratégiques :**
  * `QualSurface` = `OverallQual` × `TotalSF` *(Feature la plus explicative)*
  * `TotalSF` = Surface sous-sol + RDC + Étage
  * `TotalBath` = Indicateur composite de salles de bain et points d'eau.
* **Imputation Metier & Scaling :** Imputation contextualisée par groupe (ex: `LotFrontage` par quartier), winsorisation des extrêmes (1%-99%), et alignement de la distribution par transformation `log1p`.

### 2. Validation Croisée & Stratégie Multi-Modèles
* **Scheme CV :** 10-Fold CV (`KFold(n_splits=10, shuffle=True, SEED=42)`).
* **7 Modèles de Base :**
  * **Linéaires Régularisés :** Lasso, Ridge, ElasticNet
  * **Gradient Boosting :** Sklearn GBM (Huber Loss), XGBoost, LightGBM, CatBoost

### 3. Blend Hybride (Performance + Diversité)
Pour maximiser la capacité de généralisation sur le jeu de test Kaggle sans sur-ajuster le jeu d'entraînement, le blend final applique une matrice de poids composite :
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
├── notebook.ipynb             # Code source complet (EDA, Preprocessing, CV & Blending)
├── image.png                  # Preuve du classement Leaderboard Kaggle
├── README.md                  # Documentation du projet
└── data/                      # Données de la compétition
    ├── train.csv
    └── test.csv
