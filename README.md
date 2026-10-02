# 💶 Détection de faux billets par Machine Learning

## 📌 Contexte

Ce projet a pour objectif de développer un algorithme capable de **déterminer automatiquement si un billet est authentique ou faux** à partir de ses caractéristiques géométriques.

Le projet couvre l'ensemble d'une démarche de Machine Learning, depuis l'exploration et la préparation des données jusqu'à la création d'un modèle utilisable sur de nouveaux billets.

---

## 🎯 Objectifs

- Explorer et nettoyer les données disponibles
- Identifier les variables permettant de différencier les billets authentiques des faux
- Traiter les valeurs manquantes
- Préparer les données pour l'apprentissage automatique
- Tester plusieurs algorithmes de classification
- Comparer leurs performances
- Sélectionner un modèle final
- Construire un pipeline de Machine Learning reproductible
- Développer un script permettant de tester de nouveaux billets

---

## 📊 Données

Le jeu de données contient **1 500 billets** décrits par six caractéristiques géométriques :

- `diagonal` : diagonale du billet
- `height_left` : hauteur du côté gauche
- `height_right` : hauteur du côté droit
- `margin_low` : marge inférieure
- `margin_up` : marge supérieure
- `length` : longueur du billet

La variable cible est :

- `is_genuine` : indique si le billet est authentique (`True`) ou faux (`False`)

Le dataset contient :

- **1 000 billets authentiques**
- **500 faux billets**

Des valeurs manquantes ont été identifiées dans la variable `margin_low`.

---

## 🔎 Analyse exploratoire

L'analyse exploratoire a permis d'étudier :

- la structure et la qualité des données ;
- les valeurs manquantes ;
- les doublons ;
- les distributions des variables ;
- les valeurs atypiques ;
- les différences entre billets authentiques et faux ;
- les corrélations entre les variables.

Certaines caractéristiques, notamment `length` et `margin_low`, présentent une forte capacité à différencier les deux classes.

---

## 🧹 Prétraitement

Les données ont été séparées en :

- **80 % pour l'entraînement**
- **20 % pour le test**

Une stratification sur la variable cible a été utilisée afin de conserver la proportion de billets authentiques et faux dans les deux ensembles.

Les principales étapes de preprocessing sont :

### Imputation

Les valeurs manquantes de `margin_low` sont remplacées par la **médiane calculée sur le jeu d'entraînement**.

Cette méthode permet de conserver les observations tout en limitant l'influence des valeurs extrêmes.

### Standardisation

Les variables sont standardisées avec `StandardScaler` lorsque nécessaire afin de les placer sur des échelles comparables.

Cette étape est particulièrement importante pour les algorithmes utilisant des distances, comme **KNN** et **K-means**.

Les transformations sont apprises uniquement sur le jeu d'entraînement afin d'éviter toute **fuite de données (data leakage)**.

---

## 🤖 Modèles testés

Quatre approches ont été étudiées.

### Régression logistique

Modèle supervisé de classification permettant d'estimer la probabilité d'appartenance d'un billet à l'une des deux classes.

### K-Nearest Neighbors (KNN)

Algorithme supervisé réalisant une prédiction à partir des voisins les plus proches.

La valeur de `K` a été recherchée avec **GridSearchCV** et une **validation croisée à 5 plis**, en utilisant le recall des faux billets comme métrique de sélection.

### Random Forest

Modèle supervisé reposant sur l'agrégation de plusieurs arbres de décision.

### K-means

Algorithme non supervisé utilisé afin d'étudier si les caractéristiques géométriques permettent naturellement de faire apparaître deux groupes de billets.

---

## 📈 Évaluation des modèles

Les modèles ont notamment été comparés avec :

- Accuracy
- Precision
- Recall
- F1-score
- Matrice de confusion
- AUC

Une attention particulière a été accordée au **recall des faux billets**, afin de mesurer la capacité du modèle à détecter les billets réellement faux.

La régression logistique obtient notamment :

- **Accuracy : 99 %**
- **Recall des faux : 98 %**
- **Precision des faux : environ 99 %**
- **AUC : environ 0,999**

Sur les 300 billets du jeu de test, **297 sont correctement classés**.

Parmi les **100 faux billets**, **98 sont correctement détectés**.

---

## 🏆 Modèle retenu

La **régression logistique** a finalement été retenue.

La Random Forest présente des performances très proches, mais la régression logistique a été privilégiée pour sa **simplicité à performances comparables**.

Les performances obtenues sur les jeux d'entraînement et de test sont également très proches, ce qui indique une bonne capacité de généralisation sur les données étudiées.

---

## ⚙️ Pipeline final

Le modèle final est intégré dans un pipeline Scikit-learn :

```text
Données brutes
     ↓
SimpleImputer (médiane)
     ↓
StandardScaler
     ↓
LogisticRegression
     ↓
Prédiction True / False
```

Le pipeline garantit que les nouvelles données subissent automatiquement les mêmes transformations que celles utilisées pendant l'entraînement.

Le pipeline entraîné est sauvegardé avec **Joblib** afin de pouvoir être réutilisé sans réentraîner le modèle.

---

## 💻 Script de prédiction

Le script `detection_billet.py` permet d'utiliser le modèle entraîné sur de nouvelles données.

### Tester un fichier CSV

```bash
python detection_billet.py --csv billets_production.csv
```

Le fichier doit contenir les six variables attendues par le modèle.

Le script retourne une prédiction pour chaque billet :

```text
True
False
True
```

- `True` : billet prédit authentique
- `False` : billet prédit faux

### Tester un billet individuellement

Le script permet également de transmettre directement les six caractéristiques géométriques d'un billet :

```bash
python detection_billet.py --billet <diagonal> <height_left> <height_right> <margin_low> <margin_up> <length>
```

---

## 🛠️ Technologies utilisées

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 🧠 Compétences développées

Ce projet m'a permis de travailler sur :

- l'analyse exploratoire de données ;
- le nettoyage et le preprocessing ;
- la gestion des valeurs manquantes ;
- la prévention du data leakage ;
- la classification supervisée ;
- le clustering non supervisé ;
- la validation croisée ;
- l'optimisation d'hyperparamètres avec GridSearchCV ;
- l'évaluation et la comparaison de modèles ;
- la création de pipelines Scikit-learn ;
- la sauvegarde et le déploiement d'un modèle entraîné ;
- la création d'un script de prédiction utilisable sur de nouvelles données.

---# 💶 Détection de faux billets par Machine Learning

## 📌 Contexte

Ce projet a pour objectif de développer un algorithme capable de **déterminer automatiquement si un billet est authentique ou faux** à partir de ses caractéristiques géométriques.

Le projet couvre l'ensemble d'une démarche de Machine Learning, depuis l'exploration et la préparation des données jusqu'à la création d'un modèle utilisable sur de nouveaux billets.

---

## 🎯 Objectifs

- Explorer et nettoyer les données disponibles
- Identifier les variables permettant de différencier les billets authentiques des faux
- Traiter les valeurs manquantes
- Préparer les données pour l'apprentissage automatique
- Tester plusieurs algorithmes de classification
- Comparer leurs performances
- Sélectionner un modèle final
- Construire un pipeline de Machine Learning reproductible
- Développer un script permettant de tester de nouveaux billets

---

## 📊 Données

Le jeu de données contient **1 500 billets** décrits par six caractéristiques géométriques :

- `diagonal` : diagonale du billet
- `height_left` : hauteur du côté gauche
- `height_right` : hauteur du côté droit
- `margin_low` : marge inférieure
- `margin_up` : marge supérieure
- `length` : longueur du billet

La variable cible est :

- `is_genuine` : indique si le billet est authentique (`True`) ou faux (`False`)

Le dataset contient :

- **1 000 billets authentiques**
- **500 faux billets**

Des valeurs manquantes ont été identifiées dans la variable `margin_low`.

---

## 🔎 Analyse exploratoire

L'analyse exploratoire a permis d'étudier :

- la structure et la qualité des données ;
- les valeurs manquantes ;
- les doublons ;
- les distributions des variables ;
- les valeurs atypiques ;
- les différences entre billets authentiques et faux ;
- les corrélations entre les variables.

Certaines caractéristiques, notamment `length` et `margin_low`, présentent une forte capacité à différencier les deux classes.

---

## 🧹 Prétraitement

Les données ont été séparées en :

- **80 % pour l'entraînement**
- **20 % pour le test**

Une stratification sur la variable cible a été utilisée afin de conserver la proportion de billets authentiques et faux dans les deux ensembles.

Les principales étapes de preprocessing sont :

### Imputation

Les valeurs manquantes de `margin_low` sont remplacées par la **médiane calculée sur le jeu d'entraînement**.

Cette méthode permet de conserver les observations tout en limitant l'influence des valeurs extrêmes.

### Standardisation

Les variables sont standardisées avec `StandardScaler` lorsque nécessaire afin de les placer sur des échelles comparables.

Cette étape est particulièrement importante pour les algorithmes utilisant des distances, comme **KNN** et **K-means**.

Les transformations sont apprises uniquement sur le jeu d'entraînement afin d'éviter toute **fuite de données (data leakage)**.

---

## 🤖 Modèles testés

Quatre approches ont été étudiées.

### Régression logistique

Modèle supervisé de classification permettant d'estimer la probabilité d'appartenance d'un billet à l'une des deux classes.

### K-Nearest Neighbors (KNN)

Algorithme supervisé réalisant une prédiction à partir des voisins les plus proches.

La valeur de `K` a été recherchée avec **GridSearchCV** et une **validation croisée à 5 plis**, en utilisant le recall des faux billets comme métrique de sélection.

### Random Forest

Modèle supervisé reposant sur l'agrégation de plusieurs arbres de décision.

### K-means

Algorithme non supervisé utilisé afin d'étudier si les caractéristiques géométriques permettent naturellement de faire apparaître deux groupes de billets.

---

## 📈 Évaluation des modèles

Les modèles ont notamment été comparés avec :

- Accuracy
- Precision
- Recall
- F1-score
- Matrice de confusion
- AUC

Une attention particulière a été accordée au **recall des faux billets**, afin de mesurer la capacité du modèle à détecter les billets réellement faux.

La régression logistique obtient notamment :

- **Accuracy : 99 %**
- **Recall des faux : 98 %**
- **Precision des faux : environ 99 %**
- **AUC : environ 0,999**

Sur les 300 billets du jeu de test, **297 sont correctement classés**.

Parmi les **100 faux billets**, **98 sont correctement détectés**.

---

## 🏆 Modèle retenu

La **régression logistique** a finalement été retenue.

La Random Forest présente des performances très proches, mais la régression logistique a été privilégiée pour sa **simplicité à performances comparables**.

Les performances obtenues sur les jeux d'entraînement et de test sont également très proches, ce qui indique une bonne capacité de généralisation sur les données étudiées.

---

## ⚙️ Pipeline final

Le modèle final est intégré dans un pipeline Scikit-learn :

```text
Données brutes
     ↓
SimpleImputer (médiane)
     ↓
StandardScaler
     ↓
LogisticRegression
     ↓
Prédiction True / False
```

Le pipeline garantit que les nouvelles données subissent automatiquement les mêmes transformations que celles utilisées pendant l'entraînement.

Le pipeline entraîné est sauvegardé avec **Joblib** afin de pouvoir être réutilisé sans réentraîner le modèle.

---

## 💻 Script de prédiction

Le script `detection_billet.py` permet d'utiliser le modèle entraîné sur de nouvelles données.

### Tester un fichier CSV

```bash
python detection_billet.py --csv billets_production.csv
```

Le fichier doit contenir les six variables attendues par le modèle.

Le script retourne une prédiction pour chaque billet :

```text
True
False
True
```

- `True` : billet prédit authentique
- `False` : billet prédit faux

### Tester un billet individuellement

Le script permet également de transmettre directement les six caractéristiques géométriques d'un billet :

```bash
python detection_billet.py --billet <diagonal> <height_left> <height_right> <margin_low> <margin_up> <length>
```

---

## 🛠️ Technologies utilisées

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 📁 Structure du projet

```text
├── notebook.ipynb
├── detection_billet.py
├── modele_detection_billets.joblib
├── billets.csv
├── billets_production.csv
└── README.md
```

---

## 🧠 Compétences développées

Ce projet m'a permis de travailler sur :

- l'analyse exploratoire de données ;
- le nettoyage et le preprocessing ;
- la gestion des valeurs manquantes ;
- la prévention du data leakage ;
- la classification supervisée ;
- le clustering non supervisé ;
- la validation croisée ;
- l'optimisation d'hyperparamètres avec GridSearchCV ;
- l'évaluation et la comparaison de modèles ;
- la création de pipelines Scikit-learn ;
- la sauvegarde et le déploiement d'un modèle entraîné ;
- la création d'un script de prédiction utilisable sur de nouvelles données.

---

## 👩‍💻 Auteure

**Ines Belhoussine**  
Parcours Data Analyst — OpenClassrooms
