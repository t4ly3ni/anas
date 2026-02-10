# 🚗 Prédicteur de Prix de Voiture - Maroc

Un système de prédiction de prix de voiture basé sur le dataset Avito Maroc, utilisant le machine learning et MLflow pour le tracking des expériences. Application interactive avec Streamlit.

## 📋 Table des matières

- [À propos](#à-propos)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Pipeline de données](#pipeline-de-données)
- [Résultats du modèle](#résultats-du-modèle)
- [Structure du projet](#structure-du-projet)
- [Technologies utilisées](#technologies-utilisées)

## 📖 À propos

Ce projet développe un modèle de régression pour prédire les prix des voitures au Maroc basé sur leurs caractéristiques. Le projet intègre des bonnes pratiques MLOps avec:

- **MLflow** pour le tracking des expériences et la gestion des versions de modèles
- **DVC** pour la gestion des données et pipelines
- **Streamlit** pour l'interface utilisateur interactive
- **Scikit-learn** pour la modélisation machine learning

## ✨ Fonctionnalités

✅ **Prédiction précise** - Modèle Random Forest entraîné sur 10K+ véhicules  
✅ **Interface Web** - Application Streamlit pour prédictions en temps réel  
✅ **MLOps intégré** - Tracking complet avec MLflow et gestion des versions  
✅ **Pipeline reproductible** - DVC pour garantir la reproductibilité  
✅ **Tests unitaires** - Suite de tests complète avec pytest  
✅ **Analyse EDA** - Rapports de profiling détaillés  
✅ **Monitoring** - Métriques et visualisations de performance  

## 🏗️ Architecture

```
┌─────────────────┐
│   avito_car_    │
│ dataset_ALL.csv │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  train_with_    │ ◄── params.yaml
│   mlflow.py     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Models & Artifacts:                │
│  - models/car_model.pkl             │
│  - models/scaler.pkl                │
│  - models/encoders.pkl              │
│  - artifacts/feature_info.json      │
│  - artifacts/price_scaler_info.json │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Interface Streamlit        │
│  (main.py / main_mlflow.py) │
└─────────────────────────────┘
         │
         ▼
    Prédictions
```

## 📦 Prérequis

- Python 3.8+
- pip ou conda
- Git

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/Azaziop/detection_car_price.git
cd detection_car_price
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements/requirements.txt
```

### 4. (Optionnel) Installation pour développement

```bash
pip install -r requirements/requirements-dev.txt
```

## 💻 Utilisation

### Option 1: Lancer l'application Streamlit (Recommandé)

```bash
streamlit run main_mlflow.py
```

L'application s'ouvrira à `http://localhost:8501`

**Fonctionnalités de l'app:**
- 🎯 Formulaire pour entrer les caractéristiques du véhicule
- 💰 Prédiction du prix en DH marocain
- 📊 Visualisations des features importance
- 📈 Historique des prédictions

### Option 2: Utiliser le modèle en Python

```python
import joblib
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

# Charger les artifacts
model = joblib.load('models/car_model.pkl')
scaler = joblib.load('models/scaler.pkl')

with open('artifacts/feature_info.json', 'r') as f:
    feature_info = json.load(f)

with open('artifacts/price_scaler_info.json', 'r') as f:
    price_scaler_info = json.load(f)

# Créer les encodeurs et préparer les données
# [Voir CODE_EXAMPLES.md pour l'exemple complet]

# Faire une prédiction
prediction = model.predict(X_scaled)
```

### Option 3: Réentraîner le modèle

#### Avec DVC:
```bash
dvc repro -f dvc/dvc.yaml
```

#### Ou directement:
```bash
python scripts/train_with_mlflow.py
```

### Option 4: Lancer les tests

```bash
pytest tests/ -v
pytest tests/ --cov=.  # Avec coverage
```

## 🔄 Pipeline de données

### Étapes du pipeline:

1. **Chargement** (`load_data`)
   - Lecture du CSV Avito Maroc
   - Encodage: latin1

2. **Nettoyage** (`preprocess_data`)
   - Imputation des valeurs manquantes
   - Suppression des doublons
   - Suppression des colonnes corrélées

3. **Encodage** (`encode_features`)
   - Label encoding pour variables catégoriques
   - OneHot encoding optionnel

4. **Normalisation** (`scale_features`)
   - StandardScaler pour features numériques

5. **Entraînement** (`train_model`)
   - Random Forest Regressor
   - Hyperparamètres optimisés

6. **Évaluation** (`evaluate`)
   - MAE, MSE, R² Score
   - Sauvegarde avec MLflow

### Configuration du pipeline

Voir `params.yaml`:
```yaml
train:
  test_size: 0.2
  random_state: 42
model:
  n_estimators: 100
  max_depth: 20
  min_samples_split: 5
  min_samples_leaf: 2
  max_features: 'sqrt'
```

## 📊 Résultats du modèle

Le modèle Random Forest entraîné achieves:
- **R² Score**: ~0.87
- **MAE (Mean Absolute Error)**: Environ 15-20% du prix moyen
- **Données**: 10,000+ véhicules Avito Maroc

### Features importantes:
1. Kilométrage
2. Année-Modèle
3. Marque du véhicule
4. État général
5. Puissance fiscale

## 📁 Structure du projet

```
detection_car_price/
├── README.md                      # Ce fichier
├── requirements/requirements.txt               # Dépendances pip
├── requirements/requirements-dev.txt           # Dépendances développement
├── params.yaml                    # Hyperparamètres du modèle
├── dvc/dvc.yaml                               # Pipeline DVC
├── pytest.ini                     # Configuration pytest
│
├── data/raw/avito_car_dataset_ALL.csv      # Dataset source
├── main.py                        # App Streamlit basique
├── main_mlflow.py                 # App Streamlit avec MLflow
├── scripts/train_with_mlflow.py   # Pipeline d'entraînement
├── finalpreoject.py               # Analyse EDA
├── scripts/load_model_mlflow.py   # Chargement des modèles
│
├── tests/                         # Suite de tests
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_integration.py
│   └── test_car_pipeline.py
│
├── mlflow/mlruns/                 # Artifacts MLflow
│   ├── 1/                         # Experiment 1
│   ├── 710723541858247182/        # Experiment 2
│   └── models/                    # Registered Models
│
├── reports/htmlcov/               # Coverage reports
└── __pycache__/                   # Cache Python
```

## 🛠️ Technologies utilisées

### Data & ML:
- **pandas** - Manipulation de données
- **NumPy** - Calculs numériques
- **scikit-learn** - Machine Learning
- **joblib** - Sérialisation de modèles

### MLOps:
- **MLflow** - Tracking d'expériences et versioning
- **DVC** - Gestion de données et pipelines

### Frontend:
- **Streamlit** - Interface web interactive

### Visualisation:
- **matplotlib** - Graphiques
- **seaborn** - Visualisations avancées
- **ydata-profiling** - Rapports EDA

### DevOps & Tests:
- **pytest** - Framework de test
- **PyYAML** - Gestion de fichiers YAML
- **skops** - Sérialisation scikit-learn

## 📈 Métriques MLflow

Les expériences sont trackées dans MLflow. Pour visualiser le dashboard:

```bash
mlflow ui
```

Puis accédez à `http://localhost:5000`

Vous verrez:
- Historique des entraînements
- Comparaison des métriques
- Paramètres utilisés
- Artifacts (modèles, scalers)

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Tests avec coverage
pytest tests/ --cov=. --cov-report=html

# Tests spécifiques
pytest tests/test_pipeline.py -v
pytest tests/test_integration.py -v
```

## 📚 Documentation supplémentaire

- Voir [CODE_EXAMPLES.md](CODE_EXAMPLES.md) pour des exemples d'utilisation détaillés
- Rapport de profiling: [reports/profiling_rep.html](reports/profiling_rep.html)
- Coverage report: [reports/htmlcov/index.html](reports/htmlcov/index.html)

## 🔍 Analyse EDA

Un rapport complet de l'analyse exploratoire est généré dans `reports/profiling_rep.html`:

```bash
# Régénérer le rapport (optionnel)
python finalpreoject.py
```

Contient:
- Statistiques descriptives
- Distribution des variables
- Corrélations entre features
- Détection d'anomalies
- Valeurs manquantes

## 🐛 Troubleshooting

### L'app Streamlit ne démarre pas

```bash
# Vérifier les dépendances
pip install -r requirements/requirements.txt

# Réinstaller en cas de problème
pip install --force-reinstall -r requirements/requirements.txt
```

### Modèle non trouvé

Assurez-vous d'avoir entraîné le modèle:
```bash
python scripts/train_with_mlflow.py
# ou
dvc repro -f dvc/dvc.yaml
```

### Erreurs d'encodage CSV

Le dataset utilise l'encodage `latin1`. Ne le changez pas.

## 🤝 Contribution

Les contributions sont bienvenues! Pour contribuer:

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est open source et disponible sous la licence MIT.

## 👤 Auteur

**Azaziop**  
GitHub: [@Azaziop](https://github.com/Azaziop)

## 📞 Support

Pour des questions ou des problèmes:
- Ouvrir une issue sur GitHub
- Consulter [CODE_EXAMPLES.md](CODE_EXAMPLES.md)
- Vérifier les logs MLflow

## 🎯 Objectifs futurs

- [ ] Déploiement sur cloud (AWS/GCP/Azure)
- [ ] API REST avec FastAPI
- [ ] Dashboard de monitoring
- [ ] A/B testing de modèles
- [ ] Prédictions batch
- [ ] Explainability avec SHAP

---

**Dernière mise à jour**: Février 2026  
**Version**: 1.0.0
