# 📝 Exemples de Code - Utilisation de l'Application

## 🎯 Exemple 1: Lancer Streamlit (Le Plus Simple)

```bash
cd /Users/anass/PycharmProjects/PythonProject9
streamlit run main_mlflow.py
```

**Résultat**: L'app s'ouvre à `http://localhost:8501`

Vous verrez:
- Un formulaire pour remplir les caractéristiques de la voiture
- Un bouton "Prédire le Prix"
- Le prix estimé en DH
- Des visualisations des features

---

## 🔬 Exemple 2: Utiliser le Modèle en Python

### Code Simple
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

# Créer une voiture d'exemple
data = {
    'Kilométrage': '50 000 - 54 999',
    'Puissance fiscale': 6,
    'Année-Modèle': 2018,
    'Jantes aluminium': 0,
    # ... ajouter toutes les features
}

# Convertir en DataFrame
df = pd.DataFrame([data])

# Faire une prédiction
prediction_scaled = model.predict(df)[0]
prix_final = prediction_scaled * price_scaler_info['scale'] + price_scaler_info['mean']

print(f"Prix estimé: {prix_final:,.0f} DH")
```

---

## 📊 Exemple 3: Entraîner un Nouveau Modèle

```python
from train_with_mlflow import CarPricePipeline

# Initialiser le pipeline
pipeline = CarPricePipeline()

# Charger les données
df = pipeline.load_data('data/raw/avito_car_dataset_ALL.csv')

# Prétraiter
df = pipeline.preprocess_data(df)

# Préparer les features
X, y = pipeline.prepare_features(df)

# Entraîner le modèle
pipeline.train_model(X, y)

# Évaluer
metrics = pipeline.evaluate_model(X, y)
print(metrics)

# Sauvegarder les artifacts
pipeline.save_artifacts()
```

---

## 🧪 Exemple 4: Utiliser MLflow UI

```bash
# Démarrer MLflow UI
mlflow ui

# Cela ouvre http://localhost:5000
# Vous voyez alors:
# - Tous les runs d'entraînement
# - Les métriques de chaque run
# - Les hyperparamètres
# - Les artifacts sauvegardés
# - Possibilité de comparer les runs
```

---

## 🚀 Exemple 5: Lancer le Workflow Complet

```bash
# Terminal 1: Entraîner le modèle
python scripts/train_with_mlflow.py

# Terminal 2: Voir les expériences
mlflow ui

# Terminal 3: Lancer l'application
streamlit run main_mlflow.py
```

**Résultat**:
- Terminal 1 affiche les logs d'entraînement
- Terminal 2 ouvre MLflow UI à `http://localhost:5000`
- Terminal 3 ouvre Streamlit à `http://localhost:8501`

---

## 📈 Exemple 6: Charger Directement du Modèle MLflow

```python
import mlflow.sklearn

# Set tracking URI
mlflow.set_tracking_uri("file:./mlflow/mlruns")

# Charger un modèle spécifique
model = mlflow.sklearn.load_model("models:/CarPricePredictor/1")

# Faire une prédiction
predictions = model.predict(X_test)
```

---

## 🎨 Exemple 7: Script Personnalisé de Prédiction

```python
# predict_custom.py
import joblib
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

def load_artifacts():
    """Charger tous les artifacts"""
    model = joblib.load('models/car_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    with open('artifacts/feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    with open('artifacts/price_scaler_info.json', 'r') as f:
        price_scaler_info = json.load(f)
    
    return model, scaler, feature_info, price_scaler_info

def predict_price(car_features):
    """Prédire le prix d'une voiture"""
    model, scaler, feature_info, price_scaler_info = load_artifacts()
    
    # Créer un DataFrame avec les features
    df = pd.DataFrame([car_features])
    
    # S'assurer que les colonnes sont dans le bon ordre
    df = df[feature_info['feature_names']]
    
    # Normaliser les features numériques
    numerical_cols = [col for col in feature_info['numerical_cols'] 
                     if col in df.columns]
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Prédire
    prediction_scaled = model.predict(df)[0]
    prix_final = prediction_scaled * price_scaler_info['scale'] + price_scaler_info['mean']
    
    return prix_final

# Utilisation
car = {
    'Kilométrage': '50 000 - 54 999',
    'Puissance fiscale': 6,
    'Année-Modèle': 2018,
    # ... toutes les features
}

prix = predict_price(car)
print(f"Prix estimé: {prix:,.0f} DH")
```

---

## 🐍 Exemple 8: Boucle de Prédictions

```python
import pandas as pd
from predict_custom import predict_price

# CSV avec plusieurs voitures
cars_df = pd.read_csv('cars_to_predict.csv')

# Prédire pour toutes
results = []
for idx, row in cars_df.iterrows():
    car_dict = row.to_dict()
    price = predict_price(car_dict)
    results.append({
        'car': row['name'],
        'predicted_price': price
    })

# Sauvegarder les résultats
results_df = pd.DataFrame(results)
print(results_df)
```

---

## 🔗 Exemple 9: Intégration avec une API Flask

```python
from flask import Flask, request, jsonify
from predict_custom import predict_price
model = joblib.load('models/car_model.pkl')
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint API pour les prédictions"""
    data = request.json
    
    try:
        price = predict_price(data)
        return jsonify({
            'success': True,
            'predicted_price': price,
            'currency': 'DH'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Utilisation**:
```bash
# Démarrer l'API
python app.py

# Faire une requête
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Kilométrage": "50 000 - 54 999", ...}'
```

---

## 📊 Exemple 10: Analyse des Prédictions

```python
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Charger le modèle et les données
model = joblib.load('models/car_model.pkl')
test_data = pd.read_csv('test_data.csv')

X_test = test_data.drop('Prix', axis=1)
y_test = test_data['Prix']

# Faire des prédictions
y_pred = model.predict(X_test)

# Calculer les métriques
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R²: {r2:.4f}")

# Analyser les erreurs
errors = y_test - y_pred
print(f"Erreur moyenne: {errors.mean():,.0f} DH")
print(f"Écart-type erreur: {errors.std():,.0f} DH")
```

---

## 🎯 Exemple 11: Optimiser les Hyperparamètres

```python
from train_with_mlflow import CarPricePipeline
from sklearn.model_selection import GridSearchCV
import yaml

# Charger les paramètres
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Grille de recherche
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Pipeline
pipeline = CarPricePipeline()
df = pipeline.load_data()
df = pipeline.preprocess_data(df)
X, y = pipeline.prepare_features(df)

# GridSearch
grid_search = GridSearchCV(
    pipeline.model,
    param_grid,
    cv=5,
    scoring='r2'
)

grid_search.fit(X, y)

print(f"Meilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score: {grid_search.best_score_:.4f}")
```

---

## 🧪 Exemple 12: Exécuter les Tests

```bash
# Tous les tests
python -m pytest tests/ -v

# Avec couverture
python -m pytest tests/ --cov=. --cov-report=html

# Tests spécifiques
python -m pytest tests/test_pipeline.py -v

# Mode debug
python -m pytest tests/ -vv -x
```

---

## 📚 Fichiers à Consulter

- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Guide complet
- [main_mlflow.py](main_mlflow.py) - Code Streamlit complet
- [scripts/train_with_mlflow.py](scripts/train_with_mlflow.py) - Code d'entraînement
- [README_MLops.md](README_MLops.md) - Setup MLOps

---

**Besoin d'aide ?** Consultez les guides complètement documentés dans le projet !
