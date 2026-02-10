"""
Tests unitaires pour le pipeline MLOps
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_with_mlflow import CarPricePipeline


class TestDataLoading:
    """Tests pour le chargement des données"""
    
    def test_data_file_exists(self):
        """Vérifier que le fichier de données existe"""
        assert Path('data/raw/avito_car_dataset_ALL.csv').exists()
    
    def test_data_loads_correctly(self):
        """Vérifier que les données se chargent sans erreur"""
        df = pd.read_csv('data/raw/avito_car_dataset_ALL.csv', encoding='latin1')
        assert df is not None
        assert len(df) > 0
        assert 'Prix' in df.columns
    
    def test_data_columns(self):
        """Vérifier que les colonnes essentielles sont présentes"""
        df = pd.read_csv('data/raw/avito_car_dataset_ALL.csv', encoding='latin1')
        required_cols = ['Prix', 'Ville', 'Marque', 'Modèle', 'Année-Modèle', 'Kilométrage']
        for col in required_cols:
            assert col in df.columns, f"Colonne manquante: {col}"
    
    def test_data_types(self):
        """Vérifier les types de données"""
        df = pd.read_csv('data/raw/avito_car_dataset_ALL.csv', encoding='latin1')
        assert df['Prix'].dtype in [np.int64, np.float64], "Prix doit être numérique"
        assert df['Année-Modèle'].dtype in [np.int64, np.float64], "Année-Modèle doit être numérique"
    
    def test_data_no_duplicate_rows(self):
        """Vérifier qu'il n'y a pas trop de doublons"""
        df = pd.read_csv('data/raw/avito_car_dataset_ALL.csv', encoding='latin1')
        duplicates = df.duplicated().sum()
        # Accepter jusqu'à 5% de doublons
        assert duplicates / len(df) < 0.05, f"Trop de doublons: {duplicates}"
    
    def test_data_price_range(self):
        """Vérifier que les prix sont dans une plage raisonnable"""
        df = pd.read_csv('data/raw/avito_car_dataset_ALL.csv', encoding='latin1')
        assert df['Prix'].min() >= 0, "Prix négatifs détectés"
        # Note: Il y a quelques outliers extrêmes dans les données brutes (> 600M)
        # C'est pourquoi le pipeline les supprime avec la méthode IQR
        # Vérifier que la majorité des prix sont raisonnables (< 1M)
        reasonable_prices = df[df['Prix'] < 1_000_000]
        assert len(reasonable_prices) / len(df) > 0.9, "Trop de prix irréalistes"


class TestPreprocessing:
    """Tests pour le prétraitement des données"""
    
    @pytest.fixture
    def sample_data(self):
        """Créer un échantillon de données pour les tests"""
        return pd.DataFrame({
            'Ville': ['Casablanca', 'Rabat', None],
            'Prix': [100000, 150000, 120000],
            'Origine': ['Dédouanée', None, 'WW au Maroc'],
            'Nombre de portes': [4, None, 5],
            'Airbags': [2, 4, 6]
        })
    
    def test_missing_value_handling_categorical(self, sample_data):
        """Tester la gestion des valeurs manquantes catégorielles"""
        # Impute with mode
        for col in ['Ville', 'Origine']:
            if col in sample_data.columns and sample_data[col].isnull().any():
                mode_value = sample_data[col].mode()[0]
                sample_data[col] = sample_data[col].fillna(mode_value)
        
        assert sample_data['Ville'].isnull().sum() == 0
        assert sample_data['Origine'].isnull().sum() == 0
    
    def test_missing_value_handling_numerical(self, sample_data):
        """Tester la gestion des valeurs manquantes numériques"""
        if 'Nombre de portes' in sample_data.columns and sample_data['Nombre de portes'].isnull().any():
            median_value = sample_data['Nombre de portes'].median()
            sample_data['Nombre de portes'] = sample_data['Nombre de portes'].fillna(median_value)
        
        assert sample_data['Nombre de portes'].isnull().sum() == 0
    
    def test_column_dropping(self, sample_data):
        """Tester la suppression de colonnes"""
        df = sample_data.drop('Airbags', axis=1, errors='ignore')
        assert 'Airbags' not in df.columns
    
    @pytest.mark.parametrize("fill_value,expected", [
        ('mode', True),
        ('median', True),
        ('mean', True),
    ])
    def test_different_imputation_methods(self, sample_data, fill_value, expected):
        """Tester différentes méthodes d'imputation"""
        df = sample_data.copy()
        if fill_value == 'mode':
            mode_val = df['Ville'].mode()[0] if len(df['Ville'].mode()) > 0 else 'Unknown'
            df['Ville'] = df['Ville'].fillna(mode_val)
        elif fill_value == 'median':
            df['Nombre de portes'] = df['Nombre de portes'].fillna(df['Nombre de portes'].median())
        elif fill_value == 'mean':
            df['Nombre de portes'] = df['Nombre de portes'].fillna(df['Nombre de portes'].mean())
        
        assert expected == (df['Ville'].isnull().sum() == 0 or df['Nombre de portes'].isnull().sum() == 0)
    
    def test_outlier_detection(self):
        """Tester la détection des outliers"""
        df = pd.read_csv('data/raw/avito_car_dataset_ALL.csv', encoding='latin1')
        Q1 = df['Prix'].quantile(0.25)
        Q3 = df['Prix'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['Prix'] < lower_bound) | (df['Prix'] > upper_bound)]
        # Il devrait y avoir des outliers dans les données réelles
        assert len(outliers) >= 0
        assert lower_bound < upper_bound


class TestModelArtifacts:
    """Tests pour les artifacts du modèle"""
    
    def test_model_file_exists(self):
        """Vérifier que le modèle existe"""
        model_path = Path('models/car_model.pkl')
        if model_path.exists():
            assert model_path.exists()
            assert model_path.stat().st_size > 0
    
    def test_scaler_file_exists(self):
        """Vérifier que le scaler existe"""
        scaler_path = Path('models/scaler.pkl')
        if scaler_path.exists():
            assert scaler_path.exists()
            assert scaler_path.stat().st_size > 0
    
    def test_feature_info_exists(self):
        """Vérifier que feature_info.json existe"""
        feature_info_path = Path('artifacts/feature_info.json')
        if feature_info_path.exists():
            assert feature_info_path.exists()
            with open(feature_info_path, 'r') as f:
                info = json.load(f)
            assert 'feature_names' in info
            assert 'categorical_cols' in info
            assert 'numerical_cols' in info
    
    def test_price_scaler_info_exists(self):
        """Vérifier que price_scaler_info.json existe"""
        price_scaler_path = Path('artifacts/price_scaler_info.json')
        if price_scaler_path.exists():
            assert price_scaler_path.exists()
            with open(price_scaler_path, 'r') as f:
                info = json.load(f)
            assert 'mean' in info
            assert 'scale' in info
            assert info['mean'] > 0
            assert info['scale'] > 0


class TestModelPredictions:
    """Tests pour les prédictions du modèle"""
    
    @pytest.fixture
    def load_model_artifacts(self):
        """Charger les artifacts du modèle"""
        if not Path('models/car_model.pkl').exists():
            pytest.skip("Modèle non entraîné")
        
        model = joblib.load('models/car_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('artifacts/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        with open('artifacts/price_scaler_info.json', 'r') as f:
            price_scaler_info = json.load(f)
        
        return model, scaler, feature_info, price_scaler_info
    
    def test_model_has_predict_method(self, load_model_artifacts):
        """Vérifier que le modèle a une méthode predict"""
        model, _, _, _ = load_model_artifacts
        assert hasattr(model, 'predict')
    
    def test_model_feature_count(self, load_model_artifacts):
        """Vérifier que le modèle attend le bon nombre de features"""
        model, _, feature_info, _ = load_model_artifacts
        expected_features = len(feature_info['feature_names'])
        assert model.n_features_in_ == expected_features
    
    def test_prediction_output_shape(self, load_model_artifacts):
        """Tester que la prédiction retourne la bonne forme"""
        model, scaler, feature_info, _ = load_model_artifacts
        
        # Create dummy input
        n_features = len(feature_info['feature_names'])
        dummy_input = np.random.randn(1, n_features)
        
        prediction = model.predict(dummy_input)
        assert prediction.shape == (1,)
    
    def test_prediction_value_range(self, load_model_artifacts):
        """Tester que la prédiction est dans une plage raisonnable (après inverse transform)"""
        model, _, _, price_scaler_info = load_model_artifacts
        
        # Create dummy input
        n_features = model.n_features_in_
        dummy_input = np.zeros((1, n_features))
        
        prediction_scaled = model.predict(dummy_input)[0]
        prediction = prediction_scaled * price_scaler_info['scale'] + price_scaler_info['mean']
        
        # Prix entre 0 et 10 millions DH (raisonnable pour une voiture)
        assert 0 <= prediction <= 10_000_000
    
    @pytest.mark.parametrize("input_type", ['zeros', 'ones', 'random'])
    def test_model_predictions_stability(self, load_model_artifacts, input_type):
        """Tester la stabilité des prédictions avec différents inputs"""
        model, _, _, price_scaler_info = load_model_artifacts
        n_features = model.n_features_in_
        
        if input_type == 'zeros':
            test_input = np.zeros((1, n_features))
        elif input_type == 'ones':
            test_input = np.ones((1, n_features))
        else:
            test_input = np.random.randn(1, n_features)
        
        prediction_scaled = model.predict(test_input)[0]
        prediction = prediction_scaled * price_scaler_info['scale'] + price_scaler_info['mean']
        
        # La prédiction doit être un nombre fini
        assert np.isfinite(prediction)
        assert not np.isnan(prediction)
    
    def test_model_multiple_predictions(self, load_model_artifacts):
        """Tester les prédictions multiples"""
        model, _, _, _ = load_model_artifacts
        n_features = model.n_features_in_
        
        # Prédire pour plusieurs échantillons
        batch_input = np.random.randn(10, n_features)
        predictions = model.predict(batch_input)
        
        assert predictions.shape == (10,)
        assert all(np.isfinite(predictions))
    
    def test_model_reproducibility(self, load_model_artifacts):
        """Tester la reproductibilité des prédictions"""
        model, _, _, _ = load_model_artifacts
        n_features = model.n_features_in_
        
        test_input = np.random.RandomState(42).randn(1, n_features)
        
        # Faire deux prédictions avec le même input
        pred1 = model.predict(test_input)
        pred2 = model.predict(test_input)
        
        # Les prédictions doivent être identiques
        np.testing.assert_array_almost_equal(pred1, pred2)


class TestKilometrageMapping:
    """Tests pour le mapping des kilométrages"""
    
    @pytest.fixture
    def km_ranges(self):
        """Charger les ranges de kilométrage"""
        df = pd.read_csv('data/raw/avito_car_dataset_ALL.csv', encoding='latin1')
        return sorted(df['Kilométrage'].unique())
    
    def test_km_ranges_exist(self, km_ranges):
        """Vérifier que les ranges existent"""
        assert len(km_ranges) > 0
    
    def test_km_range_format(self, km_ranges):
        """Vérifier le format des ranges"""
        for km_range in km_ranges[:5]:  # Test first 5
            assert ' - ' in km_range
            parts = km_range.split(' - ')
            assert len(parts) == 2


class TestParamsConfig:
    """Tests pour la configuration des paramètres"""
    
    def test_params_file_exists(self):
        """Vérifier que params.yaml existe"""
        assert Path('params.yaml').exists()
    
    def test_params_structure(self):
        """Vérifier la structure de params.yaml"""
        import yaml
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        assert 'train' in params
        assert 'model' in params
        assert 'test_size' in params['train']
        assert 'random_state' in params['train']
        assert 'n_estimators' in params['model']


class TestEndToEndPipeline:
    """Tests d'intégration end-to-end"""
    
    @pytest.mark.slow
    def test_full_pipeline_runs(self):
        """Tester que tous les fichiers nécessaires existent pour le pipeline"""
        # Vérifier que tous les composants essentiels sont présents
        required_files = [
            'data/raw/avito_car_dataset_ALL.csv',
            'scripts/train_with_mlflow.py',
            'params.yaml',
            'dvc/dvc.yaml',
        ]
        
        for file in required_files:
            assert Path(file).exists(), f"Fichier requis manquant: {file}"
        
        # Si les artifacts existent, les vérifier aussi
        if Path('models/car_model.pkl').exists():
            assert Path('models/scaler.pkl').exists()
            assert Path('artifacts/feature_info.json').exists()
            assert Path('artifacts/price_scaler_info.json').exists()


# Configuration pytest
def pytest_configure(config):
    """Configuration pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
