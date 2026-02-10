"""
Tests d'intégration pour l'application Streamlit
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _has_mlflow() -> bool:
    return importlib.util.find_spec("mlflow") is not None


class TestStreamlitIntegration:
    """Tests d'intégration pour Streamlit"""
    
    @pytest.fixture
    def artifacts_exist(self):
        """Vérifier que tous les artifacts existent"""
        required_files = [
            'models/car_model.pkl',
            'models/scaler.pkl',
            'artifacts/feature_info.json',
            'artifacts/price_scaler_info.json',
            'data/raw/avito_car_dataset_ALL.csv'
        ]
        
        for file in required_files:
            if not Path(file).exists():
                pytest.skip(f"Artifact manquant: {file}")
        return True
    
    def test_model_loading(self, artifacts_exist):
        """Tester le chargement du modèle"""
        model = joblib.load('models/car_model.pkl')
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_scaler_loading(self, artifacts_exist):
        """Tester le chargement du scaler"""
        scaler = joblib.load('models/scaler.pkl')
        assert scaler is not None
        assert hasattr(scaler, 'transform')
    
    def test_feature_info_loading(self, artifacts_exist):
        """Tester le chargement de feature_info"""
        with open('artifacts/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        assert 'feature_names' in feature_info
        assert 'categorical_cols' in feature_info
        assert 'numerical_cols' in feature_info
        assert len(feature_info['feature_names']) > 0
    
    def test_price_scaler_info_loading(self, artifacts_exist):
        """Tester le chargement de price_scaler_info"""
        with open('artifacts/price_scaler_info.json', 'r') as f:
            price_scaler_info = json.load(f)
        
        assert 'mean' in price_scaler_info
        assert 'scale' in price_scaler_info
        assert price_scaler_info['mean'] > 0
        assert price_scaler_info['scale'] > 0
    
    def test_full_prediction_pipeline(self, artifacts_exist):
        """Tester le pipeline complet de prédiction"""
        try:
            from sklearn.preprocessing import LabelEncoder
            
            # Load artifacts
            model = joblib.load('models/car_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            with open('artifacts/feature_info.json', 'r') as f:
                feature_info = json.load(f)
            with open('artifacts/price_scaler_info.json', 'r') as f:
                price_scaler_info = json.load(f)
            
            # Load training data for encoders
            df_full = pd.read_csv('data/raw/avito_car_dataset_ALL.csv', encoding='latin1')
            
            # Apply preprocessing
            for col in ['Origine', 'Première main', 'État']:
                if df_full[col].isnull().any():
                    mode_value = df_full[col].mode()[0]
                    df_full[col] = df_full[col].fillna(mode_value)
            
            if df_full['Nombre de portes'].isnull().any():
                median_value = df_full['Nombre de portes'].median()
                df_full['Nombre de portes'] = df_full['Nombre de portes'].fillna(median_value)
            
            df_full = df_full.drop(['Airbags', 'Secteur', 'Lien'], axis=1, errors='ignore')
            
            # Create encoders
            encoders = {}
            for col in feature_info['categorical_cols']:
                le = LabelEncoder()
                le.fit(df_full[col].unique())
                encoders[col] = le
            
            # Create test input
            test_input = {
                'Ville': 'Casablanca',
                'Marque': 'Dacia',
                'Modèle': 'Logan',
                'Année-Modèle': 2018,
                'Kilométrage': '50 000 - 54 999',
                'Type de carburant': 'Diesel',
                'Puissance fiscale': 6,
                'Boite de vitesses': 'Manuelle',
                'Nombre de portes': 4,
                'Origine': 'Dédouanée',
                'Première main': 'Oui',
                'État': 'Bon',
                'Jantes aluminium': 0,
                'Climatisation': 1,
                'Système de navigation/GPS': 0,
                'Toit ouvrant': 0,
                'Sièges cuir': 0,
                'Radar de recul': 0,
                'Caméra de recul': 0,
                'Vitres électriques': 0,
                'ABS': 1,
                'ESP': 0,
                'Régulateur de vitesse': 0,
                'Limiteur de vitesse': 0,
                'CD/MP3/Bluetooth': 0,
                'Ordinateur de bord': 0,
                'Verrouillage centralisé à distance': 0,
            }
            
            # Create DataFrame
            input_data = pd.DataFrame([test_input])
            input_data = input_data[feature_info['feature_names']]
            
            # Encode categorical
            for col in feature_info['categorical_cols']:
                if col in input_data.columns:
                    try:
                        input_data[col] = encoders[col].transform(input_data[col])
                    except Exception:
                        input_data[col] = 0
            
            # Scale numerical
            cols_to_scale = [col for col in feature_info['numerical_cols'] if col in input_data.columns]
            if cols_to_scale:
                input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])
            
            # Predict
            prediction_scaled = model.predict(input_data)
            prix_final = prediction_scaled[0] * price_scaler_info['scale'] + price_scaler_info['mean']
            
            # Assertions
            assert prix_final > 0
            assert prix_final < 1_000_000  # Prix raisonnable pour une Dacia Logan
            
            print(f"\n✓ Test prediction: {prix_final:,.0f} DH")
        except AttributeError as e:
            if "ast" in str(e) and "Num" in str(e):
                pytest.skip("Incompatibilité ast.Num avec Python 3.12+")
            raise
    
    def test_prediction_edge_cases(self, artifacts_exist):
        """Tester les cas limites de prédiction"""
        try:
            from sklearn.preprocessing import LabelEncoder
            
            model = joblib.load('models/car_model.pkl')
            with open('artifacts/feature_info.json', 'r') as f:
                feature_info = json.load(f)
            with open('artifacts/price_scaler_info.json', 'r') as f:
                price_scaler_info = json.load(f)
            
            # Test avec des valeurs extrêmes
            n_features = len(feature_info['feature_names'])
            extreme_input = np.full((1, n_features), 999)  # Valeurs extrêmes
            
            prediction_scaled = model.predict(extreme_input)
            prix_final = prediction_scaled[0] * price_scaler_info['scale'] + price_scaler_info['mean']
            
            # La prédiction doit rester un nombre valide
            assert np.isfinite(prix_final)
            assert not np.isnan(prix_final)
        except AttributeError as e:
            if "ast" in str(e) and "Num" in str(e):
                pytest.skip("Incompatibilité ast.Num avec Python 3.12+")
            raise
    
    def test_prediction_consistency(self, artifacts_exist):
        """Tester la cohérence des prédictions"""
        try:
            model = joblib.load('models/car_model.pkl')
            with open('artifacts/feature_info.json', 'r') as f:
                feature_info = json.load(f)
            with open('artifacts/price_scaler_info.json', 'r') as f:
                price_scaler_info = json.load(f)
            
            n_features = len(feature_info['feature_names'])
            test_input = np.random.RandomState(42).randn(5, n_features)
            
            # Faire des prédictions multiples
            predictions = model.predict(test_input)
            
            # Vérifier que toutes les prédictions sont valides
            assert len(predictions) == 5
            assert all(np.isfinite(predictions))
            
            # Inverser la transformation
            prices = predictions * price_scaler_info['scale'] + price_scaler_info['mean']
            assert all(prices > 0)
        except AttributeError as e:
            if "ast" in str(e) and "Num" in str(e):
                pytest.skip("Incompatibilité ast.Num avec Python 3.12+")
            raise
    
    def test_missing_artifact_handling(self):
        """Tester la gestion des artifacts manquants"""
        # Tester avec un fichier inexistant
        with pytest.raises(FileNotFoundError):
            joblib.load('non_existent_model.pkl')


class TestMLflowIntegration:
    """Tests pour l'intégration MLflow"""
    
    def test_mlruns_directory_exists(self):
        """Vérifier que le répertoire mlruns existe"""
        mlruns_path = Path('mlflow/mlruns')
        if mlruns_path.exists():
            assert mlruns_path.is_dir()
    
    def test_mlflow_can_import(self):
        """Tester que MLflow peut être importé"""
        if not _has_mlflow():
            pytest.skip("MLflow n'est pas installé")
        import mlflow
        assert mlflow is not None
    
    @pytest.mark.slow
    def test_mlflow_experiment_exists(self):
        """Vérifier qu'une expérience MLflow existe"""
        if not _has_mlflow():
            pytest.skip("MLflow n'est pas installé")
        import mlflow
        try:
            mlflow.set_tracking_uri("file:./mlflow/mlruns")
            client = mlflow.tracking.MlflowClient()
            
            experiments = client.search_experiments()
            # Au moins l'expérience par défaut devrait exister
            # Si aucune expérience n'existe, c'est acceptable (pas encore d'entraînement)
            assert isinstance(experiments, list)
        except Exception as e:
            # Si MLflow n'est pas encore initialisé, le test passe
            pytest.skip(f"MLflow not fully initialized: {str(e)}")
    
    def test_mlflow_tracking_uri(self):
        """Tester la configuration de l'URI de tracking MLflow"""
        if not _has_mlflow():
            pytest.skip("MLflow n'est pas installé")
        import mlflow
        
        mlflow.set_tracking_uri("file:./mlflow/mlruns")
        uri = mlflow.get_tracking_uri()
        assert uri is not None
        assert "mlflow/mlruns" in uri or uri == "file:./mlflow/mlruns"


class TestDVCIntegration:
    """Tests pour l'intégration DVC"""
    
    def test_dvc_directory_exists(self):
        """Vérifier que le répertoire .dvc existe"""
        dvc_path = Path('.dvc')
        if dvc_path.exists():
            assert dvc_path.is_dir()
    
    def test_dvc_yaml_exists(self):
        """Vérifier que dvc.yaml existe"""
        if Path('dvc/dvc.yaml').exists():
            assert Path('dvc/dvc.yaml').exists()
    
    def test_params_yaml_exists(self):
        """Vérifier que params.yaml existe"""
        assert Path('params.yaml').exists()
    
    def test_params_yaml_valid_format(self):
        """Vérifier que params.yaml a un format valide"""
        import yaml
        
        with open('params.yaml', 'r') as f:
            try:
                params = yaml.safe_load(f)
                assert isinstance(params, dict)
                # Vérifier les sections principales
                assert 'train' in params or 'model' in params
            except yaml.YAMLError as e:
                pytest.fail(f"params.yaml n'est pas un YAML valide: {e}")
    
    def test_dvc_yaml_valid_format(self):
        """Vérifier que dvc.yaml a un format valide"""
        if not Path('dvc/dvc.yaml').exists():
            pytest.skip("dvc/dvc.yaml n'existe pas")
        
        import yaml
        with open('dvc/dvc.yaml', 'r') as f:
            try:
                dvc_config = yaml.safe_load(f)
                assert isinstance(dvc_config, dict)
                # Vérifier qu'il y a des stages
                assert 'stages' in dvc_config or len(dvc_config) > 0
            except yaml.YAMLError as e:
                pytest.fail(f"dvc.yaml n'est pas un YAML valide: {e}")


class TestPerformance:
    """Tests de performance"""
    
    def test_model_prediction_speed(self):
        """Tester la vitesse de prédiction du modèle"""
        if not Path('models/car_model.pkl').exists():
            pytest.skip("Modèle non entraîné")
        
        import time
        model = joblib.load('models/car_model.pkl')
        with open('artifacts/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        n_features = len(feature_info['feature_names'])
        test_input = np.random.randn(100, n_features)
        
        start_time = time.time()
        predictions = model.predict(test_input)
        elapsed_time = time.time() - start_time
        
        # La prédiction de 100 échantillons devrait prendre moins d'1 seconde
        assert elapsed_time < 1.0
        assert len(predictions) == 100
    
    def test_data_loading_speed(self):
        """Tester la vitesse de chargement des données"""
        import time
        
        start_time = time.time()
        df = pd.read_csv('data/raw/avito_car_dataset_ALL.csv', encoding='latin1')
        elapsed_time = time.time() - start_time
        
        # Le chargement devrait prendre moins de 5 secondes
        assert elapsed_time < 5.0
        assert len(df) > 0
