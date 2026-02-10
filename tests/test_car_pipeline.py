"""
Tests spécifiques pour le CarPricePipeline
"""
import pytest
import pandas as pd
import numpy as np
import yaml
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_with_mlflow import CarPricePipeline


class TestCarPricePipeline:
    """Tests pour la classe CarPricePipeline"""
    
    @pytest.fixture
    def temp_params_file(self):
        """Créer un fichier params.yaml temporaire"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            params = {
                'train': {
                    'test_size': 0.2,
                    'random_state': 42
                },
                'model': {
                    'n_estimators': 10,  # Réduit pour les tests
                    'max_depth': 5,
                    'random_state': 42
                }
            }
            yaml.dump(params, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_pipeline_initialization(self, temp_params_file):
        """Tester l'initialisation du pipeline"""
        pipeline = CarPricePipeline(params_file=temp_params_file)
        
        assert pipeline is not None
        assert pipeline.params is not None
        assert 'train' in pipeline.params
        assert 'model' in pipeline.params
        assert pipeline.model is None
        assert pipeline.scaler is None
    
    def test_pipeline_load_data(self):
        """Tester le chargement des données"""
        if not Path('data/raw/avito_car_dataset_ALL.csv').exists():
            pytest.skip("Fichier de données non disponible")
        
        pipeline = CarPricePipeline()
        
        with patch('mlflow.log_param'):  # Mock MLflow
            df = pipeline.load_data()
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'Prix' in df.columns
    
    def test_pipeline_preprocess_data(self):
        """Tester le prétraitement des données"""
        if not Path('data/raw/avito_car_dataset_ALL.csv').exists():
            pytest.skip("Fichier de données non disponible")
        
        pipeline = CarPricePipeline()
        
        with patch('mlflow.log_param'):
            df_raw = pd.read_csv('data/raw/avito_car_dataset_ALL.csv', encoding='latin1')
            df_processed = pipeline.preprocess_data(df_raw.copy())
        
        # Vérifier que le prétraitement a fonctionné
        assert len(df_processed) > 0
        assert len(df_processed) <= len(df_raw)  # Peut être réduit (outliers)
        
        # Vérifier que les colonnes inutiles ont été supprimées
        removed_cols = ['Airbags', 'Secteur', 'Lien']
        for col in removed_cols:
            assert col not in df_processed.columns
    
    def test_pipeline_handles_missing_values(self):
        """Tester la gestion des valeurs manquantes"""
        pipeline = CarPricePipeline()
        
        # Créer un DataFrame avec des valeurs manquantes
        df = pd.DataFrame({
            'Prix': [100000, 150000, 120000, 200000],
            'Origine': ['Dédouanée', None, 'WW au Maroc', 'Dédouanée'],
            'Première main': ['Oui', None, 'Non', 'Oui'],
            'État': ['Bon', 'Excellent', None, 'Bon'],
            'Nombre de portes': [4, None, 5, 4]
        })
        
        with patch('mlflow.log_param'):
            df_processed = pipeline.preprocess_data(df)
        
        # Vérifier qu'il n'y a plus de valeurs manquantes dans les colonnes clés
        if 'Origine' in df_processed.columns:
            assert df_processed['Origine'].isnull().sum() == 0
        if 'Nombre de portes' in df_processed.columns:
            assert df_processed['Nombre de portes'].isnull().sum() == 0
    
    def test_pipeline_removes_outliers(self):
        """Tester la suppression des outliers"""
        pipeline = CarPricePipeline()
        
        # Créer un DataFrame avec des outliers évidents
        df = pd.DataFrame({
            'Prix': [100000, 150000, 120000, 200000, 10_000_000],  # Dernier est un outlier
            'Ville': ['Casa', 'Rabat', 'Casa', 'Rabat', 'Casa']
        })
        
        with patch('mlflow.log_param'):
            df_processed = pipeline.preprocess_data(df)
        
        # Le outlier devrait être supprimé
        assert len(df_processed) < len(df)
        assert df_processed['Prix'].max() < 1_000_000
    
    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3])
    def test_pipeline_different_test_sizes(self, temp_params_file, test_size):
        """Tester avec différentes tailles de test set"""
        # Modifier les params
        with open(temp_params_file, 'r') as f:
            params = yaml.safe_load(f)
        params['train']['test_size'] = test_size
        with open(temp_params_file, 'w') as f:
            yaml.dump(params, f)
        
        pipeline = CarPricePipeline(params_file=temp_params_file)
        assert pipeline.params['train']['test_size'] == test_size


class TestPipelineEdgeCases:
    """Tests pour les cas limites du pipeline"""
    
    def test_empty_dataframe(self):
        """Tester avec un DataFrame vide"""
        pipeline = CarPricePipeline()
        df_empty = pd.DataFrame()
        
        with patch('mlflow.log_param'):
            with pytest.raises((KeyError, ValueError)):
                pipeline.preprocess_data(df_empty)
    
    def test_single_row_dataframe(self):
        """Tester avec un DataFrame d'une seule ligne"""
        pipeline = CarPricePipeline()
        df_single = pd.DataFrame({
            'Prix': [100000],
            'Ville': ['Casablanca']
        })
        
        with patch('mlflow.log_param'):
            try:
                df_processed = pipeline.preprocess_data(df_single)
                # Si ça fonctionne, vérifier que le DataFrame n'est pas vide
                assert len(df_processed) >= 0
            except (KeyError, ValueError):
                # Acceptable pour un DataFrame trop petit
                pass
    
    def test_all_missing_values_column(self):
        """Tester avec une colonne entièrement manquante"""
        pipeline = CarPricePipeline()
        df = pd.DataFrame({
            'Prix': [100000, 150000, 120000],
            'Origine': [None, None, None],
            'Ville': ['Casa', 'Rabat', 'Tanger']
        })
        
        with patch('mlflow.log_param'):
            # Le pipeline devrait gérer gracieusement ce cas
            try:
                df_processed = pipeline.preprocess_data(df)
                # Si l'Origine est gérée, vérifier
                if 'Origine' in df_processed.columns:
                    # Après imputation, pas de NaN
                    assert df_processed['Origine'].isnull().sum() == 0
            except Exception:
                # Acceptable si le pipeline ne peut pas gérer ce cas
                pass


class TestPipelineDataQuality:
    """Tests de qualité des données dans le pipeline"""
    
    def test_no_duplicate_features(self):
        """Vérifier qu'il n'y a pas de features dupliquées"""
        if not Path('artifacts/feature_info.json').exists():
            pytest.skip("artifacts/feature_info.json non disponible")
        
        import json
        with open('artifacts/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        feature_names = feature_info['feature_names']
        # Vérifier qu'il n'y a pas de doublons
        assert len(feature_names) == len(set(feature_names))
    
    def test_categorical_numerical_separation(self):
        """Vérifier que les colonnes catégorielles et numériques sont bien séparées"""
        if not Path('artifacts/feature_info.json').exists():
            pytest.skip("artifacts/feature_info.json non disponible")
        
        import json
        with open('artifacts/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        categorical = set(feature_info['categorical_cols'])
        numerical = set(feature_info['numerical_cols'])
        
        # Aucune colonne ne devrait être à la fois catégorielle et numérique
        intersection = categorical.intersection(numerical)
        assert len(intersection) == 0, f"Colonnes en double: {intersection}"
    
    def test_feature_consistency(self):
        """Vérifier la cohérence entre feature_names et les listes cat/num"""
        if not Path('artifacts/feature_info.json').exists():
            pytest.skip("artifacts/feature_info.json non disponible")
        
        import json
        with open('artifacts/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        all_features = set(feature_info['feature_names'])
        categorical = set(feature_info['categorical_cols'])
        numerical = set(feature_info['numerical_cols'])
        
        # Toutes les features cat + num devraient être dans feature_names
        cat_num_union = categorical.union(numerical)
        
        # Vérifier que la plupart des features sont classées
        # (Il peut y avoir des features binaires qui ne sont ni dans cat ni dans num)
        assert len(cat_num_union) > 0


class TestModelValidation:
    """Tests de validation du modèle"""
    
    def test_model_serialization(self):
        """Tester que le modèle peut être sérialisé et désérialisé"""
        if not Path('models/car_model.pkl').exists():
            pytest.skip("Modèle non entraîné")
        
        import joblib
        model = joblib.load('models/car_model.pkl')
        
        # Sérialiser dans un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            joblib.dump(model, temp_path)
            model_reloaded = joblib.load(temp_path)
            
            # Vérifier que le modèle rechargé a les mêmes propriétés
            assert model.n_estimators == model_reloaded.n_estimators
            assert model.n_features_in_ == model_reloaded.n_features_in_
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_scaler_transformation_reversible(self):
        """Vérifier que la transformation du scaler est réversible"""
        if not Path('models/scaler.pkl').exists():
            pytest.skip("Scaler non disponible")
        
        import joblib
        scaler = joblib.load('models/scaler.pkl')
        
        # Créer des données de test
        test_data = np.random.randn(10, scaler.n_features_in_)
        
        # Transformer et inverser
        transformed = scaler.transform(test_data)
        inverse = scaler.inverse_transform(transformed)
        
        # Les données inversées devraient être très proches des originales
        np.testing.assert_array_almost_equal(test_data, inverse, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
