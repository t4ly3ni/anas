"""
MLOps Training Pipeline with MLflow and DVC
"""
import os
import pandas as pd
import numpy as np
import joblib
import json
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration MLflow
mlflow.set_tracking_uri("file:./mlflow/mlruns")
EXPERIMENT_NAME = "car_price_prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

class CarPricePipeline:
    """Pipeline d'entraînement avec MLflow tracking"""
    
    def __init__(self, params_file='params.yaml'):
        # Load parameters
        with open(params_file, 'r') as f:
            self.params = yaml.safe_load(f)
        
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.target_scaler = None
        
    def load_data(self, filepath='data/raw/avito_car_dataset_ALL.csv'):
        """Chargement des données"""
        print("📂 Chargement des données...")
        df = pd.read_csv(filepath, encoding='latin1')
        print(f"✅ Données chargées: {df.shape}")
        
        # Log data info to MLflow
        mlflow.log_param("data_rows", df.shape[0])
        mlflow.log_param("data_columns", df.shape[1])
        mlflow.log_param("data_file", filepath)
        
        return df
    
    def preprocess_data(self, df):
        """Prétraitement des données"""
        print("🔧 Prétraitement des données...")
        
        # Drop columns that were removed
        cols_to_drop = ['Airbags', 'Secteur', 'Lien', 'Unnamed: 0']
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        # Handle missing values
        for col in ['Origine', 'Première main', 'État']:
            if col in df.columns and df[col].isnull().any():
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                print(f"  • {col}: rempli avec mode = '{mode_value}'")
        
        if 'Nombre de portes' in df.columns and df['Nombre de portes'].isnull().any():
            median_value = df['Nombre de portes'].median()
            df['Nombre de portes'] = df['Nombre de portes'].fillna(median_value)
            print(f"  • Nombre de portes: rempli avec médiane = {median_value}")
        
        # Remove duplicates
        df_before = len(df)
        df = df.drop_duplicates()
        duplicates_removed = df_before - len(df)
        print(f"  • Doublons supprimés: {duplicates_removed}")
        mlflow.log_param("duplicates_removed", duplicates_removed)
        
        # Remove outliers using IQR method
        Q1 = df['Prix'].quantile(0.25)
        Q3 = df['Prix'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_before = len(df)
        df = df[(df['Prix'] >= lower_bound) & (df['Prix'] <= upper_bound)]
        outliers_removed = df_before - len(df)
        
        print(f"  • Outliers supprimés: {outliers_removed}")
        mlflow.log_param("outliers_removed", outliers_removed)
        
        print(f"✅ Shape après prétraitement: {df.shape}")
        return df
    
    def prepare_features(self, df):
        """Préparation des features"""
        print("🎯 Préparation des features...")
        
        # Separate features and target
        X = df.drop('Prix', axis=1)
        y = df['Prix']
        
        feature_names = X.columns.tolist()
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"  • Features totales: {len(feature_names)}")
        print(f"  • Features catégorielles: {len(categorical_cols)}")
        print(f"  • Features numériques: {len(numerical_cols)}")
        
        mlflow.log_param("total_features", len(feature_names))
        mlflow.log_param("categorical_features", len(categorical_cols))
        mlflow.log_param("numerical_features", len(numerical_cols))
        
        # Encode categorical features
        X_encoded = X.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            self.encoders[col] = le
        
        # Save encoders
        joblib.dump(self.encoders, 'models/encoders.pkl')
        mlflow.log_artifact('models/encoders.pkl')
        
        # Scale numerical features only
        self.scaler = StandardScaler()
        X_scaled = X_encoded.copy()
        if numerical_cols:
            X_scaled[numerical_cols] = self.scaler.fit_transform(X_encoded[numerical_cols])
        
        # Scale target variable
        self.target_scaler = StandardScaler()
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        # Save feature info
        feature_info = {
            'feature_names': feature_names,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols
        }
        with open('artifacts/feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=4)
        mlflow.log_artifact('artifacts/feature_info.json')
        
        # Save price scaler info
        price_scaler_info = {
            'mean': float(self.target_scaler.mean_[0]),
            'scale': float(self.target_scaler.scale_[0])
        }
        with open('artifacts/price_scaler_info.json', 'w') as f:
            json.dump(price_scaler_info, f, indent=4)
        mlflow.log_artifact('artifacts/price_scaler_info.json')
        
        return X_scaled, y_scaled, feature_names
    
    def train_model(self, X, y):
        """Entraînement du modèle"""
        print("🤖 Entraînement du modèle...")
        
        # Get parameters
        test_size = self.params['train']['test_size']
        random_state = self.params['train']['random_state']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"  • Training set: {X_train.shape[0]} samples")
        print(f"  • Test set: {X_test.shape[0]} samples")
        
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        
        # Get model hyperparameters
        hyperparams = self.params['model'].copy()
        hyperparams['random_state'] = random_state
        hyperparams['n_jobs'] = -1
        
        # Log hyperparameters
        for param, value in hyperparams.items():
            mlflow.log_param(f"model_{param}", value)
        
        # Train model
        self.model = RandomForestRegressor(**hyperparams)
        self.model.fit(X_train, y_train)
        
        print("✅ Entraînement terminé!")
        
        # Evaluate on training set
        y_train_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        # Evaluate on test set
        y_test_pred = self.model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Log metrics
        mlflow.log_metric("train_r2_score", train_r2)
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_mae", train_mae)
        
        mlflow.log_metric("test_r2_score", test_r2)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        
        print("\n📊 Résultats:")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test MAE:  {test_mae:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save and log feature importance
        feature_importance.to_csv('artifacts/feature_importance.csv', index=False)
        mlflow.log_artifact('artifacts/feature_importance.csv')
        
        print("\n🔝 Top 10 Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance (top 15)
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(15)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig('artifacts/feature_importance.png', dpi=100, bbox_inches='tight')
        mlflow.log_artifact('artifacts/feature_importance.png')
        plt.close()
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual (scaled)')
        plt.ylabel('Predicted (scaled)')
        plt.title(f'Predictions vs Actual (Test Set)\nR² = {test_r2:.4f}')
        plt.tight_layout()
        plt.savefig('artifacts/predictions_plot.png', dpi=100, bbox_inches='tight')
        mlflow.log_artifact('artifacts/predictions_plot.png')
        plt.close()
        
        # Residuals plot
        residuals = y_test - y_test_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.tight_layout()
        plt.savefig('artifacts/residuals_plot.png', dpi=100, bbox_inches='tight')
        mlflow.log_artifact('artifacts/residuals_plot.png')
        plt.close()
        
        return X_train, X_test, y_train, y_test
    
    def save_model(self):
        """Sauvegarde du modèle"""
        print("💾 Sauvegarde du modèle...")
        
        # Save with joblib (for Streamlit)
        joblib.dump(self.model, 'models/car_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Log artifacts to MLflow
        mlflow.log_artifact('models/car_model.pkl')
        mlflow.log_artifact('models/scaler.pkl')
        
        # Log model to MLflow (for model registry)
        mlflow.sklearn.log_model(
            self.model,
            name="model",
            registered_model_name="CarPricePredictor",
            serialization_format="skops"
        )
        
        print("✅ Modèle sauvegardé!")
    
    def run(self):
        """Exécuter le pipeline complet"""
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log system info
            mlflow.set_tag("model_type", "RandomForestRegressor")
            mlflow.set_tag("purpose", "car_price_prediction")
            mlflow.set_tag("framework", "scikit-learn")
            mlflow.set_tag("mlops_tools", "MLflow + DVC")
            
            # Pipeline steps
            df = self.load_data()
            df = self.preprocess_data(df)
            X, y, feature_names = self.prepare_features(df)
            self.train_model(X, y)
            self.save_model()
            
            print("\n" + "="*50)
            print("✅ Pipeline terminé avec succès!")
            print("="*50)
            print(f"\n🔗 MLflow UI: mlflow ui")
            print(f"📂 Artifacts: ./mlflow/mlruns")
            print(f"\n💡 Pour voir les résultats:")
            print(f"   mlflow ui")
            print(f"   Puis ouvrir: http://localhost:5000")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Car Price Prediction - MLOps Pipeline")
    print("   MLflow + DVC + Scikit-learn")
    print("="*50 + "\n")
    
    pipeline = CarPricePipeline()
    pipeline.run()
