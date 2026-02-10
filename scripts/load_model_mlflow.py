"""
Load the best model from MLflow Model Registry
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def load_best_model(model_name="CarPricePredictor", stage=None):
    """
    Load the best model from MLflow registry
    
    Args:
        model_name: Name of registered model
        stage: Model stage (None, "Staging", "Production", "Archived")
    
    Returns:
        Loaded model
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlflow/mlruns")
    
    if stage:
        model_uri = f"models:/{model_name}/{stage}"
        print(f"📦 Loading model from stage: {stage}")
    else:
        # Get latest version
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No model versions found for '{model_name}'")
        
        latest_version = max([int(v.version) for v in versions])
        model_uri = f"models:/{model_name}/{latest_version}"
        print(f"📦 Loading latest model version: {latest_version}")
    
    print(f"URI: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    print("✅ Model loaded successfully!")
    
    return model

def get_model_info(model_name="CarPricePredictor"):
    """Get information about all model versions"""
    client = MlflowClient()
    mlflow.set_tracking_uri("file:./mlflow/mlruns")
    
    versions = client.search_model_versions(f"name='{model_name}'")
    
    if not versions:
        print(f"❌ No versions found for model '{model_name}'")
        return
    
    print(f"\n📋 Model: {model_name}")
    print(f"Total versions: {len(versions)}\n")
    
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        print(f"Version {v.version}:")
        print(f"  Stage: {v.current_stage}")
        print(f"  Run ID: {v.run_id}")
        print(f"  Created: {v.creation_timestamp}")
        
        # Get metrics from run
        run = client.get_run(v.run_id)
        metrics = run.data.metrics
        if metrics:
            print(f"  Metrics:")
            if 'test_r2_score' in metrics:
                print(f"    R²: {metrics['test_r2_score']:.4f}")
            if 'test_rmse' in metrics:
                print(f"    RMSE: {metrics['test_rmse']:.4f}")
        print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "info":
        # Show model info
        get_model_info()
    else:
        # Load and test model
        try:
            model = load_best_model()
            print(f"\nModel type: {type(model)}")
            print(f"Model params: {model.get_params()}")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("\n💡 Assurez-vous d'avoir entraîné un modèle avec scripts/train_with_mlflow.py")
