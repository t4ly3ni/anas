# Vehicle Price Prediction App with MLflow Integration
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder

# Configuration de la page
st.set_page_config(page_title="Predicteur de Prix Avito (MLflow)", layout="wide")

st.title(" Estimation du prix de voiture (Maroc)")
# Sidebar pour configuration MLflow
with st.sidebar:
    st.header(" Configuration MLOps")
    
    use_mlflow = st.checkbox(" Utiliser MLflow Model Registry", value=False, 
                             help="Charger le modèle depuis MLflow au lieu des fichiers locaux")
    
    if use_mlflow:
        model_stage = st.selectbox(
            "Stage du modèle",
            ["None", "Staging", "Production"],
            index=0,
            help="None = dernière version, Staging = test, Production = déployé"
        )
        
        if st.button(" Recharger le modèle"):
            st.cache_resource.clear()
            st.success("Cache cleared! Le modèle sera rechargé.")
    
    st.divider()

# 1. Chargement des ressources
@st.cache_resource
def load_assets(use_mlflow_registry=False, stage="None"):
    """Load model and preprocessing artifacts"""
    
    if use_mlflow_registry:
        # Load from MLflow
        with st.spinner(" Chargement depuis MLflow Model Registry..."):
            mlflow.set_tracking_uri("file:./mlflow/mlruns")
            
            model_name = "CarPricePredictor"
            stage_param = None if stage == "None" else stage
            
            try:
                client = MlflowClient()
                versions = client.search_model_versions(f"name='{model_name}'")
                if not versions:
                    st.error(" Aucun modèle trouvé dans MLflow!")
                    st.info("💡 Entraînez d'abord un modèle avec: python scripts/train_with_mlflow.py")
                    st.stop()
                
                model_uri = None
                # Try to find model at requested stage
                if stage_param:
                    stage_versions = [v for v in versions if v.current_stage == stage_param]
                    if stage_versions:
                        latest_stage_version = max([int(v.version) for v in stage_versions])
                        model_uri = f"models:/{model_name}/{stage_param}"
                        st.info(f"✓ Modèle trouvé au stage {stage_param} (v{latest_stage_version})")
                    else:
                        # Fallback to latest version if stage not found
                        st.warning(f"⚠ Pas de modèle au stage '{stage_param}'. Utilisation de la dernière version...")
                        latest_version = max([int(v.version) for v in versions])
                        model_uri = f"models:/{model_name}/{latest_version}"
                else:
                    # Use latest version
                    latest_version = max([int(v.version) for v in versions])
                    model_uri = f"models:/{model_name}/{latest_version}"
                
                model = mlflow.sklearn.load_model(model_uri)
                st.success(f" Modèle chargé depuis: {model_uri}")
            except Exception as e:
                st.error(f" Erreur lors du chargement MLflow: {str(e)}")
                st.info("💡 Utilisez les fichiers locaux à la place.")
                st.stop()
    else:
        # Load from local files
        try:
            model = joblib.load('models/car_model.pkl')
        except FileNotFoundError:
            st.error(" Fichier models/car_model.pkl introuvable!")
            st.info("💡 Entraînez d'abord le modèle avec: python scripts/train_with_mlflow.py")
            st.stop()
    
    # Load other artifacts
    scaler = joblib.load('models/scaler.pkl')
    
    with open('artifacts/feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    with open('artifacts/price_scaler_info.json', 'r') as f:
        price_scaler_info = json.load(f)
    
    # Load training data for encoders
    df_full = pd.read_csv('data/raw/avito_car_dataset_ALL.csv', encoding='latin1')
    
    for col in ['Origine', 'Première main', 'État']:
        if df_full[col].isnull().any():
            mode_value = df_full[col].mode()[0]
            df_full[col] = df_full[col].fillna(mode_value)
    
    if df_full['Nombre de portes'].isnull().any():
        median_value = df_full['Nombre de portes'].median()
        df_full['Nombre de portes'] = df_full['Nombre de portes'].fillna(median_value)
    
    df_full = df_full.drop(['Airbags', 'Secteur', 'Lien'], axis=1, errors='ignore')
    
    encoders = {}
    categorical_cols = feature_info['categorical_cols']
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df_full[col].unique())
        encoders[col] = le
    
    km_ranges = sorted(df_full['Kilométrage'].unique())
    
    return model, scaler, feature_info, encoders, km_ranges, price_scaler_info

# Load based on user selection
assets = load_assets(use_mlflow_registry=use_mlflow, stage=model_stage if use_mlflow else "None")
model, scaler, feature_info, encoders, km_ranges, price_scaler_info = assets

# Helper function
def km_to_range(km_value):
    """Convert numeric km value to range string"""
    for km_range in km_ranges:
        parts = km_range.split(' - ')
        if len(parts) == 2:
            low = int(parts[0].replace(' ', ''))
            high = int(parts[1].replace(' ', ''))
            if low <= km_value <= high:
                return km_range
    
    ranges_with_midpoints = []
    for km_range in km_ranges:
        parts = km_range.split(' - ')
        if len(parts) == 2:
            low = int(parts[0].replace(' ', ''))
            high = int(parts[1].replace(' ', ''))
            midpoint = (low + high) / 2
            ranges_with_midpoints.append((km_range, midpoint))
    
    if ranges_with_midpoints:
        closest = min(ranges_with_midpoints, key=lambda x: abs(x[1] - km_value))
        return closest[0]
    
    return km_ranges[0]

# 2. Interface utilisateur
with st.sidebar:
    st.header(" Caractéristiques du véhicule")
    
    villes_uniques = ['Casablanca', 'Fès', 'Marrakech', 'Rabat', 'Tanger', 'Salé', 'Agadir', 'Temara', 'Meknès', 'El Jadida']
    ville = st.selectbox("Ville", villes_uniques)
    
    marques_top = ['Dacia', 'Renault', 'Peugeot', 'Volkswagen', 'Ford', 'Toyota', 'Hyundai', 'Fiat', 'BMW', 'Mercedes-Benz']
    marque = st.selectbox("Marque", marques_top)
    
    modele = st.text_input("Modèle", "Logan")
    annee_modele = st.number_input("Année Modèle", min_value=1990, max_value=2026, value=2018)
    kilometrage = st.slider("Kilométrage", 0, 300000, 50000)
    
    carburants = ["Diesel", "Essence", "Hybride", "Electrique"]
    carburant = st.radio("Carburant", carburants)
    
    puissance_fiscale = st.number_input("Puissance Fiscale", 3, 30, 6)
    
    boites = ["Manuelle", "Automatique"]
    boite_vitesses = st.selectbox("Boite de vitesses", boites)
    
    nb_portes = st.number_input("Nombre de portes", 2, 5, 4)
    
    origines = ['WW au Maroc', 'Dédouanée', 'Importée neuve', 'Pas encore dédouanée']
    origine = st.selectbox("Origine", origines)
    
    premieres_main = ["Oui", "Non"]
    premiere_main = st.radio("Première main", premieres_main)
    
    etats = ["Très bon", "Excellent", "Bon", "Correct", "Pour Pièces", "Endommagé"]
    etat = st.selectbox("État", etats)
    
    st.subheader("🎁 Équipements")
    jantes_alu = st.checkbox("Jantes aluminium")
    climatisation = st.checkbox("Climatisation")
    gps = st.checkbox("Système de navigation/GPS")
    toit_ouvrant = st.checkbox("Toit ouvrant")
    sieges_cuir = st.checkbox("Sièges cuir")
    radar_recul = st.checkbox("Radar de recul")
    camera_recul = st.checkbox("Caméra de recul")
    vitres_electriques = st.checkbox("Vitres électriques")
    abs_active = st.checkbox("ABS")
    esp = st.checkbox("ESP")
    regulateur_vitesse = st.checkbox("Régulateur de vitesse")
    limiteur_vitesse = st.checkbox("Limiteur de vitesse")
    cd_mp3 = st.checkbox("CD/MP3/Bluetooth")
    ordinateur_bord = st.checkbox("Ordinateur de bord")
    verrouillage_central = st.checkbox("Verrouillage centralisé à distance")

# 3. Prédiction
if st.button("💰 Estimer le prix", use_container_width=True):
    try:
        input_data = pd.DataFrame({
            'Ville': [ville],
            'Marque': [marque],
            'Modèle': [modele],
            'Année-Modèle': [annee_modele],
            'Kilométrage': [kilometrage],
            'Type de carburant': [carburant],
            'Puissance fiscale': [puissance_fiscale],
            'Boite de vitesses': [boite_vitesses],
            'Nombre de portes': [nb_portes],
            'Origine': [origine],
            'Première main': [premiere_main],
            'État': [etat],
            'Jantes aluminium': [1 if jantes_alu else 0],
            'Climatisation': [1 if climatisation else 0],
            'Système de navigation/GPS': [1 if gps else 0],
            'Toit ouvrant': [1 if toit_ouvrant else 0],
            'Sièges cuir': [1 if sieges_cuir else 0],
            'Radar de recul': [1 if radar_recul else 0],
            'Caméra de recul': [1 if camera_recul else 0],
            'Vitres électriques': [1 if vitres_electriques else 0],
            'ABS': [1 if abs_active else 0],
            'ESP': [1 if esp else 0],
            'Régulateur de vitesse': [1 if regulateur_vitesse else 0],
            'Limiteur de vitesse': [1 if limiteur_vitesse else 0],
            'CD/MP3/Bluetooth': [1 if cd_mp3 else 0],
            'Ordinateur de bord': [1 if ordinateur_bord else 0],
            'Verrouillage centralisé à distance': [1 if verrouillage_central else 0],
        })
        
        input_data = input_data[feature_info['feature_names']]
        
        if 'Kilométrage' in input_data.columns:
            km_value = input_data['Kilométrage'].values[0]
            input_data['Kilométrage'] = [km_to_range(km_value)]
        
        categorical_cols = feature_info['categorical_cols']
        warnings = []
        
        for col in categorical_cols:
            if col in input_data.columns:
                le = encoders[col]
                try:
                    input_data[col] = le.transform(input_data[col])
                except ValueError:
                    val_str = str(input_data[col].values[0])
                    if len(val_str) > 30:
                        val_str = val_str[:30] + "..."
                    warnings.append(f" Valeur inconnue: '{val_str}' pour {col}")
                    input_data[col] = 0
        
        numerical_cols = feature_info['numerical_cols']
        cols_to_scale = [col for col in numerical_cols if col in input_data.columns]
        
        if cols_to_scale:
            input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])
        
        prediction_scaled = model.predict(input_data)
        prix_final = prediction_scaled[0] * price_scaler_info['scale'] + price_scaler_info['mean']
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("Prédiction réussie!")
        with col2:
            st.metric("Prix estimé", f"{prix_final:,.0f} DH")
        
        if warnings:
            with st.expander("Avertissements"):
                st.write("Les valeurs suivantes n'ont pas été vues pendant l'entraînement:")
                for w in warnings:
                    st.write(f"• {w}")
        
        with st.expander("📊 Détails du véhicule"):
            info_cols = st.columns(3)
            with info_cols[0]:
                st.write(f"**Marque:** {marque}")
                st.write(f"**Modèle:** {modele}")
                st.write(f"**Année:** {annee_modele}")
            with info_cols[1]:
                st.write(f"**Kilométrage:** {kilometrage:,} km")
                st.write(f"**Puissance:** {puissance_fiscale} CV")
                st.write(f"**Carburant:** {carburant}")
            with info_cols[2]:
                st.write(f"**Boîte:** {boite_vitesses}")
                st.write(f"**État:** {etat}")
                st.write(f"**Portes:** {nb_portes}")
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        with st.expander("Détails de l'erreur"):
            st.error(f"Type: {type(e).__name__}")
            import traceback
            st.code(traceback.format_exc())

# Footer with MLflow info
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(" Framework: Scikit-learn")
with col2:
    st.caption("Tracking: MLflow")
with col3:
    st.caption("Version Control: DVC")
