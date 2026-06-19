import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

import config
import data_pipeline
import impact_engine

def train_and_evaluate():
    """
    Runs the model training and evaluation pipeline:
    - Loads and preprocesses data
    - Calculates PII scores for ranking
    - Prepares aggregated training dataset
    - Encodes categorical columns
    - Trains XGBoost and LightGBM models
    - Compares performance and saves the best model
    """
    print("Starting Model Training Pipeline...")
    
    # 1. Load and process historical violations
    df, agg_df = data_pipeline.run_pipeline(config.DATA_PATH, fill_zeros=True)
    
    # 2. Compute PII scores (to save in model artifact for dashboard)
    pii_df = impact_engine.compute_pii_scores(df)
    
    # 3. Create cluster profiles for looking up coordinates and names during inference
    cluster_profiles = data_pipeline.build_cluster_profiles(df)
    
    # 4. Fit Label Encoders
    print("Fitting categorical label encoders...")
    junction_encoder = LabelEncoder()
    # Handle unseen categories by adding a default class in the encoder classes if needed,
    # but since we fit on all possible values from raw and aggregated data, we are safe.
    all_junctions = list(df['junction_name'].unique()) + ['No Junction', 'Unknown']
    junction_encoder.fit(all_junctions)
    
    police_encoder = LabelEncoder()
    all_police = list(df['police_station'].unique()) + ['Unknown']
    police_encoder.fit(all_police)
    
    # Encode categorical columns in aggregated training data
    agg_df['junction_name_encoded'] = junction_encoder.transform(agg_df['junction_name'])
    agg_df['police_station_encoded'] = police_encoder.transform(agg_df['police_station'])
    
    # 5. Define features and target
    feature_cols = [
        'Hour', 'DayOfWeek', 'Month', 'Weekend_Flag', 'Cluster_ID',
        'vehicle_severity', 'violation_severity', 
        'junction_name_encoded', 'police_station_encoded'
    ]
    target_col = 'violation_count'
    
    X = agg_df[feature_cols]
    y = agg_df[target_col]
    
    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # 7. Train XGBoost
    print("Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42, 
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    print(f"XGBoost Evaluation - R^2: {xgb_r2:.4f}, RMSE: {xgb_rmse:.4f}")
    
    # 8. Train LightGBM
    print("Training LightGBM Regressor...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42, 
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    lgb_r2 = r2_score(y_test, lgb_preds)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
    print(f"LightGBM Evaluation - R^2: {lgb_r2:.4f}, RMSE: {lgb_rmse:.4f}")
    
    # 9. Compare and select the best model
    if lgb_r2 > xgb_r2:
        best_model = lgb_model
        best_type = "LightGBM"
        best_r2 = lgb_r2
        best_rmse = lgb_rmse
    else:
        best_model = xgb_model
        best_type = "XGBoost"
        best_r2 = xgb_r2
        best_rmse = xgb_rmse
        
    print(f"Best model selected: {best_type} with R^2={best_r2:.4f} and RMSE={best_rmse:.4f}")
    
    # Save optimized historical coordinate array (only lat, lon) for dashboard heatmap
    heatmap_coords = df[['latitude', 'longitude']].values
    
    # Save model artifacts
    model_artifact = {
        'model': best_model,
        'model_type': best_type,
        'r2': best_r2,
        'rmse': best_rmse,
        'junction_encoder': junction_encoder,
        'police_encoder': police_encoder,
        'cluster_profiles': cluster_profiles,
        'pii_df': pii_df,
        'heatmap_coords': heatmap_coords
    }
    
    print(f"Saving model artifacts to {config.MODEL_PATH}...")
    with open(config.MODEL_PATH, 'wb') as f:
        pickle.dump(model_artifact, f)
        
    print("Model training pipeline completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
