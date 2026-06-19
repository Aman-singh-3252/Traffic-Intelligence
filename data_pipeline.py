import pandas as pd
import numpy as np
import json
from sklearn.cluster import DBSCAN
import config

def load_clean_data(file_path=config.DATA_PATH):
    """
    Ingests violations dataset and applies cleaning rules:
    - Drop rows with invalid or missing latitude/longitude
    - Drop duplicate records
    - Drop rows with missing timestamps
    - Strictly retain only records where validation_status is 'Approved' (case-insensitive)
    """
    print(f"Loading data from {file_path}...")
    try:
        # Load only necessary columns to save memory and speed up processing
        cols = [
            'id', 'latitude', 'longitude', 'location', 'junction_name', 
            'center_code', 'police_station', 'violation_type', 
            'offence_code', 'validation_status', 'vehicle_type', 
            'created_datetime'
        ]
        df = pd.read_csv(file_path, usecols=cols)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        # Try loading without column filter if there is a mismatch
        df = pd.read_csv(file_path)
    
    initial_len = len(df)
    
    # Drop rows with invalid or missing latitude/longitude
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Drop rows with missing timestamps
    df = df.dropna(subset=['created_datetime'])
    
    # Retain only Approved validation_status (case-insensitive)
    df = df.dropna(subset=['validation_status'])
    df = df[df['validation_status'].str.lower() == 'approved']
    
    # Drop duplicate records
    df = df.drop_duplicates()
    
    print(f"Cleaned data: Reduced from {initial_len} to {len(df)} records.")
    return df

def engineer_temporal_features(df):
    """
    Parses created_datetime and extracts Hour, DayOfWeek, Month, and Weekend_Flag.
    """
    print("Engineering temporal features...")
    df['created_datetime'] = pd.to_datetime(df['created_datetime'])
    
    df['Hour'] = df['created_datetime'].dt.hour
    df['DayOfWeek'] = df['created_datetime'].dt.dayofweek
    df['Month'] = df['created_datetime'].dt.month
    df['Weekend_Flag'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    return df

def apply_dbscan(df):
    """
    Applies DBSCAN clustering using latitude and longitude.
    Assigns Cluster_ID to each violation.
    """
    print("Applying DBSCAN clustering...")
    # Convert lat/long to radians for haversine distance
    coords = np.radians(df[['latitude', 'longitude']].values)
    
    db = DBSCAN(
        eps=config.DBSCAN_EPS, 
        min_samples=config.DBSCAN_MIN_SAMPLES, 
        metric='haversine', 
        n_jobs=-1
    )
    df['Cluster_ID'] = db.fit_predict(coords)
    
    n_clusters = len(set(df['Cluster_ID'])) - (1 if -1 in df['Cluster_ID'].values else 0)
    n_noise = (df['Cluster_ID'] == -1).sum()
    print(f"DBSCAN complete. Found {n_clusters} clusters. Noise points: {n_noise}.")
    
    return df

def _parse_violation_severity(violation_str):
    """
    Parses violation_type JSON list and returns the maximum severity weight.
    """
    if not isinstance(violation_str, str):
        return 1.0
    
    try:
        v_list = json.loads(violation_str)
        if not isinstance(v_list, list):
            v_list = [v_list]
    except Exception:
        # If not JSON, treat it as a single string
        v_list = [violation_str]
        
    weights = []
    for v in v_list:
        if isinstance(v, str):
            clean_v = v.strip().upper()
            weight = config.VIOLATION_SEVERITY_MAPPING.get(clean_v, 1.0)
            weights.append(weight)
        else:
            weights.append(1.0)
            
    return float(max(weights)) if weights else 1.0

def encode_severity(df):
    """
    Encodes vehicle_type and violation_type into numeric weights.
    """
    print("Encoding severity weights...")
    # Map vehicle type
    df['vehicle_severity'] = df['vehicle_type'].str.upper().map(config.VEHICLE_SEVERITY_MAPPING).fillna(3.0)
    
    # Parse and map violation type
    df['violation_severity'] = df['violation_type'].apply(_parse_violation_severity)
    
    return df

def build_cluster_profiles(df):
    """
    Builds a profile/metadata dataframe for each non-noise cluster.
    Includes: centroid latitude/longitude, mode of junction_name/police_station, 
    and average historical severities.
    """
    non_noise = df[df['Cluster_ID'] != -1]
    
    def get_mode(x):
        modes = x.mode()
        return modes.iloc[0] if not modes.empty else 'Unknown'

    profiles = non_noise.groupby('Cluster_ID').agg(
        lat_center=('latitude', 'mean'),
        lon_center=('longitude', 'mean'),
        junction_name=('junction_name', get_mode),
        police_station=('police_station', get_mode),
        avg_vehicle_severity=('vehicle_severity', 'mean'),
        avg_violation_severity=('violation_severity', 'mean'),
        total_violations=('id', 'count')
    ).reset_index()
    
    return profiles

def aggregate_data(df, fill_zeros=True):
    """
    Groups the data by Cluster_ID, Hour, DayOfWeek, and Month.
    Calculates: Violation Count (Target), Average Vehicle Severity, and Average Violation Severity.
    """
    print("Aggregating data by cluster and time features...")
    # We train models only on non-noise points
    df_train = df[df['Cluster_ID'] != -1].copy()
    
    # Group the active records
    grouped = df_train.groupby(['Cluster_ID', 'Hour', 'DayOfWeek', 'Month']).agg(
        violation_count=('id', 'count'),
        vehicle_severity=('vehicle_severity', 'mean'),
        violation_severity=('violation_severity', 'mean')
    ).reset_index()
    
    if not fill_zeros:
        # Get cluster profiles to join junction and police station
        profiles = build_cluster_profiles(df)
        agg_df = pd.merge(grouped, profiles[['Cluster_ID', 'junction_name', 'police_station']], on='Cluster_ID')
        agg_df['Weekend_Flag'] = agg_df['DayOfWeek'].isin([5, 6]).astype(int)
        return agg_df
        
    # Cartesian product reindexing to fill zeros
    print("Reindexing to fill zero-violation periods (creating full grid)...")
    all_clusters = sorted(df_train['Cluster_ID'].unique())
    all_hours = list(range(24))
    all_days = list(range(7))
    all_months = sorted(df_train['Month'].unique())
    
    full_index = pd.MultiIndex.from_product(
        [all_clusters, all_hours, all_days, all_months],
        names=['Cluster_ID', 'Hour', 'DayOfWeek', 'Month']
    )
    
    grouped = grouped.set_index(['Cluster_ID', 'Hour', 'DayOfWeek', 'Month'])
    grouped = grouped.reindex(full_index).reset_index()
    
    # Fill violation count with 0
    grouped['violation_count'] = grouped['violation_count'].fillna(0).astype(int)
    
    # Join cluster profiles to fill severities and categorical features
    profiles = build_cluster_profiles(df)
    agg_df = pd.merge(grouped, profiles, on='Cluster_ID', how='left')
    
    # For severity values, if count is > 0 we keep the group average, else use the cluster average
    agg_df['vehicle_severity'] = agg_df['vehicle_severity'].fillna(agg_df['avg_vehicle_severity'])
    agg_df['violation_severity'] = agg_df['violation_severity'].fillna(agg_df['avg_violation_severity'])
    
    # Drop temporary average columns
    agg_df = agg_df.drop(columns=['avg_vehicle_severity', 'avg_violation_severity'])
    
    agg_df['Weekend_Flag'] = agg_df['DayOfWeek'].isin([5, 6]).astype(int)
    
    print(f"Aggregated training dataset shape: {agg_df.shape}")
    return agg_df

def run_pipeline(file_path=config.DATA_PATH, fill_zeros=True):
    """
    Executes the entire preprocessing pipeline.
    """
    df = load_clean_data(file_path)
    df = engineer_temporal_features(df)
    df = apply_dbscan(df)
    df = encode_severity(df)
    
    agg_df = aggregate_data(df, fill_zeros=fill_zeros)
    return df, agg_df
