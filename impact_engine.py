import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, QhullError
import config

def calculate_cluster_area(coords):
    """
    Computes the area of a cluster (in square meters) using Convex Hull.
    Applies spatial projection to convert degrees (lat, lon) to meters.
    """
    if len(coords) < 3 or len(np.unique(coords, axis=0)) < 3:
        return 1.0  # minimal area fallback
    
    # Extract coordinates
    lat = coords[:, 0]
    lon = coords[:, 1]
    
    # Local Mercator projection in meters
    lat_mean = np.mean(lat)
    lat_rad = np.radians(lat_mean)
    
    # 1 degree lat = approx 110,574 meters
    # 1 degree lon = approx 111,320 * cos(lat) meters
    x = lon * 111320.0 * np.cos(lat_rad)
    y = lat * 110574.0
    
    projected = np.column_stack((x, y))
    
    try:
        hull = ConvexHull(projected)
        # For 2D, volume represents the area of the convex hull
        return max(hull.volume, 1.0)
    except (QhullError, ValueError):
        # Fallback to bounding box area if ConvexHull fails
        dx = np.max(x) - np.min(x)
        dy = np.max(y) - np.min(y)
        area = dx * dy
        return max(area, 1.0)

def compute_pii_scores(df):
    """
    Calculates the 6 proxy indicators for each cluster, normalizes them, 
    and computes the final Parking Impact Index (PII).
    
    Input:
        df: The cleaned and engineered violations DataFrame (contains Cluster_ID).
        
    Output:
        ranked_df: Ranked clusters sorted descending by PII with all components.
    """
    print("Computing Parking Impact Index (PII) for clusters...")
    # Filter out noise points
    df_clusters = df[df['Cluster_ID'] != -1].copy()
    
    if len(df_clusters) == 0:
        return pd.DataFrame()
        
    # Get total observation days in the entire dataset
    # (using unique dates in df to capture the full duration)
    df['date_only'] = df['created_datetime'].dt.date
    total_observation_days = df['date_only'].nunique()
    if total_observation_days == 0:
        total_observation_days = 1
        
    cluster_records = []
    grouped = df_clusters.groupby('Cluster_ID')
    
    for cluster_id, grp in grouped:
        total_violations = len(grp)
        
        # 1. Density Score: violations / cluster area (in sq meters)
        coords = grp[['latitude', 'longitude']].values
        area = calculate_cluster_area(coords)
        raw_density = total_violations / area
        
        # 2. Peak Hour Score: (violations during 8-10 and 17-20) / total violations
        # Note: 8-10 means Hour in [8, 9], 17-20 means Hour in [17, 18, 19]
        peak_violations = grp[grp['Hour'].isin([8, 9, 17, 18, 19])]
        raw_peak = len(peak_violations) / total_violations
        
        # 3. Persistence Score: active days / total observation days
        active_days = grp['created_datetime'].dt.date.nunique()
        raw_persistence = active_days / total_observation_days
        
        # 4. Vehicle Impact Score: average vehicle severity weight
        raw_vehicle = grp['vehicle_severity'].mean()
        
        # 5. Violation Severity Score: average violation severity weight
        raw_violation = grp['violation_severity'].mean()
        
        # 6. Junction Criticality Score: (violations at junction) / total violations
        # Junction is defined as junction_name not equal to 'No Junction'
        junction_violations = grp[grp['junction_name'] != 'No Junction']
        raw_junction = len(junction_violations) / total_violations
        
        # Capture cluster location metadata (centroid)
        lat_center = grp['latitude'].mean()
        lon_center = grp['longitude'].mean()
        
        cluster_records.append({
            'Cluster_ID': cluster_id,
            'lat': lat_center,
            'lon': lon_center,
            'total_violations': total_violations,
            'area_sqm': area,
            'raw_density': raw_density,
            'raw_peak': raw_peak,
            'raw_persistence': raw_persistence,
            'raw_vehicle_impact': raw_vehicle,
            'raw_violation_severity': raw_violation,
            'raw_junction_criticality': raw_junction
        })
        
    res_df = pd.DataFrame(cluster_records)
    
    # Normalization helper to map values to [0, 100]
    def normalize(series):
        s_min = series.min()
        s_max = series.max()
        if s_max == s_min:
            return pd.Series(0.0, index=series.index)
        return 100.0 * (series - s_min) / (s_max - s_min)
        
    # Apply normalization
    res_df['Density_Score'] = normalize(res_df['raw_density'])
    res_df['Peak_Score'] = normalize(res_df['raw_peak'])
    res_df['Persistence_Score'] = normalize(res_df['raw_persistence'])
    res_df['Vehicle_Impact_Score'] = normalize(res_df['raw_vehicle_impact'])
    res_df['Violation_Severity_Score'] = normalize(res_df['raw_violation_severity'])
    res_df['Junction_Criticality_Score'] = normalize(res_df['raw_junction_criticality'])
    
    # Calculate Final Parking Impact Index (PII)
    w = config.PII_WEIGHTS
    res_df['PII'] = (
        w['density'] * res_df['Density_Score'] +
        w['peak'] * res_df['Peak_Score'] +
        w['persistence'] * res_df['Persistence_Score'] +
        w['vehicle_impact'] * res_df['Vehicle_Impact_Score'] +
        w['violation_severity'] * res_df['Violation_Severity_Score'] +
        w['junction_criticality'] * res_df['Junction_Criticality_Score']
    )
    
    # Apply PII Severity Classification
    res_df['PII_Severity'] = res_df['PII'].apply(config.get_pii_severity)
    
    # Rank and sort descending by PII
    res_df = res_df.sort_values(by='PII', ascending=False).reset_index(drop=True)
    res_df['Rank'] = res_df.index + 1
    
    return res_df
