import os

# File paths
DATA_PATH = "violations.csv"
MODEL_PATH = "best_model.pkl"

# DBSCAN parameters
# eps in radians for haversine metric: 200 meters / radius of earth (6,371,000 meters)
DBSCAN_EPS = 200.0 / 6371000.0
DBSCAN_MIN_SAMPLES = 25

# Vehicle Severity Mappings
# Two Wheeler (1), Auto Rickshaw (2), Car (3), SUV (4), Bus (5), Truck (6), Tanker (7)
VEHICLE_SEVERITY_MAPPING = {
    'SCOOTER': 1,
    'MOTOR CYCLE': 1,
    'MOPED': 1,
    'PASSENGER AUTO': 2,
    'GOODS AUTO': 2,
    'CAR': 3,
    'MAXI-CAB': 3,
    'VAN': 3,
    'TEMPO': 3,
    'JEEP': 4,
    'PRIVATE BUS': 5,
    'BUS (BMTC/KSRTC)': 5,
    'TOURIST BUS': 5,
    'SCHOOL VEHICLE': 5,
    'FACTORY BUS': 5,
    'LGV': 6,
    'HGV': 6,
    'LORRY/GOODS VEHICLE': 6,
    'MINI LORRY': 6,
    'TRACTOR': 6,
    'TANKER': 7,
    'OTHERS': 3,  # default weight
}

# Violation Severity Mappings
# Minor Violation (1), Improper Parking (3), Footpath Encroachment (4), Roadside Obstruction (4), Blocking Junction (5), No Parking Zone (5)
VIOLATION_SEVERITY_MAPPING = {
    # Minor Violations
    '2W/3W - USING MOBILE PHONE': 1,
    'CARRYING LENGHTY MATERIAL': 1,
    'DEFECTIVE NUMBER PLATE': 1,
    'DEMANDING EXCESS FARE': 1,
    'FAIL TO USE SAFETY BELTS': 1,
    'OBSTRUCTING DRIVER': 1,
    'OTHER - USING MOBILE PHONE': 1,
    'REFUSE TO GO FOR HIRE': 1,
    'RIDER NOT WEARING HELMET': 1,
    'STOPING ON WHITE/STOP LINE': 1,
    'USING BLACK FILM/OTHER MATERIALS': 1,
    'VIOLATING LANE DISIPLINE': 1,
    'WITHOUT SIDE MIRROR': 1,
    
    # Improper Parking
    'DOUBLE PARKING': 3,
    'PARKING OPPOSITE TO ANOTHER PARKED VEHICLE': 3,
    'PARKING OTHER THAN BUS STOP': 3,
    'WRONG PARKING': 3,
    
    # Footpath Encroachment
    'PARKING ON FOOTPATH': 4,
    
    # Roadside Obstruction
    'PARKING IN A MAIN ROAD': 4,
    'PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC': 4,
    'H T V PROHIBITED': 4,
    'AGAINST ONE WAY/NO ENTRY': 4,
    
    # Blocking Junction
    'PARKING NEAR ROAD CROSSING': 5,
    'PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS': 5,
    'JUMPING TRAFFIC SIGNAL': 5,
    'U TURN PROHIBITED': 5,
    
    # No Parking Zone
    'NO PARKING': 5
}

# PII weights
PII_WEIGHTS = {
    'density': 0.30,
    'peak': 0.20,
    'persistence': 0.15,
    'vehicle_impact': 0.15,
    'violation_severity': 0.10,
    'junction_criticality': 0.10
}

# PII Severity classification
# Low: 0 - 15, Medium: 16 - 25, High: 26 - 35, Critical: 36+
def get_pii_severity(pii):
    if pii <= 15:
        return 'Low'
    elif pii <= 25:
        return 'Medium'
    elif pii <= 35:
        return 'High'
    else:
        return 'Critical'

# Hotspot Forecasting categories
# Low: 0 - 2, Medium: 3 - 5, High: 6 - 10, Critical: 10+
def get_forecasting_category(count):
    if count <= 2:
        return 'Low'
    elif count <= 5:
        return 'Medium'
    elif count <= 10:
        return 'High'
    else:
        return 'Critical'
