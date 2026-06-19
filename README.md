# AI-Driven Parking Intelligence Platform

A **Dual-Engine Spatial Prioritization & Predictive Hotspot Forecasting System** designed to identify and forecast high-risk parking violation zones to optimize urban enforcement resources.

---

## 🚀 Key Features

1. **Spatial Prioritization Engine (Parking Impact Index - PII):**
   * Computes a multi-criteria score (0-100) for discovered hotspots using six proxy indicators:
     * **Density:** Violations per square meter (calculated via Scipy Convex Hull).
     * **Peak Hours:** Percentage of violations occurring during peak traffic hours.
     * **Persistence:** Temporal consistency of violations over observation days.
     * **Vehicle Weight:** Aggregated vehicle severity index (e.g., Tankers/Trucks weighted higher than scooters).
     * **Violation Severity:** Aggregated severity mapping based on traffic hazard levels.
     * **Junction Criticality:** Proximity and impact on major traffic junctions.
   * Classifies hotspots into Low, Medium, High, and Critical risk categories.

2. **Predictive Machine Learning Engine:**
   * Trains and compares **XGBoost** and **LightGBM** regressors on aggregated spatial-temporal features.
   * Dynamically forecasts violation counts for any future date/hour on a per-cluster basis.
   * Auto-selects the best model and serializes weights and metadata into a binary pipeline artifact.

3. **Interactive Visual Dashboard:**
   * **Map Overlays:** Renders interactive maps using Folium, displaying historical density heatmaps and colored markers representing priority zones.
   * **Interactive Analytics:** Dynamic Plotly visualizations including a multi-dimensional radar chart profiling top priority hotspots.
   * **Future Forecasting Panel:** Interactive inputs enabling real-time risk predictions and visualization of future risk distributions.

---

## 📁 Repository Structure

* `app.py`: The Streamlit-based interactive web dashboard.
* `config.py`: Platform configuration containing weights, vehicle/violation severity mappings, and classification thresholds.
* `data_pipeline.py`: Preprocessing pipeline handling data ingestion, cleaning, temporal feature engineering, DBSCAN clustering, and training set grid preparation.
* `impact_engine.py`: Numerical module computing Convex Hull area and calculating normalized PII scores.
* `model_training.py`: Training engine that trains XGBoost/LightGBM, compares performance metrics ($R^2$ and RMSE), and serializes model weights.
* `requirements.txt`: Python package dependencies.
* `.gitignore`: Configured to exclude heavy datasets, models, and cache files from Git version control.

---

## 🛠️ Setup & Running Instructions

### 1. Prerequisites
Make sure you have Python 3.8+ installed. Install the package dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
Place the dataset `violations.csv` in the root directory.

### 3. Run the Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```
*Note: If the serialized model (`best_model.pkl`) is missing, the dashboard will automatically run the training and evaluation pipeline on startup to generate it.*

### 4. Run Model Training Manually
To manually retrain the machine learning models and recalculate the PII scores:
```bash
python model_training.py
```
