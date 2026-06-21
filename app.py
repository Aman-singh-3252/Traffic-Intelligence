import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
import plotly.express as px

import config

# Page configuration
st.set_page_config(
    page_title="AI-Driven Parking Intelligence Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling, dark mode accents, and custom typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title and Header Accent */
    .main-title {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #B2B2B2;
        font-weight: 300;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 75, 75, 0.4);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #FAFAFA 0%, #D1D1D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8C8C8C;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model artifacts
@st.cache_resource
def load_model_artifacts():
    if not os.path.exists(config.MODEL_PATH):
        st.info("Model artifacts not found. Automatically running the model training pipeline, please wait...")
        with st.spinner("Executing model training pipeline (XGBoost & LightGBM)..."):
            import model_training
            model_training.train_and_evaluate()
            st.success("Model training complete!")
            
    with open(config.MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

# Load artifacts
try:
    artifacts = load_model_artifacts()
    model = artifacts['model']
    model_type = artifacts['model_type']
    model_r2 = artifacts['r2']
    model_rmse = artifacts['rmse']
    pipeline = artifacts['feature_pipeline']
    cluster_profiles = artifacts['cluster_profiles']
    pii_df = artifacts['pii_df']
    heatmap_coords = artifacts['heatmap_coords']
    
    # SHAP and explainability artifacts
    explainer = artifacts.get('explainer')
    shap_values = artifacts.get('shap_values')
    feature_cols = artifacts.get('feature_cols')
    X_test_engineered = artifacts.get('X_test_engineered')
    y_test = artifacts.get('y_test')
except Exception as e:
    st.error(f"Failed to load or train model: {e}")
    st.stop()

# Sidebar Information
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #FF4B4B; margin-bottom: 0;">PARKING AI</h2>
        <small style="color: #8C8C8C; text-transform: uppercase; letter-spacing: 1px;">Intelligence Dashboard</small>
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.subheader("System Status")
st.sidebar.markdown(f"**ML Model Engine:** `{model_type}`")
st.sidebar.markdown(f"**Model R² Value:** `{model_r2:.4f}`")
st.sidebar.markdown(f"**Model RMSE:** `{model_rmse:.4f}`")
st.sidebar.markdown(f"**Hotspots Monitored:** `{len(cluster_profiles)}` clusters")

st.sidebar.markdown("---")
st.sidebar.subheader("PII Weights Configuration")
for component, weight in config.PII_WEIGHTS.items():
    st.sidebar.markdown(f"- {component.replace('_', ' ').title()}: `{weight * 100:.0f}%`")

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Severity Legend")
st.sidebar.markdown(
    """<div style="background: rgba(255, 255, 255, 0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 1rem;">
<div style="display: flex; align-items: center; margin-bottom: 8px;">
<span style="height: 10px; width: 10px; background-color: #D32F2F; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
<span style="font-weight: 600; color: #FAFAFA; font-size: 0.85rem;">Critical Risk</span>
</div>
<div style="font-size: 0.75rem; color: #8C8C8C; margin-left: 20px; margin-bottom: 8px; margin-top: -6px;">
PII Score 36+ | Forecast 11+
</div>
<div style="display: flex; align-items: center; margin-bottom: 8px;">
<span style="height: 10px; width: 10px; background-color: #F57C00; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
<span style="font-weight: 600; color: #FAFAFA; font-size: 0.85rem;">High Risk</span>
</div>
<div style="font-size: 0.75rem; color: #8C8C8C; margin-left: 20px; margin-bottom: 8px; margin-top: -6px;">
PII Score 26-35 | Forecast 6-10
</div>
<div style="display: flex; align-items: center; margin-bottom: 8px;">
<span style="height: 10px; width: 10px; background-color: #FBC02D; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
<span style="font-weight: 600; color: #FAFAFA; font-size: 0.85rem;">Medium Risk</span>
</div>
<div style="font-size: 0.75rem; color: #8C8C8C; margin-left: 20px; margin-bottom: 8px; margin-top: -6px;">
PII Score 16-25 | Forecast 3-5
</div>
<div style="display: flex; align-items: center; margin-bottom: 4px;">
<span style="height: 10px; width: 10px; background-color: #388E3C; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
<span style="font-weight: 600; color: #FAFAFA; font-size: 0.85rem;">Low Risk</span>
</div>
<div style="font-size: 0.75rem; color: #8C8C8C; margin-left: 20px; margin-top: -2px;">
PII Score 0-15 | Forecast 0-2
</div>
</div>
<div style="font-size: 0.75rem; color: #8C8C8C; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 6px;">
* <b>PII:</b> Parking Impact Index (Current)<br/>
* <b>Forecast:</b> Predicted Violations/Hour (Future)
</div>""",
    unsafe_allow_html=True
)

# Header Section
st.markdown('<h1 class="main-title">AI-Driven Parking Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Dual-Engine Spatial Prioritization & Predictive Hotspot Forecasting System</p>', unsafe_allow_html=True)

# Metrics Grid (Header Summary Cards)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Clusters</div>
            <div class="metric-val">{len(cluster_profiles)}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Max Violations / Cluster</div>
            <div class="metric-val">{cluster_profiles['total_violations'].max():,}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
with m3:
    critical_count = len(pii_df[pii_df['PII_Severity'] == 'Critical'])
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Critical Hotspots</div>
            <div class="metric-val" style="color: #FF4B4B;">{critical_count}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
with m4:
    best_hotspot = pii_df.iloc[0]['Cluster_ID']
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Top Priority Hotspot</div>
            <div class="metric-val" style="color: #FF8F00;">ID {int(best_hotspot)}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.write("")
st.write("")

# Navigation Tabs
tab1, tab2 = st.tabs(["📊 Impact Panel (Current Operations)", "🔮 Prediction Panel (Future Operations)"])

# Define PII severity colors
SEVERITY_COLORS = {
    'Critical': '#D32F2F',  # Red
    'High': '#F57C00',      # Orange
    'Medium': '#FBC02D',    # Yellow
    'Low': '#388E3C'        # Green
}

# ----------------- TAB 1: CURRENT OPERATIONS -----------------
with tab1:
    st.subheader("Spatial Hotspot Map & Historical Density")
    
    col_map_options, col_legend = st.columns([3, 2])
    with col_map_options:
        map_view_type = st.radio(
            "Select Map View",
            ["Hotspot Severity Markers", "Historical Density Heatmap", "Combined View"],
            horizontal=True
        )
    with col_legend:
        st.markdown(
            """
            <div style="display: flex; gap: 15px; justify-content: flex-end; align-items: center; height: 100%;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="height: 10px; width: 10px; background-color: #D32F2F; border-radius: 50%; display: inline-block;"></span>
                    <span style="font-size: 0.85rem; color: #B2B2B2; font-weight: 600;">Critical (PII 36+)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="height: 10px; width: 10px; background-color: #F57C00; border-radius: 50%; display: inline-block;"></span>
                    <span style="font-size: 0.85rem; color: #B2B2B2; font-weight: 600;">High (PII 26-35)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="height: 10px; width: 10px; background-color: #FBC02D; border-radius: 50%; display: inline-block;"></span>
                    <span style="font-size: 0.85rem; color: #B2B2B2; font-weight: 600;">Medium (PII 16-25)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="height: 10px; width: 10px; background-color: #388E3C; border-radius: 50%; display: inline-block;"></span>
                    <span style="font-size: 0.85rem; color: #B2B2B2; font-weight: 600;">Low (PII 0-15)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # Create Folium Map
    center_lat = pii_df['lat'].mean()
    center_lon = pii_df['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")
    
    # 1. Overlay Heatmap
    if map_view_type in ["Historical Density Heatmap", "Combined View"]:
        HeatMap(heatmap_coords.tolist(), radius=15, blur=10, min_opacity=0.4).add_to(m)
        
    # 2. Overlay Hotspots Markers
    if map_view_type in ["Hotspot Severity Markers", "Combined View"]:
        max_vol = pii_df['total_violations'].max()
        for idx, row in pii_df.iterrows():
            # Calculate marker radius based on total violations volume
            r = float(5 + (row['total_violations'] / max_vol) * 15)
            color = SEVERITY_COLORS.get(row['PII_Severity'], '#388E3C')
            
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 220px; font-size: 12px; color: #333;">
                <h4 style="margin: 0 0 5px 0; color:#FF4B4B;">Cluster {int(row['Cluster_ID'])} (Rank #{int(row['Rank'])})</h4>
                <hr style="margin: 3px 0;"/>
                <b>PII Score:</b> {row['PII']:.2f} ({row['PII_Severity']})<br/>
                <b>Total Violations:</b> {int(row['total_violations'])}<br/>
                <b>Junction:</b> {row['raw_junction_criticality']*100:.1f}% of violations at Junction<br/>
                <b>Area:</b> {row['area_sqm']:.1f} m²<br/>
                <hr style="margin: 3px 0;"/>
                <b>Sub-Scores (Normalized):</b><br/>
                - Density: {row['Density_Score']:.1f}<br/>
                - Peak: {row['Peak_Score']:.1f}<br/>
                - Persistence: {row['Persistence_Score']:.1f}<br/>
                - Vehicle Severity: {row['Vehicle_Impact_Score']:.1f}<br/>
                - Violation Severity: {row['Violation_Severity_Score']:.1f}<br/>
                - Junction Criticality: {row['Junction_Criticality_Score']:.1f}
            </div>
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=r,
                popup=folium.Popup(popup_html, max_width=250),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                weight=1
            ).add_to(m)
            
    # Render map
    st_folium(m, width="100%", height=550, returned_objects=[])
    
    st.write("---")
    
    st.subheader("Enforcement Priority Ranking")
    st.write("This table prioritizes parking hotspots using the multi-criteria Parking Impact Index (PII). All normalized sub-scores are displayed for operational transparency.")
    
    # Format dataframe for display
    display_df = pii_df[[
        'Rank', 'Cluster_ID', 'PII', 'PII_Severity', 'total_violations', 'area_sqm',
        'Density_Score', 'Peak_Score', 'Persistence_Score', 'Vehicle_Impact_Score', 
        'Violation_Severity_Score', 'Junction_Criticality_Score'
    ]].copy()
    
    # Round float values for presentation
    for col in ['PII', 'Density_Score', 'Peak_Score', 'Persistence_Score', 'Vehicle_Impact_Score', 'Violation_Severity_Score', 'Junction_Criticality_Score']:
        display_df[col] = display_df[col].round(2)
    display_df['area_sqm'] = display_df['area_sqm'].round(1)
    
    # Rename columns for clarity
    display_df.columns = [
        'Rank', 'Cluster ID', 'PII (Index)', 'PII Severity', 'Total Violations', 'Cluster Area (m²)',
        'Density (0-100)', 'Peak Hours (0-100)', 'Persistence (0-100)', 
        'Vehicle Weight (0-100)', 'Violation Severity (0-100)', 'Junction Criticality (0-100)'
    ]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.write("---")
    
    st.subheader("Top 5 Hotspot Comparison & Analytics")
    top5_df = pii_df.head(5)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Plotly Bar Chart for Component Scores of Top 5 Clusters
        fig_scores = go.Figure()
        fig_scores.add_trace(go.Bar(
            name="Vehicle Impact Score (Norm)",
            x=[f"Cluster {int(cid)}" for cid in top5_df['Cluster_ID']],
            y=top5_df['Vehicle_Impact_Score'],
            marker_color='#1E88E5'
        ))
        fig_scores.add_trace(go.Bar(
            name="Violation Severity Score (Norm)",
            x=[f"Cluster {int(cid)}" for cid in top5_df['Cluster_ID']],
            y=top5_df['Violation_Severity_Score'],
            marker_color='#D81B60'
        ))
        fig_scores.update_layout(
            barmode='group',
            title="Vehicle vs. Violation Severity (Top 5 Hotspots)",
            xaxis_title="Hotspot Cluster",
            yaxis_title="Normalized Score (0 - 100)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#FAFAFA"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scores, use_container_width=True)
        
    with col_chart2:
        # Plotly Radar / Spider-web style chart of all components for the #1 ranked cluster
        top_cluster = pii_df.iloc[0]
        categories = ['Density', 'Peak Hours', 'Persistence', 'Vehicle Weight', 'Violation Severity', 'Junction Criticality']
        scores = [
            top_cluster['Density_Score'],
            top_cluster['Peak_Score'],
            top_cluster['Persistence_Score'],
            top_cluster['Vehicle_Impact_Score'],
            top_cluster['Violation_Severity_Score'],
            top_cluster['Junction_Criticality_Score']
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(255, 75, 75, 0.2)',
            line=dict(color='#FF4B4B', width=2),
            marker=dict(size=8)
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#B2B2B2"),
                angularaxis=dict(color="#B2B2B2")
            ),
            showlegend=False,
            title=f"Detailed Profile: Priority #1 Hotspot (Cluster {int(top_cluster['Cluster_ID'])})",
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#FAFAFA")
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ----------------- TAB 2: PREDICTION PANEL -----------------
with tab2:
    st.subheader("Predictive Analytics & Future Violation Forecasting")
    st.write("Select a future date and hour below to forecast parking violation hotspots and risk distribution. Predictions are generated using the active Machine Learning Regressor.")
    
    # Inputs row
    col_date, col_hour, col_btn = st.columns([1.5, 2, 1])
    with col_date:
        # Dataset datetime range: 2023-11-09 to 2024-03-29
        # Default to a date in that range to show historical forecasting, or select future
        # Let's default to April 1st, 2024 (immediately after dataset span)
        pred_date = st.date_input("Select Future Date", value=pd.to_datetime("2024-04-01").date())
    with col_hour:
        pred_hour = st.slider("Select Future Hour (0-23)", min_value=0, max_value=23, value=12)
    with col_btn:
        st.write("")  # padding
        st.write("")  # padding
        run_prediction = st.button("Generate Forecast", type="primary", use_container_width=True)
        
    if run_prediction or 'predictions_made' not in st.session_state or st.session_state.get('last_pred_date') != pred_date or st.session_state.get('last_pred_hour') != pred_hour:
        # Perform forecasting
        with st.spinner("Calculating predictions for all clusters..."):
            # Prepare inputs
            day_of_week = pred_date.weekday()  # Monday=0, Sunday=6
            month = pred_date.month
            weekend_flag = 1 if day_of_week in [5, 6] else 0
            
            # Predict for all clusters
            pred_records = []
            for idx, cluster in cluster_profiles.iterrows():
                cid = cluster['Cluster_ID']
                
                # Prepare a DataFrame matching the feature pipeline input shape
                features = pd.DataFrame([{
                    'Hour': pred_hour,
                    'DayOfWeek': day_of_week,
                    'Month': month,
                    'Weekend_Flag': weekend_flag,
                    'Cluster_ID': cid,
                    'vehicle_severity': cluster['avg_vehicle_severity'],
                    'violation_severity': cluster['avg_violation_severity'],
                    'junction_name': cluster['junction_name'],
                    'police_station': cluster['police_station']
                }])
                
                # Transform using the upgraded pipeline
                features_engineered = pipeline.transform(features)
                model_features = features_engineered[feature_cols]
                
                pred_val = model.predict(model_features)[0]
                # Clip prediction to >= 0
                pred_val = max(0.0, float(pred_val))
                
                # Get severity category
                sev_cat = config.get_forecasting_category(pred_val)
                
                pred_records.append({
                    'Cluster_ID': int(cid),
                    'lat': cluster['lat_center'],
                    'lon': cluster['lon_center'],
                    'junction_name': cluster['junction_name'],
                    'police_station': cluster['police_station'],
                    'predicted_violations': pred_val,
                    'Risk_Category': sev_cat
                })
                
            pred_df = pd.DataFrame(pred_records)
            pred_df = pred_df.sort_values(by='predicted_violations', ascending=False).reset_index(drop=True)
            
            # Save in session state to avoid recalculation on page rerender
            st.session_state['predictions_made'] = True
            st.session_state['last_pred_date'] = pred_date
            st.session_state['last_pred_hour'] = pred_hour
            st.session_state['pred_df'] = pred_df
            
    # Retrieve forecast from session state
    pred_df = st.session_state.get('pred_df')
    
    if pred_df is not None:
        # Display forecast overview
        st.write("")
        st.subheader(f"Risk Distribution Forecast for {pred_date.strftime('%A, %b %d, %Y')} at {pred_hour:02d}:00")
        
        # Risk summaries
        counts = pred_df['Risk_Category'].value_counts()
        c_crit = counts.get('Critical', 0)
        c_high = counts.get('High', 0)
        c_med = counts.get('Medium', 0)
        c_low = counts.get('Low', 0)
        
        # Display small progress/indicators of future risk
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.metric("Critical Risk Clusters", c_crit, delta="Future Forecast", delta_color="inverse")
        with f2:
            st.metric("High Risk Clusters", c_high)
        with f3:
            st.metric("Medium Risk Clusters", c_med)
        with f4:
            st.metric("Low Risk Clusters", c_low)
            
        st.write("")
        
        # Render Future Risk Map
        col_pred_title, col_pred_legend = st.columns([1, 1])
        with col_pred_title:
            st.write("**Future Hotspot Risk Map**")
        with col_pred_legend:
            st.markdown(
                """
                <div style="display: flex; gap: 15px; justify-content: flex-end; align-items: center; height: 100%;">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="height: 10px; width: 10px; background-color: #D32F2F; border-radius: 50%; display: inline-block;"></span>
                        <span style="font-size: 0.85rem; color: #B2B2B2; font-weight: 600;">Critical (FC 11+)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="height: 10px; width: 10px; background-color: #F57C00; border-radius: 50%; display: inline-block;"></span>
                        <span style="font-size: 0.85rem; color: #B2B2B2; font-weight: 600;">High (FC 6-10)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="height: 10px; width: 10px; background-color: #FBC02D; border-radius: 50%; display: inline-block;"></span>
                        <span style="font-size: 0.85rem; color: #B2B2B2; font-weight: 600;">Medium (FC 3-5)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="height: 10px; width: 10px; background-color: #388E3C; border-radius: 50%; display: inline-block;"></span>
                        <span style="font-size: 0.85rem; color: #B2B2B2; font-weight: 600;">Low (FC 0-2)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        m_pred = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")
        
        max_pred_val = pred_df['predicted_violations'].max()
        for idx, row in pred_df.iterrows():
            r = float(5 + (row['predicted_violations'] / max(1.0, max_pred_val)) * 15)
            color = SEVERITY_COLORS.get(row['Risk_Category'], '#388E3C')
            
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 200px; font-size: 12px; color: #333;">
                <h4 style="margin: 0 0 5px 0; color:#FF4B4B;">Cluster {int(row['Cluster_ID'])}</h4>
                <hr style="margin: 3px 0;"/>
                <b>Forecasted Count:</b> {row['predicted_violations']:.2f}<br/>
                <b>Risk Level:</b> {row['Risk_Category']}<br/>
                <b>Junction:</b> {row['junction_name']}<br/>
                <b>Station:</b> {row['police_station']}
            </div>
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=r,
                popup=folium.Popup(popup_html, max_width=250),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                weight=1
            ).add_to(m_pred)
            
        st_folium(m_pred, width="100%", height=500, key="future_map", returned_objects=[])
        
        st.write("---")
        
        # Predict priority list
        col_list, col_chart = st.columns([3, 2])
        with col_list:
            st.write("**Top Prescriptive Patrol & Action Planning**")
            
            # Step 7: Apply the predictive & prescriptive operational rules engine
            import data_pipeline
            prescribed_df = data_pipeline.prescribe_operational_actions(pred_df['predicted_violations'])
            
            # Combine the predictions and prescriptive recommendations
            pred_df_full = pd.concat([pred_df, prescribed_df.drop(columns=['Predicted_Violations'])], axis=1)
            
            pred_display = pred_df_full[[
                'Cluster_ID', 'predicted_violations', 'Risk_Level', 
                'Suggested_Patrol_Units', 'Enforcement_Priority', 
                'junction_name', 'police_station'
            ]].copy()
            pred_display['predicted_violations'] = pred_display['predicted_violations'].round(2)
            pred_display.columns = [
                'Cluster ID', 'Predicted Violations', 'Risk Level', 
                'Patrol Units Needed', 'Priority Rank', 
                'Junction Name', 'Police Station'
            ]
            st.dataframe(pred_display.head(15), use_container_width=True, hide_index=True)
            
        with col_chart:
            st.write("**Violation Forecast by Risk Category**")
            fig_pie = px.pie(
                names=counts.index,
                values=counts.values,
                color=counts.index,
                color_discrete_map=SEVERITY_COLORS,
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#FAFAFA"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.write("---")
        st.subheader("🔍 Hackathon Special: Model Explainability (SHAP Diagnostics)")
        st.write("Below are the SHAP plots generated on the validation dataset to explain the model predictions to judges.")
        
        col_shap1, col_shap2 = st.columns(2)
        with col_shap1:
            st.markdown("**Global Feature Importance (SHAP Summary Plot)**")
            if shap_values is not None:
                import matplotlib.pyplot as plt
                import shap
                
                # Apply dark theme styling globally for matplotlib
                plt.rcParams.update({
                    "text.color": "#FAFAFA",
                    "axes.labelcolor": "#FAFAFA",
                    "xtick.color": "#FAFAFA",
                    "ytick.color": "#FAFAFA",
                    "figure.facecolor": "none",
                    "axes.facecolor": "none",
                    "axes.edgecolor": "#FAFAFA"
                })
                
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.summary_plot(shap_values, X_test_engineered, show=False, plot_size=None)
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)
            else:
                st.info("SHAP values not available.")
                
        with col_shap2:
            st.markdown("**Individual Prediction Breakdown (SHAP Waterfall Plot)**")
            if shap_values is not None and y_test is not None:
                import matplotlib.pyplot as plt
                import shap
                
                # Matplotlib styles persist, but reaffirming them for safety
                plt.rcParams.update({
                    "text.color": "#FAFAFA",
                    "axes.labelcolor": "#FAFAFA",
                    "xtick.color": "#FAFAFA",
                    "ytick.color": "#FAFAFA",
                    "figure.facecolor": "none",
                    "axes.facecolor": "none",
                    "axes.edgecolor": "#FAFAFA"
                })
                
                # Find the index of the highest violation prediction in the test set
                y_test_reset = y_test.reset_index(drop=True)
                max_idx = int(y_test_reset.idxmax())
                
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                shap.plots.waterfall(shap_values[max_idx], show=False)
                st.pyplot(fig2, bbox_inches='tight')
                plt.close(fig2)
            else:
                st.info("SHAP values/Test set not available.")
