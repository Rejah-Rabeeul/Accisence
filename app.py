
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import os
import networkx as nx
import osmnx as ox
from realtime_inference_utils import get_live_weather, get_temporal_features, prepare_live_features
from risk_aware_navigation import analyze_route, get_coordinates, calculate_curvature, get_maxspeed

# --- Configuration ---
st.set_page_config(
    page_title="Accisence - AI Accident Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "accident_model.pkl"
METRICS_FILE = "final_scores.txt"

# --- Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .metric-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_resource
def load_graph():
    # Cache the graph for Kozhikode to speed up subsequent requests
    try:
        return ox.graph_from_place('Kozhikode, Kerala, India', network_type='drive')
    except Exception as e:
        return None

@st.cache_data(ttl=600)
def get_cached_weather():
    # Cache weather for 10 minutes to avoid API timeouts on every refresh
    return get_live_weather()

# --- Main App ---
def main():
    st.title("🛡️ Accisence: AI Road Safety Assistant")
    st.markdown("### Real-time Accident Risk Prediction & Navigation")

    # Load Resources
    with st.spinner("Loading AI Nervous System..."):
        model = load_model()
        G = load_graph()

    if not model:
        st.error(f"Critical Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        st.stop()
        
    if not G:
        st.warning("Warning: Could not load Road Network Graph. Navigation features may be slower or unavailable.")

    # --- Sidebar: Environment Status ---
    st.sidebar.title("🌍 Live Environment")
    
    # Use cached weather to prevent blocking/blinking
    weather = get_cached_weather()
    time_ctx = get_temporal_features()
    
    # Weather Icon
    weather_icons = {'Clear': '☀️', 'Rain': '🌧️', 'Fog': '🌫️'}
    st.sidebar.header(f"{weather_icons.get(weather, '☁️')} {weather}")
    
    # Time Data
    st.sidebar.markdown(f"**Time:** {time_ctx['hour_of_day']}:00")
    st.sidebar.markdown(f"**Condition:** {'🌙 Night' if time_ctx['is_night'] else '☀️ Day'}")
    st.sidebar.markdown(f"**Status:** {'🏖️ Holiday/Weekend' if time_ctx['is_holiday'] else '🏢 Workday'}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("System connected to Kozhikode Traffic Grid.")

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["🗺️ Journey Planner", "📍 Point Predictor", "📊 Metrics"])

    # --- Tab 1: Journey Planner ---
    with tab1:
        st.header("Plan a Safe Journey")
        
        # Use a Form to prevent page reloads on every character type
        with st.form("journey_planner"):
            col1, col2 = st.columns(2)
            with col1:
                origin = st.text_input("From", placeholder="Current Location (or enter place)")
            with col2:
                dest = st.text_input("To", placeholder="e.g. Calicut Beach")
            
            # Form Submit Button
            submitted = st.form_submit_button("Analyze Route Risk")
            
        if submitted:
            if not dest:
                st.warning("Please enter a destination.")
            else:
                with st.spinner("Calculating Fastest & Safest Route..."):
                    # Store result in session state
                    st.session_state['route_result'] = analyze_route(origin, dest, model=model, G=G)

        # Check if we have a result to display (outside form to persist)
        if 'route_result' in st.session_state:
            result = st.session_state['route_result']
            
            if not result or "error" in result:
                st.error(f"Analysis Failed: {result.get('error', 'Unknown Error')}")
            else:
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                path_len = len(result['route_nodes'])
                max_prob = result['top_5_risks'][0]['prob'] if result['top_5_risks'] else 0
                
                m1.metric("Route Segments", path_len)
                m2.metric("Max Risk Score", f"{max_prob:.1%}", delta_color="inverse")
                m3.metric("High Risk Spots", len(result['top_5_risks']))
                
                # Map Visualization
                # Create Folium Map centered on Origin or first point
                center_lat = result['route_coords'][0][0]
                center_lon = result['route_coords'][0][1]
                m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
                
                # Draw Route
                folium.PolyLine(
                    result['route_coords'],
                    color="blue",
                    weight=5,
                    opacity=0.7
                ).add_to(m)
                
                # Mark Start/End
                folium.Marker(result['route_coords'][0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
                folium.Marker(result['route_coords'][-1], popup="End", icon=folium.Icon(color="red")).add_to(m)
                
                # Mark Risk Points
                for risk in result['top_5_risks']:
                    folium.CircleMarker(
                        location=[risk['lat'], risk['lon']],
                        radius=8,
                        popup=f"Risk: {risk['prob']:.1%}<br>Curvature: {risk['features']['curvature_score']:.2f}",
                        color="red",
                        fill=True,
                        fill_color="red"
                    ).add_to(m)
                
                st_folium(m, width=700, height=500)
                
                st.success("Route Analysis Complete!")
                st.markdown(f"### [🚀 Open Safe Navigation Route in Google Maps]({result['maps_url']})")

    # --- Tab 2: Point Predictor ---
    with tab2:
        st.header("Manual Risk Assessment")
        st.markdown("Predict accident probability for a specific road condition.")
        
        c1, c2 = st.columns(2)
        with c1:
            p_curv = st.slider("Curvature Score", 1.0, 1.5, 1.0, help="1.0 is straight, higher is curvier.")
            p_speed = st.slider("Max Speed (km/h)", 20, 100, 40)
            p_junc = st.selectbox("Is Junction?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        
        with c2:
            p_weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Fog"])
            p_night = st.selectbox("Time of Day", [0, 1], format_func=lambda x: "Night" if x else "Day")
            p_holiday = st.selectbox("Is Holiday/Weekend?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            
        if st.button("Predict Probability"):
            # Construct feature dict matched to utils
            # We need to manually construct the input dataframe structure expected by model
            # Or use prepare_live_features logic
            
            # Mimic tools
            feat = {'curvature_score': p_curv, 'maxspeed': float(p_speed), 'is_junction': p_junc}
            # Mock get_temporal_features style output for our inputs
            time_d = {'is_night': p_night, 'hour_of_day': 20 if p_night else 12, 'is_holiday': p_holiday}
            
            df_in = prepare_live_features(feat, p_weather, time_d)
            prob = model.predict_proba(df_in)[0][1]
            
            st.metric("Accident Probability", f"{prob:.2%}")
            
            if prob > 0.7:
                st.error("⚠️ HIGH RISK! Drive with extreme caution.")
            elif prob > 0.4:
                st.warning("⚠️ MODERATE RISK. Be careful.")
            else:
                st.success("✅ LOW RISK. Safe conditions.")

    # --- Tab 3: Metrics ---
    with tab3:
        st.header("Model Performance Metrics")
        
        if os.path.exists(METRICS_FILE):
             with open(METRICS_FILE, "r") as f:
                metrics_text = f.read()
                
             # Parse simple metrics
             lines = metrics_text.split('\n')
             cols = st.columns(2)
             for i, line in enumerate(lines):
                 if ':' in line:
                     k, v = line.split(':', 1)
                     cols[i % 2].metric(k.strip(), v.strip())
        else:
            st.info("Metrics file not found. Run evaluation script first.")

if __name__ == "__main__":
    main()
