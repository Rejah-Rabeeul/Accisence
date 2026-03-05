
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import os
import networkx as nx
import osmnx as ox
import streamlit.components.v1 as components
import base64
from realtime_inference_utils import get_live_weather, get_temporal_features, prepare_live_features
from risk_aware_navigation import analyze_route, get_coordinates, calculate_curvature, get_maxspeed, haversine_distance

# --- Configuration ---
st.set_page_config(
    page_title="Accisence - AI Accident Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit hack to serve static ServiceWorker file precisely at root for scoping
import streamlit.components.v1 as components
import base64
with open("static/sw.js", "r") as f:
    sw_js = f.read()
sw_b64 = base64.b64encode(sw_js.encode()).decode()

# Inject the SW payload silently onto the page head so it can be registered from a Blob safely 
# if static serving from Streamlit is blocked by the proxy
st.markdown(f"""
<script>
    window.SW_CODE_B64 = "{sw_b64}";
</script>
""", unsafe_allow_html=True)

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
        
    # Get User Location (Browser) - Mobile Safe Method via Custom JS
    st.markdown("### 📍 Location Services (Required)")
    st.markdown("**Mobile Users:** Tap the button below to allow GPS access. This will prompt your mobile browser securely.")
    
    # Custom HTML/JS to trigger native browser prompt and send data back to Streamlit via query params
    location_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .btn {
                background-color: #FF4B4B;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
                font-family: sans-serif;
            }
            .btn:hover { background-color: #ff3333; }
            #status { margin-top: 10px; font-family: sans-serif; font-size: 14px; color: #333; }
        </style>
    </head>
    <body>
        <button class="btn" onclick="getLocation()">📍 Get My Location</button>
        <div id="status"></div>
        <script>
            function getLocation() {
                const status = document.getElementById('status');
                status.textContent = "Locating...";
                
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(
                        (position) => {
                            const lat = position.coords.latitude;
                            const lon = position.coords.longitude;
                            const acc = position.coords.accuracy;
                            status.textContent = "Location found! Sending to app...";
                            status.style.color = "green";
                            
                            // Send data back to Streamlit URL silently
                            const urlParams = new URLSearchParams(window.parent.location.search);
                            urlParams.set('lat', lat);
                            urlParams.set('lon', lon);
                            urlParams.set('acc', acc);
                            window.parent.history.replaceState(null, '', '?' + urlParams.toString());
                            
                            // Force Streamlit to rerun by simulating a parent window reload without full refresh
                            window.parent.location.reload(); 
                        },
                        (error) => {
                            let msg = "Error: ";
                            switch(error.code) {
                                case error.PERMISSION_DENIED: msg += "Permission Denied. Please enable GPS."; break;
                                case error.POSITION_UNAVAILABLE: msg += "Position Unavailable."; break;
                                case error.TIMEOUT: msg += "Request Timeout."; break;
                                default: msg += "Unknown Error."; break;
                            }
                            status.textContent = msg;
                            status.style.color = "red";
                        },
                        { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
                    );
                } else {
                    status.textContent = "Geolocation is not supported by this browser.";
                    status.style.color = "red";
                }
            }
        </script>
    </body>
    </html>
    """
    
    # Render the JS component
    components.html(location_html, height=100)
    
    user_coords = None
    
    # Read from query params
    params = st.query_params
    if 'lat' in params and 'lon' in params:
        try:
            user_coords = (float(params['lat']), float(params['lon']))
            loc_acc = float(params.get('acc', 0.0))
            st.success(f"✅ Location Acquired! ({user_coords[0]:.4f}, {user_coords[1]:.4f})")
            
            # Create a mock 'loc' object for the bottom tab to read from
            loc = {'latitude': user_coords[0], 'longitude': user_coords[1], 'accuracy': loc_acc}
        except:
            st.warning("⚠️ Error parsing location. Please try tapping the button again.")
            loc = None
    else:
        st.warning("⚠️ Waiting for location... Please tap the button above to enable GPS.")
        loc = None

    if not model:
        st.error(f"Critical Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        st.stop()
        
    if not G:
        st.warning("Warning: Could not load Road Network Graph. Navigation features may be slower or unavailable.")

    # --- Sidebar: Environment Status ---
    st.sidebar.title("🌍 Live Environment")
    
    # GPS Status
    if user_coords:
        st.sidebar.success(f"📡 GPS: Connected\n({user_coords[0]:.3f}, {user_coords[1]:.3f})")
    else:
        st.sidebar.warning("📡 GPS: Not Detected\n(Using IP Location)")

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
    
    if st.sidebar.button("🔄 Reload Model"):
        st.cache_resource.clear()
        st.success("Model Cache Cleared! Reloading...")
        st.rerun()

    # --- Tabs ---
    tab1, tab2 = st.tabs(["🗺️ Journey Planner", "📍 Point Predictor"])

    # --- Tab 1: Journey Planner ---
    with tab1:
        st.header("Plan a Safe Journey")
        
        if not user_coords:
            st.info("⚠️ **Note:** GPS is inactive (requires HTTPS on mobile). 'Current Location' will use approximate IP-based location.")
        
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
                    st.session_state['route_result'] = analyze_route(origin, dest, model=model, G=G, user_location=user_coords)
                    
                    st.success("Route Analyzed Successfully!")

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
                
                if max_prob > 0.7:
                    risk_level = "High"
                elif max_prob > 0.4:
                    risk_level = "Moderate"
                else:
                    risk_level = "Low"
                    
                m1.metric("Route Segments", path_len)
                m2.metric("Max Risk Level", risk_level)
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
                        popup=f"Risk Level: {'High' if risk['prob']>0.7 else 'Moderate' if risk['prob']>0.4 else 'Low'}<br>Curvature: {risk['features']['curvature_score']:.2f}",
                        color="red",
                        fill=True,
                        fill_color="red"
                    ).add_to(m)
                
                st_folium(m, width=700, height=500)
                
                # Native Fallback (In case JS blocks it)
                if 'top_5_risks' in result and result['top_5_risks']:
                    risk_spots_text = "\\n".join([
                        f"{i+1}. {r.get('name', 'Unknown Road')} (Risk: {'High' if r['prob']>0.7 else 'Mod' if r['prob']>0.4 else 'Low'})"
                        for i, r in enumerate(result['top_5_risks'])
                    ])
                    st.info(f"**🚨 Top 5 Accident Spots on Your Route:**\\n\\n{risk_spots_text.replace('\\n', '  \\n')}")
                else:
                    risk_spots_text = "No severe risks found on this route."

                st.success("Route Analysis Complete!")
                st.markdown(f"### [🚀 Open Safe Navigation Route in Google Maps]({result['maps_url']})")
                
                # Render Web Push Notification Button
                risk_spots_encoded = risk_spots_text.replace('\\n', '\\\\n')
                notification_html = f"""
                <div style="font-family: sans-serif; text-align: center; margin-top: 5px;">
                    <button id="notifyBtn" style="background:#FF4B4B; color:white; border:none; padding:8px 15px; border-radius:5px; cursor:pointer; font-weight:bold;">
                        🔔 Show Top 5 Accident Spots Alert
                    </button>
                </div>
                <script>
                    document.getElementById('notifyBtn').addEventListener('click', async function() {{
                        const title = "🚨 Top 5 Accident Spots on Your Route:";
                        const options = {{
                            body: "{risk_spots_encoded}",
                            icon: "https://unpkg.com/lucide-static@0.263.1/icons/alert-triangle.svg",
                            requireInteraction: true
                        }};

                        try {{
                            if (!("Notification" in window) || !("serviceWorker" in navigator)) {{
                                alert("This browser lacks necessary support for native notifications.\\n\\n" + title + "\\n" + options.body);
                                return;
                            }}

                            let perm = Notification.permission;
                            if (perm !== "granted" && perm !== "denied") {{
                                perm = await Notification.requestPermission();
                            }}

                            if (perm === "granted") {{
                                // Try registering the Service Worker via the injected b64 payload 
                                // Streamlit makes hosting vanilla JS files at the exact root scope extremely difficult,
                                // so we decode it on the client and serve it via Blob, but more robustly this time.
                                
                                let swUrl;
                                try {{
                                    if (window.parent.window.SW_CODE_B64) {{
                                        const swCode = atob(window.parent.window.SW_CODE_B64);
                                        const blob = new Blob([swCode], {{type: 'application/javascript'}});
                                        swUrl = URL.createObjectURL(blob);
                                    }} else {{
                                        // Fallback minimal SW
                                        const swCode = "self.addEventListener('install', e => self.skipWaiting()); self.addEventListener('activate', e => event.waitUntil(clients.claim()));";
                                        const blob = new Blob([swCode], {{type: 'application/javascript'}});
                                        swUrl = URL.createObjectURL(blob);
                                    }}
                                }} catch(e) {{
                                    const swCode = "self.addEventListener('install', e => self.skipWaiting()); self.addEventListener('activate', e => event.waitUntil(clients.claim()));";
                                    const blob = new Blob([swCode], {{type: 'application/javascript'}});
                                    swUrl = URL.createObjectURL(blob);
                                }}

                                navigator.serviceWorker.register(swUrl).then(function(reg) {{
                                    return navigator.serviceWorker.ready;
                                }}).then(function(reg) {{
                                    if (reg) {{
                                        reg.showNotification(title, options);
                                        document.getElementById('notifyBtn').innerText = "🔔 Alert Sent to Device!";
                                    }} else {{
                                        alert(title + "\\n" + options.body);
                                    }}
                                }}).catch(function(err) {{
                                    console.error('SW Error:', err);
                                    // Fallback to old Notification object if SW fails
                                    try {{
                                        new Notification(title, options);
                                    }} catch(e) {{
                                        alert(title + "\\n" + options.body);
                                    }}
                                }});
                            }} else {{
                                alert("Permissions blocked. Manual Alert:\\n\\n" + title + "\\n" + options.body);
                            }}
                        }} catch (e) {{
                            console.error("Push Error:", e);
                            alert("Notification Error.\\n\\n" + title + "\\n" + options.body);
                        }}
                    }});
                </script>
                """
                components.html(notification_html, height=60)

                # --- Real-Time GPS Tracker ---
                st.markdown("---")
                st.subheader("📡 Real-Time GPS Tracker")
                
                # GPS Toggle
                gps_active = st.checkbox("Enable Live GPS Tracking")
                
                if gps_active:

                    
                    # Instructions for mobile updates
                    st.info("💡 **Tip:** To update your live position as you move, tap the 'Get Location' button at the **very top of the main page** again.")
                        
                    # Check if we have coordinates from the global top-level widget
                    if user_coords:
                        curr_lat = user_coords[0]
                        curr_lon = user_coords[1]
                        accuracy = loc.get('accuracy', 0.0) if loc and loc.get('accuracy') is not None else 0.0
                        
                        st.info(f"📍 **Your Location:** {curr_lat:.5f}, {curr_lon:.5f} (±{accuracy:.1f}m)")
                        
                        # 3. Check Proximity to Risks
                        risk_nearby = False
                        risk_points = result['top_5_risks']
                        
                        for risk in risk_points:
                            r_lat = risk['lat']
                            r_lon = risk['lon']
                            
                            dist = haversine_distance(curr_lat, curr_lon, r_lat, r_lon)
                            
                            # 300m Warning Threshold
                            if dist < 300:
                                risk_nearby = True
                                risk_level_str = 'High' if risk['prob']>0.7 else 'Moderate' if risk['prob']>0.4 else 'Low'
                                msg = f"⚠️ **DANGER AHEAD ({int(dist)}m)**\n" \
                                      f"Risk Level: `{risk_level_str}` | Curve: `{risk['features']['curvature_score']:.2f}`"
                                st.toast(msg, icon="🚨")
                                st.error(msg)
                        
                        if not risk_nearby:
                            st.success("✅ You are in a Safe Zone.")
                            
                    else:
                        st.error("⚠️ GPS Signal Not Detected.")
                        st.warning("Please enable Location Services in your browser and system settings, then click 'Refresh My Location'.")
                        st.markdown("**Note:** GPS access requires **HTTPS** or **localhost**.")

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
            
            risk_level = "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"
            st.metric("Accident Risk Level", risk_level)
            
            if prob > 0.7:
                st.error("⚠️ HIGH RISK! Drive with extreme caution.")
            elif prob > 0.4:
                st.warning("⚠️ MODERATE RISK. Be careful.")
            else:
                st.success("✅ LOW RISK. Safe conditions.")



if __name__ == "__main__":
    main()
