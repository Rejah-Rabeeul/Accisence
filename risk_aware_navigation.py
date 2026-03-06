import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import shapely.geometry
from shapely.geometry import Point, LineString
import joblib
import os
import requests
from realtime_inference_utils import prepare_live_features, get_live_weather, get_temporal_features

# --- Connectors ---
MODEL_PATH = "accident_model.pkl"

def calculate_curvature(geom):
    """
    Calculates sinuosity: Curve Length / Euclidean Distance.
    """
    if not isinstance(geom, LineString):
        return 1.0
    length = geom.length
    if length == 0: return 1.0
    
    start = Point(geom.coords[0])
    end = Point(geom.coords[-1])
    euclidean = start.distance(end)
    
    if euclidean == 0: return 1.0
    return length / euclidean

def get_maxspeed(speed_val):
    """Parses maxspeed tag to float."""
    if isinstance(speed_val, list):
        speed_val = speed_val[0]
    
    try:
        if pd.isna(speed_val): return 40.0
        return float(str(speed_val).split()[0])
    except:
        return 40.0

def get_current_location():
    """
    Tries to get approximate location via IP.
    Fallback: Kunnamangalam, Kerala.
    """
    print("   Fetching approximate location via IP...")
    try:
        r = requests.get('https://ipinfo.io/json')
        if r.status_code == 200:
            data = r.json()
            if 'loc' in data:
                lat, lon = map(float, data['loc'].split(','))
                print(f"   Detected Location: {data.get('city', 'Unknown')} ({lat}, {lon})")
                return lat, lon
    except Exception as e:
        print(f"   Location fetch failed: {e}")
    
    print("   Using Default Fallback: Kunnamangalam")
    # Kunnamangalam Coordinates
    return 11.3067, 75.8767

def get_coordinates(place_name):
    """
    Geocodes a place name to (lat, lon).
    Handling 'Current Location' specially.
    """
    place_name = place_name.strip()
    
    if not place_name or place_name.lower() in ["current location", "here", "me"]:
        return get_current_location() # For analysis logic
        
    print(f"   Geocoding '{place_name}'...")
    try:
        # Use OSmnx's geocoder (Nominatim)
        lat, lon = ox.geocode(place_name)
        return lat, lon
    except Exception as e:
        print(f"   Error finding '{place_name}': {e}")
        return None

def main():
    print("1. Loading Real-Time Nervous System...")
    if not os.path.exists(MODEL_PATH):
        print("Error: Trained model not found.")
        return
    model = joblib.load(MODEL_PATH)
    
    weather = get_live_weather()
    time_ctx = get_temporal_features()
    print(f"   Context: {weather}, Night={time_ctx['is_night']}, Hour={time_ctx['hour_of_day']}")

    # --- User Interaction ---
    print("\n--- Plan Your Journey ---")
    origin_input = input("Enter Origin (Press Enter for Current Location): ").strip()
    dest_input = input("Enter Destination: ").strip()
    
    if not dest_input:
        print("Error: Destination is required.")
        return

    # 1. Resolve Coordinates for Analysis
    orig_coords = get_coordinates(origin_input)
    dest_coords = get_coordinates(dest_input)
    
    if not orig_coords or not dest_coords:
        print("Could not resolve locations. Try explicit names (e.g. 'Calicut Beach').")
        return
        
    orig_lat, orig_lon = orig_coords
    dest_lat, dest_lon = dest_coords

def analyze_route(origin_input, dest_input, model=None, G=None, user_location=None):
    """
    Analyzes the route between origin and destination.
    Returns a dictionary with route details, risks, and map link, or None if failed.
    """
    if not model:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            return {"error": "Model not found."}

    # 1. Resolve Coordinates
    # First, normalize the origin input to catch variations of "current location" safely
    norm_origin = str(origin_input).strip().lower() if origin_input else ""
    
    # If they passed accurate GPS coordinates and didn't type a specific city name, USE IT.
    if (not norm_origin or norm_origin in ["current location", "here", "me"]) and user_location is not None:
         print(f"   Using high-accuracy browser GPS: {user_location}")
         orig_lat, orig_lon = user_location
         # Use destination resolution as usual
         dest_coords = get_coordinates(dest_input)
         if not dest_coords:
             return {"error": "Could not resolve destination."}
         dest_lat, dest_lon = dest_coords
    else:
        # Standard Resolution
        orig_coords = get_coordinates(origin_input)
        dest_coords = get_coordinates(dest_input)
        
        if not orig_coords or not dest_coords:
            return {"error": "Could not resolve locations."}
            
        orig_lat, orig_lon = orig_coords
        dest_lat, dest_lon = dest_coords

    # 2. Get Graph
    if G is None:
        try:
            # optimize: cache this in the calling app
            G = ox.graph_from_place('Kozhikode, Kerala, India', network_type='drive')
        except Exception as e:
            return {"error": f"Graph download failed: {e}"}

    # 3. Enrich Graph (if not already enriched)
    # Check if a random edge has 'curvature_score' to see if enriched
    first_edge = list(G.edges(data=True))[0][2]
    if 'curvature_score' not in first_edge:
        node_degrees = dict(G.degree())
        for u, v, k, data in G.edges(keys=True, data=True):
            if 'geometry' in data:
                data['curvature_score'] = calculate_curvature(data['geometry'])
            else:
                data['curvature_score'] = 1.0
            data['maxspeed_clean'] = get_maxspeed(data.get('maxspeed', 40))
            is_junc = 1 if (node_degrees[u] > 2 or node_degrees[v] > 2) else 0
            data['is_junction'] = is_junc
            
            # Travel Time
            length_m = data.get('length', 10)
            speed_mps = max(data['maxspeed_clean'], 10.0) / 3.6
            data['travel_time'] = length_m / speed_mps

    # 4. Find Path
    orig_node = ox.distance.nearest_nodes(G, orig_lon, orig_lat)
    dest_node = ox.distance.nearest_nodes(G, dest_lon, dest_lat)
    
    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight='length')
    except nx.NetworkXNoPath:
        return {"error": "No route found."}

    # 5. Audit Risks
    weather = get_live_weather()
    time_ctx = get_temporal_features()
    
    segment_risks = []
    route_coords = []
    
    for i in range(len(route)):
        node_data = G.nodes[route[i]]
        route_coords.append((node_data['y'], node_data['x']))
        
        if i < len(route) - 1:
            u = route[i]
            v = route[i+1]
            edge_data = G.get_edge_data(u, v)[0]
            
            features = {
                'curvature_score': edge_data.get('curvature_score', 1.0),
                'maxspeed': edge_data.get('maxspeed_clean', 40.0),
                'is_junction': edge_data.get('is_junction', 0)
            }
            
            input_df = prepare_live_features(features, weather, time_ctx)
            prob = model.predict_proba(input_df)[0][1]
            
            name = edge_data.get('name', 'Unknown Road')
            if isinstance(name, list):
                name = name[0]
            
            node_data_v = G.nodes[v]
            segment_risks.append({
                'lat': node_data_v['y'],
                'lon': node_data_v['x'],
                'prob': prob,
                'features': features,
                'order': i,
                'name': name
            })

    # Top 5 Risks
    segment_risks.sort(key=lambda x: x['prob'], reverse=True)
    top_5 = segment_risks[:5]
    top_5.sort(key=lambda x: x['order'])
    
    # Overall Route Risk
    if segment_risks:
        avg_prob = sum(r['prob'] for r in segment_risks) / len(segment_risks)
        if avg_prob > 0.6:
            overall_risk = "High"
        elif avg_prob > 0.3:
            overall_risk = "Medium"
        else:
            overall_risk = "Low"
    else:
        overall_risk = "Unknown"
        avg_prob = 0.0

    # URL Construction
    # Use resolved coordinates for Origin to match analysis exactly
    url_origin = f"{orig_lat},{orig_lon}"
    url_dest = f"{dest_lat},{dest_lon}"
    waypoints = [f"{risk['lat']},{risk['lon']}" for risk in top_5]
    wp_str = "%7C".join(waypoints)
    maps_url = f"https://www.google.com/maps/dir/?api=1&origin={url_origin}&destination={url_dest}&waypoints={wp_str}"
    
    return {
        "route_nodes": route,
        "route_coords": route_coords,
        "top_5_risks": top_5,
        "overall_risk": overall_risk,
        "avg_risk_prob": float(avg_prob),
        "maps_url": maps_url,
        "weather": weather,
        "time_ctx": time_ctx,
        "G": G  # Return graph in case it was created here
    }

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth surface.
    Returns distance in meters.
    """
    import math
    R = 6371000  # Earth radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def main():
    print("1. Loading Real-Time Nervous System...")
    # Loading model strictly for CLI main check; analyze_route handles it too but efficient to pass it if we had it.
    if not os.path.exists(MODEL_PATH):
        print("Error: Trained model not found.")
        return
    model = joblib.load(MODEL_PATH)
    
    # Just printing context for CLI user
    weather = get_live_weather()
    time_ctx = get_temporal_features()
    print(f"   Context: {weather}, Night={time_ctx['is_night']}, Hour={time_ctx['hour_of_day']}")

    # --- User Interaction ---
    print("\n--- Plan Your Journey ---")
    origin_input = input("Enter Origin (Press Enter for Current Location): ").strip()
    dest_input = input("Enter Destination: ").strip()
    
    if not dest_input:
        print("Error: Destination is required.")
        return

    print("\n2. Analyzing Route...")
    result = analyze_route(origin_input, dest_input, model=model)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
        
    print(f"   Route found with {len(result['route_nodes'])-1} segments.")

    print("\n" + "="*40)
    print("TOP 5 MOST DANGEROUS POINTS ON ROUTE")
    print("="*40)
    
    for i, risk in enumerate(result['top_5_risks']):
        risk_level = 'High' if risk['prob']>0.7 else 'Moderate' if risk['prob']>0.4 else 'Low'
        print(f"{i+1}. Risk Level: {risk_level} | Curv: {risk['features']['curvature_score']:.2f}")
        
    print("\n" + "="*40)
    print("NAVIGATION LINK (Safe Route)")
    print("="*40)
    print(result['maps_url'])
    print("="*40)

if __name__ == "__main__":
    main()
