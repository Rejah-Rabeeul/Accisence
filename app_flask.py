from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import joblib
import networkx as nx
import osmnx as ox
from risk_aware_navigation import analyze_route, get_coordinates
from realtime_inference_utils import get_live_weather, get_temporal_features

app = Flask(__name__)

# Basic CORS implementation to explicitly allow Pinggy traffic
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

MODEL_PATH = "accident_model.pkl"
GRAPH_PATH = "kozhikode_graph.graphml" # Hypothetical cache for faster loading

# --- Global Resources ---
model = None
G = None

def load_resources():
    global model, G
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    
    # Load or download graph
    try:
        # For production, we'd use a local file, but for now we'll use the ox method
        # or load from a pre-saved graphml if it exists to be fast.
        G = ox.graph_from_place('Kozhikode, Kerala, India', network_type='drive')
    except Exception as e:
        print(f"Error loading graph: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sw.js')
def service_worker():
    return send_from_directory('.', 'sw.js', mimetype='application/javascript')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    origin = data.get('origin', '')
    destination = data.get('destination', '')
    user_lat = data.get('lat')
    user_lon = data.get('lon')
    
    user_coords = (float(user_lat), float(user_lon)) if user_lat and user_lon else None
    
    # Use the existing analyze_route function which we've already optimized for shortest path
    result = analyze_route(origin, destination, model=model, G=G, user_location=user_coords)
    
    if "error" in result:
        return jsonify(result), 400
    
    # Deep-remove non-serializable objects (like the Graph G)
    # and convert numpy floats/ints to native Python types
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items() if 'graph' not in str(type(v)).lower()}
        elif isinstance(d, list):
            return [clean_dict(x) for x in d]
        elif hasattr(d, 'item'): # Catches numpy float32/int64
            return d.item()
        return d

    clean_result = clean_dict(result)
        
    return jsonify(clean_result)

if __name__ == '__main__':
    print("Loading AI Nervous System...")
    load_resources()
    # Debug = False is critical here. Flask's Werkzeug debugger violently rejects
    # external Host headers when proxying through services like Pinggy or Ngrok 
    # as a security measure against DNS rebinding. 
    app.run(host='0.0.0.0', port=5000, debug=False)
