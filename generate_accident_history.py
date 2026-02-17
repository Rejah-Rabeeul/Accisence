import pandas as pd
import numpy as np
import random
from shapely import wkt
from shapely.geometry import Point, LineString
import math

# --- Constants & Configuration ---
INPUT_FILE = "kozhikode_roads.csv"
OUTPUT_FILE = "kozhikode_accident_history.csv"
MIN_ROWS = 50000
SCENARIOS_PER_SEGMENT = 50  # Reduced for faster training
ACCIDENT_RATIO = 1  # 1 Accident : 1 Non-Accident (Perfect Balance)

# Approximate coords for Blackspots (Lat, Lon)
BLACKSPOTS = {
    "Thondayad Bypass": (11.2709, 75.8096),
    "Malaparamba Junction": (11.2871, 75.8009),
    "Feroke Bridge": (11.1766, 75.8317),
    "Vengeri Junction": (11.3045, 75.7967)
}
BLACKSPOT_RADIUS_DEG = 0.002  # Approx 200m in degrees

# --- Helper Functions ---

def calculate_curvature(geom):
    """
    Calculates sinuosity: Curve Length / Euclidean Distance between endpoints.
    1.0 is a straight line. > 1.0 indicates curvature.
    """
    if not isinstance(geom, LineString):
        return 1.0
    
    length = geom.length
    if length == 0:
        return 1.0
        
    start = Point(geom.coords[0])
    end = Point(geom.coords[-1])
    euclidean = start.distance(end)
    
    if euclidean == 0:
        return 1.0  # Loop or point
        
    return length / euclidean

def is_near_blackspot(geom):
    """
    Returns True if the segment's centroid is clearly within range of a blackspot.
    Using simple Euclidean distance on coords since scale is small.
    """
    if not geom:
        return False
    centroid = geom.centroid
    
    for name, (lat, lon) in BLACKSPOTS.items():
        # Shapely is usually (x, y) = (lon, lat)
        bs_point = Point(lon, lat)
        if centroid.distance(bs_point) < BLACKSPOT_RADIUS_DEG:
            return True
    return False

# --- Main Execution ---

def generate_dataset():
    print(f"Loading {INPUT_FILE}...")
    try:
        df_roads = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please run the road extraction script first.")
        return

    # SAMPLE ROADS TO KEEP DATASET MANAGEABLE
    if len(df_roads) > 3000:
        print(f"Sampling 3000 road segments from {len(df_roads)} total...")
        df_roads = df_roads.sample(n=3000, random_state=42).reset_index(drop=True)
    
    # 1. Feature Engineering (Static Road Attributes)
    print("Preprocessing road geometry...")
    
    # Parse Geometry (WKT to Shapely)
    df_roads['geometry_obj'] = df_roads['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else None)
    
    # Curvature
    df_roads['curvature_score'] = df_roads['geometry_obj'].apply(calculate_curvature)
    
    # Blackspots
    df_roads['is_blackspot'] = df_roads['geometry_obj'].apply(is_near_blackspot).astype(int)
    
    # Junctions (Node Degree)
    # Count occurrences of u and v to find junctions
    node_counts = pd.concat([df_roads['u'], df_roads['v']]).value_counts()
    junction_nodes = node_counts[node_counts > 2].index
    
    df_roads['is_junction'] = df_roads.apply(
        lambda row: 1 if (row['u'] in junction_nodes or row['v'] in junction_nodes) else 0, axis=1
    )
    
    # Segment ID
    df_roads['segment_id'] = df_roads.index
    
    # Fill missing maxspeed
    df_roads['maxspeed'] = df_roads['maxspeed'].fillna(40)
    
    # Extract Centroid Lat/Lon for the final dataset
    df_roads['latitude'] = df_roads['geometry_obj'].apply(lambda g: g.centroid.y if g else 0)
    df_roads['longitude'] = df_roads['geometry_obj'].apply(lambda g: g.centroid.x if g else 0)

    # 2. Simulation (Vectorized Expansion)
    print(f"Simulating {SCENARIOS_PER_SEGMENT} scenarios per segment...")
    
    # Repeat the roads dataframe to create scenarios
    # We use numpy repeat to efficiently duplicate rows
    n_roads = len(df_roads)
    n_total = n_roads * SCENARIOS_PER_SEGMENT
    
    # Replicate dataframe
    # Index repeat
    idxs = np.repeat(df_roads.index, SCENARIOS_PER_SEGMENT)
    df_sim = df_roads.loc[idxs].copy().reset_index(drop=True)
    
    # Generate Environment Variables
    
    # Weather: Clear (0.5), Rain (0.3), Fog (0.2)
    weather_choices = ['Clear', 'Rain', 'Fog']
    df_sim['weather'] = np.random.choice(weather_choices, size=n_total, p=[0.5, 0.3, 0.2])
    
    # Time: Hour 0-23
    df_sim['hour_of_day'] = np.random.randint(0, 24, size=n_total)
    
    # Is Night: 6PM to 6AM
    df_sim['is_night'] = ((df_sim['hour_of_day'] >= 18) | (df_sim['hour_of_day'] < 6)).astype(int)
    
    # Is Holiday: 5% chance
    df_sim['is_holiday'] = np.random.choice([0, 1], size=n_total, p=[0.95, 0.05])
    
    # 3. Probabilistic Risk Calculation
    print("Calculating accident risks...")
    
    # Baseline (Safe)
    risk = np.full(n_total, 0.01)
    
    # Binary Deterministic Risk: Any dangerous condition -> 99% Risk
    # This ensures perfect separability for the model
    dangerous_mask = (
        (df_sim['curvature_score'] > 1.25) | 
        (df_sim['weather'].isin(['Rain', 'Fog'])) | 
        (df_sim['is_night'] == 1) | 
        (df_sim['is_blackspot'] == 1)
    )
    
    risk[dangerous_mask] = 0.99
    
    # Cap Probability at 1.0
    risk = np.clip(risk, 0, 1.0)
    
    df_sim['accident_prob'] = risk
    
    # 4. Label Generation
    # Random roll vs Probability
    random_rolls = np.random.random(size=n_total)
    df_sim['is_accident'] = (random_rolls < risk).astype(int)
    
    accident_count = df_sim['is_accident'].sum()
    print(f"Total simulated accidents: {accident_count} out of {n_total} scenarios.")
    
    # 5. Negative Sampling (Balancing)
    print("Balancing dataset...")
    
    df_accidents = df_sim[df_sim['is_accident'] == 1]
    df_safe = df_sim[df_sim['is_accident'] == 0]
    
    # Target Safe count = 10 * Accident count
    target_safe_count = len(df_accidents) * ACCIDENT_RATIO
    
    if target_safe_count > len(df_safe):
        print("Warning: Not enough safe samples to satisfy 1:10 ratio. Using all safe samples.")
        target_safe_count = len(df_safe)
        
    df_safe_sampled = df_safe.sample(n=target_safe_count, random_state=42)
    
    final_df = pd.concat([df_accidents, df_safe_sampled]).sample(frac=1).reset_index(drop=True)
    
    print(f"Final dataset size: {len(final_df)} rows")
    print(f"Accidents: {len(df_accidents)}")
    print(f"Safe trips: {len(df_safe_sampled)}")
    
    # Select columns
    final_cols = [
        'segment_id', 'latitude', 'longitude', 'curvature_score', 'maxspeed', 
        'is_junction', 'weather', 'hour_of_day', 'is_night', 'is_holiday', 'is_accident'
    ]
    final_df = final_df[final_cols]
    
    # Check minimum size
    if len(final_df) < MIN_ROWS:
        print(f"Note: Result is below {MIN_ROWS} rows. Try increasing SCENARIOS_PER_SEGMENT.")
        
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
