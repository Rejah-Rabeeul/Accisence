import osmnx as ox
import pandas as pd
import numpy as np

def clean_value(val):
    """
    OSM data often contains lists where it should be a single value 
    (e.g., maxspeed or lanes). This helper returns the first element or a default.
    """
    if isinstance(val, list):
        return val[0]
    return val

def extract_kozhikode_roads():
    print("Connecting to OpenStreetMap to download Kozhikode road network...")
    # 1. Define Area & Filter Network
    place_name = "Kozhikode, Kerala, India"
    # Download the 'drive' network
    G = ox.graph_from_place(place_name, network_type='drive')
    
    print("Processing edges into a table...")
    # 2. Convert to GeoDataFrame
    # nodes, edges = ox.graph_to_gdfs(G)
    # We only need edges for the road shapes
    edges = ox.graph_to_gdfs(G, nodes=False)
    
    # 3. Process Columns as requested
    # OSMnx edges index is usually (u, v, key)
    # Reset index to get u and v as columns
    df = edges.reset_index()
    
    # Required columns: u, v, highway, length, maxspeed, geometry, lanes
    target_columns = ['u', 'v', 'highway', 'length', 'maxspeed', 'geometry', 'lanes']
    
    # Ensure all target columns exist (some might be missing in certain regions)
    for col in target_columns:
        if col not in df.columns:
            df[col] = np.nan
            
    # Keep only target columns
    df = df[target_columns].copy()
    
    # Clean data (handle lists in maxspeed and lanes)
    print("Cleaning data (handling lists and types)...")
    df['maxspeed'] = df['maxspeed'].apply(clean_value)
    df['lanes'] = df['lanes'].apply(clean_value)
    
    # Convert maxspeed to Int (strip ' km/h' if present)
    def parse_speed(val):
        if pd.isna(val): return np.nan
        try:
            return int(str(val).split(' ')[0])
        except (ValueError, IndexError):
            return np.nan

    df['maxspeed'] = df['maxspeed'].apply(parse_speed)
    
    # Convert lanes to Int
    df['lanes'] = pd.to_numeric(df['lanes'], errors='coerce')
    
    # Convert length to float (should already be float)
    df['length'] = pd.to_numeric(df['length'], errors='coerce')
    
    # 4. Export to CSV
    output_file = "kozhikode_roads.csv"
    print(f"Saving {len(df)} road segments to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Extraction complete!")

if __name__ == "__main__":
    extract_kozhikode_roads()
