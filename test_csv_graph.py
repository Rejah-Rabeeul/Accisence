import pandas as pd
import networkx as nx
from shapely import wkt
import os

def load_graph_from_csv(csv_path="kozhikode_roads.csv"):
    """
    Loads the street network into a NetworkX graph directly from the CSV
    to save memory, bypassing OSMnx entirely.
    """
    print(f"Loading graph from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    G = nx.MultiDiGraph()
    
    for _, row in df.iterrows():
        u = row['u']
        v = row['v']
        
        # Build edge data dictionary
        data = {
            'length': float(row['length']),
            'highway': row['highway']
        }
        
        if pd.notna(row['maxspeed']):
            data['maxspeed'] = str(row['maxspeed'])
            
        if pd.notna(row['lanes']):
            data['lanes'] = str(row['lanes'])
            
        if pd.notna(row['geometry']):
            data['geometry'] = wkt.loads(row['geometry'])
            
        G.add_edge(u, v, 0, **data)
        
    return G

if __name__ == "__main__":
    import tracemalloc
    tracemalloc.start()
    
    G = load_graph_from_csv()
    print(f"Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
