import pandas as pd
import networkx as nx
import math
from shapely import wkt
import os

def load_graph_from_csv(csv_path="kozhikode_roads.csv"):
    """
    Loads the street network into a NetworkX graph directly from the CSV
    to save memory, bypassing OSMnx entirely.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing routing graph data: {csv_path}")

    print(f"Loading lightweight routing graph from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    G = nx.MultiDiGraph()
    
    # We need to construct node data manually so we can do nearest neighbor lookups later.
    # The CSV gives us LineStrings in the 'geometry' column. 
    # We can extract node coordinates from the start/endpoints of these LineStrings.
    
    nodes_added = set()

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
            geom = wkt.loads(row['geometry'])
            data['geometry'] = geom
            
            # Since the original CSV drops node locations (only keeps edges), 
            # we recreate node coordinate lookups from the start/end points of the edge geometry!
            if u not in nodes_added:
                G.add_node(u, x=geom.coords[0][0], y=geom.coords[0][1])
                nodes_added.add(u)
            if v not in nodes_added:
                G.add_node(v, x=geom.coords[-1][0], y=geom.coords[-1][1])
                nodes_added.add(v)
            
        G.add_edge(u, v, 0, **data)
        
    print(f"Created NetworkX graph: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    return G

def haversine(lon1, lat1, lon2, lat2):
    import math
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371000 # Radius of earth in meters
    return c * r

def nearest_nodes(G, point, return_dist=False):
    """
    Very fast brute-force nearest neighbor lookup to replace osmnx.distance.nearest_nodes.
    G: networkx Graph
    point: tuple of (longitude, latitude)
    """
    nearest_node = None
    min_dist = float('inf')
    lon, lat = point
    
    for n, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            dist = haversine(lon, lat, data['x'], data['y'])
            if dist < min_dist:
                min_dist = dist
                nearest_node = n
                
    if return_dist:
        return nearest_node, min_dist
    return nearest_node
