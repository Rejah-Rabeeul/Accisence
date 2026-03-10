import osmnx as ox
import networkx as nx

print("Downloading graph for Kozhikode...")
# Use simplify=True to reduce memory footprint
G = ox.graph_from_place('Kozhikode, Kerala, India', network_type='drive', simplify=True)
print("Saving graph to kozhikode_graph.graphml...")
ox.save_graphml(G, "kozhikode_graph.graphml")
print("Done!")
