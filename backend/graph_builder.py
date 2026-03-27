import os
import time
import osmnx as ox
import networkx as nx
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from dotenv import load_dotenv

# =====================================================
# CONFIG
# =====================================================

load_dotenv()

PLACE_NAME = "Wayanad, India"
K = 4                       # nearest neighbors
MAX_TRAVEL_TIME_MIN = 60
CHUNK_SIZE = 50             # ORS matrix limit per request
PAIR_BATCH = 25             # cross-chunk pairs per request (25 pairs = 50 nodes max)
RATE_LIMIT_SLEEP = 1.5      # seconds between ORS calls (free tier: 40 req/min)

ORS_API_KEY = os.getenv("OPENROUTESERVICE_API_KEY")
ORS_MATRIX_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"

if not ORS_API_KEY:
    raise Exception("Set OPENROUTESERVICE_API_KEY in .env file")

# =====================================================
# STEP 1: FETCH TOURIST PLACES (OSM)
# =====================================================

tags = {
    "tourism": ["attraction", "museum", "viewpoint", "theme_park", "zoo"]
}

gdf = ox.features_from_place(PLACE_NAME, tags=tags)

# Keep only point geometries with names
gdf = gdf[gdf.geometry.type == "Point"]
gdf = gdf[gdf["name"].notna()]
gdf = gdf.reset_index(drop=True)

print(f"Fetched {len(gdf)} tourist places")

# =====================================================
# STEP 2: CREATE GRAPH NODES
# =====================================================

G = nx.Graph()

for i, (_, row) in enumerate(gdf.iterrows()):
    G.add_node(
        i,
        name=row["name"],
        lat=row.geometry.y,
        lon=row.geometry.x,
        category=row.get("tourism")
    )

# =====================================================
# STEP 3: KNN PRUNING (avoid n²)
# =====================================================

nodes = list(G.nodes(data=True))

coords = np.array([
    [data["lat"], data["lon"]]
    for _, data in nodes
])

tree = BallTree(np.radians(coords), metric="haversine")
distances, indices = tree.query(np.radians(coords), k=K + 1)

candidate_edges = set()

for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:
        u = nodes[i][0]
        v = nodes[j][0]
        candidate_edges.add(tuple(sorted((u, v))))

print(f"Candidate edges after KNN: {len(candidate_edges)}")

# =====================================================
# STEP 4: ORS MATRIX — CHUNKED (fixes 400 error)
# =====================================================
# ORS matrix API supports max 50 locations per request.
# Strategy:
#   - Extract only unique nodes used in candidate edges
#   - Split into chunks of 50
#   - Same-chunk pairs: reuse chunk matrix result
#   - Cross-chunk pairs: build tiny combined matrices (25 pairs max)

def ors_matrix_chunk(coord_subset):
    """Call ORS matrix for a list of (lat, lon) tuples. Max 50 items."""
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "locations": [[lon, lat] for lat, lon in coord_subset],
        "metrics": ["distance", "duration"]
    }
    r = requests.post(ORS_MATRIX_URL, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


# Only query nodes that actually appear in candidate edges
involved_nodes = sorted(set(n for edge in candidate_edges for n in edge))
print(f"Unique nodes in candidate edges: {len(involved_nodes)}")

# Chunk into groups of CHUNK_SIZE
chunks = [
    involved_nodes[i:i + CHUNK_SIZE]
    for i in range(0, len(involved_nodes), CHUNK_SIZE)
]
print(f"ORS matrix chunks needed: {len(chunks)}")

# Map each node to its chunk index
chunk_assignments = {}
for orig_idx in involved_nodes:
    for c_i, chunk in enumerate(chunks):
        if orig_idx in chunk:
            chunk_assignments[orig_idx] = c_i
            break

# Separate same-chunk and cross-chunk edges
same_chunk_edges = []
cross_chunk_edges = []

for u, v in candidate_edges:
    if chunk_assignments[u] == chunk_assignments[v]:
        same_chunk_edges.append((u, v))
    else:
        cross_chunk_edges.append((u, v))

print(f"Same-chunk edges: {len(same_chunk_edges)} | Cross-chunk edges: {len(cross_chunk_edges)}")

# Storage for edge results: (u, v) → (dist_km, dur_min)
edge_data = {}

# --- Query same-chunk edges using full chunk matrix ---
chunk_matrices = {}

for c_i, chunk in enumerate(chunks):
    chunk_coords = [coords[n] for n in chunk]
    try:
        result = ors_matrix_chunk(chunk_coords)
        local_map = {orig: pos for pos, orig in enumerate(chunk)}
        chunk_matrices[c_i] = {
            "dist": result["distances"],
            "time": result["durations"],
            "local_map": local_map
        }
        print(f"  Chunk {c_i} ({len(chunk)} nodes) → OK")
    except requests.exceptions.HTTPError as e:
        print(f"  Chunk {c_i} failed: {e}")
        chunk_matrices[c_i] = None

    time.sleep(RATE_LIMIT_SLEEP)

for u, v in same_chunk_edges:
    c_i = chunk_assignments[u]
    m = chunk_matrices.get(c_i)
    if m is None:
        continue
    lu = m["local_map"][u]
    lv = m["local_map"][v]
    dist_km  = m["dist"][lu][lv] / 1000
    dur_min  = m["time"][lu][lv] / 60
    edge_data[(u, v)] = (dist_km, dur_min)

# --- Query cross-chunk edges in batches of PAIR_BATCH ---
def batch_cross_chunk(pairs):
    for i in range(0, len(pairs), PAIR_BATCH):
        batch = pairs[i:i + PAIR_BATCH]

        # Deduplicate nodes while preserving order
        seen = {}
        for p in batch:
            for n in p:
                if n not in seen:
                    seen[n] = len(seen)
        unique_in_batch = list(seen.keys())

        batch_coords = [coords[n] for n in unique_in_batch]
        local = {orig: pos for pos, orig in enumerate(unique_in_batch)}

        try:
            result = ors_matrix_chunk(batch_coords)
            for u, v in batch:
                lu, lv = local[u], local[v]
                dist_km = result["distances"][lu][lv] / 1000
                dur_min = result["durations"][lu][lv] / 60
                edge_data[(u, v)] = (dist_km, dur_min)
            print(f"  Cross-chunk batch {i // PAIR_BATCH} ({len(batch)} pairs) → OK")
        except requests.exceptions.HTTPError as e:
            print(f"  Cross-chunk batch {i // PAIR_BATCH} failed: {e}")

        time.sleep(RATE_LIMIT_SLEEP)

if cross_chunk_edges:
    batch_cross_chunk(cross_chunk_edges)

print(f"Edge data collected: {len(edge_data)} pairs")

# =====================================================
# STEP 5: BUILD GRAPH EDGES WITH REAL ROAD DATA
# =====================================================

for (u, v), (distance_km, duration_min) in edge_data.items():
    if duration_min <= MAX_TRAVEL_TIME_MIN:
        G.add_edge(
            u,
            v,
            road_distance_km=round(distance_km, 2),
            duration_min=round(duration_min, 1),
            edge_type="road"
        )

print(f"Final graph edges: {G.number_of_edges()}")
print(f"Final graph nodes: {G.number_of_nodes()}")

# =====================================================
# STEP 6: VISUALIZE GRAPH (GEO-AWARE)
# =====================================================

pos = {
    n: (data["lon"], data["lat"])
    for n, data in G.nodes(data=True)
}

labels = nx.get_node_attributes(G, "name")
edge_labels = {
    (u, v): f"{d['duration_min']}m"
    for u, v, d in G.edges(data=True)
}

plt.figure(figsize=(10, 10))
nx.draw(
    G,
    pos,
    node_size=40,
    node_color="steelblue",
    edge_color="gray",
    alpha=0.7,
    width=0.8
)
nx.draw_networkx_labels(
    G,
    pos,
    labels=labels,
    font_size=6,
    font_color="black"
)
plt.title(f"Tourist Place Graph — {PLACE_NAME} (ORS road distances)")
plt.axis("off")
plt.tight_layout()
plt.savefig("tourist_place_graph.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graph visualization saved as tourist_place_graph.png")

# =====================================================
# STEP 7: SAVE GRAPH
# =====================================================

nx.write_graphml(G, "tourist_place_graph.graphml")
print("Graph saved as tourist_place_graph.graphml")

# Print a summary
print("\n--- Summary ---")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
if G.number_of_edges() > 0:
    durations = [d["duration_min"] for _, _, d in G.edges(data=True)]
    print(f"Avg travel time: {np.mean(durations):.1f} min")
    print(f"Max travel time: {np.max(durations):.1f} min")