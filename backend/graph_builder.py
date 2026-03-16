"""
graph_builder.py
================
Pipeline: Geocode → OSM places → KNN + ORS graph → Save to DB

Terminal output shows every step with progress, counts, and timing.
"""
import os, re, uuid, traceback, argparse, time
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from typing import Optional

import networkx as nx
import numpy as np
import requests
from sklearn.neighbors import BallTree

from database_models import (
    Region, Place, PlaceGraph, PlaceEdge,
    PlaceCategory, GraphStatus, get_session_maker,
)

# ── Config ─────────────────────────────────────────────────────────────────────
ORS_API_KEY    = os.getenv("OPENROUTESERVICE_API_KEY", "")
ORS_MATRIX_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"
OVERPASS_API   = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL  = "https://nominatim.openstreetmap.org/search"

K_NEIGHBORS    = 5
MAX_TRAVEL_MIN = 90
MAX_PLACES     = 40
RADIUS_M       = 15000
PROXIMITY_M    = 500

_CAT_MAP = {
    "attraction":       PlaceCategory.ATTRACTION,
    "museum":           PlaceCategory.MUSEUM,
    "viewpoint":        PlaceCategory.VIEWPOINT,
    "theme_park":       PlaceCategory.THEME_PARK,
    "zoo":              PlaceCategory.ZOO,
    "historic":         PlaceCategory.HISTORICAL,
    "place_of_worship": PlaceCategory.RELIGIOUS,
    "park":             PlaceCategory.PARK,
    "waterfall":        PlaceCategory.ATTRACTION,
    "peak":             PlaceCategory.VIEWPOINT,
}

def _id(): return str(uuid.uuid4())

def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower().strip())

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    return _haversine_m(lat1, lon1, lat2, lon2) / 1000.0

# ── Pretty print helpers ───────────────────────────────────────────────────────
def _sep(char="─", width=60):
    print(char * width)

def _step(n, total, label):
    print(f"\n  [{n}/{total}] {label}")

def _ok(msg):   print(f"  ✅ {msg}")
def _warn(msg): print(f"  ⚠️  {msg}")
def _info(msg): print(f"  ℹ️  {msg}")
def _err(msg):  print(f"  ❌ {msg}")

def _progress_bar(current, total, width=30, label=""):
    filled = int(width * current / total) if total else 0
    bar    = "█" * filled + "░" * (width - filled)
    pct    = int(100 * current / total) if total else 0
    print(f"\r  [{bar}] {pct:3d}%  {label}", end="", flush=True)

def _elapsed(t0): return f"{time.time()-t0:.1f}s"


# ── Step 1: Geocode ────────────────────────────────────────────────────────────
def _geocode(dest: str) -> Optional[dict]:
    print(f"  → Querying Nominatim for: {dest!r}")
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"q": dest, "format": "json", "limit": 1},
            headers={"User-Agent": "TravelCopilot/2.0"},
            timeout=10,
        )
        d = r.json()
        if d:
            lat, lon = float(d[0]["lat"]), float(d[0]["lon"])
            print(f"  → Found: {d[0].get('display_name','')[:60]}")
            print(f"  → Coordinates: lat={lat:.5f}, lon={lon:.5f}")
            return {"lat": lat, "lon": lon}
        _warn("Nominatim returned no results")
    except Exception as e:
        _err(f"Geocode error: {e}")
    return None


# ── Step 2: Fetch + deduplicate OSM places ─────────────────────────────────────
def _deduplicate_places(raw: list) -> list:
    kept = []
    removed = []
    for candidate in raw:
        c_norm = _normalise(candidate["name"])
        is_dup = False
        for existing in kept:
            if _normalise(existing["name"]) == c_norm:
                dist = _haversine_m(
                    candidate["lat"], candidate["lon"],
                    existing["lat"],  existing["lon"],
                )
                if dist <= PROXIMITY_M:
                    removed.append((candidate["name"], existing["name"], dist))
                    is_dup = True
                    break
        if not is_dup:
            kept.append(candidate)
    if removed:
        print(f"  → Removed {len(removed)} duplicate(s):")
        for dup_name, kept_name, dist in removed:
            print(f"      '{dup_name}' ≈ '{kept_name}' ({dist:.0f}m)")
    return kept


def _fetch_osm(lat: float, lon: float) -> list:
    t0 = time.time()
    print(f"  → Search radius: {RADIUS_M/1000:.1f}km around ({lat:.4f}, {lon:.4f})")
    print(f"  → Categories: attraction, museum, viewpoint, theme_park, zoo,")
    print(f"                historic, waterfall, peak, park, place_of_worship")

    q = f"""[out:json][timeout:35];(
      node["tourism"="attraction"](around:{RADIUS_M},{lat},{lon});
      node["tourism"="museum"](around:{RADIUS_M},{lat},{lon});
      node["tourism"="viewpoint"](around:{RADIUS_M},{lat},{lon});
      node["tourism"="theme_park"](around:{RADIUS_M},{lat},{lon});
      node["tourism"="zoo"](around:{RADIUS_M},{lat},{lon});
      node["historic"]["name"](around:{RADIUS_M},{lat},{lon});
      node["natural"="waterfall"]["name"](around:{RADIUS_M},{lat},{lon});
      node["natural"="peak"]["name"](around:{RADIUS_M},{lat},{lon});
      node["leisure"="park"]["name"](around:{RADIUS_M},{lat},{lon});
      node["amenity"="place_of_worship"]["name"](around:{RADIUS_M},{lat},{lon});
    );out body;"""
    try:
        print(f"  → Sending Overpass query...")
        r = requests.post(OVERPASS_API, data={"data": q}, timeout=40)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        print(f"  → Overpass returned {len(elements)} raw elements ({_elapsed(t0)})")

        raw = []
        cat_counts: dict = {}
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name", "").strip()
            if not name:
                continue
            cat = (
                tags.get("tourism")
                or ("historic" if "historic" in tags else None)
                or tags.get("natural")
                or tags.get("leisure")
                or tags.get("amenity")
                or "attraction"
            )
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            raw.append({
                "name":     name,
                "lat":      float(el["lat"]),
                "lon":      float(el["lon"]),
                "category": cat,
                "osm_id":   str(el.get("id", "")),
            })

        print(f"  → Named places: {len(raw)}")
        print(f"  → By category: " + ", ".join(f"{k}={v}" for k,v in sorted(cat_counts.items())))

        print(f"  → Deduplicating (same name within {PROXIMITY_M}m)...")
        places = _deduplicate_places(raw)
        places = places[:MAX_PLACES]

        print(f"  → Final place list ({len(places)}):")
        for i, p in enumerate(places, 1):
            print(f"      {i:2d}. {p['name']:<35} [{p['category']}]  ({p['lat']:.4f}, {p['lon']:.4f})")

        return places

    except Exception as e:
        _err(f"Overpass error: {e}")
        return []


# ── Step 3: Build NetworkX graph ───────────────────────────────────────────────
def _ors_matrix(coords: list) -> Optional[tuple]:
    t0 = time.time()
    print(f"  → Sending {len(coords)}×{len(coords)} matrix request to OpenRouteService...")
    try:
        r = requests.post(
            ORS_MATRIX_URL,
            json={"locations": [[lon, lat] for lat, lon in coords],
                  "metrics": ["distance", "duration"]},
            headers={"Authorization": ORS_API_KEY, "Content-Type": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()
        print(f"  → ORS matrix received ({_elapsed(t0)})")
        return d["distances"], d["durations"]
    except Exception as e:
        _warn(f"ORS error — falling back to haversine: {e}")
        return None


def _build_nx_graph(places: list) -> nx.Graph:
    t0 = time.time()
    n  = len(places)
    print(f"  → Adding {n} nodes to graph...")
    G = nx.Graph()
    for i, p in enumerate(places):
        G.add_node(i, name=p["name"], lat=p["lat"], lon=p["lon"],
                   category=p["category"], osm_id=p["osm_id"])

    if n < 2:
        _warn("Too few places to build edges")
        return G

    coords = np.array([[p["lat"], p["lon"]] for p in places])
    k      = min(K_NEIGHBORS + 1, n)

    print(f"  → Running KNN (k={K_NEIGHBORS}) with BallTree haversine metric...")
    _, indices = BallTree(np.radians(coords), metric="haversine").query(np.radians(coords), k=k)

    candidates = set()
    for i, nbrs in enumerate(indices):
        for j in nbrs[1:]:
            candidates.add(tuple(sorted((int(i), int(j)))))
    print(f"  → KNN candidate edges: {len(candidates)}")

    # Road distances
    if ORS_API_KEY:
        ors = _ors_matrix(coords.tolist())
    else:
        _warn("OPENROUTESERVICE_API_KEY not set — using haversine distances")
        ors = None

    edge_count = 0
    skipped    = 0
    road_count = 0
    est_count  = 0

    for u, v in candidates:
        if ors:
            dist_km = ors[0][u][v] / 1000.0
            dur_min = ors[1][u][v] / 60.0
            etype   = "road"
            road_count += 1
        else:
            dist_km = _haversine_km(*coords[u], *coords[v])
            dur_min = (dist_km / 40.0) * 60.0
            etype   = "estimated"
            est_count += 1

        if dur_min <= MAX_TRAVEL_MIN:
            G.add_edge(u, v, road_distance_km=round(dist_km, 3),
                       duration_min=round(dur_min, 1), edge_type=etype)
            edge_count += 1
        else:
            skipped += 1

    print(f"  → Edges added:   {edge_count} "
          f"({'road' if road_count else 'estimated'})")
    if skipped:
        print(f"  → Edges skipped: {skipped} (>{MAX_TRAVEL_MIN} min travel time)")

    # Print a sample of edges
    sample_edges = list(G.edges(data=True))[:5]
    if sample_edges:
        print(f"  → Sample connections:")
        for u, v, d in sample_edges:
            print(f"      {places[u]['name']:<28} → {places[v]['name']:<28} "
                  f"{d['duration_min']:.0f}min  {d['road_distance_km']:.1f}km  [{d['edge_type']}]")

    print(f"  → Graph built in {_elapsed(t0)}: "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── Step 4: Save to DB ─────────────────────────────────────────────────────────
def _save_to_db(G: nx.Graph, places: list,
                region_name: str, state: Optional[str], country: str) -> dict:
    t0 = time.time()
    S  = get_session_maker()
    with S() as s:
        try:
            # ── Region ────────────────────────────────────────────────────
            print(f"  → Looking up region '{region_name}' in DB...")
            region = s.query(Region).filter(
                Region.name.ilike(region_name),
                Region.country.ilike(country),
            ).first()

            if not region:
                lats   = [p["lat"] for p in places]
                lons   = [p["lon"] for p in places]
                region = Region(
                    id=_id(), name=region_name, state=state, country=country,
                    minLat=min(lats), maxLat=max(lats),
                    minLon=min(lons), maxLon=max(lons),
                    createdAt=datetime.utcnow(), updatedAt=datetime.utcnow(),
                )
                s.add(region); s.flush()
                print(f"  → Created new region: {region.name} (id={region.id[:8]}…)")
            else:
                print(f"  → Reusing existing region: {region.name} (id={region.id[:8]}…)")

            # ── Load existing places ───────────────────────────────────────
            existing_db_places = s.query(Place).filter_by(regionId=region.id).all()
            existing_osm_ids   = {p.osmId for p in existing_db_places if p.osmId}
            print(f"  → Existing places in DB for this region: {len(existing_db_places)}")

            # ── Map nodes → place ids ──────────────────────────────────────
            node_to_place_id: dict = {}
            new_places   = []
            reused_places = []

            for ni, nd in G.nodes(data=True):
                osm_id  = nd.get("osm_id", "")
                name    = nd["name"]
                n_lat   = float(nd["lat"])
                n_lon   = float(nd["lon"])
                n_norm  = _normalise(name)
                matched = None

                # Match by osm_id
                if osm_id and osm_id in existing_osm_ids:
                    ep = next((p for p in existing_db_places if p.osmId == osm_id), None)
                    if ep:
                        matched = ep.id
                        reused_places.append((name, "osm_id"))

                # Match by name + proximity
                if not matched:
                    for ep in existing_db_places:
                        if _normalise(ep.name) == n_norm:
                            dist = _haversine_m(n_lat, n_lon, ep.latitude, ep.longitude)
                            if dist <= PROXIMITY_M:
                                matched = ep.id
                                reused_places.append((name, f"name+proximity {dist:.0f}m"))
                                break

                if matched:
                    node_to_place_id[ni] = matched
                    continue

                # Insert new place
                new_p = Place(
                    id=_id(), regionId=region.id, name=name,
                    category=_CAT_MAP.get(str(nd.get("category","")).lower(), PlaceCategory.OTHER),
                    latitude=n_lat, longitude=n_lon,
                    osmId=osm_id or None,
                    createdAt=datetime.utcnow(), updatedAt=datetime.utcnow(),
                )
                s.add(new_p)
                node_to_place_id[ni] = new_p.id
                existing_db_places.append(new_p)
                if osm_id:
                    existing_osm_ids.add(osm_id)
                new_places.append(name)
                if len(new_places) % 50 == 0:
                    s.flush()

            s.flush()

            print(f"  → Places inserted (new): {len(new_places)}")
            for n in new_places:
                print(f"      + {n}")
            print(f"  → Places reused from DB: {len(reused_places)}")
            for n, reason in reused_places:
                print(f"      = {n:<35} [{reason}]")

            # ── Mark old graphs OUTDATED ───────────────────────────────────
            old_graphs = s.query(PlaceGraph).filter_by(
                regionId=region.id, status=GraphStatus.ACTIVE
            ).all()
            for og in old_graphs:
                og.status    = GraphStatus.OUTDATED
                og.updatedAt = datetime.utcnow()
            new_version = (max(og.version or 1 for og in old_graphs) + 1) if old_graphs else 1
            if old_graphs:
                print(f"  → Marked {len(old_graphs)} old graph(s) as OUTDATED")

            # ── New graph record ───────────────────────────────────────────
            graph_rec = PlaceGraph(
                id=_id(), regionId=region.id,
                name=f"{region_name} Tourist Graph v{new_version}",
                version=new_version, status=GraphStatus.BUILDING,
                buildStartedAt=datetime.utcnow(),
                createdAt=datetime.utcnow(), updatedAt=datetime.utcnow(),
            )
            s.add(graph_rec); s.flush()
            print(f"  → Created graph record: v{new_version} (id={graph_rec.id[:8]}…)")

            # ── Edges ──────────────────────────────────────────────────────
            ec = 0
            total_edges = G.number_of_edges()
            print(f"  → Saving {total_edges * 2} directed edges (both directions)...")
            for idx, (u, v, ed) in enumerate(G.edges(data=True)):
                fid = node_to_place_id.get(u)
                tid = node_to_place_id.get(v)
                if not fid or not tid:
                    continue
                for src, dst in [(fid, tid), (tid, fid)]:
                    s.add(PlaceEdge(
                        id=_id(), graphId=graph_rec.id,
                        fromPlaceId=src, toPlaceId=dst,
                        roadDistanceKm=float(ed.get("road_distance_km", 0)),
                        durationMin=float(ed.get("duration_min", 0)),
                        transportMode="driving",
                        edgeType=ed.get("edge_type", "road"),
                        createdAt=datetime.utcnow(), updatedAt=datetime.utcnow(),
                    ))
                    ec += 1
                if (idx + 1) % 100 == 0:
                    _progress_bar(idx + 1, total_edges, label=f"{ec} edges")
                    s.flush()
            if total_edges > 0:
                _progress_bar(total_edges, total_edges, label=f"{ec} edges")
                print()  # newline after progress bar
            s.flush()

            # ── Finalise ───────────────────────────────────────────────────
            graph_rec.totalNodes         = G.number_of_nodes()
            graph_rec.totalEdges         = ec
            graph_rec.avgDegree          = round(ec / G.number_of_nodes(), 2) if G.number_of_nodes() else 0
            graph_rec.status             = GraphStatus.ACTIVE
            graph_rec.buildCompletedAt   = datetime.utcnow()
            graph_rec.buildDurationSeconds = int(
                (graph_rec.buildCompletedAt - graph_rec.buildStartedAt).total_seconds()
            )
            graph_rec.updatedAt = datetime.utcnow()
            s.commit()

            print(f"\n  ┌─ DB Summary ──────────────────────────────────────┐")
            print(f"  │  Region    : {region.name} ({region.id[:8]}…)")
            print(f"  │  Graph     : v{new_version} ({graph_rec.id[:8]}…) — ACTIVE")
            print(f"  │  Places    : {G.number_of_nodes()} ({len(new_places)} new, {len(reused_places)} reused)")
            print(f"  │  Edges     : {ec} directed ({G.number_of_edges()} undirected)")
            print(f"  │  Avg degree: {graph_rec.avgDegree}")
            print(f"  │  Build time: {graph_rec.buildDurationSeconds}s")
            print(f"  └──────────────────────────────────────────────────────┘")

            # ── Build return dict ──────────────────────────────────────────
            places_dict = {}
            for ni, nd in G.nodes(data=True):
                pid = node_to_place_id[ni]
                places_dict[pid] = {
                    "id": pid, "name": nd["name"],
                    "category":  nd.get("category", "attraction"),
                    "latitude":  float(nd["lat"]),
                    "longitude": float(nd["lon"]),
                    "description": None, "rating": None,
                    "typical_duration": None, "entry_fee": None,
                    "opening_hours": None, "best_time": None,
                    "tags": [], "popularity": None,
                }

            adj = {}; dl = {}
            for u, v, ed in G.edges(data=True):
                fid = node_to_place_id[u]
                tid = node_to_place_id[v]
                for src, dst in [(fid, tid), (tid, fid)]:
                    adj.setdefault(src, []).append({
                        "to":          dst,
                        "distance_km": ed.get("road_distance_km", 0),
                        "duration_min":ed.get("duration_min", 0),
                        "transport":   "driving",
                        "cost":        None,
                    })
                    dl[f"{src}:{dst}"] = {
                        "distance_km":  ed.get("road_distance_km", 0),
                        "duration_min": ed.get("duration_min", 0),
                        "transport":    "driving",
                    }

            return {
                "region_id":    region.id,
                "region_name":  region.name,
                "graph_id":     graph_rec.id,
                "graph_version":graph_rec.version,
                "places":       places_dict,
                "adjacency":    adj,
                "distance_lookup": dl,
                "total_nodes":  len(places_dict),
                "total_edges":  ec,
                "has_complete_graph": True,
            }

        except Exception:
            s.rollback()
            _err("DB transaction rolled back")
            traceback.print_exc()
            raise


# ── Public API ─────────────────────────────────────────────────────────────────
def build_and_save_graph(
    destination: str,
    state: Optional[str] = None,
    country: str = "India",
) -> Optional[dict]:

    t_total = time.time()
    _sep("═")
    print(f"  GRAPH BUILDER — {destination.upper()}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _sep("═")

    STEPS = 4

    # ── Step 1: Geocode ────────────────────────────────────────────────────
    _step(1, STEPS, "GEOCODING")
    t0 = time.time()
    loc = _geocode(destination)
    if not loc:
        _err(f"Could not geocode '{destination}' — aborting")
        return None
    _ok(f"Geocoded in {_elapsed(t0)}")

    # ── Step 2: Fetch OSM places ───────────────────────────────────────────
    _step(2, STEPS, "FETCHING PLACES FROM OPENSTREETMAP (Overpass API)")
    t0 = time.time()
    places = _fetch_osm(loc["lat"], loc["lon"])
    if not places:
        _err("No places found — aborting")
        return None
    _ok(f"{len(places)} unique places fetched and deduplicated in {_elapsed(t0)}")

    # ── Step 3: Build graph ────────────────────────────────────────────────
    _step(3, STEPS, f"BUILDING KNN GRAPH (k={K_NEIGHBORS}, max_travel={MAX_TRAVEL_MIN}min)")
    t0 = time.time()
    G = _build_nx_graph(places)
    if G.number_of_nodes() == 0:
        _err("Empty graph — aborting")
        return None
    _ok(f"Graph built in {_elapsed(t0)}: "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Step 4: Save to DB ─────────────────────────────────────────────────
    _step(4, STEPS, "SAVING TO DATABASE")
    t0 = time.time()
    try:
        gd = _save_to_db(G, places, destination, state, country)
        _ok(f"Saved to DB in {_elapsed(t0)}")
    except Exception as e:
        _err(f"DB save failed: {e}")
        return None

    _sep("═")
    print(f"  ✅ COMPLETE  —  total time: {_elapsed(t_total)}")
    print(f"     {gd['total_nodes']} places  |  {gd['total_edges']} directed edges  |  v{gd['graph_version']}")
    _sep("═")
    print()
    return gd


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build and save a tourist place graph")
    ap.add_argument("--destination", required=True, help="e.g. Munnar")
    ap.add_argument("--state",   default=None,    help="e.g. Kerala")
    ap.add_argument("--country", default="India", help="e.g. India")
    a = ap.parse_args()
    build_and_save_graph(a.destination, a.state, a.country)