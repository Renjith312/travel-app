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
ORS_API_KEY        = os.getenv("OPENROUTESERVICE_API_KEY", "")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
ORS_MATRIX_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"
# Multiple Overpass mirrors — tried in order, first success wins
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
NOMINATIM_URL  = "https://nominatim.openstreetmap.org/search"

K_NEIGHBORS    = 5
MAX_TRAVEL_MIN = 90
MAX_PLACES     = 50
RADIUS_M       = 40000
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
# Mirrors that have already failed this process run — skip them immediately
_dead_mirrors: set = set()

def _overpass_post(query: str, timeout: int = 12):
    """POST an Overpass query, trying each mirror until one succeeds.
    Dead mirrors (failed earlier this run) are skipped instantly.
    """
    last_err = None
    for mirror in OVERPASS_MIRRORS:
        if mirror in _dead_mirrors:
            continue
        try:
            r = requests.post(mirror, data={"data": query}, timeout=timeout)
            if r.status_code == 429:
                _warn(f"429 on {mirror} — marking dead, trying next mirror")
                _dead_mirrors.add(mirror)
                continue
            if r.status_code in (502, 503, 504):
                _warn(f"HTTP {r.status_code} on {mirror} — marking dead, trying next mirror")
                _dead_mirrors.add(mirror)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            _dead_mirrors.add(mirror)
            _warn(f"{mirror} failed: {type(e).__name__} — marking dead, trying next mirror")
    _warn(f"All Overpass mirrors exhausted.")
    return None


def _robust_json_parse(text: str):
    """Parse JSON from LLM output tolerating fences, trailing commas, comments."""
    import json as _j, ast as _a
    t = text.strip()
    # Remove markdown fences
    fence = "```"
    if t.startswith(fence):
        t = t[t.find("\n")+1:] if "\n" in t else t[3:]
    if t.endswith(fence):
        t = t[:t.rfind(fence)]
    t = t.strip()
    # Extract first [...] or {...} block
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = t.find(open_ch)
        if start == -1:
            continue
        depth, in_str, esc, end = 0, False, False, -1
        for i, ch in enumerate(t[start:], start):
            if esc: esc = False; continue
            if ch == "\\": esc = True; continue
            if ch == '"' and not esc: in_str = not in_str
            if in_str: continue
            if ch == open_ch: depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0: end = i; break
        if end != -1:
            t = t[start:end + 1]
            break
    # Remove // comments and trailing commas
    t = re.sub(r"//[^\n]*", "", t)
    t = re.sub(r",\s*([\]}])", r"\1", t)
    try:
        return _j.loads(t)
    except Exception:
        pass
    try:
        return _a.literal_eval(t)
    except Exception:
        return None


def _llm_places_fallback(destination: str, lat: float, lon: float) -> list:
    """
    Ask Gemini (then OpenRouter) for tourist places when all Overpass mirrors fail.
    Returns places in the same dict format as _fetch_osm().
    """
    import json as _json
    prompt = (
        f"You are a travel data API. Return ONLY a valid JSON array — "
        f"no markdown fences, no comments, no trailing commas, no extra text. "
        f"List 25 top tourist attractions in {destination}, India. "
        f"Each element must have exactly these keys: "
        f'"name" (string), "category" (one of: attraction/museum/viewpoint/beach/fort/'
        f"nature/park/waterfall/market/historic), "
        f'"latitude" (number), "longitude" (number). '
        f"Use accurate GPS coordinates near ({lat:.4f}, {lon:.4f}). "
        f"Start your response with [ and end with ]. No other text."
    )

    raw_text = None

    if GEMINI_API_KEY:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}},
                timeout=20,
            )
            if r.status_code == 200:
                raw_text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                print(f"  → LLM fallback: got response from Gemini")
        except Exception as e:
            _warn(f"Gemini LLM fallback error: {e}")

    if not raw_text and OPENROUTER_API_KEY:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": OPENROUTER_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.2, "max_tokens": 2048},
                timeout=20,
            )
            if r.status_code == 200:
                raw_text = r.json()["choices"][0]["message"]["content"]
                print(f"  → LLM fallback: got response from OpenRouter")
        except Exception as e:
            _warn(f"OpenRouter LLM fallback error: {e}")

    if not raw_text:
        _warn("LLM fallback: no response from any LLM")
        return []

    data = _robust_json_parse(raw_text)
    if data is None:
        _warn("LLM fallback: could not parse JSON from response")
        _warn(f"  Raw text preview: {raw_text[:300]!r}")
        return []

    if not isinstance(data, list):
        _warn("LLM fallback: response is not a list")
        return []

    places = []
    seen = set()
    for item in data:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"]).strip()
        norm = re.sub(r"[^a-z0-9]", "", name.lower())
        if norm in seen:
            continue
        seen.add(norm)
        places.append({
            "name":     name,
            "lat":      float(item.get("latitude", lat)),
            "lon":      float(item.get("longitude", lon)),
            "category": str(item.get("category", "attraction")),
            "osm_id":   "",
            "priority": "llm_fallback",
        })
    print(f"  → LLM fallback: {len(places)} places for {destination}")
    return places


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


def _fetch_osm(lat: float, lon: float, destination: str = "") -> list:
    """
    Fetch tourist places in priority order:
      P1 — Pure tourism + nature features (highest signal, no worship)
      P2 — Hidden gems: scenic roads, gardens, waterbodies, heritage
      P3 — Famous religious sites ONLY (wikipedia/wikidata tagged)
    Generic temples/churches/mosques are excluded unless famous.
    """
    t0 = time.time()
    R  = RADIUS_M
    print(f"  → Search radius: {R/1000:.1f}km around ({lat:.4f}, {lon:.4f})")
    print(f"  → Priority queries: tourism > nature > hidden gems > famous landmarks")

    # ── Priority 1: Pure tourism + natural features ───────────────────────────
    q1 = f"""[out:json][timeout:30];(
      node["tourism"="attraction"](around:{R},{lat},{lon});
      node["tourism"="museum"](around:{R},{lat},{lon});
      node["tourism"="viewpoint"](around:{R},{lat},{lon});
      node["tourism"="theme_park"](around:{R},{lat},{lon});
      node["tourism"="zoo"](around:{R},{lat},{lon});
      node["tourism"="aquarium"](around:{R},{lat},{lon});
      node["tourism"="gallery"](around:{R},{lat},{lon});
      node["tourism"="picnic_site"]["name"](around:{R},{lat},{lon});
      node["natural"="waterfall"]["name"](around:{R},{lat},{lon});
      node["natural"="peak"]["name"](around:{R},{lat},{lon});
      node["natural"="beach"]["name"](around:{R},{lat},{lon});
      node["natural"="cave_entrance"]["name"](around:{R},{lat},{lon});
      node["natural"="hot_spring"]["name"](around:{R},{lat},{lon});
      node["natural"="spring"]["name"](around:{R},{lat},{lon});
      node["natural"="geyser"]["name"](around:{R},{lat},{lon});
      node["natural"="glacier"]["name"](around:{R},{lat},{lon});
      way["tourism"="attraction"](around:{R},{lat},{lon});
      way["natural"="water"]["name"](around:{R},{lat},{lon});
      relation["natural"="water"]["name"](around:{R},{lat},{lon});
    );out center body;"""

    # ── Priority 2: Hidden gems & outdoor spots ───────────────────────────────
    q2 = f"""[out:json][timeout:30];(
      node["leisure"="nature_reserve"]["name"](around:{R},{lat},{lon});
      node["leisure"="park"]["name"](around:{R},{lat},{lon});
      node["leisure"="garden"]["name"](around:{R},{lat},{lon});
      node["leisure"="water_park"]["name"](around:{R},{lat},{lon});
      node["leisure"="bird_hide"]["name"](around:{R},{lat},{lon});
      node["leisure"="fishing"]["name"](around:{R},{lat},{lon});
      node["landuse"="reservoir"]["name"](around:{R},{lat},{lon});
      node["waterway"="waterfall"]["name"](around:{R},{lat},{lon});
      node["waterway"="dam"]["name"](around:{R},{lat},{lon});
      node["man_made"="lighthouse"]["name"](around:{R},{lat},{lon});
      node["man_made"="dam"]["name"](around:{R},{lat},{lon});
      node["amenity"="arts_centre"]["name"](around:{R},{lat},{lon});
      node["amenity"="theatre"]["name"](around:{R},{lat},{lon});
      node["boundary"="national_park"]["name"](around:{R},{lat},{lon});
      way["leisure"="nature_reserve"](around:{R},{lat},{lon});
      way["landuse"="reservoir"](around:{R},{lat},{lon});
      way["waterway"="dam"](around:{R},{lat},{lon});
    );out center body;"""

    # ── Priority 3: Historic landmarks (forts, monuments — not generic temples) ─
    q3 = f"""[out:json][timeout:25];(
      node["historic"="fort"]["name"](around:{R},{lat},{lon});
      node["historic"="castle"]["name"](around:{R},{lat},{lon});
      node["historic"="monument"]["name"](around:{R},{lat},{lon});
      node["historic"="memorial"]["name"](around:{R},{lat},{lon});
      node["historic"="ruins"]["name"](around:{R},{lat},{lon});
      node["historic"="archaeological_site"]["name"](around:{R},{lat},{lon});
      node["historic"="palace"]["name"](around:{R},{lat},{lon});
      node["historic"="battlefield"]["name"](around:{R},{lat},{lon});
      way["historic"="fort"](around:{R},{lat},{lon});
      way["historic"="castle"](around:{R},{lat},{lon});
    );out center body;"""

    # ── Priority 4: ONLY famous religious sites (globally known, wiki-tagged) ──
    q4 = f"""[out:json][timeout:20];(
      node["tourism"="attraction"]["amenity"="place_of_worship"](around:{R},{lat},{lon});
      node["amenity"="place_of_worship"]["wikidata"](around:{R},{lat},{lon});
      node["amenity"="place_of_worship"]["wikipedia"](around:{R},{lat},{lon});
    );out center body;"""

    all_queries = [("tourism+nature", q1), ("hidden_gems", q2),
                   ("historic", q3), ("famous_landmarks", q4)]

    raw = []
    seen_names = set()
    cat_counts: dict = {}

    for qname, q in all_queries:
        if len(raw) >= MAX_PLACES * 2:  # collect 2x then trim at end
            break
        data = _overpass_post(q, timeout=40)
        if data is None:
            _warn(f"Query {qname} failed on all mirrors — skipping")
            continue
        try:
            elements = data.get("elements", [])
            batch_added = 0
            for el in elements:
                tags = el.get("tags", {})
                name = tags.get("name", "").strip()
                if not name:
                    continue

                # Skip generic worship places — only keep famous ones (q4 already filters)
                if (tags.get("amenity") == "place_of_worship"
                        and tags.get("tourism") != "attraction"
                        and not tags.get("wikidata")
                        and not tags.get("wikipedia")):
                    continue

                # Skip low-quality names
                name_lower = name.lower()
                if any(skip in name_lower for skip in [
                    "kingdom hall", "csi church", "station church",
                    "st joseph church", "st george church", "st mary",
                    "jacobite church", "zion church", "bishop house",
                ]):
                    continue

                norm = re.sub(r"[^a-z0-9]", "", name_lower)
                if norm in seen_names:
                    continue
                seen_names.add(norm)

                # Get coordinates (node direct or way center)
                p_lat = el.get("lat") or (el.get("center") or {}).get("lat")
                p_lon = el.get("lon") or (el.get("center") or {}).get("lon")
                if not p_lat or not p_lon:
                    continue

                cat = (tags.get("tourism")
                       or tags.get("natural")
                       or tags.get("leisure")
                       or tags.get("historic")
                       or tags.get("waterway")
                       or tags.get("man_made")
                       or ("historic" if "historic" in tags else None)
                       or "attraction")

                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                raw.append({
                    "name":     name,
                    "lat":      float(p_lat),
                    "lon":      float(p_lon),
                    "category": cat,
                    "osm_id":   str(el.get("id", "")),
                    "priority": qname,
                })
                batch_added += 1

            print(f"  → [{qname}] +{batch_added} places ({_elapsed(t0)})")

        except Exception as e:
            _warn(f"Query {qname} processing error: {e}")

    print(f"  → Total raw places: {len(raw)}")
    print(f"  → By category: " + ", ".join(f"{k}={v}" for k,v in sorted(cat_counts.items())))
    print(f"  → Deduplicating (same name within {PROXIMITY_M}m)...")
    places = _deduplicate_places(raw)
    places = places[:MAX_PLACES]

    # ── LLM fallback when all Overpass mirrors returned nothing ──────────────
    if not places:
        _warn("All Overpass mirrors returned 0 places — switching to LLM fallback")
        places = _llm_places_fallback(destination, lat, lon)
        places = places[:MAX_PLACES]

    print(f"  → Final place list ({len(places)}):")
    for i, p in enumerate(places, 1):
        print(f"      {i:2d}. {p['name']:<35} [{p['category']}]  ({p['lat']:.4f}, {p['lon']:.4f})")

    return places


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
    places = _fetch_osm(loc["lat"], loc["lon"], destination)
    if not places:
        _err("No places found even after LLM fallback — aborting")
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