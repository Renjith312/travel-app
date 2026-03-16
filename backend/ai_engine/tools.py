"""
ai_engine/tools.py  — deterministic data-fetching tools (no LLM)
"""
import os, requests
from typing import List, Dict, Any, Optional

OVERPASS_API = "https://overpass-api.de/api/interpreter"
NOMINATIM    = "https://nominatim.openstreetmap.org/search"
ORS_API_KEY  = os.getenv("OPENROUTESERVICE_API_KEY", "")
ORS_MATRIX   = "https://api.openrouteservice.org/v2/matrix/driving-car"


# ── Geocode ────────────────────────────────────────────────────────────────────
def _geocode(destination: str) -> Optional[Dict]:
    try:
        r = requests.get(NOMINATIM,
                         params={"q": destination, "format": "json", "limit": 1},
                         headers={"User-Agent": "TravelCopilot/1.0"}, timeout=10)
        data = r.json()
        if data:
            return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
    except Exception as e:
        print(f"[TOOLS] Geocode error: {e}")
    return None


# ── Haversine ──────────────────────────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2) -> float:
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── Fetch tourist places ───────────────────────────────────────────────────────
def fetch_places(destination: str, interests: List[str] = None, limit: int = 30) -> List[Dict]:
    print(f"[TOOL] Fetching places for: {destination}")
    loc = _geocode(destination)
    if not loc:
        return []
    lat, lon = loc["lat"], loc["lon"]
    radius = 12000
    parts = [
        f'node["tourism"="attraction"](around:{radius},{lat},{lon});',
        f'node["tourism"="museum"](around:{radius},{lat},{lon});',
        f'node["tourism"="viewpoint"](around:{radius},{lat},{lon});',
        f'node["historic"]["name"](around:{radius},{lat},{lon});',
        f'node["natural"="waterfall"]["name"](around:{radius},{lat},{lon});',
        f'node["natural"="peak"]["name"](around:{radius},{lat},{lon});',
        f'node["leisure"="park"]["name"](around:{radius},{lat},{lon});',
        f'node["amenity"="place_of_worship"]["name"](around:{radius},{lat},{lon});',
    ]
    if interests:
        for i in interests:
            il = i.lower()
            if "beach" in il:    parts.append(f'node["natural"="beach"](around:{radius},{lat},{lon});')
            if "food" in il:     parts.append(f'node["amenity"="restaurant"]["name"](around:{radius},{lat},{lon});')
            if "shop" in il:     parts.append(f'node["shop"]["name"](around:{radius},{lat},{lon});')
    query = f"[out:json][timeout:30];\n(\n{''.join(parts)}\n);\nout body;"
    try:
        r = requests.post(OVERPASS_API, data={"data": query}, timeout=35)
        r.raise_for_status()
        seen, places = set(), []
        for el in r.json().get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name or name in seen: continue
            seen.add(name)
            cat = (tags.get("tourism") or ("historic" if "historic" in tags else None)
                   or tags.get("natural") or tags.get("leisure") or tags.get("amenity") or "attraction")
            places.append({"name": name, "category": cat,
                           "latitude": el["lat"], "longitude": el["lon"], "tags": tags})
            if len(places) >= limit: break
        print(f"[TOOL] Fetched {len(places)} places")
        return places
    except Exception as e:
        print(f"[TOOL] Overpass error: {e}")
        return []


# ── Fetch stays ────────────────────────────────────────────────────────────────
def fetch_stays(destination: str, budget: str = None, limit: int = 20) -> List[Dict]:
    print(f"[TOOL] Fetching stays for: {destination}")
    loc = _geocode(destination)
    if not loc:
        return _fallback_stays(destination)
    lat, lon = loc["lat"], loc["lon"]
    radius = 10000
    query = f"""[out:json][timeout:25];
(
  node["tourism"="hotel"](around:{radius},{lat},{lon});
  node["tourism"="hostel"](around:{radius},{lat},{lon});
  node["tourism"="guest_house"](around:{radius},{lat},{lon});
  node["tourism"="motel"](around:{radius},{lat},{lon});
  way["tourism"="hotel"](around:{radius},{lat},{lon});
);
out body;"""
    try:
        r = requests.post(OVERPASS_API, data={"data": query}, timeout=30)
        r.raise_for_status()
        stays = []
        for el in r.json().get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name: continue
            if el["type"] == "node":
                s_lat, s_lon = el["lat"], el["lon"]
            elif el["type"] == "way" and "center" in el:
                s_lat, s_lon = el["center"]["lat"], el["center"]["lon"]
            else:
                continue
            stays.append({"name": name, "type": tags.get("tourism", "hotel"),
                          "latitude": s_lat, "longitude": s_lon, "tags": tags})
            if len(stays) >= limit: break
        print(f"[TOOL] Fetched {len(stays)} stays")
        return stays if stays else _fallback_stays(destination)
    except Exception as e:
        print(f"[TOOL] fetch_stays error: {e}")
        return _fallback_stays(destination)


def _fallback_stays(destination: str) -> List[Dict]:
    return [
        {"name": f"{destination} Grand Hotel", "type": "hotel", "latitude": 0.0, "longitude": 0.0, "tags": {}},
        {"name": f"{destination} Homestay",    "type": "guest_house", "latitude": 0.0, "longitude": 0.0, "tags": {}},
    ]


# ── ORS distance (single pair) ─────────────────────────────────────────────────
def _ors_distance(coord1, coord2) -> Dict:
    if not ORS_API_KEY:
        d = _haversine(*coord1, *coord2)
        return {"distance_km": d, "duration_min": (d/30)*60}
    try:
        r = requests.post(ORS_MATRIX,
                          json={"locations": [[coord1[1],coord1[0]],[coord2[1],coord2[0]]],
                                "metrics": ["distance","duration"]},
                          headers={"Authorization": ORS_API_KEY, "Content-Type": "application/json"},
                          timeout=10)
        if r.status_code == 200:
            d = r.json()
            return {"distance_km": d["distances"][0][1]/1000, "duration_min": d["durations"][0][1]/60}
    except Exception:
        pass
    d = _haversine(*coord1, *coord2)
    return {"distance_km": d, "duration_min": (d/30)*60}
