"""
ai_engine/tools.py  — v3
========================
Data sources:
  1. OpenRouteService (ORS)   -> fetch_road_info()  real road distance + time (FREE, no limits)
  2. Booking.com via RapidAPI -> fetch_stays()       real hotel names + prices
  3. OpenStreetMap / Overpass -> fetch_places()      tourist attractions (FREE)
  4. Nominatim (OSM)          -> _geocode()          city to lat/lon (FREE)

Required .env keys:
  OPENROUTESERVICE_API_KEY  — already in your .env (free, no limits)
  RAPIDAPI_KEY              — for Booking.com hotels
"""

import os, requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta

NOMINATIM         = "https://nominatim.openstreetmap.org/search"
OVERPASS_API      = "https://overpass-api.de/api/interpreter"
ORS_BASE          = "https://api.openrouteservice.org/v2"
RAPIDAPI_KEY      = os.getenv("RAPIDAPI_KEY", "")
ORS_API_KEY       = os.getenv("OPENROUTESERVICE_API_KEY", "")
BOOKING_HOST      = "booking-com.p.rapidapi.com"
BOOKING_SEARCH_URL = "https://booking-com.p.rapidapi.com/v1/hotels/search-by-coordinates"


# ── Geocode ────────────────────────────────────────────────────────────────────
def _geocode(place: str) -> Optional[Dict]:
    try:
        r = requests.get(
            NOMINATIM,
            params={"q": place + ", India", "format": "json", "limit": 1},
            headers={"User-Agent": "TravelCopilot/3.0"},
            timeout=10,
        )
        data = r.json()
        if data:
            return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
    except Exception as e:
        print(f"[TOOLS][geocode] {place}: {e}")
    return None


# ── Haversine ──────────────────────────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2) -> float:
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def _mins_to_text(mins: float) -> str:
    h = int(mins // 60); m = int(mins % 60)
    if h == 0: return f"{m} mins"
    if m == 0: return f"{h} hr{'s' if h>1 else ''}"
    return f"{h} hr{'s' if h>1 else ''} {m} mins"


# ── Road info via ORS (primary) ────────────────────────────────────────────────
def fetch_road_info(origin: str, destination: str) -> Dict:
    """
    Real road distance + driving time using ORS (free, no request limits).
    Falls back to haversine estimate if ORS unavailable.

    Returns: {distance_km, duration_min, distance_text, duration_text, source}
    """
    c1 = _geocode(origin)
    c2 = _geocode(destination)

    if not c1 or not c2:
        print(f"[TOOLS][road] Geocode failed — {origin}={c1} {destination}={c2}")
        return {"distance_km": 0, "duration_min": 0,
                "distance_text": "unknown", "duration_text": "unknown", "source": "none"}

    if ORS_API_KEY:
        try:
            r = requests.post(
                f"{ORS_BASE}/directions/driving-car",
                headers={"Authorization": ORS_API_KEY, "Content-Type": "application/json"},
                # ORS expects [lon, lat] order
                json={"coordinates": [[c1["lon"], c1["lat"]], [c2["lon"], c2["lat"]]], "units": "km"},
                timeout=12,
            )
            if r.status_code == 200:
                s       = r.json()["routes"][0]["summary"]
                dist_km = round(s["distance"], 1)
                dur_min = round(s["duration"] / 60, 1)
                print(f"[TOOLS][ors] {origin}->{destination}: {dist_km} km, {_mins_to_text(dur_min)}")
                return {
                    "distance_km":   dist_km,
                    "duration_min":  dur_min,
                    "distance_text": f"{dist_km:.0f} km",
                    "duration_text": _mins_to_text(dur_min),
                    "source":        "ors",
                }
            print(f"[TOOLS][ors] HTTP {r.status_code}: {r.text[:150]}")
        except Exception as e:
            print(f"[TOOLS][ors] Error: {e}")

    # Haversine fallback — Kerala hill roads avg ~40 km/h
    km   = _haversine(c1["lat"], c1["lon"], c2["lat"], c2["lon"])
    mins = (km / 40) * 60
    print(f"[TOOLS][haversine] {origin}->{destination}: ~{km:.0f} km straight-line")
    return {
        "distance_km":   round(km, 1),
        "duration_min":  round(mins, 1),
        "distance_text": f"~{km:.0f} km",
        "duration_text": _mins_to_text(mins) + " (est.)",
        "source":        "haversine",
    }


# ── Hotels via Booking.com (RapidAPI) ─────────────────────────────────────────
def fetch_stays(destination: str, budget: str = None,
                checkin: str = None, checkout: str = None,
                num_adults: int = 1, limit: int = 15) -> List[Dict]:
    print(f"[TOOLS][fetch_stays] {destination} | budget={budget} | adults={num_adults}")
    if RAPIDAPI_KEY:
        result = _booking_search(destination, budget, checkin, checkout, num_adults, limit)
        if result:
            return result
        print("[TOOLS][fetch_stays] Booking.com empty — falling back to OSM")
    return _osm_stays(destination, limit)


def _booking_search(destination, budget, checkin, checkout, num_adults, limit):
    loc = _geocode(destination)
    if not loc: return []

    today = datetime.now()
    ci    = checkin  or (today + timedelta(days=1)).strftime("%Y-%m-%d")
    co    = checkout or (today + timedelta(days=2)).strftime("%Y-%m-%d")

    params = {
        "latitude": loc["lat"], "longitude": loc["lon"],
        "checkin_date": ci, "checkout_date": co,
        "adults_number": str(num_adults), "room_number": "1",
        "units": "metric", "locale": "en-gb", "currency": "INR",
        "order_by": "popularity", "filter_by_currency": "INR",
        "page_number": "0", "categories_filter_ids": "class::1,class::2,class::3,class::4",
    }
    if budget:
        try:
            b = float(str(budget).replace(",","").replace("₹",""))
            max_night = int(b * 0.30 / 2)
            if max_night > 500:
                params["price_filter_currencycode"] = "INR"
                params["price_filter_max"] = str(max_night)
        except Exception: pass

    try:
        r = requests.get(
            BOOKING_SEARCH_URL,
            headers={"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": BOOKING_HOST},
            params=params, timeout=15,
        )
        print(f"[TOOLS][booking] HTTP {r.status_code}")
        if r.status_code == 401: print("[TOOLS][booking] Invalid RapidAPI key"); return []
        if r.status_code == 403: print("[TOOLS][booking] Not subscribed — rapidapi.com/tipsters/api/booking-com"); return []
        if r.status_code != 200: return []

        raw    = r.json().get("result", [])
        nights = max(1, (datetime.strptime(co,"%Y-%m-%d") - datetime.strptime(ci,"%Y-%m-%d")).days)
        stays  = []
        for h in raw[:limit]:
            pt = h.get("min_total_price") or (h.get("price_breakdown") or {}).get("all_inclusive_price")
            stays.append({
                "name":            h.get("hotel_name", "Unknown Hotel"),
                "type":            {1:"budget",2:"budget",3:"mid-range",4:"premium",5:"luxury"}.get(h.get("class",0),"hotel"),
                "price_per_night": round(float(pt)/nights) if pt else None,
                "currency":        "INR",
                "rating":          h.get("review_score"),
                "review_count":    h.get("review_nr"),
                "latitude":        h.get("latitude", loc["lat"]),
                "longitude":       h.get("longitude", loc["lon"]),
                "booking_url":     h.get("url") or f"https://www.booking.com/hotel/in/{h.get('hotel_id','')}.html",
                "address":         h.get("address",""),
                "source":          "booking",
            })
        print(f"[TOOLS][booking] {len(stays)} hotels for {destination}")
        return stays
    except Exception as e:
        print(f"[TOOLS][booking] Error: {e}"); return []


def _osm_stays(destination: str, limit: int = 15) -> List[Dict]:
    loc = _geocode(destination)
    if not loc: return _fallback_stays(destination)
    lat, lon = loc["lat"], loc["lon"]
    query = f"""[out:json][timeout:25];
(
  node["tourism"="hotel"](around:10000,{lat},{lon});
  node["tourism"="hostel"](around:10000,{lat},{lon});
  node["tourism"="guest_house"](around:10000,{lat},{lon});
  node["tourism"="motel"](around:10000,{lat},{lon});
  way["tourism"="hotel"](around:10000,{lat},{lon});
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
            s_lat = el.get("lat", lat); s_lon = el.get("lon", lon)
            if el.get("type") == "way" and "center" in el:
                s_lat, s_lon = el["center"]["lat"], el["center"]["lon"]
            stays.append({
                "name": name, "type": tags.get("tourism","hotel"),
                "price_per_night": None, "currency": "INR",
                "rating": None, "review_count": None,
                "latitude": s_lat, "longitude": s_lon,
                "booking_url": None, "address": tags.get("addr:full",""), "source": "osm",
            })
            if len(stays) >= limit: break
        print(f"[TOOLS][osm] {len(stays)} stays for {destination}")
        return stays if stays else _fallback_stays(destination)
    except Exception as e:
        print(f"[TOOLS][osm_stays] Error: {e}"); return _fallback_stays(destination)


def _fallback_stays(destination: str) -> List[Dict]:
    return [
        {"name": f"{destination} Grand Hotel", "type": "hotel", "price_per_night": None,
         "currency": "INR", "rating": None, "review_count": None,
         "latitude": 0.0, "longitude": 0.0, "booking_url": None, "source": "fallback"},
        {"name": f"{destination} Homestay", "type": "guest_house", "price_per_night": None,
         "currency": "INR", "rating": None, "review_count": None,
         "latitude": 0.0, "longitude": 0.0, "booking_url": None, "source": "fallback"},
    ]


# ── Tourist places via OSM/Overpass ───────────────────────────────────────────
def fetch_places(destination: str, interests: List[str] = None, limit: int = 30) -> List[Dict]:
    """
    Fetch tourist places in priority order:
      1. Tourism tags (attraction, museum, viewpoint, zoo, etc.)
      2. Natural features (waterfall, peak, beach, lake)
      3. Parks & leisure
      4. Notable historic sites (fort, monument — NOT generic temples)
      5. Interest-specific extras (food, shopping, adventure)

    Generic places of worship (temples, mosques, churches) are EXCLUDED
    unless they are explicitly tagged as tourism=attraction or have a
    Wikipedia/Wikidata entry (meaning they are genuinely famous landmarks).
    """
    print(f"[TOOLS][places] {destination}")
    loc = _geocode(destination)
    if not loc: return []
    lat, lon = loc["lat"], loc["lon"]
    radius = 15000  # 15 km radius

    # ── Query 1: Pure tourism + nature (highest quality) ──────────────────────
    q1 = f"""[out:json][timeout:25];
(
  node["tourism"="attraction"](around:{radius},{lat},{lon});
  node["tourism"="museum"](around:{radius},{lat},{lon});
  node["tourism"="viewpoint"](around:{radius},{lat},{lon});
  node["tourism"="theme_park"](around:{radius},{lat},{lon});
  node["tourism"="zoo"](around:{radius},{lat},{lon});
  node["tourism"="aquarium"](around:{radius},{lat},{lon});
  node["tourism"="gallery"](around:{radius},{lat},{lon});
  node["natural"="waterfall"]["name"](around:{radius},{lat},{lon});
  node["natural"="peak"]["name"](around:{radius},{lat},{lon});
  node["natural"="beach"]["name"](around:{radius},{lat},{lon});
  node["natural"="lake"]["name"](around:{radius},{lat},{lon});
  node["natural"="hot_spring"]["name"](around:{radius},{lat},{lon});
  node["leisure"="nature_reserve"]["name"](around:{radius},{lat},{lon});
  node["leisure"="park"]["name"](around:{radius},{lat},{lon});
  node["leisure"="water_park"]["name"](around:{radius},{lat},{lon});
  node["leisure"="garden"]["name"](around:{radius},{lat},{lon});
  way["tourism"="attraction"](around:{radius},{lat},{lon});
  way["leisure"="nature_reserve"](around:{radius},{lat},{lon});
  way["natural"="water"]["name"](around:{radius},{lat},{lon});
);
out center body;"""

    # ── Query 2: Famous historic landmarks only ────────────────────────────────
    q2 = f"""[out:json][timeout:20];
(
  node["historic"="fort"]["name"](around:{radius},{lat},{lon});
  node["historic"="castle"]["name"](around:{radius},{lat},{lon});
  node["historic"="monument"]["name"](around:{radius},{lat},{lon});
  node["historic"="memorial"]["name"](around:{radius},{lat},{lon});
  node["historic"="ruins"]["name"](around:{radius},{lat},{lon});
  node["historic"="archaeological_site"]["name"](around:{radius},{lat},{lon});
  way["historic"="fort"](around:{radius},{lat},{lon});
  way["historic"="castle"](around:{radius},{lat},{lon});
);
out center body;"""

    # ── Query 3: ONLY famous religious sites (wikidata = globally known) ──────
    q3 = f"""[out:json][timeout:20];
(
  node["tourism"="attraction"]["amenity"="place_of_worship"](around:{radius},{lat},{lon});
  node["amenity"="place_of_worship"]["wikidata"](around:{radius},{lat},{lon});
  node["amenity"="place_of_worship"]["wikipedia"](around:{radius},{lat},{lon});
);
out center body;"""

    # ── Query 4: Interest-specific extras ─────────────────────────────────────
    extra_parts = []
    if interests:
        for i in (interests or []):
            il = i.lower()
            if "food" in il or "culinary" in il:
                extra_parts.append(f'node["amenity"="restaurant"]["cuisine"]["name"](around:{radius},{lat},{lon});')
            if "shop" in il:
                extra_parts.append(f'node["shop"="mall"]["name"](around:{radius},{lat},{lon});')
                extra_parts.append(f'node["shop"="market"]["name"](around:{radius},{lat},{lon});')
            if "adventure" in il or "sport" in il:
                extra_parts.append(f'node["sport"]["name"](around:{radius},{lat},{lon});')
                extra_parts.append(f'node["leisure"="sports_centre"]["name"](around:{radius},{lat},{lon});')

    queries = [q1, q2, q3]
    if extra_parts:
        queries.append(f"[out:json][timeout:15];\n(\n{''.join(extra_parts)}\n);\nout center body;")

    seen, places = set(), []

    for q in queries:
        if len(places) >= limit:
            break
        try:
            r = requests.post(OVERPASS_API, data={"data": q}, timeout=35)
            r.raise_for_status()
            for el in r.json().get("elements", []):
                if len(places) >= limit:
                    break
                tags = el.get("tags", {})
                name = tags.get("name", "").strip()
                if not name or name in seen:
                    continue

                # Skip generic worship places — only keep famous ones (already filtered in q3)
                if (tags.get("amenity") == "place_of_worship"
                        and tags.get("tourism") != "attraction"
                        and not tags.get("wikidata")
                        and not tags.get("wikipedia")):
                    continue

                seen.add(name)
                cat = (tags.get("tourism")
                       or tags.get("natural")
                       or tags.get("leisure")
                       or tags.get("historic")
                       or "attraction")

                # Handle both node (lat/lon direct) and way (center)
                el_lat = el.get("lat") or (el.get("center") or {}).get("lat", lat)
                el_lon = el.get("lon") or (el.get("center") or {}).get("lon", lon)

                places.append({
                    "name":      name,
                    "category":  cat,
                    "latitude":  el_lat,
                    "longitude": el_lon,
                    "tags":      tags,
                })
        except Exception as e:
            print(f"[TOOLS][places] Query error: {e}")

    print(f"[TOOLS][places] {len(places)} places for {destination}")
    return places


# ── ORS matrix — place-to-place distances inside itinerary ────────────────────
def _ors_distance(coord1, coord2) -> Dict:
    """coord format: (lat, lon)"""
    if ORS_API_KEY:
        try:
            r = requests.post(
                f"{ORS_BASE}/matrix/driving-car",
                headers={"Authorization": ORS_API_KEY, "Content-Type": "application/json"},
                json={"locations": [[coord1[1],coord1[0]], [coord2[1],coord2[0]]], "metrics": ["distance","duration"]},
                timeout=10,
            )
            if r.status_code == 200:
                d = r.json()
                return {"distance_km": d["distances"][0][1]/1000, "duration_min": d["durations"][0][1]/60}
        except Exception: pass
    d = _haversine(*coord1, *coord2)
    return {"distance_km": d, "duration_min": (d/30)*60}