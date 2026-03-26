"""
ai_engine/transport.py
======================
Fetches REAL transport options between two Indian cities using:

  1. RapidAPI — IRCTC / Indian Railway API  → live train schedules & fares
  2. tools.fetch_road_info()               → real road distance & drive time
     (Google Maps if GOOGLE_MAPS_API_KEY set, else ORS/haversine fallback)
  3. Grounded bus fare estimates           → calculated from real road km
     (Rs 1.2–1.8/km for KSRTC, more accurate than LLM guessing)

Required .env keys:
  RAPIDAPI_KEY         — RapidAPI key (for IRCTC train API)
  GOOGLE_MAPS_API_KEY  — Google Maps Distance Matrix (for road time)

Fallback behavior:
  - If train API unavailable → skip train options, show buses only
  - If road API unavailable → use haversine estimate
  - All estimated options are clearly labelled
"""

import os, requests
from typing import Optional, Dict, List
from datetime import datetime, timedelta

from ai_engine.tools import fetch_road_info

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")

# IRCTC train search (RapidAPI)
IRCTC_HOST       = "irctc1.p.rapidapi.com"
IRCTC_TRAINS_URL = "https://irctc1.p.rapidapi.com/api/v3/trainBetweenStations"
IRCTC_STATION_URL= "https://irctc1.p.rapidapi.com/api/v1/searchStation"

# Kerala KSRTC bus fare (Rs/km, 2024 approximate)
KSRTC_ORDINARY_RATE = 1.20
KSRTC_EXPRESS_RATE  = 1.50


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def fetch_transport_options(origin: str, destination: str,
                             date: str, departure_time: str) -> Dict:
    """
    Returns:
    {
      "options": [...],
      "data_source": "live" | "estimated",
      "booking_links": {"train": str, "bus": str},
      "road_distance": str,
      "road_duration": str,
    }
    """
    print(f"[TRANSPORT] {origin} -> {destination} on {date}")

    # 1. Get real road info
    road      = fetch_road_info(origin, destination)
    dist_km   = road.get("distance_km", 0)
    dur_min   = road.get("duration_min", 0)
    dist_text = road.get("distance_text", f"~{dist_km:.0f} km")
    dur_text  = road.get("duration_text", f"~{dur_min:.0f} min")
    print(f"[TRANSPORT] Road: {dist_text}, {dur_text} (source={road.get('source')})")

    dep_dt   = _parse_time(departure_time)
    options  = []
    has_live = False
    opt_id   = 1

    # 2. Try live train data
    trains = _fetch_trains(origin, destination, date)
    if trains:
        has_live = True
        for t in trains[:2]:
            arr_dt  = _add_duration(dep_dt, t["duration_min"]) if dep_dt else None
            arr_txt = arr_dt.strftime("%I:%M %p") if arr_dt else t.get("arrival", "?")
            dep_txt = dep_dt.strftime("%I:%M %p") if dep_dt else departure_time
            sight   = (arr_dt.hour < 16) if arr_dt else False
            options.append({
                "id":              opt_id,
                "mode":            "Train",
                "route":           t["route"],
                "departure":       t.get("departure", dep_txt),
                "arrival":         arr_txt,
                "duration":        t.get("duration_text", ""),
                "cost_per_person": t.get("fare", 0),
                "sightseeing_day1": sight,
                "notes":           t.get("notes", ""),
                "data_source":     "live",
            })
            opt_id += 1

    # 3. Bus options grounded on real distance
    if dist_km > 0:
        for b in _build_bus_options(origin, destination, dep_dt, dist_km, dur_min):
            b["id"] = opt_id
            options.append(b)
            opt_id += 1

    # 4. Last-resort fallback
    if not options:
        fb = _estimated_fallback(origin, destination, dist_km, dur_min, dep_dt, dur_text)
        for i, o in enumerate(fb, 1):
            o["id"] = i
        options = fb

    return {
        "options":       options,
        "data_source":   "live" if has_live else "estimated",
        "booking_links": {
            "train": "https://www.irctc.co.in/nget/train-search",
            "bus":   "https://www.ksrtc.in/oprs-web/",
        },
        "road_distance": dist_text,
        "road_duration": dur_text,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINS via IRCTC RapidAPI
# ══════════════════════════════════════════════════════════════════════════════
def _search_station_code(city: str) -> Optional[str]:
    if not RAPIDAPI_KEY:
        return None
    try:
        r = requests.get(
            IRCTC_STATION_URL,
            headers={"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": IRCTC_HOST},
            params={"query": city},
            timeout=5,
        )
        if r.status_code == 200:
            stations = r.json().get("data", [])
            for s in stations:
                if city.lower() in (s.get("stationName") or "").lower():
                    code = s.get("stationCode")
                    print(f"[TRANSPORT][irctc] Station match: {s.get('stationName')} ({code})")
                    return code
            if stations:
                return stations[0].get("stationCode")
        else:
            print(f"[TRANSPORT][irctc] Station search HTTP {r.status_code}")
    except Exception as e:
        print(f"[TRANSPORT][irctc] Station search error: {e}")
    return None


# Cities with no railway station — skip IRCTC entirely for these
_NO_RAIL_CITIES = {
    "munnar", "wayanad", "vagamon", "thekkady", "periyar", "varkala",
    "kovalam", "kumarakom", "athirappilly", "ooty", "coorg",
    "kodaikanal", "chikmagalur", "kasauli", "mcleod ganj",
    "idukki", "ponmudi", "bekal",
}

# Hardcoded station codes for common Kerala/India cities
# Avoids IRCTC API timeout for well-known stations
_KNOWN_STATIONS = {
    # Kerala
    "kottayam":       "KTYM",
    "kozhikode":      "CLT",
    "calicut":        "CLT",
    "thiruvananthapuram": "TVC",
    "trivandrum":     "TVC",
    "ernakulam":      "ERS",
    "kochi":          "ERS",
    "cochin":         "ERS",
    "thrissur":       "TCR",
    "palakkad":       "PGT",
    "kollam":         "QLN",
    "alappuzha":      "ALLP",
    "alleppey":       "ALLP",
    "kannur":         "CAN",
    "kasaragod":      "KGQ",
    "pathanamthitta": "PAA",
    "kottarakkara":   "KTU",
    "tirur":          "TIR",
    "shoranur":       "SRR",
    "ottapalam":      "OTP",
    "kayamkulam":     "KYJ",
    "changanacherry": "CGY",
    "ettumanoor":     "ETM",
    "vadakara":       "BDJ",
    "nilambur":       "NBR",
    # Tamil Nadu
    "chennai":        "MAS",
    "coimbatore":     "CBE",
    "madurai":        "MDU",
    "trichy":         "TPJ",
    "salem":          "SA",
    "tirunelveli":    "TEN",
    "nagercoil":      "NCJ",
    # Karnataka
    "bangalore":      "SBC",
    "bengaluru":      "SBC",
    "mysore":         "MYS",
    "mangalore":      "MAQ",
    "hubli":          "UBL",
    # Others
    "mumbai":         "CSTM",
    "delhi":          "NDLS",
    "hyderabad":      "HYB",
    "pune":           "PUNE",
    "ahmedabad":      "ADI",
    "kolkata":        "HWH",
    "goa":            "MAO",
}

def _fetch_trains(origin: str, destination: str, date: str) -> List[Dict]:
    if not RAPIDAPI_KEY:
        print("[TRANSPORT][irctc] No RAPIDAPI_KEY")
        return []

    # Skip immediately for places with no railway station
    if destination.lower().strip() in _NO_RAIL_CITIES:
        print(f"[TRANSPORT][irctc] {destination} has no rail station — skipping")
        return []

    # Use hardcoded codes first — avoids API timeout
    from_code = _KNOWN_STATIONS.get(origin.lower().strip())
    to_code   = _KNOWN_STATIONS.get(destination.lower().strip())

    # Only call API if not in hardcoded list
    if not from_code:
        from_code = _search_station_code(origin)
    if not to_code:
        to_code = _search_station_code(destination)

    if not from_code or not to_code:
        print(f"[TRANSPORT][irctc] Station not found: {origin}({from_code}) -> {destination}({to_code})")
        return []

    try:
        d = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
    except Exception:
        d = datetime.now().strftime("%Y%m%d")

    try:
        r = requests.get(
            IRCTC_TRAINS_URL,
            headers={"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": IRCTC_HOST},
            params={"fromStationCode": from_code, "toStationCode": to_code, "dateOfJourney": d},
            timeout=6,
        )
        print(f"[TRANSPORT][irctc] Trains HTTP {r.status_code}")
        if r.status_code in (401, 403):
            print("[TRANSPORT][irctc] Unauthorized — check RAPIDAPI_KEY and subscription")
            return []
        if r.status_code != 200:
            return []

        trains_raw = r.json().get("data", [])
        result = []
        for t in trains_raw[:3]:
            dep = t.get("departureTime", "")
            arr = t.get("arrivalTime", "")
            dur_min = _time_diff_mins(dep, arr)
            # Get cheapest fare class available
            fare = 0
            for cls in ["SL", "2S", "CC", "3A", "2A", "1A"]:
                for f in t.get("fares", []):
                    if f.get("classType") == cls:
                        try:
                            fare = int(f.get("fare", 0))
                            break
                        except Exception:
                            pass
                if fare:
                    break
            result.append({
                "route":         f"{origin} -> {destination} ({t.get('trainName','')} #{t.get('trainNumber','')})",
                "departure":     dep,
                "arrival":       arr,
                "duration_text": _mins_to_text(dur_min),
                "duration_min":  dur_min,
                "fare":          fare,
                "notes":         f"{t.get('trainName','')} ({t.get('trainNumber','')}) — book at irctc.co.in",
            })
        print(f"[TRANSPORT][irctc] Found {len(result)} trains")
        return result
    except Exception as e:
        print(f"[TRANSPORT][irctc] Error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# BUS OPTIONS  — grounded on real road distance
# ══════════════════════════════════════════════════════════════════════════════
def _build_bus_options(origin: str, destination: str,
                        dep_dt, dist_km: float, road_dur_min: float) -> List[Dict]:
    if dist_km <= 0:
        return []

    dep_txt = dep_dt.strftime("%I:%M %p") if dep_dt else "Morning"
    options = []

    # KSRTC Ordinary
    dur_ord  = road_dur_min * 1.25
    fare_ord = max(30, round(dist_km * KSRTC_ORDINARY_RATE / 10) * 10)
    arr_ord  = _add_duration(dep_dt, dur_ord)
    options.append({
        "mode":              "Bus (KSRTC Ordinary)",
        "route":             f"{origin} -> {destination} via KSRTC",
        "departure":         dep_txt,
        "arrival":           arr_ord.strftime("%I:%M %p") if arr_ord else "?",
        "duration":          _mins_to_text(dur_ord),
        "cost_per_person":   fare_ord,
        "sightseeing_day1":  (arr_ord.hour < 16) if arr_ord else False,
        "notes":             f"KSRTC Ordinary bus. ~{dist_km:.0f} km. Fare based on Rs {KSRTC_ORDINARY_RATE}/km rate. Verify at ksrtc.in.",
        "data_source":       "estimated",
    })

    # KSRTC Super Fast / Express
    dur_exp  = road_dur_min * 1.10
    fare_exp = max(50, round(dist_km * KSRTC_EXPRESS_RATE / 10) * 10)
    arr_exp  = _add_duration(dep_dt, dur_exp)
    options.append({
        "mode":              "Bus (KSRTC Super Fast)",
        "route":             f"{origin} -> {destination} via KSRTC Super Fast",
        "departure":         dep_txt,
        "arrival":           arr_exp.strftime("%I:%M %p") if arr_exp else "?",
        "duration":          _mins_to_text(dur_exp),
        "cost_per_person":   fare_exp,
        "sightseeing_day1":  (arr_exp.hour < 16) if arr_exp else False,
        "notes":             f"KSRTC Super Fast Express. Fewer stops. ~{dist_km:.0f} km. Verify at ksrtc.in.",
        "data_source":       "estimated",
    })

    return options


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK
# ══════════════════════════════════════════════════════════════════════════════
def _estimated_fallback(origin: str, destination: str, dist_km: float,
                          dur_min: float, dep_dt, dur_text: str) -> List[Dict]:
    dep_txt = dep_dt.strftime("%I:%M %p") if dep_dt else "Morning"
    return [{
        "mode":              "Bus (KSRTC)",
        "route":             f"{origin} -> {destination}",
        "departure":         dep_txt,
        "arrival":           "Check local schedule",
        "duration":          dur_text or "Check schedule",
        "cost_per_person":   max(30, int(dist_km * KSRTC_ORDINARY_RATE)) if dist_km else 0,
        "sightseeing_day1":  False,
        "notes":             "Verify times and fares at ksrtc.in or local bus stand.",
        "data_source":       "estimated",
    }]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _parse_time(time_str: str) -> Optional[datetime]:
    if not time_str:
        return None
    for fmt in ["%I:%M %p", "%H:%M", "%I %p", "%I:%M%p"]:
        try:
            t = datetime.strptime(time_str.strip().upper(), fmt.upper())
            now = datetime.now()
            return now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        except ValueError:
            continue
    return None


def _add_duration(base: Optional[datetime], minutes: float) -> Optional[datetime]:
    if base is None or not minutes:
        return None
    return base + timedelta(minutes=float(minutes))


def _mins_to_text(mins: float) -> str:
    h = int(mins // 60); m = int(mins % 60)
    if h == 0:  return f"{m} mins"
    if m == 0:  return f"{h} hr{'s' if h > 1 else ''}"
    return f"{h} hr{'s' if h > 1 else ''} {m} mins"


def _time_diff_mins(dep: str, arr: str) -> float:
    for fmt in ["%H:%M", "%I:%M %p", "%I:%M%p"]:
        try:
            d = datetime.strptime(dep.strip(), fmt)
            a = datetime.strptime(arr.strip(), fmt)
            diff = (a - d).total_seconds() / 60
            if diff < 0:
                diff += 24 * 60
            return diff
        except Exception:
            continue
    return 0