"""
Run this on your LOCAL machine (not in Claude sandbox) to verify both API keys.
  python test_keys.py
"""
import requests, json

RAPIDAPI_KEY    = "a907016344msheed8cb3412dcd24p10868djsnfde2c45695fd"
GOOGLE_MAPS_KEY = "AIzaSyB5aw_Q3RCGGGI7N9csV_e1cF7iTWHJp94"

PASS = "✅"; FAIL = "❌"; WARN = "⚠️ "

def sep(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")

# ── Test 1: Google Maps Distance Matrix ───────────────────────────────────────
sep("TEST 1 — Google Maps: Kochi → Munnar")
try:
    r = requests.get(
        "https://maps.googleapis.com/maps/api/distancematrix/json",
        params={
            "origins":      "Kochi, Kerala, India",
            "destinations": "Munnar, Kerala, India",
            "mode":         "driving",
            "units":        "metric",
            "key":          GOOGLE_MAPS_KEY,
        },
        timeout=10,
    )
    d = r.json()
    status = d.get("status")
    if status == "OK":
        el = d["rows"][0]["elements"][0]
        if el.get("status") == "OK":
            print(f"{PASS} Distance : {el['distance']['text']}")
            print(f"{PASS} Duration : {el['duration']['text']}")
            print(f"{PASS} Google Maps key is VALID and working!")
        else:
            print(f"{FAIL} Route element status: {el.get('status')}")
    elif status == "REQUEST_DENIED":
        print(f"{FAIL} Key rejected: {d.get('error_message','')}")
        print("   → Make sure 'Distance Matrix API' is enabled in Google Cloud Console")
    else:
        print(f"{FAIL} Unexpected status: {status}")
        print(json.dumps(d, indent=2))
except Exception as e:
    print(f"{FAIL} Error: {e}")

# ── Test 2: RapidAPI — IRCTC Station Search ───────────────────────────────────
sep("TEST 2 — RapidAPI IRCTC: Search 'Ernakulam'")
try:
    r = requests.get(
        "https://irctc1.p.rapidapi.com/api/v1/searchStation",
        headers={
            "X-RapidAPI-Key":  RAPIDAPI_KEY,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com",
        },
        params={"query": "Ernakulam"},
        timeout=10,
    )
    print(f"HTTP {r.status_code}")
    if r.status_code == 200:
        stations = r.json().get("data", [])
        for s in stations[:3]:
            print(f"  Station: {s.get('stationName')} ({s.get('stationCode')})")
        print(f"{PASS} IRCTC API key is VALID!")
    elif r.status_code == 401:
        print(f"{FAIL} Unauthorized — key invalid")
    elif r.status_code == 403:
        print(f"{FAIL} Not subscribed — go to:")
        print("   https://rapidapi.com/IRCTC/api/irctc1  → Subscribe (free tier)")
    else:
        print(f"{FAIL} HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    print(f"{FAIL} Error: {e}")

# ── Test 3: RapidAPI — Booking.com Hotel Search ───────────────────────────────
sep("TEST 3 — RapidAPI Booking.com: Hotels in Munnar")
try:
    r = requests.get(
        "https://booking-com.p.rapidapi.com/v1/hotels/search-by-coordinates",
        headers={
            "X-RapidAPI-Key":  RAPIDAPI_KEY,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com",
        },
        params={
            "latitude":       "10.0892",
            "longitude":      "77.0595",
            "checkin_date":   "2026-04-01",
            "checkout_date":  "2026-04-02",
            "adults_number":  "2",
            "room_number":    "1",
            "locale":         "en-gb",
            "currency":       "INR",
            "order_by":       "popularity",
            "page_number":    "0",
        },
        timeout=15,
    )
    print(f"HTTP {r.status_code}")
    if r.status_code == 200:
        hotels = r.json().get("result", [])
        print(f"Hotels found: {len(hotels)}")
        for h in hotels[:3]:
            price = h.get("min_total_price", "?")
            name  = h.get("hotel_name", "?")
            score = h.get("review_score", "?")
            print(f"  {name} — ₹{price}/stay — Score: {score}/10")
        print(f"{PASS} Booking.com API key is VALID!")
    elif r.status_code == 401:
        print(f"{FAIL} Unauthorized — key invalid")
    elif r.status_code == 403:
        print(f"{FAIL} Not subscribed — go to:")
        print("   https://rapidapi.com/tipsters/api/booking-com  → Subscribe (free tier)")
    else:
        print(f"{WARN} HTTP {r.status_code}: {r.text[:300]}")
except Exception as e:
    print(f"{FAIL} Error: {e}")

print(f"\n{'='*55}")
print("  Done. Fix any FAIL items above before deploying.")
print(f"{'='*55}\n")