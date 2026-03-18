"""
ai_engine/graph.py  — v5
========================
New in this version:
  1. Asks departure TIME ("10:30 AM") as part of gathering
  2. Asks travel mode: Private or Public
     - Public → searches bus/train/flight options (via LLM knowledge),
       presents them to user, asks to select one
     - Private → calculates drive time, adjusts Day 1 itinerary
  3. Consistent hotel pricing — uses one price per hotel per night
  4. Total cost breakdown shown with labels (what each amount is for)
  5. 401 API key error surfaced clearly to user
  6. If travel time allows sightseeing on Day 1 → includes it
"""
from typing import TypedDict, Literal, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    try:
        from langgraph.checkpoint import MemorySaver
    except ImportError:
        MemorySaver = None

import os, json, re, uuid, traceback
from datetime import datetime, timedelta

from ai_engine.llm import llm_chat_with_retry, _extract_json
from ai_engine.tools import fetch_places, fetch_stays, fetch_road_info
from database_models import (
    Trip, Itinerary, ItineraryActivity,
    Place, Region, PlaceGraph, PlaceEdge,
    TripStatus, ActivityType, ActivityStatus,
    get_session_maker,
)


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
class TravelState(TypedDict):
    user_id:              Optional[str]
    trip_id:              Optional[str]
    itinerary_id:         Optional[str]
    user_message:         str
    current_phase:        Literal["planning","preparation","trip","post_trip"]
    detected_intent:      Optional[str]
    # trip fields
    origin:               Optional[str]
    destination:          Optional[str]
    start_date:           Optional[str]
    departure_time:       Optional[str]   # e.g. "10:30 AM"
    travel_mode:          Optional[str]   # "private" or "public"
    selected_transport:   Optional[str]   # e.g. "Kottayam → Ernakulam train at 08:15, then bus"
    duration_days:        Optional[int]
    num_travelers:        Optional[int]
    budget:               Optional[str]
    user_preferences:     Dict[str, Any]
    # data
    places:               List[Dict[str, Any]]
    stays:                List[Dict[str, Any]]
    graph_data:           Optional[Dict[str, Any]]
    travel_options:       Optional[Dict[str, Any]]  # public transport options shown to user
    itinerary:            Optional[Dict[str, Any]]
    # response
    final_response:       str
    conversation_history: List[Dict[str, str]]
    # flags
    core_info_complete:   bool
    has_graph_data:       bool
    graph_ready:          bool
    awaiting_transport_selection: bool   # True when we've shown options and waiting for choice


# ══════════════════════════════════════════════════════════════════════════════
# REGEX / FAST-PATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
_MONTHS = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "january":1,"february":2,"march":3,"april":4,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}
_CONFIRM = {
    "yes","yeah","yep","yup","sure","ok","okay","go","go ahead","proceed",
    "generate","create","make it","let's go","do it","create it","generate it",
    "yes please","please","absolutely","of course","sure thing","sounds good",
}
_BAD_DEST = _CONFIRM | {"hi","hello","hey","thanks","thank you","no","nope","cancel","stop"}

def _is_confirmation(msg: str) -> bool:
    return msg.strip().lower() in _CONFIRM

def _missing_fields(state: TravelState) -> list:
    required = ["origin","destination","start_date","departure_time",
                "travel_mode","duration_days","num_travelers","budget"]
    missing = [f for f in required if not state.get(f)]
    # travel_mode selected → also need selected_transport for public
    if state.get("travel_mode") == "public" and not state.get("selected_transport"):
        if "selected_transport" not in missing:
            missing.append("selected_transport")
    return missing

def _core_complete(state: TravelState) -> bool:
    required = ["origin","destination","start_date","departure_time",
                "travel_mode","duration_days","num_travelers","budget"]
    if not all(state.get(f) for f in required):
        return False
    if state.get("travel_mode") == "public" and not state.get("selected_transport"):
        return False
    return True

def _regex_extract(msg: str, state: TravelState) -> dict:
    mc = msg.strip().lower()
    result = {}

    # "from X to Y" / "I am from X going to Y"
    if not state.get("origin") or not state.get("destination"):
        from_to = re.search(
            r"(?:from|i'?m? from|i am from|travelling from|starting from)\s+([a-zA-Z\s]+?)"
            r"\s+(?:to|and (?:want to )?go to|visiting|heading to|going to)\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|!)",
            msg, re.IGNORECASE
        )
        if from_to:
            origin = from_to.group(1).strip().title()
            dest   = from_to.group(2).strip().title()
            if origin and origin.lower() not in _BAD_DEST: result["origin"] = origin
            if dest   and dest.lower()   not in _BAD_DEST: result["destination"] = dest

    # Standalone "I am from X"
    if not state.get("origin") and not result.get("origin"):
        origin_only = re.search(
            r"\b(?:i(?:'?m| am)? from|my (?:home|city|place) is|i live in|based in|starting from)\s+"
            r"([a-zA-Z][a-zA-Z\s]{1,30})(?:\s*$|,|\.|!)",
            msg, re.IGNORECASE
        )
        if origin_only:
            candidate = origin_only.group(1).strip().title()
            if candidate.lower() not in _BAD_DEST and len(candidate.split()) <= 4:
                result["origin"] = candidate

    # Date
    if not state.get("start_date"):
        iso = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", mc)
        if iso:
            result["start_date"] = f"{iso.group(1)}-{int(iso.group(2)):02d}-{int(iso.group(3)):02d}"
        else:
            m = re.search(r"(?:(\d{1,2})\s+([a-z]+)|([a-z]+)\s+(\d{1,2}))(?:\s+(\d{4}))?", mc)
            if m:
                if m.group(1) and m.group(2):   day, mon = int(m.group(1)), m.group(2)
                elif m.group(3) and m.group(4): mon, day = m.group(3), int(m.group(4))
                else:                            mon, day = None, None
                if mon and mon[:3] in _MONTHS and day:
                    yr = int(m.group(5)) if m.group(5) else datetime.now().year
                    result["start_date"] = f"{yr}-{_MONTHS[mon[:3]]:02d}-{day:02d}"

    # Departure time: "10:30 AM", "10.30am", "6 AM", "at 8pm"
    # IMPORTANT: only match if am/pm is explicit OR a colon/dot separates hours:minutes.
    # A bare number like "28" (from "march 28") must NOT be treated as a time.
    if not state.get("departure_time") and not result.get("departure_time"):
        t = re.search(
            r"\b(\d{1,2})(?:[:.]\s*(\d{2}))\s*(?:am|pm|AM|PM)?\b"   # "10:30" or "10:30 AM"
            r"|\b(\d{1,2})\s*(am|pm|AM|PM)\b",                         # "6 AM" or "8pm" — explicit am/pm required
            mc
        )
        if t:
            if t.group(1):  # matched HH:MM[am/pm] form
                h = int(t.group(1)); mins = t.group(2) or "00"
                # look for trailing am/pm right after this match
                ampm_m = re.search(r"\b" + re.escape(t.group(0).strip()) + r"\s*(am|pm)", mc, re.IGNORECASE)
                ampm = (ampm_m.group(1) if ampm_m else "").upper()
                if not ampm:
                    ampm = "PM" if 12 <= h < 24 else "AM"
                    h = h % 12 or 12
            else:           # matched "N am/pm" form
                h = int(t.group(3)); mins = "00"; ampm = t.group(4).upper()
                if ampm == "PM" and h != 12: h += 12
                if ampm == "AM" and h == 12: h = 0
                h = h % 12 or 12
            result["departure_time"] = f"{h:02d}:{mins} {ampm}"

    # Travel mode
    if not state.get("travel_mode"):
        if re.search(r"\bprivate\b|\bmy (?:car|bike|vehicle)\b|\bown (?:car|vehicle)\b|\bdrive\b|\bdriving\b", mc):
            result["travel_mode"] = "private"
        elif re.search(r"\bpublic\b|\bbus\b|\btrain\b|\bflight\b|\bfly\b|\btransport\b", mc):
            result["travel_mode"] = "public"

    # Duration
    if not state.get("duration_days") and not result.get("departure_time"):
        d = re.search(r"\b(\d+)\s*(?:days?|d\b|nights?)", mc)
        if d: result["duration_days"] = int(d.group(1))
        elif re.fullmatch(r"\d+", mc.strip()):
            val = int(mc.strip())
            missing = _missing_fields(state)
            # A bare number goes to departure_time first if that's the next missing field
            if "departure_time" in missing and 0 <= val <= 23:
                # Treat bare hour like "8" → "08:00 AM/PM"
                ampm = "PM" if 12 <= val < 24 else "AM"
                h = val % 12 or 12
                result["departure_time"] = f"{h:02d}:00 {ampm}"
            elif "duration_days" in missing:
                result["duration_days"] = val

    # Travelers — explicit keyword match
    if not state.get("num_travelers"):
        t2 = re.search(r"\b(\d+)\s*(?:people|persons?|travelers?|adults?|pax|of us|friends?)\b", mc)
        if t2:
            result["num_travelers"] = int(t2.group(1))
        elif re.fullmatch(r"\d+", mc.strip()):
            val = int(mc.strip())
            missing = _missing_fields(state)
            # Assign bare number to num_travelers if:
            #   - duration_days already filled (or just extracted)
            #   - num_travelers is still missing
            #   - value is small (1-30) — travelers not a duration or budget
            if "num_travelers" in missing and "duration_days" not in result                     and state.get("duration_days") and 1 <= val <= 30:
                result["num_travelers"] = val

    # Budget
    if not state.get("budget"):
        b = re.search(r"[\$₹]?\s*(\d[\d,]+)\s*(?:rupees?|inr|usd|dollars?|eur|euros?)?", mc)
        if b:
            raw = b.group(1).replace(",","")
            if int(raw) > 100: result["budget"] = raw

    # Destination (fallback)
    if not state.get("destination") and not result.get("destination") and not result:
        if not re.search(r"\d", mc) and len(mc.split()) <= 4:
            if mc not in _BAD_DEST: result["destination"] = msg.strip().title()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TRANSPORT OPTIONS (public mode)  —  REAL DATA via Railway API + KSRTC
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_transport_options(state: TravelState) -> dict:
    """
    Fetch REAL transport options using:
      - irctc1.p.rapidapi.com  → live train schedules + fares
      - onlineksrtcswift.com   → KSRTC Kerala bus data
      - AbhiBus                → fallback bus data
    Falls back to labelled estimates only if both APIs fail.
    """
    from ai_engine.transport import fetch_transport_options as _real_fetch
    origin      = state.get("origin", "")
    destination = state.get("destination", "")
    date        = state.get("start_date", "")
    dep_time    = state.get("departure_time", "")
    try:
        return _real_fetch(origin, destination, date, dep_time)
    except Exception as e:
        print(f"[TRANSPORT] fetch_transport_options error: {e}")
        return None


def _format_transport_options(options_data: dict, n_travelers: int) -> str:
    """Format transport options as a readable message for the user."""
    if not options_data or not options_data.get("options"):
        return None

    opts        = options_data["options"]
    data_source = options_data.get("data_source", "live")
    booking     = options_data.get("booking_links", {})

    # Header
    if data_source == "estimated":
        lines = [
            "⚠️ **Live transport data unavailable** — showing estimated options.\n"
            "Please verify times & fares before booking.\n"
        ]
    else:
        lines = ["🔴 **Live transport data** from Railway API & KSRTC:\n"]

    for opt in opts:
        total      = opt.get("cost_per_person", 0) * n_travelers
        sight      = "✅ Sightseeing possible on arrival day!" if opt.get("sightseeing_day1") else "❌ Late arrival — sightseeing starts Day 2"
        src_badge  = " *(estimated)*" if opt.get("data_source") == "estimated" else " *(live)*"
        dep        = opt.get("departure","?")
        arr        = opt.get("arrival","?")
        dur        = opt.get("duration","")
        cost       = opt.get("cost_per_person", 0)

        lines.append(
            f"**Option {opt['id']}: {opt['mode']}**{src_badge}\n"
            f"  🛤️  Route: {opt['route']}\n"
            f"  🕐  Departs: {dep} → Arrives: {arr}"
            + (f"  ⏱️  ({dur})" if dur else "") + "\n"
            f"  💰  ₹{cost:,}/person"
            + (f"  (₹{total:,} for {n_travelers})" if n_travelers > 1 else "") + "\n"
            f"  {sight}\n"
            + (f"  📝  {opt['notes']}\n" if opt.get("notes") else "")
        )

    # Booking links footer
    link_parts = []
    if booking.get("train"):
        link_parts.append(f"🚆 Trains: {booking['train']}")
    if booking.get("ksrtc"):
        link_parts.append(f"🚌 KSRTC: {booking['ksrtc']}")
    if booking.get("redbus"):
        link_parts.append(f"🚌 RedBus: {booking['redbus']}")
    if link_parts:
        lines.append("\n**Book tickets:**\n" + "\n".join(link_parts))

    lines.append("\nWhich option would you like? Reply with the **option number** (e.g. *Option 1*).")
    return "\n".join(lines)


def _parse_transport_selection(msg: str, options_data: dict) -> Optional[dict]:
    """Parse user's transport option selection."""
    if not options_data: return None
    opts = options_data.get("options", [])
    # Match "option 1", "1", "first", "option 2" etc.
    m = re.search(r"\b(?:option\s*)?(\d+)\b", msg.lower())
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(opts):
            return opts[idx]
    # "first" → 1, "second" → 2, etc.
    words = {"first":0,"second":1,"third":2,"fourth":3,"fifth":4}
    for word, idx in words.items():
        if word in msg.lower() and idx < len(opts):
            return opts[idx]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# COST CALCULATOR  (terminal only)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_and_print_cost(state: TravelState):
    itin = state.get("itinerary", {})
    if not itin: return
    n = state.get("num_travelers", 1) or 1

    # Build hotel price map — one price per hotel to ensure consistency
    hotel_prices: dict = {}
    for dp in itin.get("daily_plans", []):
        stay = dp.get("stay_name","")
        if not stay: continue
        stay_act = next((a for a in dp.get("activities",[])
                        if any(w in (a.get("name","") + a.get("details","")).lower()
                               for w in ["hotel","stay","check-in","resort","hostel","accommodation"])), None)
        if stay_act and stay not in hotel_prices:
            try:
                hotel_prices[stay] = float(str(stay_act.get("estimatedCost",2000) or 2000).replace(",",""))
            except: hotel_prices[stay] = 2000

    stay_total = transport_total = activity_total = 0.0

    print("\n" + "─"*65)
    print(f"  💰 TRIP COST BREAKDOWN  ({n} traveler{'s' if n>1 else ''})")
    print(f"  {state.get('origin','')} → {state.get('destination','')} → {state.get('origin','')}")
    print("─"*65)

    for dp in itin.get("daily_plans", []):
        day_label = f"Day {dp['day_number']} — {dp.get('theme','')}"
        print(f"\n  {day_label}")
        stay = dp.get("stay_name","")
        if stay and stay != "null":
            night_cost = hotel_prices.get(stay, 2000)
            total_stay = night_cost * n
            stay_total += total_stay
            print(f"    🏨 Accommodation: {stay}")
            print(f"       ₹{night_cost:,.0f}/night × {n} = ₹{total_stay:,.0f}")

        for act in dp.get("activities", []):
            cost = 0.0
            try: cost = float(str(act.get("estimatedCost",0) or 0).replace(",",""))
            except: pass
            name = act.get("name") or act.get("title") or "Activity"
            atype = act.get("type","")
            is_transport = any(w in (name+atype).lower()
                              for w in ["train","bus","flight","taxi","travel","depart","arrive","return"])
            icon = "🚗" if is_transport else "🎯"
            cost_total = cost * n
            print(f"    {icon} {name:<42} ₹{cost:>6,.0f}/person  ₹{cost_total:>8,.0f} total")
            if is_transport: transport_total += cost_total
            else:            activity_total  += cost_total

    grand_total = stay_total + transport_total + activity_total
    budget      = float(str(state.get("budget","0")).replace(",","") or 0)
    per_person  = grand_total / n if n else grand_total

    print("\n" + "─"*65)
    print(f"  🏨 Accommodation total   : ₹{stay_total:>10,.0f}  (all nights, all travelers)")
    print(f"  🚗 Transport total       : ₹{transport_total:>10,.0f}  (travel to/from destination)")
    print(f"  🎯 Activities total      : ₹{activity_total:>10,.0f}  (entry fees, food, etc.)")
    print(f"  {'─'*45}")
    print(f"  💰 Grand total           : ₹{grand_total:>10,.0f}  ({n} traveler{'s' if n>1 else ''})")
    print(f"  💰 Per person            : ₹{per_person:>10,.0f}")
    print(f"  📊 Budget                : ₹{budget:>10,.0f}")
    diff = budget - grand_total
    if diff >= 0: print(f"  ✅ Under budget          : ₹{diff:>10,.0f} remaining")
    else:         print(f"  ⚠️  Over budget           : ₹{abs(diff):>10,.0f} over limit")
    print("─"*65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fetch_graph_data_from_db(destination: str) -> Optional[Dict]:
    S = get_session_maker()
    with S() as s:
        try:
            region = s.query(Region).filter(Region.name.ilike(f"%{destination}%")).first()
            if not region: print(f"[DB] Region not found: {destination}"); return None
            graph = s.query(PlaceGraph).filter(
                PlaceGraph.regionId == region.id, PlaceGraph.status == "ACTIVE",
            ).order_by(PlaceGraph.version.desc()).first()
            if not graph: print(f"[DB] No active graph: {region.name}"); return None
            print(f"[DB] Graph: {graph.name} ({graph.totalNodes} nodes, {graph.totalEdges} edges)")
            places = s.query(Place).filter_by(regionId=region.id).all()
            edges  = s.query(PlaceEdge).filter_by(graphId=graph.id).all()
            pd = {p.id: {
                "id":p.id,"name":p.name,"category":p.category.value,
                "latitude":p.latitude,"longitude":p.longitude,
                "description":p.description,"rating":p.rating,
                "typical_duration":p.typicalVisitDuration,"entry_fee":p.entryFee,
                "opening_hours":p.openingHours,"best_time":p.bestTimeToVisit,
                "tags":p.tags or [],"popularity":p.popularityScore,
            } for p in places}
            adj={}; dl={}
            for e in edges:
                adj.setdefault(e.fromPlaceId,[]).append({
                    "to":e.toPlaceId,"distance_km":e.roadDistanceKm,
                    "duration_min":e.durationMin,"transport":e.transportMode,"cost":e.travelCost,
                })
                dl[f"{e.fromPlaceId}:{e.toPlaceId}"]={"distance_km":e.roadDistanceKm,"duration_min":e.durationMin,"transport":e.transportMode}
            return {"region_id":region.id,"region_name":region.name,
                "graph_id":graph.id,"graph_version":graph.version,
                "places":pd,"adjacency":adj,"distance_lookup":dl,
                "total_nodes":len(pd),"total_edges":len(edges),"has_complete_graph":True}
        except Exception as e:
            print(f"[DB ERROR] {e}"); traceback.print_exc(); return None


def _upsert_trip(state: TravelState) -> str:
    S = get_session_maker()
    with S() as s:
        try:
            budget_val = None
            if state.get("budget"):
                try: budget_val = float(str(state["budget"]).replace("$","").replace(",","").replace("₹",""))
                except: pass
            if state.get("trip_id"):
                t = s.query(Trip).filter_by(id=state["trip_id"]).first()
                if t:
                    if state.get("destination"):    t.destination       = state["destination"]
                    if state.get("start_date"):     t.startDate         = datetime.fromisoformat(state["start_date"])
                    if state.get("duration_days"):  t.duration          = state["duration_days"]
                    if state.get("num_travelers"):  t.numberOfTravelers = state["num_travelers"]
                    if budget_val is not None:      t.totalBudget       = budget_val
                    if state.get("origin"):
                        t.description = f"From {state['origin']}"
                        ctx = t.tripContext or {}
                        ctx["origin"]           = state["origin"]
                        ctx["departure_time"]   = state.get("departure_time")
                        ctx["travel_mode"]      = state.get("travel_mode")
                        ctx["selected_transport"] = state.get("selected_transport")
                        t.tripContext = ctx
                    t.status = TripStatus.PLANNING; t.updatedAt = datetime.utcnow()
                    s.commit(); return t.id
            tid = str(uuid.uuid4())
            ctx = {
                "origin": state.get("origin"),
                "departure_time": state.get("departure_time"),
                "travel_mode": state.get("travel_mode"),
                "selected_transport": state.get("selected_transport"),
            }
            s.add(Trip(
                id=tid, userId=state["user_id"],
                destination=state.get("destination","Unknown"),
                title=f"Trip to {state.get('destination','Unknown')}",
                description=f"From {state.get('origin','')}" if state.get("origin") else None,
                startDate=datetime.fromisoformat(state["start_date"]) if state.get("start_date") else None,
                duration=state.get("duration_days"),
                numberOfTravelers=state.get("num_travelers",1),
                totalBudget=budget_val, status=TripStatus.PLANNING,
                conversationPhase="gathering", tripContext=ctx,
                createdAt=datetime.utcnow(), updatedAt=datetime.utcnow(),
            ))
            s.commit(); print(f"[DB] Trip: {tid}"); return tid
        except Exception as e:
            s.rollback(); print(f"[DB ERROR] _upsert_trip: {e}"); traceback.print_exc(); raise


def _save_itinerary(state: TravelState) -> str:
    S = get_session_maker()
    with S() as s:
        try:
            data = state["itinerary"]; tid = state["trip_id"]
            ex = s.query(Itinerary).filter_by(tripId=tid).first()
            if ex:
                itin = ex; itin.fullItinerary = data; itin.updatedAt = datetime.utcnow()
                s.query(ItineraryActivity).filter_by(itineraryId=itin.id).delete()
            else:
                itin = Itinerary(
                    id=str(uuid.uuid4()), tripId=tid,
                    summary=f"{state.get('duration_days')}-day trip to {state.get('destination')} from {state.get('origin','')}",
                    fullItinerary=data, createdAt=datetime.utcnow(), updatedAt=datetime.utcnow(),
                )
                s.add(itin); s.flush()
            for dp in data.get("daily_plans",[]):
                for idx, act in enumerate(dp.get("activities",[])):
                    title = act.get("name") or act.get("title") or "Activity"
                    desc  = act.get("details") or act.get("description") or ""
                    atype = ActivityType.SIGHTSEEING
                    combo = f"{title} {desc}".lower()
                    if any(w in combo for w in ["restaurant","lunch","dinner","breakfast","food","cafe"]): atype = ActivityType.FOOD
                    elif any(w in combo for w in ["hotel","stay","check-in","resort","hostel"]):           atype = ActivityType.ACCOMMODATION
                    elif any(w in combo for w in ["train","bus","flight","taxi","depart","arrive","travel"]): atype = ActivityType.TRANSPORT
                    loc = act.get("location",{})
                    loc_str = (f"{loc.get('lat',0)},{loc.get('lon',0)}" if isinstance(loc,dict) else str(loc or title))
                    cost = None
                    for k in ["cost","estimatedCost","estimated_cost"]:
                        if act.get(k):
                            try: cost=float(str(act[k]).replace("$","").replace(",","").replace("₹","")); break
                            except: pass
                    s.add(ItineraryActivity(
                        id=str(uuid.uuid4()), itineraryId=itin.id, dayNumber=dp["day_number"],
                        date=datetime.fromisoformat(dp["date"]) if dp.get("date") else None,
                        title=title, description=desc, type=atype, status=ActivityStatus.SUGGESTED,
                        location=loc_str,
                        latitude=loc.get("lat") if isinstance(loc,dict) else None,
                        longitude=loc.get("lon") if isinstance(loc,dict) else None,
                        startTime=act.get("start_time") or act.get("startTime"),
                        endTime=act.get("end_time") or act.get("endTime"),
                        orderIndex=idx, estimatedCost=cost,
                        createdAt=datetime.utcnow(), updatedAt=datetime.utcnow(),
                    ))
            s.commit()
            trip = s.query(Trip).filter_by(id=tid).first()
            if trip: trip.status=TripStatus.PLANNED; trip.conversationPhase="generated"; trip.updatedAt=datetime.utcnow(); s.commit()
            print(f"[DB] Itinerary: {itin.id}"); return itin.id
        except Exception as e:
            s.rollback(); print(f"[DB ERROR] _save_itinerary: {e}"); traceback.print_exc(); raise


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
def _prepare_graph(state: TravelState) -> TravelState:
    destination = state.get("destination","")
    if not destination or state.get("graph_ready"): return state
    print(f"\n[GRAPH_PREP] {destination}")
    graph_data = fetch_graph_data_from_db(destination)
    if not graph_data:
        print("[GRAPH_PREP] Building from OSM...")
        try:
            from graph_builder import build_and_save_graph
            graph_data = build_and_save_graph(destination, country="India")
        except Exception as e:
            print(f"[GRAPH_PREP] Build error: {e}"); graph_data = None
    if graph_data and graph_data.get("has_complete_graph"):
        state["graph_data"]=graph_data; state["has_graph_data"]=True; state["graph_ready"]=True
        state["places"]=list(graph_data["places"].values())
        print(f"[GRAPH_PREP] ✅ {graph_data['total_nodes']} places (v{graph_data.get('graph_version',1)})")
    else:
        print("[GRAPH_PREP] ⚠️ API fallback")
        state["has_graph_data"]=False; state["graph_ready"]=True
        state["places"]=fetch_places(destination=destination, interests=state["user_preferences"].get("interests",[]))
        print(f"[GRAPH_PREP] {len(state['places'])} places")
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════════════════════
def user_message_node(state: TravelState) -> TravelState:
    print(f"\n[USER_MESSAGE_NODE] {state['user_message']}")
    state.setdefault("conversation_history",[])
    state["conversation_history"].append({"role":"user","content":state["user_message"]})
    state.setdefault("current_phase","planning")
    state.setdefault("user_preferences",{})
    state.setdefault("has_graph_data",False)
    state.setdefault("graph_ready",False)
    state.setdefault("places",[])
    state.setdefault("stays",[])
    state.setdefault("awaiting_transport_selection",False)

    if state.get("trip_id"):
        try:
            S = get_session_maker()
            with S() as s:
                trip = s.query(Trip).filter_by(id=state["trip_id"]).first()
                if trip:
                    if trip.destination and trip.destination not in ("New Trip","Unknown"):
                        state["destination"] = trip.destination
                    if trip.startDate and not state.get("start_date"):
                        state["start_date"] = trip.startDate.strftime("%Y-%m-%d")
                    if trip.duration and not state.get("duration_days"):
                        state["duration_days"] = trip.duration
                    if trip.numberOfTravelers and not state.get("num_travelers"):
                        state["num_travelers"] = trip.numberOfTravelers
                    if trip.totalBudget and not state.get("budget"):
                        state["budget"] = str(int(trip.totalBudget))
                    if trip.tripContext:
                        ctx = trip.tripContext
                        if not state.get("origin")             and ctx.get("origin"):             state["origin"]             = ctx["origin"]
                        if not state.get("departure_time")     and ctx.get("departure_time"):     state["departure_time"]     = ctx["departure_time"]
                        if not state.get("travel_mode")        and ctx.get("travel_mode"):        state["travel_mode"]        = ctx["travel_mode"]
                        if not state.get("selected_transport") and ctx.get("selected_transport"): state["selected_transport"] = ctx["selected_transport"]
                        if not state["user_preferences"]       and ctx.get("preferences"):        state["user_preferences"]   = ctx["preferences"]
                    if not state.get("itinerary") and trip.itinerary:
                        state["itinerary"] = trip.itinerary.fullItinerary
                    state["core_info_complete"] = _core_complete(state)
                    print(f"[USER_MESSAGE_NODE] Restored: origin={state.get('origin')} dest={state.get('destination')} "
                          f"mode={state.get('travel_mode')} complete={state['core_info_complete']}")
        except Exception as e:
            print(f"[USER_MESSAGE_NODE] DB restore error: {e}")

    if state.get("destination") and not state.get("graph_ready"):
        state = _prepare_graph(state)
    return state


def phase_router_node(state):
    print(f"\n[PHASE_ROUTER_NODE] phase={state.get('current_phase','planning')}")
    if not state.get("itinerary"): state["current_phase"] = "planning"
    return state

def planning_node(state):   print("\n[PLANNING_NODE]"); return state
def preparation_node(state): state["final_response"]="Preparation phase coming soon."; return state
def trip_node(state):        state["final_response"]="Trip phase coming soon."; return state
def post_trip_node(state):   state["final_response"]="Post-trip phase coming soon."; return state


def planning_intent_node(state: TravelState) -> TravelState:
    print("\n[PLANNING_INTENT_NODE]")
    msg           = state["user_message"]
    core_complete = _core_complete(state)
    has_itinerary = state.get("itinerary") is not None

    # If waiting for transport selection, this message is a transport choice
    if state.get("awaiting_transport_selection") and state.get("travel_options"):
        sel = _parse_transport_selection(msg, state["travel_options"])
        if sel:
            state["detected_intent"] = "transport_selected"
            return state
        # Not a valid selection — treat as gather_info
        state["detected_intent"] = "gather_info"
        return state

    if _is_confirmation(msg):
        intent = "create_itinerary" if core_complete else "gather_info"
        print(f"[PLANNING_INTENT_NODE] Fast confirm → {intent}")
        state["detected_intent"] = intent; return state

    if _regex_extract(msg, state):
        print("[PLANNING_INTENT_NODE] Fast regex → gather_info")
        state["detected_intent"] = "gather_info"; return state

    # LLM for ambiguous
    sys_p = f"""Intent classifier. core_complete={core_complete}, has_itinerary={has_itinerary}
Fields: origin={state.get('origin')}, dest={state.get('destination')}, date={state.get('start_date')},
dep_time={state.get('departure_time')}, mode={state.get('travel_mode')},
days={state.get('duration_days')}, travelers={state.get('num_travelers')}, budget={state.get('budget')}

Intents: gather_info, create_itinerary, update_itinerary, ask_question, casual_chat
Return ONLY JSON: {{"intent":"...","confidence":0.0}}"""
    try:
        r = _extract_json(llm_chat_with_retry(
            [{"role":"system","content":sys_p},{"role":"user","content":f"User: {msg}"}], temperature=0.0))
        state["detected_intent"] = r["intent"] if r and "intent" in r else "gather_info"
    except Exception as e:
        print(f"[PLANNING_INTENT_NODE] Error: {e}"); state["detected_intent"] = "gather_info"
    return state


def extract_info_node(state: TravelState) -> TravelState:
    print("\n[EXTRACT_INFO_NODE]")
    msg = state["user_message"]; changed = False

    # Handle transport selection
    if state.get("awaiting_transport_selection") and state.get("travel_options"):
        sel = _parse_transport_selection(msg, state["travel_options"])
        if sel:
            desc = f"{sel['mode']}: {sel['route']} (departs {sel['departure']}, arrives {sel['arrival']}, ₹{sel.get('cost_per_person',0)}/person)"
            state["selected_transport"] = desc
            state["awaiting_transport_selection"] = False
            state["core_info_complete"] = _core_complete(state)
            if state.get("user_id"):
                try: state["trip_id"] = _upsert_trip(state)
                except Exception as e: print(f"[EXTRACT] DB: {e}")
            print(f"[EXTRACT_INFO_NODE] Transport selected: {desc}")
            return state

    rx = _regex_extract(msg, state)
    if rx:
        print(f"[EXTRACT_INFO_NODE] Regex: {rx}")
        for f in ["origin","destination","start_date","departure_time","travel_mode",
                  "duration_days","num_travelers","budget"]:
            if rx.get(f) is not None: state[f]=rx[f]; changed=True
    else:
        current = {k:state.get(k) for k in ["origin","destination","start_date","departure_time",
                                              "travel_mode","duration_days","num_travelers","budget"]}
        sys_p = """Extract trip info. Return ONLY JSON:
{"origin":str|null,"destination":str|null,"start_date":"YYYY-MM-DD"|null,
"departure_time":str|null,"travel_mode":"private"|"public"|null,
"num_travelers":int|null,"duration_days":int|null,"budget":str|null,
"preferences":[str],"new_info_extracted":bool}
departure_time = when they plan to leave (e.g. "08:30 AM")
travel_mode = "private" (own car) or "public" (bus/train/flight)
num_travelers = number of people travelling (asked BEFORE duration_days)"""
        try:
            r = _extract_json(llm_chat_with_retry(
                [{"role":"system","content":sys_p},
                 {"role":"user","content":f"Current: {current}\nMessage: {msg}"}], temperature=0.0))
            if r:
                for f in ["origin","destination","start_date","departure_time","travel_mode",
                          "duration_days","num_travelers","budget"]:
                    if r.get(f) is not None: state[f]=r[f]; changed=True
                if r.get("preferences"):
                    state["user_preferences"].setdefault("interests",[])
                    state["user_preferences"]["interests"].extend(r["preferences"]); changed=True
        except Exception as e:
            print(f"[EXTRACT_INFO_NODE] LLM error: {e}"); traceback.print_exc()

    state["core_info_complete"] = _core_complete(state)

    if changed and state.get("user_id"):
        try: state["trip_id"] = _upsert_trip(state)
        except Exception as e: print(f"[EXTRACT] DB: {e}")

    # Build graph as soon as destination is known
    if state.get("destination") and not state.get("graph_ready"):
        state = _prepare_graph(state)

    # If travel_mode just became "public" and we have all other fields, fetch options
    if (state.get("travel_mode") == "public"
            and not state.get("selected_transport")
            and not state.get("awaiting_transport_selection")
            and state.get("origin") and state.get("destination")
            and state.get("start_date") and state.get("departure_time")):
        print("[EXTRACT_INFO_NODE] Fetching public transport options...")
        opts = _fetch_transport_options(state)
        if opts:
            state["travel_options"] = opts
            state["awaiting_transport_selection"] = True

    print(f"[EXTRACT_INFO_NODE] complete={state['core_info_complete']} "
          f"mode={state.get('travel_mode')} awaiting_transport={state.get('awaiting_transport_selection')}")
    return state


def generate_question_node(state: TravelState) -> TravelState:
    print("\n[GENERATE_QUESTION_NODE]")

    # Show transport options if awaiting selection
    if state.get("awaiting_transport_selection") and state.get("travel_options"):
        n = state.get("num_travelers",1) or 1
        msg = _format_transport_options(state["travel_options"], n)
        if msg:
            state["final_response"] = msg
            return state

    qs = [
        ("origin",          "🏠 Where are you travelling **from**? (e.g. 'I am from Kottayam')"),
        ("destination",     "🌍 Where would you like to go?"),
        ("start_date",      "📅 What date are you planning to travel? (e.g. April 15)"),
        ("departure_time",  "🕐 What time are you planning to leave? (e.g. 8:30 AM)"),
        ("travel_mode",     "🚗 How are you travelling? Reply **private** (own car/bike) or **public** (bus/train/flight)"),
        ("duration_days",   "⏳ How many days will you spend at the destination?"),
        ("num_travelers",   "👥 How many people are travelling?"),
        ("budget",          "💰 What's your total budget for the trip? (e.g. 50000)"),
    ]
    for field, question in qs:
        if not state.get(field):
            # For public mode: skip duration_days and num_travelers until transport is selected
            if field in ("duration_days", "num_travelers") and state.get("travel_mode") == "public" \
                    and not state.get("selected_transport"):
                continue
            state["final_response"] = question
            return state

    # All collected
    n = state.get("num_travelers",1) or 1
    transport_info = ""
    if state.get("selected_transport"):
        transport_info = f"\n🚌 **Transport:** {state.get('selected_transport')}"
    elif state.get("travel_mode") == "private":
        transport_info = f"\n🚗 **Travel:** By private vehicle"

    state["final_response"] = (
        f"Perfect! Here's your trip summary:\n"
        f"🏠 **From:** {state.get('origin')}\n"
        f"🌍 **To:** {state.get('destination')}\n"
        f"📅 **Date:** {state.get('start_date')} at {state.get('departure_time')}\n"
        f"⏳ **Duration:** {state.get('duration_days')} days at destination\n"
        f"👥 **Travelers:** {n}\n"
        f"💰 **Budget:** ₹{state.get('budget')}"
        f"{transport_info}\n\n"
        "Shall I generate your itinerary? Just say **yes**! 🗺️"
    )
    return state


def _fmt_places(gd: dict, limit: int = 25) -> str:
    places=gd["places"]; adj=gd["adjacency"]; lines=[]
    for pid,p in list(places.items())[:limit]:
        cs=adj.get(pid,[]); lat=p.get("latitude",0); lon=p.get("longitude",0)
        line=f"{p['name']} [{p['category']}] ({lat:.4f},{lon:.4f})"
        nb=[f"{places[c['to']]['name']} {c['duration_min']:.0f}min" for c in cs[:2] if c["to"] in places]
        if nb: line += " → " + ", ".join(nb)
        lines.append(line)
    return "\n".join(lines)


def create_itinerary_node(state: TravelState) -> TravelState:
    destination = state.get("destination","")
    origin      = state.get("origin","your starting location")
    if not destination or destination.strip().lower() in _BAD_DEST:
        state["final_response"] = "I need to know your destination. Where would you like to go?"
        return state

    print(f"\n[CREATE_ITINERARY_NODE] {origin} → {destination}")
    try:
        if not state.get("graph_ready"): state = _prepare_graph(state)

        stays = fetch_stays(destination=destination, budget=state.get("budget"))
        state["stays"] = stays
        print(f"  → {len(stays)} stays")

        # Build consistent hotel price map (one price per hotel)
        hotel_price_notes = "IMPORTANT: Use ONE consistent price per hotel for ALL days. Do not vary the price."

        graph_data = state.get("graph_data")
        if state.get("has_graph_data") and graph_data:
            places_block = _fmt_places(graph_data, 25)
            data_label   = f"GRAPH DATA ({graph_data['total_nodes']} places with connections)"
        else:
            places_block = json.dumps(state.get("places",[])[:18], indent=2)
            data_label   = "AVAILABLE PLACES"

        stays_compact = [{"name":s["name"],"type":s.get("type","hotel"),
                          "lat":s.get("latitude",0),"lon":s.get("longitude",0)}
                         for s in stays[:8]]

        n         = int(state.get("num_travelers") or 1)
        duration  = int(state.get("duration_days") or 3)
        dep_time  = state.get("departure_time","morning")
        mode      = state.get("travel_mode","private")
        transport = state.get("selected_transport","")
        total_days = duration + 2  # travel + destination days + return

        start_dt = datetime.fromisoformat(state["start_date"]) if state.get("start_date") else datetime.now()
        dates    = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(total_days)]

        # Estimate arrival time for Day 1 sightseeing check
        if mode == "private":
            travel_note = f"Traveling by private vehicle from {origin}. Departure: {dep_time}."
            day1_sight  = "If arrival before 3 PM, include 1-2 nearby sightseeing spots on Day 1."
        elif transport:
            # Parse arrival from selected transport
            arr_match = re.search(r"arrives?\s+(\d{1,2}[:.]\d{2}\s*(?:AM|PM)?)", transport, re.IGNORECASE)
            arr_time  = arr_match.group(1) if arr_match else "evening"
            travel_note = f"Traveling by public transport: {transport}. Arrives ~{arr_time}."
            day1_sight  = f"If arrival before 3 PM ({arr_time}), include 1-2 sightseeing spots on Day 1 after check-in."
        else:
            travel_note = f"Traveling from {origin} to {destination}. Departure: {dep_time}."
            day1_sight  = "If arrival before 3 PM, include 1-2 nearby sightseeing spots on Day 1."

        old_itin_section = ""
        if state.get("itinerary"):
            old_itin_section = "\n\nEXISTING ITINERARY (improve, keep what is good):\n" + json.dumps(state["itinerary"], indent=2)

        prompt = f"""You are a travel planner. Return ONLY a JSON itinerary.

Trip: {origin} → {destination}, {state.get('start_date')}, {dep_time} departure
{n} travelers, budget ₹{state.get('budget')}, {duration} days at destination

{data_label}:
{places_block}

Stays (pick one per destination day, USE CONSISTENT PRICE for same hotel):
{json.dumps(stays_compact, separators=(',',':'))}

{hotel_price_notes}

Transport: {travel_note}
{old_itin_section}

RULES:
1. Day 1 = Travel day from {origin}.
   Include: depart {origin}, journey details, arrive {destination}, check-in.
   {day1_sight}
   estimatedCost = transport cost per person (₹).
2. Days 2 to {duration+1} = Full sightseeing days at {destination}.
   Use ONLY places from list. 4-5 activities/day. stay_name from stays list.
   Next day starts from previous stay.
3. Day {total_days} = Return travel {destination} → {origin}.
   No stay_name. estimatedCost = return transport cost.
4. estimatedCost is PER PERSON in ₹ for every activity.
5. For accommodation: estimatedCost = price per room per night (SAME price every time same hotel appears).
6. Return raw JSON only — no markdown.

Schema:
{{"destination":"{destination}","origin":"{origin}","start_date":"{state.get('start_date')}","duration_days":{total_days},"num_travelers":{n},"daily_plans":[{{"day_number":1,"date":"{dates[0]}","theme":"Travel Day: {origin} → {destination}","stay_name":"hotel name","activities":[{{"time":"Morning","name":"activity","details":"description","start_time":"09:00 AM","end_time":"11:00 AM","location":{{"lat":0.0,"lon":0.0}},"travel_from_previous":"X min by car","estimatedCost":500}}]}}],"notes":{{"packing":"...","tips":"..."}}}}"""

        print("[CREATE_ITINERARY_NODE] Calling LLM...")
        resp = llm_chat_with_retry(
            [{"role":"system","content":prompt},
             {"role":"user","content":"Generate the complete itinerary JSON now."}],
            temperature=0.7, max_tokens=4000)

        itin = _extract_json(resp)
        if not itin: raise ValueError("LLM did not return valid JSON")

        itin["num_travelers"] = n
        state["itinerary"] = itin
        print("[CREATE_ITINERARY_NODE] ✅ Done")
        _compute_and_print_cost(state)

        if state.get("trip_id"):
            try: state["itinerary_id"] = _save_itinerary(state); print(f"  Saved: {state['itinerary_id']}")
            except Exception as e: print(f"  DB save (non-fatal): {e}")

        state["final_response"] = (
            f"✅ Your {total_days}-day itinerary is ready!\n"
            f"🏠 **{origin}** → 📍 **{destination}** → 🏠 **{origin}**\n"
            f"({duration} days sightseeing, {n} traveler{'s' if n>1 else ''})\n\n"
            f"The full cost breakdown has been printed to the server terminal. 💰"
        )
    except ValueError as e:
        if "API key" in str(e):
            state["final_response"] = "❌ The OpenRouter API key is invalid or expired. Please check your `.env` file and restart the server."
        else:
            print(f"[CREATE_ITINERARY_NODE] ❌ {e}"); traceback.print_exc()
            state["final_response"] = "I had trouble generating your itinerary. Please try again."
    except Exception as e:
        print(f"[CREATE_ITINERARY_NODE] ❌ {e}"); traceback.print_exc()
        state["final_response"] = "I had trouble generating your itinerary. Please try again."
    return state


def update_itinerary_node(state: TravelState) -> TravelState:
    print("\n[UPDATE_ITINERARY_NODE]")
    if not state.get("itinerary"):
        state["final_response"] = "No itinerary to update yet."; return state
    try:
        sys_p = (f"Modify this travel itinerary.\n\nCURRENT:\n{json.dumps(state['itinerary'],indent=2)}\n\n"
                 f"REQUEST: {state['user_message']}\n\nReturn ONLY complete updated itinerary as raw JSON. "
                 f"Keep hotel prices CONSISTENT — same price for same hotel throughout.")
        updated = _extract_json(llm_chat_with_retry(
            [{"role":"system","content":sys_p},{"role":"user","content":"Return updated JSON."}],
            temperature=0.5, max_tokens=4000))
        if updated:
            state["itinerary"] = updated
            if state.get("trip_id"):
                try: _save_itinerary(state)
                except Exception as e: print(f"[UPDATE] DB: {e}")
            _compute_and_print_cost(state)
            state["final_response"] = "✅ Itinerary updated!"
        else:
            raise ValueError("Could not parse updated itinerary")
    except Exception as e:
        print(f"[UPDATE] {e}")
        state["final_response"] = "I had trouble updating. Could you rephrase?"
    return state


def answer_question_node(state: TravelState) -> TravelState:
    try:
        resp = llm_chat_with_retry(
            [{"role":"system","content":f"Helpful travel assistant. Trip: {state.get('origin')} → {state.get('destination')}"},
             {"role":"user","content":state["user_message"]}])
        state["final_response"] = resp
    except Exception: state["final_response"] = "I'm here to help with your travel planning!"
    return state


def casual_chat_node(state: TravelState) -> TravelState:
    try:
        resp = llm_chat_with_retry(
            [{"role":"system","content":"Friendly travel assistant. 1-2 sentences then redirect to planning."},
             {"role":"user","content":state["user_message"]}])
        state["final_response"] = resp
        if not _core_complete(state): state = generate_question_node(state)
    except Exception:
        state["final_response"] = "Hello! I'm your travel planning assistant. Where would you like to go?"
    return state


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════
def route_phase(state) -> str:
    p = state.get("current_phase","planning")
    return {"planning":"planning_node","preparation":"preparation_node",
            "trip":"trip_node","post_trip":"post_trip_node"}.get(p,"planning_node")

def route_intent(state) -> str:
    i = state.get("detected_intent","gather_info")
    print(f"[ROUTER] intent → {i}")
    if i == "transport_selected": return "extract_info_node"
    if i == "gather_info":        return "extract_info_node"
    if i == "create_itinerary":   return "create_itinerary_node" if _core_complete(state) else "generate_question_node"
    if i == "update_itinerary":   return "update_itinerary_node" if state.get("itinerary") else "generate_question_node"
    if i == "ask_question":       return "answer_question_node"
    if i == "casual_chat":        return "casual_chat_node"
    return "generate_question_node"

def route_after_extract(state) -> str:
    return "generate_question_node"


# ══════════════════════════════════════════════════════════════════════════════
# BUILD LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════
def create_travel_graph():
    wf = StateGraph(TravelState)
    for name, fn in [
        ("user_message_node",      user_message_node),
        ("phase_router_node",      phase_router_node),
        ("planning_node",          planning_node),
        ("preparation_node",       preparation_node),
        ("trip_node",              trip_node),
        ("post_trip_node",         post_trip_node),
        ("planning_intent_node",   planning_intent_node),
        ("extract_info_node",      extract_info_node),
        ("generate_question_node", generate_question_node),
        ("create_itinerary_node",  create_itinerary_node),
        ("update_itinerary_node",  update_itinerary_node),
        ("answer_question_node",   answer_question_node),
        ("casual_chat_node",       casual_chat_node),
    ]: wf.add_node(name, fn)

    wf.set_entry_point("user_message_node")
    wf.add_edge("user_message_node","phase_router_node")
    wf.add_conditional_edges("phase_router_node", route_phase,
        {"planning_node":"planning_node","preparation_node":"preparation_node",
         "trip_node":"trip_node","post_trip_node":"post_trip_node"})
    wf.add_edge("planning_node","planning_intent_node")
    wf.add_edge("preparation_node",END); wf.add_edge("trip_node",END); wf.add_edge("post_trip_node",END)
    wf.add_conditional_edges("planning_intent_node", route_intent,
        {"extract_info_node":"extract_info_node","create_itinerary_node":"create_itinerary_node",
         "update_itinerary_node":"update_itinerary_node","answer_question_node":"answer_question_node",
         "casual_chat_node":"casual_chat_node","generate_question_node":"generate_question_node"})
    wf.add_conditional_edges("extract_info_node", route_after_extract,
        {"generate_question_node":"generate_question_node"})
    for n in ["generate_question_node","create_itinerary_node","update_itinerary_node",
              "answer_question_node","casual_chat_node"]:
        wf.add_edge(n, END)

    checkpointer = MemorySaver() if MemorySaver else None
    return wf.compile(checkpointer=checkpointer)