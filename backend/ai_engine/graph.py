"""
ai_engine/graph.py  — v4
========================
Changes in this version:
  1. TravelState gains: origin (starting city), travel_plan (how to get there),
     total_estimated_cost (computed after itinerary), num_travelers restored correctly.
  2. Info gathering now asks for ORIGIN first.
  3. Graph built as soon as destination is known.
  4. Itinerary includes:
       Day 0  — travel from origin to destination (transport options + cost)
       Day 1…N — activities at destination
       Last day — return travel from destination back to origin
  5. Total cost computed server-side and printed to terminal only (not sent to frontend).
  6. num_travelers bug fixed: DB restore now always reads from DB.
  7. stepfun removed from default model list (reasoning model).
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
from ai_engine.tools import fetch_places, fetch_stays
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
    origin:               Optional[str]   # NEW: where the user is travelling FROM
    destination:          Optional[str]
    start_date:           Optional[str]
    duration_days:        Optional[int]
    num_travelers:        Optional[int]
    budget:               Optional[str]
    user_preferences:     Dict[str, Any]
    # data
    places:               List[Dict[str, Any]]
    stays:                List[Dict[str, Any]]
    graph_data:           Optional[Dict[str, Any]]
    travel_plan:          Optional[Dict[str, Any]]  # NEW: origin→dest travel info
    itinerary:            Optional[Dict[str, Any]]
    # response
    final_response:       str
    conversation_history: List[Dict[str, str]]
    # flags
    core_info_complete:   bool
    has_graph_data:       bool
    graph_ready:          bool


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
    # origin is now required too
    return [f for f in ["origin","destination","start_date","duration_days","num_travelers","budget"]
            if not state.get(f)]

def _core_complete(state: TravelState) -> bool:
    return all(state.get(f) for f in
               ["origin","destination","start_date","duration_days","num_travelers","budget"])

def _regex_extract(msg: str, state: TravelState) -> dict:
    mc = msg.strip().lower()
    result = {}

    # Pattern 1: "I am from Kozhikode and want to go to Kochi" / "from X to Y"
    if not state.get("origin") or not state.get("destination"):
        from_to = re.search(
            r"(?:from|i'?m? from|i am from|travelling from|starting from)\s+([a-zA-Z\s]+?)"
            r"\s+(?:to|and (?:want to )?go to|visiting|heading to|going to)\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|!)",
            msg, re.IGNORECASE
        )
        if from_to:
            origin = from_to.group(1).strip().title()
            dest   = from_to.group(2).strip().title()
            if origin and origin.lower() not in _BAD_DEST:
                result["origin"] = origin
            if dest and dest.lower() not in _BAD_DEST:
                result["destination"] = dest

    # Pattern 2: standalone "i am from X" / "I'm from X" / "from X" (origin only, no destination part)
    if not state.get("origin") and not result.get("origin"):
        origin_only = re.search(
            r"\b(?:i(?:'?m| am)? from|my (?:home|city|place) is|i live in|based in|starting from)\s+([a-zA-Z][a-zA-Z\s]{1,30})(?:\s*$|,|\.|!)",
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

    # Duration
    if not state.get("duration_days"):
        d = re.search(r"\b(\d+)\s*(?:days?|d\b|nights?)", mc)
        if d:
            result["duration_days"] = int(d.group(1))
        elif re.fullmatch(r"\d+", mc.strip()) and "duration_days" in _missing_fields(state):
            result["duration_days"] = int(mc.strip())

    # Travelers
    if not state.get("num_travelers"):
        t = re.search(r"\b(\d+)\s*(?:people|persons?|travelers?|adults?|pax|of us)\b", mc)
        if t:
            result["num_travelers"] = int(t.group(1))
        elif re.fullmatch(r"\d+", mc.strip()) \
                and "num_travelers" in _missing_fields(state) \
                and "duration_days" not in result:
            result["num_travelers"] = int(mc.strip())

    # Budget
    if not state.get("budget"):
        b = re.search(r"[\$₹]?\s*(\d[\d,]+)\s*(?:rupees?|inr|usd|dollars?|eur|euros?)?", mc)
        if b:
            raw = b.group(1).replace(",", "")
            if int(raw) > 100:
                result["budget"] = raw

    # Single destination (only if neither origin nor dest found yet)
    if not state.get("destination") and not result.get("destination") and not result:
        if not re.search(r"\d", mc) and len(mc.split()) <= 4:
            if mc not in _BAD_DEST:
                result["destination"] = msg.strip().title()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# COST CALCULATOR  (terminal only — never sent to frontend)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_and_print_cost(state: TravelState):
    """Calculate total estimated cost and print to terminal only."""
    itin = state.get("itinerary", {})
    if not itin:
        return

    activity_total = 0.0
    stay_total     = 0.0
    travel_total   = 0.0
    n_travelers    = state.get("num_travelers", 1) or 1

    print("\n" + "─"*60)
    print(f"  💰 COST BREAKDOWN  ({n_travelers} traveler{'s' if n_travelers>1 else ''})")
    print("─"*60)

    for dp in itin.get("daily_plans", []):
        day_cost = 0.0
        print(f"\n  Day {dp['day_number']} — {dp.get('theme','')}")
        if dp.get("stay_name"):
            # Estimate stay cost from activity if tagged accommodation
            stay_act = next((a for a in dp.get("activities",[])
                             if "hotel" in (a.get("name","") + a.get("details","")).lower()
                             or "stay" in (a.get("name","") + a.get("details","")).lower()
                             or "accommodation" in (a.get("type","")).lower()), None)
            stay_cost = float(stay_act.get("estimatedCost", 2000) if stay_act else 2000)
            stay_total += stay_cost
            print(f"    🏨 Stay: {dp['stay_name']}  ₹{stay_cost:,.0f}/night")

        for act in dp.get("activities", []):
            cost = 0.0
            try:
                cost = float(str(act.get("estimatedCost", 0) or 0).replace(",",""))
            except Exception:
                pass
            name = act.get("name") or act.get("title") or "Activity"
            print(f"    • {name:<40} ₹{cost:>7,.0f}")
            day_cost += cost
            if "transport" in (act.get("type","")).lower() or \
               any(w in name.lower() for w in ["train","bus","flight","taxi","travel","return"]):
                travel_total += cost
            else:
                activity_total += cost

    total_per_person = stay_total + activity_total + travel_total
    total_all        = total_per_person * n_travelers
    budget           = float(str(state.get("budget","0")).replace(",","") or 0)

    print("\n" + "─"*60)
    print(f"  Stay costs       : ₹{stay_total:>10,.0f}")
    print(f"  Activity costs   : ₹{activity_total:>10,.0f}")
    print(f"  Transport costs  : ₹{travel_total:>10,.0f}")
    print(f"  ─────────────────────────────────")
    print(f"  Total (1 person) : ₹{total_per_person:>10,.0f}")
    if n_travelers > 1:
        print(f"  Total ({n_travelers} people): ₹{total_all:>10,.0f}")
    print(f"  Budget           : ₹{budget:>10,.0f}")
    diff = budget - total_all
    if diff >= 0:
        print(f"  ✅ Under budget  : ₹{diff:>10,.0f} remaining")
    else:
        print(f"  ⚠️  Over budget   : ₹{abs(diff):>10,.0f} over")
    print("─"*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fetch_graph_data_from_db(destination: str) -> Optional[Dict]:
    S = get_session_maker()
    with S() as s:
        try:
            region = s.query(Region).filter(Region.name.ilike(f"%{destination}%")).first()
            if not region:
                print(f"[DB] Region not found: {destination}"); return None
            graph = s.query(PlaceGraph).filter(
                PlaceGraph.regionId == region.id, PlaceGraph.status == "ACTIVE",
            ).order_by(PlaceGraph.version.desc()).first()
            if not graph:
                print(f"[DB] No active graph: {region.name}"); return None
            print(f"[DB] Graph: {graph.name} ({graph.totalNodes} nodes)")
            places = s.query(Place).filter_by(regionId=region.id).all()
            edges  = s.query(PlaceEdge).filter_by(graphId=graph.id).all()
            pd = {p.id: {
                "id": p.id, "name": p.name, "category": p.category.value,
                "latitude": p.latitude, "longitude": p.longitude,
                "description": p.description, "rating": p.rating,
                "typical_duration": p.typicalVisitDuration, "entry_fee": p.entryFee,
                "opening_hours": p.openingHours, "best_time": p.bestTimeToVisit,
                "tags": p.tags or [], "popularity": p.popularityScore,
            } for p in places}
            adj = {}; dl = {}
            for e in edges:
                adj.setdefault(e.fromPlaceId,[]).append({
                    "to": e.toPlaceId, "distance_km": e.roadDistanceKm,
                    "duration_min": e.durationMin, "transport": e.transportMode, "cost": e.travelCost,
                })
                dl[f"{e.fromPlaceId}:{e.toPlaceId}"] = {
                    "distance_km": e.roadDistanceKm, "duration_min": e.durationMin,
                    "transport": e.transportMode,
                }
            return {
                "region_id": region.id, "region_name": region.name,
                "graph_id": graph.id, "graph_version": graph.version,
                "places": pd, "adjacency": adj, "distance_lookup": dl,
                "total_nodes": len(pd), "total_edges": len(edges), "has_complete_graph": True,
            }
        except Exception as e:
            print(f"[DB ERROR] fetch_graph_data_from_db: {e}"); traceback.print_exc(); return None


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
                    if state.get("origin"):
                        t.description = f"From {state['origin']}"
                        # Also store in tripContext for reliable restore
                        ctx = t.tripContext or {}
                        ctx["origin"] = state["origin"]
                        t.tripContext = ctx
                    if state.get("destination"):  t.destination       = state["destination"]
                    if state.get("start_date"):   t.startDate         = datetime.fromisoformat(state["start_date"])
                    if state.get("duration_days"):t.duration          = state["duration_days"]
                    if state.get("num_travelers"):t.numberOfTravelers = int(state["num_travelers"])
                    if budget_val is not None:    t.totalBudget       = budget_val
                    t.status = TripStatus.PLANNING; t.updatedAt = datetime.utcnow()
                    s.commit(); return t.id
            tid = str(uuid.uuid4())
            s.add(Trip(
                id=tid, userId=state["user_id"],
                destination=state.get("destination","Unknown"),
                description=f"From {state.get('origin','')}" if state.get("origin") else None,
                tripContext={"origin": state.get("origin")} if state.get("origin") else None,
                title=f"Trip to {state.get('destination','Unknown')}",
                startDate=datetime.fromisoformat(state["start_date"]) if state.get("start_date") else None,
                duration=state.get("duration_days"),
                numberOfTravelers=int(state.get("num_travelers") or 1),
                totalBudget=budget_val, status=TripStatus.PLANNING,
                conversationPhase="gathering",
                createdAt=datetime.utcnow(), updatedAt=datetime.utcnow(),
            ))
            s.commit(); print(f"[DB] Trip created: {tid}"); return tid
        except Exception as e:
            s.rollback(); print(f"[DB ERROR] _upsert_trip: {e}"); traceback.print_exc(); raise


def _save_itinerary(state: TravelState) -> str:
    S = get_session_maker()
    with S() as s:
        try:
            data = state["itinerary"]; tid = state["trip_id"]
            ex   = s.query(Itinerary).filter_by(tripId=tid).first()
            if ex:
                itin = ex; itin.fullItinerary = data; itin.updatedAt = datetime.utcnow()
                s.query(ItineraryActivity).filter_by(itineraryId=itin.id).delete()
            else:
                itin = Itinerary(
                    id=str(uuid.uuid4()), tripId=tid,
                    summary=f"{state.get('duration_days')}-day trip to {state.get('destination')} from {state.get('origin','')}",
                    fullItinerary=data,
                    createdAt=datetime.utcnow(), updatedAt=datetime.utcnow(),
                )
                s.add(itin); s.flush()
            for dp in data.get("daily_plans", []):
                for idx, act in enumerate(dp.get("activities", [])):
                    title = act.get("name") or act.get("title") or "Activity"
                    desc  = act.get("details") or act.get("description") or ""
                    atype = ActivityType.SIGHTSEEING
                    combo = f"{title} {desc}".lower()
                    if any(w in combo for w in ["restaurant","lunch","dinner","breakfast","food","cafe"]):
                        atype = ActivityType.FOOD
                    elif any(w in combo for w in ["hotel","stay","check-in","resort","hostel"]):
                        atype = ActivityType.ACCOMMODATION
                    elif any(w in combo for w in ["train","bus","flight","taxi","transport","travel","depart","arrive","return","journey"]):
                        atype = ActivityType.TRANSPORT
                    loc = act.get("location", {})
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
            if trip:
                trip.status=TripStatus.PLANNED; trip.conversationPhase="generated"
                trip.updatedAt=datetime.utcnow(); s.commit()
            print(f"[DB] Itinerary saved: {itin.id}"); return itin.id
        except Exception as e:
            s.rollback(); print(f"[DB ERROR] _save_itinerary: {e}"); traceback.print_exc(); raise


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH PREPARATION — triggered as soon as destination is known
# ══════════════════════════════════════════════════════════════════════════════
def _prepare_graph(state: TravelState) -> TravelState:
    destination = state.get("destination","")
    if not destination or state.get("graph_ready"):
        return state
    print(f"\n[GRAPH_PREP] Preparing graph for: {destination}")
    graph_data = fetch_graph_data_from_db(destination)
    if not graph_data:
        print("[GRAPH_PREP] Not in DB — building now...")
        try:
            from graph_builder import build_and_save_graph
            graph_data = build_and_save_graph(destination, country="India")
        except Exception as e:
            print(f"[GRAPH_PREP] Build error: {e}"); graph_data = None
    if graph_data and graph_data.get("has_complete_graph"):
        state["graph_data"]  = graph_data; state["has_graph_data"] = True
        state["graph_ready"] = True
        state["places"]      = list(graph_data["places"].values())
        print(f"[GRAPH_PREP] ✅ {graph_data['total_nodes']} places, {graph_data['total_edges']} edges")
    else:
        print("[GRAPH_PREP] ⚠️ API fallback")
        state["has_graph_data"] = False; state["graph_ready"] = True
        state["places"] = fetch_places(destination, state["user_preferences"].get("interests",[]))
        print(f"[GRAPH_PREP] API: {len(state['places'])} places")
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
    state.setdefault("origin",None)

    # Always restore from DB
    if state.get("trip_id"):
        try:
            S = get_session_maker()
            with S() as s:
                trip = s.query(Trip).filter_by(id=state["trip_id"]).first()
                if trip:
                    if trip.destination and trip.destination not in ("New Trip","Unknown"):
                        state["destination"] = trip.destination
                    # Restore origin — check tripContext first, then description fallback
                    if not state.get("origin"):
                        if trip.tripContext and trip.tripContext.get("origin"):
                            state["origin"] = trip.tripContext["origin"]
                        elif trip.description and trip.description.startswith("From "):
                            state["origin"] = trip.description[5:].strip()
                    if trip.startDate and not state.get("start_date"):
                        state["start_date"] = trip.startDate.strftime("%Y-%m-%d")
                    if trip.duration and not state.get("duration_days"):
                        state["duration_days"] = trip.duration
                    if trip.numberOfTravelers and not state.get("num_travelers"):
                        state["num_travelers"] = int(trip.numberOfTravelers)
                    if trip.totalBudget and not state.get("budget"):
                        state["budget"] = str(int(trip.totalBudget))
                    if trip.tripContext and not state["user_preferences"]:
                        prefs = trip.tripContext.get("preferences",{})
                        if prefs: state["user_preferences"] = prefs
                    if not state.get("itinerary") and trip.itinerary:
                        state["itinerary"] = trip.itinerary.fullItinerary
                    state["core_info_complete"] = _core_complete(state)
                    print(f"[USER_MESSAGE_NODE] Restored: origin={state.get('origin')} "
                          f"dest={trip.destination} travelers={state.get('num_travelers')} "
                          f"complete={state['core_info_complete']}")
        except Exception as e:
            print(f"[USER_MESSAGE_NODE] DB restore error: {e}")

    # Build graph as soon as destination is known
    if state.get("destination") and not state.get("graph_ready"):
        state = _prepare_graph(state)
    return state


def phase_router_node(state):
    print(f"\n[PHASE_ROUTER_NODE] phase={state.get('current_phase','planning')}")
    if not state.get("itinerary"): state["current_phase"] = "planning"
    return state

def planning_node(state):   print("\n[PLANNING_NODE]");   return state
def preparation_node(state): state["final_response"]="Preparation phase coming soon."; return state
def trip_node(state):        state["final_response"]="Trip phase coming soon.";        return state
def post_trip_node(state):   state["final_response"]="Post-trip phase coming soon.";   return state


def planning_intent_node(state: TravelState) -> TravelState:
    print("\n[PLANNING_INTENT_NODE] Classifying intent")
    msg           = state["user_message"]
    core_complete = state.get("core_info_complete", False)
    has_itinerary = state.get("itinerary") is not None

    if _is_confirmation(msg):
        intent = "create_itinerary" if core_complete else "gather_info"
        print(f"[PLANNING_INTENT_NODE] Fast-path confirm → {intent}")
        state["detected_intent"] = intent; return state

    if _regex_extract(msg, state):
        print("[PLANNING_INTENT_NODE] Fast-path regex → gather_info")
        state["detected_intent"] = "gather_info"; return state

    sys_prompt = f"""Intent classifier for travel planning assistant.
Context: has_itinerary={has_itinerary}, core_complete={core_complete}
Origin={state.get('origin')}, Destination={state.get('destination')},
Start={state.get('start_date')}, Duration={state.get('duration_days')},
Travelers={state.get('num_travelers')}, Budget={state.get('budget')}

Classify: gather_info | create_itinerary | update_itinerary | ask_question | casual_chat
Return ONLY JSON: {{"intent":"...","confidence":0.0,"reasoning":"..."}}"""
    try:
        resp = llm_chat_with_retry(
            [{"role":"system","content":sys_prompt},{"role":"user","content":f"User: {msg}"}],
            temperature=0.0)
        r = _extract_json(resp)
        state["detected_intent"] = r["intent"] if r and "intent" in r else "gather_info"
        print(f"[PLANNING_INTENT_NODE] LLM → {state['detected_intent']}")
    except Exception as e:
        print(f"[PLANNING_INTENT_NODE] Error: {e}"); state["detected_intent"] = "gather_info"
    return state


def extract_info_node(state: TravelState) -> TravelState:
    print("\n[EXTRACT_INFO_NODE] Extracting")
    msg = state["user_message"]; changed = False

    rx = _regex_extract(msg, state)
    if rx:
        print(f"[EXTRACT_INFO_NODE] Regex: {rx}")
        for f in ["origin","destination","start_date","duration_days","num_travelers","budget"]:
            if rx.get(f) is not None:
                state[f] = rx[f]; changed = True
    else:
        current = {k: state.get(k) for k in
                   ["origin","destination","start_date","duration_days","num_travelers","budget"]}
        sys_prompt = """Extract trip info. Return ONLY JSON:
{"origin":str|null,"destination":str|null,"start_date":"YYYY-MM-DD"|null,
"duration_days":int|null,"num_travelers":int|null,"budget":str|null,
"preferences":[str],"new_info_extracted":bool}
origin = the city/place the user is TRAVELLING FROM (their home city).
destination = where they WANT TO GO."""
        try:
            resp = llm_chat_with_retry(
                [{"role":"system","content":sys_prompt},
                 {"role":"user","content":f"Current: {current}\nMessage: {msg}"}],
                temperature=0.0)
            r = _extract_json(resp)
            if r:
                for f in ["origin","destination","start_date","duration_days","num_travelers","budget"]:
                    if r.get(f) is not None: state[f]=r[f]; changed=True
                if r.get("preferences"):
                    state["user_preferences"].setdefault("interests",[])
                    state["user_preferences"]["interests"].extend(r["preferences"]); changed=True
        except Exception as e:
            print(f"[EXTRACT_INFO_NODE] LLM error: {e}"); traceback.print_exc()

    # Ensure num_travelers is int
    if state.get("num_travelers"):
        try: state["num_travelers"] = int(state["num_travelers"])
        except: state["num_travelers"] = 1

    state["core_info_complete"] = _core_complete(state)

    if changed and state.get("user_id"):
        try: state["trip_id"] = _upsert_trip(state)
        except Exception as e: print(f"[EXTRACT_INFO_NODE] DB error: {e}")

    # Build graph as soon as destination is known
    if state.get("destination") and not state.get("graph_ready"):
        print("[EXTRACT_INFO_NODE] Destination captured — preparing graph...")
        state = _prepare_graph(state)

    print(f"[EXTRACT_INFO_NODE] complete={state['core_info_complete']}, "
          f"graph_ready={state.get('graph_ready')}, places={len(state.get('places',[]))}")
    return state


def generate_question_node(state: TravelState) -> TravelState:
    print("\n[GENERATE_QUESTION_NODE]")
    qs = [
        ("origin",        "🏠 Where are you travelling **from**? (e.g. 'I am from Kozhikode')"),
        ("destination",   "🌍 Where would you like to go?"),
        ("start_date",    "📅 When are you planning to start? (e.g. April 15)"),
        ("duration_days", "⏳ How many days will you spend at the destination?"),
        ("num_travelers", "👥 How many people will be traveling?"),
        ("budget",        "💰 What is your total budget? (e.g. 50000)"),
    ]
    for field, question in qs:
        if not state.get(field):
            state["final_response"] = question; return state

    n         = state.get("num_travelers",1)
    n_places  = len(state.get("places",[]))
    src       = "graph DB" if state.get("has_graph_data") else "API"
    graph_line = (f"\n\n🗺️ Already loaded **{n_places} places** for {state.get('destination')} ({src})!"
                  if state.get("graph_ready") and n_places > 0 else "")

    state["final_response"] = (
        f"Perfect! Here's your trip summary:\n"
        f"🏠 **From:** {state.get('origin')}\n"
        f"📍 **To:** {state.get('destination')}\n"
        f"📅 **Start date:** {state.get('start_date')}\n"
        f"⏳ **Duration:** {state.get('duration_days')} days at destination\n"
        f"👥 **Travelers:** {n} person{'s' if n>1 else ''}\n"
        f"💰 **Budget:** ₹{state.get('budget')}"
        f"{graph_line}\n\n"
        f"The itinerary will include:\n"
        f"• **Day 1** — Travel from {state.get('origin')} to {state.get('destination')}\n"
        f"• **Days 2–{int(state.get('duration_days',1))+1}** — Sightseeing at {state.get('destination')}\n"
        f"• **Last day** — Return journey to {state.get('origin')}\n\n"
        "Shall I generate your itinerary? Just say **yes**! 🗺️"
    )
    return state


def _fmt_places(gd: dict, limit: int = 25) -> str:
    places = gd["places"]; adj = gd["adjacency"]; lines = []
    for pid, p in list(places.items())[:limit]:
        cs   = adj.get(pid,[])
        lat  = p.get("latitude",0); lon = p.get("longitude",0)
        line = f"{p['name']} [{p['category']}] ({lat:.4f},{lon:.4f})"
        nb   = [f"{places[c['to']]['name']} {c['duration_min']:.0f}min"
                for c in cs[:2] if c["to"] in places]
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
        if not state.get("graph_ready"):
            state = _prepare_graph(state)

        print("[CREATE_ITINERARY_NODE] Fetching stays...")
        stays = fetch_stays(destination=destination, budget=state.get("budget"))
        state["stays"] = stays
        print(f"  → {len(stays)} stays found")

        graph_data = state.get("graph_data")
        if state.get("has_graph_data") and graph_data:
            places_block = _fmt_places(graph_data, 25)
            data_label   = f"GRAPH DATA ({graph_data['total_nodes']} places with KNN connections)"
        else:
            places_block = json.dumps(state.get("places",[])[:18], indent=2)
            data_label   = "AVAILABLE PLACES"

        stays_compact = [{"name":s["name"],"type":s.get("type","hotel"),
                          "lat":s.get("latitude",0),"lon":s.get("longitude",0)}
                         for s in stays[:8]]

        n = int(state.get("num_travelers") or 1)
        duration = int(state.get("duration_days") or 3)

        # Total days in itinerary = 1 travel day + duration + 1 return day
        total_days = duration + 2
        # Date calculations
        start_dt = datetime.fromisoformat(state["start_date"]) if state.get("start_date") else datetime.now()
        dates = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(total_days)]

        old_itin_section = ""
        if state.get("itinerary"):
            old_itin_section = (
                "\n\nEXISTING ITINERARY (improve this, keep what is good):\n"
                + json.dumps(state["itinerary"], indent=2)
            )

        prompt = f"""You are a travel planner. Return ONLY a JSON itinerary, no explanation.

Trip: {origin} → {destination}
Start date: {state.get('start_date')} | Destination days: {duration} | Total days: {total_days}
Travelers: {n} | Budget: ₹{state.get('budget')}

{data_label}:
{places_block}

Stays at {destination} (pick one per destination day):
{json.dumps(stays_compact, separators=(',',':'))}
{old_itin_section}

RULES:
1. Day 1 MUST be travel day: {origin} → {destination}.
   Include real transport options: train/bus/flight with estimated travel cost per person.
   No sightseeing on Day 1 — only travel + check-in activity.
2. Days 2 to {duration+1}: sightseeing days at {destination}.
   Use ONLY places from the list. Group nearby places. 4-5 activities/day.
   Each day needs stay_name from stays list.
   Next day starts from previous day's stay.
3. Day {total_days} MUST be return travel day: {destination} → {origin}.
   Include transport back (train/bus). NO stay_name on last day.
   Last activity = arrive at {origin}.
4. Include estimatedCost (₹) per person for EVERY activity.
5. Return raw JSON only — no markdown.

JSON schema:
{{"destination":"{destination}","origin":"{origin}","start_date":"{state.get('start_date')}","duration_days":{total_days},"num_travelers":{n},"daily_plans":[{{"day_number":1,"date":"{dates[0]}","theme":"Travel Day: {origin} to {destination}","stay_name":"[hotel name]","activities":[{{"time":"Morning","name":"Depart from {origin}","details":"Board train/bus to {destination}","start_time":"06:00 AM","end_time":"12:00 PM","location":{{"lat":0.0,"lon":0.0}},"travel_from_previous":"Start of journey","estimatedCost":500}}]}},{{"day_number":{total_days},"date":"{dates[-1]}","theme":"Return: {destination} to {origin}","stay_name":null,"activities":[{{"time":"Morning","name":"Depart {destination}","details":"Board return train/bus to {origin}","start_time":"08:00 AM","end_time":"02:00 PM","location":{{"lat":0.0,"lon":0.0}},"travel_from_previous":"From last stay","estimatedCost":500}},{{"time":"Afternoon","name":"Arrive at {origin}","details":"Journey complete","start_time":"02:00 PM","end_time":"02:30 PM","location":{{"lat":0.0,"lon":0.0}},"travel_from_previous":"Final leg","estimatedCost":0}}]}}],"notes":{{"packing":"...","tips":"..."}}}}"""

        print("[CREATE_ITINERARY_NODE] Calling LLM...")
        resp = llm_chat_with_retry(
            [{"role":"system","content":prompt},
             {"role":"user","content":"Generate the complete itinerary JSON now."}],
            temperature=0.7, max_tokens=4000)

        itin = _extract_json(resp)
        if not itin:
            raise ValueError("LLM did not return valid JSON")

        # Ensure num_travelers is stored correctly in itinerary
        itin["num_travelers"] = n
        state["itinerary"] = itin
        print("[CREATE_ITINERARY_NODE] ✅ Itinerary generated")

        # Print cost breakdown to terminal only
        _compute_and_print_cost(state)

        if state.get("trip_id"):
            try:
                state["itinerary_id"] = _save_itinerary(state)
                print(f"[CREATE_ITINERARY_NODE] Saved: {state['itinerary_id']}")
            except Exception as e:
                print(f"[CREATE_ITINERARY_NODE] DB save error (non-fatal): {e}")

        src = "graph DB" if state.get("has_graph_data") else "API"
        state["final_response"] = (
            f"✅ Your {total_days}-day itinerary is ready!\n"
            f"🏠 **{origin}** → 📍 **{destination}** → 🏠 **{origin}**\n"
            f"({duration} days sightseeing + travel days, {n} traveler{'s' if n>1 else ''})"
        )

    except Exception as e:
        print(f"[CREATE_ITINERARY_NODE] ❌ {e}"); traceback.print_exc()
        state["final_response"] = "I had trouble generating your itinerary. Please try again."
    return state


def update_itinerary_node(state: TravelState) -> TravelState:
    print("\n[UPDATE_ITINERARY_NODE]")
    if not state.get("itinerary"):
        state["final_response"] = "No itinerary to update yet."; return state
    try:
        sys_prompt = (
            "Modify this travel itinerary per the user's request.\n\n"
            f"CURRENT:\n{json.dumps(state['itinerary'], indent=2)}\n\n"
            f"REQUEST: {state['user_message']}\n\n"
            "Return ONLY complete updated itinerary as raw JSON. No markdown."
        )
        resp = llm_chat_with_retry(
            [{"role":"system","content":sys_prompt},
             {"role":"user","content":"Return updated JSON."}],
            temperature=0.5, max_tokens=4000)
        updated = _extract_json(resp)
        if updated:
            state["itinerary"] = updated
            _compute_and_print_cost(state)
            if state.get("trip_id"):
                try: _save_itinerary(state)
                except Exception as e: print(f"[UPDATE] DB: {e}")
            state["final_response"] = "✅ Itinerary updated!"
        else:
            raise ValueError("Could not parse updated itinerary")
    except Exception as e:
        print(f"[UPDATE_ITINERARY_NODE] {e}")
        state["final_response"] = "I had trouble updating. Could you rephrase?"
    return state


def answer_question_node(state: TravelState) -> TravelState:
    try:
        resp = llm_chat_with_retry(
            [{"role":"system","content":f"Helpful travel assistant. Trip: {state.get('origin')} → {state.get('destination')}"},
             {"role":"user","content":state["user_message"]}])
        state["final_response"] = resp
    except Exception:
        state["final_response"] = "I'm here to help with your travel planning!"
    return state


def casual_chat_node(state: TravelState) -> TravelState:
    try:
        resp = llm_chat_with_retry(
            [{"role":"system","content":"Friendly travel assistant. 1-2 warm sentences then redirect to trip planning."},
             {"role":"user","content":state["user_message"]}])
        state["final_response"] = resp
        if not state.get("core_info_complete"):
            state = generate_question_node(state)
    except Exception:
        state["final_response"] = "Hello! I'm your travel assistant. Where would you like to go? 🌍"
    return state


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════
def route_phase(state) -> str:
    p = state.get("current_phase","planning")
    print(f"[ROUTER] phase → {p}")
    return {"planning":"planning_node","preparation":"preparation_node",
            "trip":"trip_node","post_trip":"post_trip_node"}.get(p,"planning_node")

def route_intent(state) -> str:
    i = state.get("detected_intent","gather_info")
    print(f"[ROUTER] intent → {i}")
    if i=="gather_info":       return "extract_info_node"
    if i=="create_itinerary":  return "create_itinerary_node" if state.get("core_info_complete") else "generate_question_node"
    if i=="update_itinerary":  return "update_itinerary_node" if state.get("itinerary") else "generate_question_node"
    if i=="ask_question":      return "answer_question_node"
    if i=="casual_chat":       return "casual_chat_node"
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