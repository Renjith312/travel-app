"""
ai_engine/graph.py
==================
Workflow:
1. Classify message → core_info | general_question | itinerary_request
2. core_info       → extract fields → ask next missing field
3. general_question → LLM answer
4. itinerary_request:
   - core incomplete → ask missing fields
   - core complete + has itinerary → modify
   - core complete + no itinerary  → generate
5. Itinerary: single-prompt, all days, each day starts from previous day's ending location

FIX NOTES:
- LangGraph nodes must return a DICT of only the keys they changed.
  Mutating state in-place and returning the full state object causes
  double-appends on list fields (conversation_history etc.).
  All nodes now return only the keys they actually updated.
"""

from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
import json, uuid, traceback
from datetime import datetime, timedelta

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    try:
        from langgraph.checkpoint import MemorySaver
    except ImportError:
        MemorySaver = None

from ai_engine.llm import llm_chat_with_retry, _extract_json
from ai_engine.tools import fetch_places, fetch_stays
from database_models import (
    Trip, Itinerary, ItineraryActivity,
    Place, Region, PlaceGraph, PlaceEdge,
    TripStatus, ActivityType, ActivityStatus,
    get_session_maker,
)


# ══════════════════════════════════════════════════════════════
# TERMINAL DEBUG HELPERS
# ══════════════════════════════════════════════════════════════
W = 65

def _line(char="─"):    print(char * W)
def _header(title):     print(f"\n{'═'*W}\n  ▶  {title}\n{'═'*W}")
def _section(title):    print(f"\n  ┌─ {title}")
def _item(k, v):        print(f"  │   {k:<20} = {v!r}")
def _ok(msg):           print(f"  ✅  {msg}")
def _warn(msg):         print(f"  ⚠️   {msg}")
def _err(msg):          print(f"  ❌  {msg}")
def _arrow(a, b):       print(f"  {'─'*20}  {a}  →  {b}")

def _print_state_snapshot(state, label="STATE SNAPSHOT"):
    _section(label)
    for field, _ in CORE_FIELDS:
        val    = state.get(field)
        status = "✅" if val else "❌"
        print(f"  │   {status} {field:<18} = {val!r}")
    print(f"  │   {'core_complete':<20} = {_core_complete(state)}")
    print(f"  │   {'has_itinerary':<20} = {bool(state.get('itinerary'))}")
    print(f"  │   {'graph_ready':<20} = {state.get('graph_ready')}")
    print(f"  │   {'history_turns':<20} = {len(state.get('conversation_history') or [])}")


# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════
class TravelState(TypedDict):
    user_id:              Optional[str]
    trip_id:              Optional[str]
    itinerary_id:         Optional[str]
    user_message:         str
    detected_intent:      Optional[str]     # core_info | general_question | itinerary_request
    origin:               Optional[str]
    destination:          Optional[str]
    start_date:           Optional[str]
    departure_time:       Optional[str]
    travel_mode:          Optional[str]
    duration_days:        Optional[int]
    num_travelers:        Optional[int]
    budget:               Optional[str]
    user_preferences:     Dict[str, Any]
    places:               List[Dict[str, Any]]
    stays:                List[Dict[str, Any]]
    graph_data:           Optional[Dict[str, Any]]
    itinerary:            Optional[Dict[str, Any]]
    final_response:       str
    conversation_history: List[Dict[str, str]]
    core_info_complete:   bool
    graph_ready:          bool
    has_graph_data:       bool


# ══════════════════════════════════════════════════════════════
# CORE FIELDS — ordered by collection priority
# ══════════════════════════════════════════════════════════════
CORE_FIELDS = [
    ("destination",    "🌍 Where would you like to go?"),
    ("origin",         "🏠 Where are you travelling from? (e.g. 'I am from Kochi')"),
    ("start_date",     "📅 What date are you planning to travel? (e.g. April 15 or 2025-04-15)"),
    ("departure_time", "🕐 What time are you planning to leave? (e.g. 8:30 AM)"),
    ("travel_mode",    "🚗 How are you travelling? Reply **private** (own car/bike) or **public** (bus/train/flight)"),
    ("num_travelers",  "👥 How many people are travelling?"),
    ("duration_days",  "⏳ How many days will you spend at the destination?"),
    ("budget",         "💰 What is your total budget for the trip? (in ₹, e.g. 50000)"),
]

def _missing_fields(state) -> list:
    return [(f, q) for f, q in CORE_FIELDS if not state.get(f)]

def _core_complete(state) -> bool:
    return all(state.get(f) for f, _ in CORE_FIELDS)


# ══════════════════════════════════════════════════════════════
# MESSAGE BUILDER  (pass full conversation history to every LLM call)
# ══════════════════════════════════════════════════════════════
def _build_messages(state, system: str, include_current: bool = True) -> list:
    msgs = [{"role": "system", "content": system}]
    for turn in (state.get("conversation_history") or [])[-20:]:
        role = turn.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        msgs.append({"role": role, "content": turn.get("content", "")})
    if include_current:
        msg = state["user_message"]
        if not msgs or msgs[-1].get("content") != msg or msgs[-1].get("role") != "user":
            msgs.append({"role": "user", "content": msg})
    return msgs


# ══════════════════════════════════════════════════════════════
# NODE 1 — ENTRY  (initialise defaults, restore DB, add to history)
# ══════════════════════════════════════════════════════════════
def entry_node(state: TravelState) -> dict:
    _header("NODE: entry_node")
    print(f"  User message: {state['user_message']!r}")

    updates: dict = {}

    # Only set list/dict defaults if completely absent
    if not state.get("conversation_history"):
        updates["conversation_history"] = []
    if not state.get("user_preferences"):
        updates["user_preferences"] = {}
    if not state.get("places"):
        updates["places"] = []
    if not state.get("stays"):
        updates["stays"] = []
    if "has_graph_data" not in state:
        updates["has_graph_data"] = False
    if "graph_ready" not in state:
        updates["graph_ready"] = False

    # Merge for subsequent reads in this function
    merged = {**state, **updates}

    # ── Restore from DB ──────────────────────────────────────
    if merged.get("trip_id"):
        _section("DB restore")
        try:
            S = get_session_maker()
            with S() as s:
                trip = s.query(Trip).filter_by(id=merged["trip_id"]).first()
                if trip:
                    for db_field, st_field in [
                        ("destination",       "destination"),
                        ("duration",          "duration_days"),
                        ("numberOfTravelers", "num_travelers"),
                        ("totalBudget",       "budget"),
                    ]:
                        val = getattr(trip, db_field, None)
                        if val and not merged.get(st_field):
                            updates[st_field] = str(int(val)) if db_field == "totalBudget" else val
                            _item(f"restored {st_field}", updates[st_field])
                    if trip.startDate and not merged.get("start_date"):
                        updates["start_date"] = trip.startDate.strftime("%Y-%m-%d")
                    ctx = trip.tripContext or {}
                    for fld in ["origin", "departure_time", "travel_mode"]:
                        if not merged.get(fld) and ctx.get(fld):
                            updates[fld] = ctx[fld]
                            _item(f"restored {fld}", updates[fld])
                    if not merged.get("itinerary") and hasattr(trip, "itinerary") and trip.itinerary:
                        updates["itinerary"] = trip.itinerary.fullItinerary
                        _ok("Restored existing itinerary from DB")
                else:
                    _warn(f"trip_id {merged['trip_id']!r} not found in DB")
        except Exception as e:
            _err(f"DB restore failed: {e}")

    # ── Append user message to history (ONCE) ────────────────
    history = list(merged.get("conversation_history") or [])
    history.append({"role": "user", "content": state["user_message"]})
    updates["conversation_history"] = history

    # ── Recompute derived fields with latest merged state ─────
    merged.update(updates)
    # If the trip was submitted via form ("ready") or already generated ("generated"),
    # treat core info as complete — never ask for missing fields mid-chat.
    _phase_complete = False
    try:
        if merged.get("trip_id"):
            S = get_session_maker()
            with S() as s:
                _t = s.query(Trip).filter_by(id=merged["trip_id"]).first()
                if _t and _t.conversationPhase in ("ready", "generated"):
                    _phase_complete = True
    except Exception:
        pass
    updates["core_info_complete"] = True if _phase_complete else _core_complete(merged)

    # ── Prepare graph if destination already known ────────────
    if merged.get("destination") and not merged.get("graph_ready"):
        graph_updates = _prepare_graph(merged)
        updates.update(graph_updates)

    _print_state_snapshot({**merged, **updates}, "AFTER ENTRY")
    return updates


# ══════════════════════════════════════════════════════════════
# NODE 2 — CLASSIFY MESSAGE
# ══════════════════════════════════════════════════════════════
_CLASSIFY_SYSTEM = """\
Classify the user's message into exactly one of these intents:

- core_info         : message contains travel details (destination, dates, duration, budget, travelers, origin, travel mode)
- general_question  : user is asking a travel-related question not about changing the itinerary
- itinerary_request : user wants to generate, see, or modify/change the itinerary in any way

Respond ONLY with valid JSON: {"intent": "<one of the three>"}

Rules:
- If the message contains ANY trip detail, classify as core_info even if it also has a question.
- "yes", "go ahead", "generate it", "create it", "proceed", "ok", "sure" → itinerary_request
- A bare place name ("goa", "munnar") → core_info
- A single number likely to be duration or traveler count → core_info
- ANY request to add, remove, change, swap, replace, modify, update activities or days → itinerary_request
- "add X", "remove X", "change day N", "make it more X", "include X", "replace X with Y" → itinerary_request
- "more adventure", "budget friendly", "less activities", "add beach", "spa day" → itinerary_request
- "regenerate", "redo", "start over", "different plan" → itinerary_request
"""

CONFIRMATIONS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go", "proceed",
    "generate", "create", "do it", "go ahead", "let's go", "absolutely",
    "yes please", "sounds good", "sure thing",
}

def classify_message_node(state: TravelState) -> dict:
    _header("NODE: classify_message_node")
    msg = state["user_message"].strip()
    print(f"  Input: {msg!r}")

    # Fast-path for obvious confirmations
    if msg.lower() in CONFIRMATIONS:
        intent = "itinerary_request"
        _ok(f"Fast-path confirmation → {intent}")
        return {"detected_intent": intent}

    try:
        msgs = _build_messages(state, _CLASSIFY_SYSTEM)
        print(f"  Sending {len(msgs)} messages to LLM...")
        raw    = llm_chat_with_retry(msgs, temperature=0.0, max_tokens=60)
        print(f"  LLM raw: {raw!r}")
        r      = _extract_json(raw)
        intent = (r or {}).get("intent", "core_info")
        if intent not in ("core_info", "general_question", "itinerary_request"):
            _warn(f"Unknown intent {intent!r} → defaulting to core_info")
            intent = "core_info"
    except Exception as e:
        _err(f"LLM classify failed: {e}")
        intent = "core_info"

    _ok(f"Classified → {intent}")
    _arrow("classify_message_node", intent)
    return {"detected_intent": intent}


# ══════════════════════════════════════════════════════════════
# NODE 3 — EXTRACT CORE INFO
# ══════════════════════════════════════════════════════════════
_EXTRACT_SYSTEM = """\
You are a strict travel info extractor.

Extract ONLY what the user EXPLICITLY stated. Never guess or infer.

Return ONLY valid JSON (all fields nullable):
{
  "destination":    null,
  "origin":         null,
  "start_date":     null,
  "departure_time": null,
  "travel_mode":    null,
  "duration_days":  null,
  "num_travelers":  null,
  "budget":         null
}

Rules:
- destination   : where they want to GO. A bare place name = destination.
- origin        : only if they say "from X", "I live in X", "starting from X".
- start_date    : only if a date is stated. Format YYYY-MM-DD.
- departure_time: only if a time is stated. Format "HH:MM AM/PM".
- travel_mode   : "private" or "public" only if explicitly stated.
- duration_days : integer, only if "X days" or "X nights" stated.
- num_travelers : integer, only if "X people / X of us / X friends" stated.
- budget        : plain number string, only if a budget number is stated.
- Do NOT copy values from Already Known. Extract NEW info only.
"""

_NULL_VALUES = {None, "", "null", "None", "unknown",
                "not mentioned", "not specified", "n/a", "none"}

def extract_core_info_node(state: TravelState) -> dict:
    _header("NODE: extract_core_info_node")
    msg = state["user_message"]
    print(f"  Input: {msg!r}")

    already_known = {f: state.get(f) for f, _ in CORE_FIELDS}
    _section("Already known")
    for k, v in already_known.items():
        _item(k, v)

    system = _EXTRACT_SYSTEM + f"\n\nAlready known (do NOT re-extract): {json.dumps(already_known)}"

    try:
        msgs = _build_messages(state, system)
        print(f"\n  Sending {len(msgs)} messages to LLM...")
        raw       = llm_chat_with_retry(msgs, temperature=0.0, max_tokens=300)
        print(f"  LLM raw: {raw!r}")
        extracted = _extract_json(raw) or {}
    except Exception as e:
        _err(f"LLM extraction failed: {e}")
        extracted = {}

    _section("Extraction result")
    updates: dict = {}
    for field, _ in CORE_FIELDS:
        val = extracted.get(field)
        if val is None:
            continue
        str_val = str(val).strip().lower()
        if str_val in _NULL_VALUES:
            continue
        if already_known.get(field):
            print(f"  │   SKIP {field} (already {already_known[field]!r})")
        else:
            updates[field] = val
            _ok(f"NEW  {field} = {val!r}")

    if not updates:
        _warn("No new fields extracted from this message")

    # Merge to compute derived state
    merged = {**state, **updates}
    updates["core_info_complete"] = _core_complete(merged)

    # Persist to DB
    if merged.get("user_id") and updates:
        try:
            trip_id = _upsert_trip(merged)
            updates["trip_id"] = trip_id
            _ok(f"DB upserted trip_id={trip_id}")
        except Exception as e:
            _err(f"DB upsert: {e}")

    # Prepare graph if destination just became known
    if merged.get("destination") and not merged.get("graph_ready"):
        graph_updates = _prepare_graph(merged)
        updates.update(graph_updates)

    _print_state_snapshot({**state, **updates}, "AFTER EXTRACT")
    return updates


# ══════════════════════════════════════════════════════════════
# NODE 4 — ASK NEXT MISSING FIELD (or show summary)
# ══════════════════════════════════════════════════════════════
def ask_next_field_node(state: TravelState) -> dict:
    _header("NODE: ask_next_field_node")
    missing = _missing_fields(state)

    _section("Field status")
    for field, _ in CORE_FIELDS:
        val    = state.get(field)
        status = "✅" if val else "❌"
        print(f"  │   {status} {field:<18} = {val!r}")

    if missing:
        field, question = missing[0]
        _ok(f"Next question → asking for: {field}")
        response = question
    else:
        n = state.get("num_travelers", 1) or 1
        response = (
            f"Great! Here's your trip summary:\n\n"
            f"🏠 **From:** {state.get('origin')}\n"
            f"🌍 **To:** {state.get('destination')}\n"
            f"📅 **Date:** {state.get('start_date')} at {state.get('departure_time')}\n"
            f"🚗 **Mode:** {state.get('travel_mode')}\n"
            f"⏳ **Duration:** {state.get('duration_days')} days\n"
            f"👥 **Travelers:** {n}\n"
            f"💰 **Budget:** ₹{state.get('budget')}\n\n"
            "Shall I generate your itinerary? Just say **yes**! 🗺️"
        )
        _ok("All fields collected — showing summary")

    print(f"\n  Response: {response[:120]}")
    return {"final_response": response}


# ══════════════════════════════════════════════════════════════
# NODE 5 — ANSWER GENERAL QUESTION
# ══════════════════════════════════════════════════════════════
def answer_question_node(state: TravelState) -> dict:
    _header("NODE: answer_question_node")
    print(f"  Question: {state['user_message']!r}")
    try:
        system = (
            "You are a helpful travel assistant. "
            f"Trip context: {state.get('origin','?')} → {state.get('destination','?')}. "
            "Answer clearly and concisely."
        )
        msgs     = _build_messages(state, system)
        print(f"  Sending {len(msgs)} messages to LLM...")
        response = llm_chat_with_retry(msgs, max_tokens=600)
        _ok("Answer generated")
    except Exception as e:
        _err(f"LLM failed: {e}")
        response = "I'm here to help! Could you rephrase your question?"

    print(f"  Response: {response[:120]}")
    return {"final_response": response}


# ══════════════════════════════════════════════════════════════
# NODE 6a — GENERATE ITINERARY  (single prompt, all days)
# ══════════════════════════════════════════════════════════════
def _build_itinerary_prompt(state, place_names: list, stays: list) -> str:
    origin      = state.get("origin", "your city")
    destination = state.get("destination")
    n           = int(state.get("num_travelers") or 1)
    duration    = int(state.get("duration_days") or 3)
    dep_time    = state.get("departure_time", "morning")
    mode        = state.get("travel_mode", "private")
    budget      = int(
        str(state.get("budget", "10000"))
        .replace(",", "").replace("₹", "").replace("Rs", "")
    )

    total_days = duration + 2   # Day 1: travel out | middle: sightseeing | last: return
    start_dt   = (datetime.fromisoformat(state["start_date"])
                  if state.get("start_date") else datetime.now())
    dates      = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
                  for i in range(total_days)]

    first_hotel = stays[0]["name"] if stays else f"{destination} Hotel"
    hotel_names = ", ".join(s["name"] for s in stays[:4]) if stays else first_hotel
    places_list = ", ".join(place_names[:20]) if place_names else f"popular spots in {destination}"

    # Explicit per-day blueprint — LLM must honour wakes_at / sleeps_at
    blueprint = []
    for i in range(total_days):
        day  = i + 1
        date = dates[i]
        if i == 0:
            wakes_at  = origin
            plan      = (f"Travel day. Depart {origin} at {dep_time} by {mode}. "
                         f"Arrive {destination}, check in to {first_hotel}.")
            sleeps_at = first_hotel
        elif i == total_days - 1:
            wakes_at  = first_hotel
            plan      = (f"Return day. Check out from {first_hotel}. "
                         f"Travel from {destination} back to {origin}.")
            sleeps_at = "home (no accommodation)"
        else:
            wakes_at  = first_hotel
            plan      = f"Full sightseeing day in {destination}."
            sleeps_at = first_hotel
        blueprint.append(
            f"Day {day} ({date}): wakes_at=[{wakes_at}] | {plan} | sleeps_at=[{sleeps_at}]"
        )

    schema = json.dumps({
        "destination": destination, "origin": origin,
        "start_date": dates[0], "duration_days": total_days, "num_travelers": n,
        "daily_plans": [{
            "day_number": 1, "date": "YYYY-MM-DD",
            "theme": "short theme",
            "stay_name": "hotel name or null for return day",
            "activities": [{
                "name": "Activity name", "details": "One sentence description",
                "start_time": "09:00 AM", "end_time": "11:00 AM",
                "location": {"lat": 0.0, "lon": 0.0}, "estimatedCost": 200,
            }],
        }],
        "notes": {"packing": "...", "tips": "..."},
    }, indent=2)

    return (
        f"Generate a complete {total_days}-day travel itinerary as JSON.\n\n"
        f"TRIP DETAILS:\n"
        f"- Route      : {origin} → {destination} → {origin}\n"
        f"- Travelers  : {n}\n"
        f"- Budget     : ₹{budget} total (₹{budget // max(n, 1)}/person)\n"
        f"- Transport  : {mode}, departing at {dep_time}\n"
        f"- Hotels     : {hotel_names}\n"
        f"- Places     : {places_list}\n\n"
        f"DAY-BY-DAY BLUEPRINT (follow exactly — each day MUST start from its wakes_at):\n"
        + "\n".join(blueprint) +
        "\n\nRULES:\n"
        f"- Day 1 (travel)   : 3 activities — depart → en-route stop → check-in.\n"
        f"- Sightseeing days : 4–5 activities — morning → lunch → afternoon → evening/dinner.\n"
        f"- Return day       : 3 activities — checkout → travel → arrive home.\n"
        f"- stay_name        : hotel for that night; null on return day.\n"
        f"- estimatedCost    : plain INR integer per person (not a string).\n"
        f"- All coordinates  : realistic lat/lon near {destination}.\n"
        f"- Output ALL {total_days} days. Do NOT skip any day.\n\n"
        f"Output ONLY valid JSON matching this exact structure:\n{schema}"
    )


def generate_itinerary_node(state: TravelState) -> dict:
    _header("NODE: generate_itinerary_node")
    print(f"  Route: {state.get('origin')} → {state.get('destination')}")
    _print_state_snapshot(state, "INPUT STATE")

    updates: dict = {}

    # Prepare graph if not ready yet
    if not state.get("graph_ready"):
        graph_updates = _prepare_graph(state)
        updates.update(graph_updates)
        state = {**state, **graph_updates}   # local merge for this function only

    destination = state.get("destination")

    _section("Fetching stays")
    stays = fetch_stays(destination=destination, budget=state.get("budget"))
    updates["stays"] = stays
    _ok(f"{len(stays)} stays fetched")

    # Resolve place names
    graph_data = state.get("graph_data") or updates.get("graph_data")
    if state.get("has_graph_data") and graph_data:
        place_names = [p["name"] for p in list(graph_data["places"].values())[:20]]
        _ok(f"{len(place_names)} places from graph DB")
    else:
        place_names = [p["name"] for p in state.get("places", [])[:20]]
        _ok(f"{len(place_names)} places from API fallback")

    try:
        prompt = _build_itinerary_prompt({**state, **updates}, place_names, stays)
        system = (
            "You are a JSON-only travel itinerary generator. "
            "Output ONLY valid JSON. No markdown, no explanation. "
            "Your entire response must start with { and end with }."
        )
        msgs = _build_messages(state, system, include_current=False)
        msgs.append({"role": "user", "content": prompt})

        _section("Calling LLM — single prompt for full itinerary")
        print(f"  Prompt: {len(prompt)} chars  |  Total messages: {len(msgs)}")
        raw  = llm_chat_with_retry(msgs, temperature=0.15, max_tokens=6000)
        _ok(f"LLM responded ({len(raw)} chars)")
        print(f"  Raw excerpt: {raw[:200]}...")

        itin = _extract_json(raw)
        if not itin or not itin.get("daily_plans"):
            raise ValueError(f"LLM did not return a parseable itinerary.\nRaw: {raw[:500]}")

        itin.setdefault("destination",   destination)
        itin.setdefault("origin",        state.get("origin"))
        itin.setdefault("num_travelers", int(state.get("num_travelers") or 1))
        itin.setdefault("notes", {
            "packing": "Light clothes, sunscreen, comfortable shoes",
            "tips":    "Carry cash for local markets. Stay hydrated.",
        })

        updates["itinerary"] = itin
        total_days = len(itin["daily_plans"])
        _ok(f"Itinerary parsed: {total_days} days")

        _section("Day summary")
        for dp in itin["daily_plans"]:
            acts = len(dp.get("activities", []))
            stay = dp.get("stay_name") or "—"
            print(f"  │   Day {dp['day_number']:>2} ({dp.get('date','?')})  "
                  f"{dp.get('theme',''):<30}  {acts} acts  stay={stay}")

        # Save to DB
        merged = {**state, **updates}
        if merged.get("trip_id"):
            try:
                itin_id = _save_itinerary(merged)
                updates["itinerary_id"] = itin_id
                _ok(f"Saved to DB: itinerary_id={itin_id}")
            except Exception as e:
                _warn(f"DB save failed (non-fatal): {e}")

        n = int(state.get("num_travelers") or 1)
        updates["final_response"] = (
            f"✅ Your {total_days}-day itinerary is ready!\n"
            f"🏠 **{state.get('origin')}** → 📍 **{destination}** → 🏠 **{state.get('origin')}**\n"
            f"({state.get('duration_days')} sightseeing days, "
            f"{n} traveler{'s' if n > 1 else ''})\n\n"
            "Ask me anything about it or say **modify** to make changes! ✏️"
        )

    except Exception as e:
        _err(f"Itinerary generation failed: {e}")
        traceback.print_exc()
        updates["final_response"] = "I had trouble generating your itinerary. Please try again."

    return updates


# ══════════════════════════════════════════════════════════════
# NODE 6b — MODIFY ITINERARY
# ══════════════════════════════════════════════════════════════
def modify_itinerary_node(state: TravelState) -> dict:
    _header("NODE: modify_itinerary_node")
    print(f"  Request: {state['user_message']!r}")

    modification_request = state["user_message"]
    current_itin = state.get("itinerary")

    if not current_itin:
        _warn("No itinerary in state — cannot modify")
        return {"final_response": "I don't have an itinerary to modify yet. Please generate one first."}

    def _compact_itin(itin: dict) -> dict:
        """Strip heavy fields (details, location coords) to reduce token count."""
        out = {k: v for k, v in itin.items() if k != "daily_plans"}
        out["daily_plans"] = []
        for dp in itin.get("daily_plans", []):
            day = {
                "day_number": dp.get("day_number"),
                "date":       dp.get("date"),
                "theme":      dp.get("theme"),
                "stay_name":  dp.get("stay_name"),
                "activities": [],
            }
            for act in dp.get("activities", []):
                day["activities"].append({
                    "name":          act.get("name") or act.get("title"),
                    "start_time":    act.get("start_time"),
                    "end_time":      act.get("end_time"),
                    "estimatedCost": act.get("estimatedCost"),
                    "type":          act.get("type"),
                })
            out["daily_plans"].append(day)
        return out

    try:
        compact = _compact_itin(current_itin)
        system = (
            "You are a travel itinerary editor. "
            "Output ONLY valid JSON — no markdown, no explanation. "
            "Your entire response must start with { and end with }.\n\n"
            f"CURRENT ITINERARY (compact):\n{json.dumps(compact, indent=2)}"
        )
        user_prompt = (
            f"MODIFICATION REQUEST: {modification_request}\n\n"
            "Apply this change and return the COMPLETE updated itinerary JSON. "
            "Keep all unchanged days IDENTICAL to the original. "
            "Add a 'details' field to each activity with a one-sentence description. "
            "Keep the same JSON structure: destination, origin, start_date, duration_days, "
            "num_travelers, daily_plans (each with day_number, date, theme, stay_name, activities), notes."
        )
        msgs = [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user_prompt},
        ]
        print(f"  Sending modification request to LLM...")
        raw     = llm_chat_with_retry(msgs, temperature=0.3, max_tokens=8000)
        _ok(f"LLM responded ({len(raw)} chars)")
        updated = _extract_json(raw)

        if not updated:
            raise ValueError(f"Could not parse updated itinerary. Raw: {raw[:300]}")
        # Partial response (truncated) — fill missing days from the original
        if not updated.get("daily_plans"):
            raise ValueError(f"Parsed itinerary has no daily_plans. Raw: {raw[:300]}")
        orig_days = {dp["day_number"]: dp for dp in current_itin.get("daily_plans", [])}
        new_days  = {dp["day_number"]: dp for dp in updated.get("daily_plans", [])}
        if len(new_days) < len(orig_days):
            _warn(f"LLM returned {len(new_days)}/{len(orig_days)} days — filling missing days from original")
            for dn, dp in orig_days.items():
                if dn not in new_days:
                    updated["daily_plans"].append(dp)
            updated["daily_plans"].sort(key=lambda d: d.get("day_number", 0))

        _ok(f"Modification parsed: {len(updated['daily_plans'])} days")
        updates: dict = {"itinerary": updated}

        merged = {**state, **updates}
        if merged.get("trip_id"):
            try:
                _save_itinerary(merged)
                _ok("Saved modified itinerary to DB")
            except Exception as e:
                _warn(f"DB save failed (non-fatal): {e}")

        updates["final_response"] = "✅ Done! Your itinerary has been updated. Ask for more changes anytime."
        return updates

    except Exception as e:
        _err(f"Modify failed: {e}")
        traceback.print_exc()
        return {"final_response": "I had trouble applying that change. Could you rephrase your request?"}


# ══════════════════════════════════════════════════════════════
# NODE 7 — APPEND ASSISTANT RESPONSE TO HISTORY
# ══════════════════════════════════════════════════════════════
def add_assistant_response_node(state: TravelState) -> dict:
    _header("NODE: add_assistant_response_node")
    response = state.get("final_response", "")
    if response:
        history = list(state.get("conversation_history") or [])
        history.append({"role": "assistant", "content": response})
        _ok(f"History now {len(history)} turns")
        _line()
        print(f"  FINAL RESPONSE:\n  {response[:300]}{'...' if len(response) > 300 else ''}")
        _line()
        return {"conversation_history": history}
    _warn("No final_response to append")
    return {}


# ══════════════════════════════════════════════════════════════
# GRAPH DATA  (DB → OSM builder → API fallback)
# Returns a dict of state updates — never mutates input.
# ══════════════════════════════════════════════════════════════
def fetch_graph_data_from_db(destination: str):
    S = get_session_maker()
    with S() as s:
        try:
            region = s.query(Region).filter(Region.name.ilike(f"%{destination}%")).first()
            if not region:
                _warn(f"Region not found in DB: {destination!r}")
                return None
            graph = s.query(PlaceGraph).filter_by(
                regionId=region.id, status="ACTIVE"
            ).order_by(PlaceGraph.version.desc()).first()
            if not graph:
                _warn(f"No active PlaceGraph for: {region.name}")
                return None
            places = s.query(Place).filter_by(regionId=region.id).all()
            edges  = s.query(PlaceEdge).filter_by(graphId=graph.id).all()
            pd_ = {p.id: {
                "id": p.id, "name": p.name, "category": p.category.value,
                "latitude": p.latitude, "longitude": p.longitude,
                "description": p.description, "rating": p.rating,
            } for p in places}
            adj = {}
            for e in edges:
                adj.setdefault(e.fromPlaceId, []).append({
                    "to": e.toPlaceId, "distance_km": e.roadDistanceKm,
                    "duration_min": e.durationMin, "transport": e.transportMode,
                })
            _ok(f"DB graph loaded: {len(pd_)} places, {len(edges)} edges")
            return {
                "region_name":        region.name,
                "places":             pd_,
                "adjacency":          adj,
                "total_nodes":        len(pd_),
                "total_edges":        len(edges),
                "has_complete_graph": True,
            }
        except Exception as e:
            _err(f"DB graph fetch error: {e}")
            return None


def _prepare_graph(state: dict) -> dict:
    """Returns a dict of graph-related state updates. Never mutates input."""
    destination = state.get("destination", "")
    if not destination or state.get("graph_ready"):
        return {}

    _section(f"Preparing graph for {destination!r}")
    graph_data = fetch_graph_data_from_db(destination)

    if not graph_data:
        _warn("DB miss — attempting OSM graph builder")
        try:
            from graph_builder import build_and_save_graph
            graph_data = build_and_save_graph(destination, country="India")
            _ok("Graph built from OSM")
        except Exception as e:
            _warn(f"OSM build failed: {e}")
            graph_data = None

    if graph_data and graph_data.get("has_complete_graph"):
        _ok(f"Graph ready: {graph_data['total_nodes']} places")
        return {
            "graph_data":     graph_data,
            "has_graph_data": True,
            "graph_ready":    True,
            "places":         list(graph_data["places"].values()),
        }
    else:
        _warn("Falling back to API place fetch")
        places = fetch_places(
            destination=destination,
            interests=(state.get("user_preferences") or {}).get("interests", [])
        )
        _ok(f"API fallback: {len(places)} places")
        return {
            "has_graph_data": False,
            "graph_ready":    True,
            "places":         places,
        }


# ══════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════
def _upsert_trip(state: dict) -> str:
    S = get_session_maker()
    with S() as s:
        try:
            budget_val = None
            try:
                budget_val = float(
                    str(state.get("budget", "") or "")
                    .replace(",", "").replace("₹", "").replace("Rs", "")
                )
            except Exception:
                pass

            if state.get("trip_id"):
                t = s.query(Trip).filter_by(id=state["trip_id"]).first()
                if t:
                    if state.get("destination"):   t.destination       = state["destination"]
                    if state.get("start_date"):    t.startDate         = datetime.fromisoformat(state["start_date"])
                    if state.get("duration_days"): t.duration          = state["duration_days"]
                    if state.get("num_travelers"): t.numberOfTravelers = state["num_travelers"]
                    if budget_val is not None:     t.totalBudget       = budget_val
                    ctx = t.tripContext or {}
                    ctx.update({k: state.get(k) for k in ["origin", "departure_time", "travel_mode"]})
                    t.tripContext = ctx
                    t.updatedAt   = datetime.utcnow()
                    s.commit()
                    return t.id

            tid = str(uuid.uuid4())
            s.add(Trip(
                id=tid,
                userId=state.get("user_id"),
                destination=state.get("destination", "Unknown"),
                title=f"Trip to {state.get('destination', 'Unknown')}",
                startDate=(datetime.fromisoformat(state["start_date"])
                           if state.get("start_date") else None),
                duration=state.get("duration_days"),
                numberOfTravelers=state.get("num_travelers", 1),
                totalBudget=budget_val,
                status=TripStatus.PLANNING,
                tripContext={k: state.get(k) for k in ["origin", "departure_time", "travel_mode"]},
                createdAt=datetime.utcnow(),
                updatedAt=datetime.utcnow(),
            ))
            s.commit()
            return tid
        except Exception as e:
            s.rollback()
            _err(f"_upsert_trip: {e}")
            raise


def _save_itinerary(state: dict) -> str:
    S = get_session_maker()
    with S() as s:
        try:
            data = state["itinerary"]
            tid  = state["trip_id"]
            ex   = s.query(Itinerary).filter_by(tripId=tid).first()
            if ex:
                itin           = ex
                itin.fullItinerary = data
                itin.updatedAt     = datetime.utcnow()
                s.query(ItineraryActivity).filter_by(itineraryId=itin.id).delete()
            else:
                itin = Itinerary(
                    id=str(uuid.uuid4()), tripId=tid,
                    summary=f"{state.get('duration_days')}-day trip to {state.get('destination')}",
                    fullItinerary=data,
                    createdAt=datetime.utcnow(), updatedAt=datetime.utcnow(),
                )
                s.add(itin)
                s.flush()

            for dp in data.get("daily_plans", []):
                for idx, act in enumerate(dp.get("activities", [])):
                    title = act.get("name") or act.get("title") or "Activity"
                    combo = f"{title} {act.get('details', '')}".lower()
                    atype = ActivityType.SIGHTSEEING
                    if any(w in combo for w in ["restaurant","lunch","dinner","breakfast","food","cafe"]):
                        atype = ActivityType.FOOD
                    elif any(w in combo for w in ["hotel","stay","check-in","resort"]):
                        atype = ActivityType.ACCOMMODATION
                    elif any(w in combo for w in ["train","bus","flight","taxi","depart","arrive","travel"]):
                        atype = ActivityType.TRANSPORT
                    loc  = act.get("location", {})
                    cost = None
                    try:
                        cost = float(str(act.get("estimatedCost", 0) or 0).replace(",", ""))
                    except Exception:
                        pass
                    s.add(ItineraryActivity(
                        id=str(uuid.uuid4()), itineraryId=itin.id,
                        dayNumber=dp["day_number"],
                        date=(datetime.fromisoformat(dp["date"]) if dp.get("date") else None),
                        title=title, description=act.get("details", ""),
                        type=atype, status=ActivityStatus.SUGGESTED,
                        location=(f"{loc.get('lat',0)},{loc.get('lon',0)}"
                                  if isinstance(loc, dict) else str(loc or title)),
                        latitude=(loc.get("lat")  if isinstance(loc, dict) else None),
                        longitude=(loc.get("lon") if isinstance(loc, dict) else None),
                        startTime=act.get("start_time"), endTime=act.get("end_time"),
                        orderIndex=idx, estimatedCost=cost,
                        createdAt=datetime.utcnow(), updatedAt=datetime.utcnow(),
                    ))

            s.commit()
            trip = s.query(Trip).filter_by(id=tid).first()
            if trip:
                trip.status            = TripStatus.PLANNED
                trip.conversationPhase = "generated"
                trip.updatedAt         = datetime.utcnow()
                s.commit()
            return itin.id
        except Exception as e:
            s.rollback()
            _err(f"_save_itinerary: {e}")
            raise


# ══════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════
def route_after_classify(state: TravelState) -> str:
    intent         = state.get("detected_intent", "core_info")
    # Use the flag set by entry_node (respects form phase) rather than recomputing
    core_complete  = state.get("core_info_complete") or _core_complete(state)
    has_itinerary  = bool(state.get("itinerary"))
    _section(f"ROUTER  intent={intent!r}  core_complete={core_complete}  has_itinerary={has_itinerary}")

    if intent == "core_info":
        dest = "extract_core_info_node"

    elif intent == "general_question":
        dest = "answer_question_node"

    else:   # itinerary_request
        if not core_complete:
            _warn("Core info incomplete — redirecting to ask_next_field_node")
            dest = "ask_next_field_node"
        elif has_itinerary:
            dest = "modify_itinerary_node"
        else:
            dest = "generate_itinerary_node"

    _arrow("classify_message_node", dest)
    return dest


def route_after_extract(state: TravelState) -> str:
    _section("ROUTER  after extract → ask_next_field_node")
    return "ask_next_field_node"


# ══════════════════════════════════════════════════════════════
# BUILD LANGGRAPH
# ══════════════════════════════════════════════════════════════
def create_travel_graph():
    print("\n" + "═" * W)
    print("  Building LangGraph travel planning graph...")
    print("═" * W)

    wf = StateGraph(TravelState)

    nodes = {
        "entry_node":                  entry_node,
        "classify_message_node":       classify_message_node,
        "extract_core_info_node":      extract_core_info_node,
        "ask_next_field_node":         ask_next_field_node,
        "answer_question_node":        answer_question_node,
        "generate_itinerary_node":     generate_itinerary_node,
        "modify_itinerary_node":       modify_itinerary_node,
        "add_assistant_response_node": add_assistant_response_node,
    }
    for name, fn in nodes.items():
        wf.add_node(name, fn)
        print(f"  ✅ Registered node: {name}")

    # ── Edges ────────────────────────────────────────────────
    wf.set_entry_point("entry_node")
    wf.add_edge("entry_node", "classify_message_node")
    print("\n  Edge: entry_node → classify_message_node")

    wf.add_conditional_edges("classify_message_node", route_after_classify, {
        "extract_core_info_node":  "extract_core_info_node",
        "answer_question_node":    "answer_question_node",
        "ask_next_field_node":     "ask_next_field_node",
        "generate_itinerary_node": "generate_itinerary_node",
        "modify_itinerary_node":   "modify_itinerary_node",
    })
    print("  Conditional edges: classify_message_node → [extract|answer|ask|generate|modify]")

    wf.add_conditional_edges("extract_core_info_node", route_after_extract, {
        "ask_next_field_node": "ask_next_field_node",
    })
    print("  Conditional edge : extract_core_info_node → ask_next_field_node")

    terminal_nodes = [
        "ask_next_field_node",
        "answer_question_node",
        "generate_itinerary_node",
        "modify_itinerary_node",
    ]
    for node in terminal_nodes:
        wf.add_edge(node, "add_assistant_response_node")
        print(f"  Edge: {node} → add_assistant_response_node")

    wf.add_edge("add_assistant_response_node", END)
    print("  Edge: add_assistant_response_node → END")

    checkpointer = MemorySaver() if MemorySaver else None
    graph = wf.compile(checkpointer=checkpointer)

    print("\n  ✅ Graph compiled successfully")
    print("═" * W + "\n")
    return graph