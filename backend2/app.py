"""
app.py  — Flask application
All routes: auth, trips, chat (LangGraph), itinerary
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from database import get_db, init_db, generate_uuid, User, Trip, Itinerary, ItineraryActivity, ChatMessage
from database_models import TripStatus, ActivityType, ActivityStatus, MessageRole
from auth import token_required, hash_password, verify_password, create_token
from ai_engine.graph import create_travel_graph
import os, re
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=[os.getenv("FRONTEND_URL","http://localhost:3000")])
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY","dev-secret")

travel_graph = create_travel_graph()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/auth/register", methods=["POST"])
def register():
    db   = get_db()
    data = request.json or {}
    email    = data.get("email","").strip().lower()
    password = data.get("password","")
    if not email or not password:
        return jsonify({"error":"Email and password required"}), 400
    if db.query(User).filter_by(email=email).first():
        return jsonify({"error":"Email already registered"}), 400
    user = User(id=generate_uuid(), email=email, passwordHash=hash_password(password),
                firstName=data.get("firstName"), lastName=data.get("lastName"))
    db.add(user); db.commit()
    return jsonify({"user":{"id":user.id,"email":user.email,
        "firstName":user.firstName,"lastName":user.lastName},
        "token":create_token(user.id,user.email)}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    db   = get_db()
    data = request.json or {}
    user = db.query(User).filter_by(email=data.get("email","").lower()).first()
    if not user or not verify_password(data.get("password",""), user.passwordHash):
        return jsonify({"error":"Invalid credentials"}), 401
    return jsonify({"user":{"id":user.id,"email":user.email,
        "firstName":user.firstName,"lastName":user.lastName},
        "token":create_token(user.id,user.email)}), 200

@app.route("/api/auth/me", methods=["GET"])
@token_required
def me(current_user):
    db   = get_db()
    user = db.query(User).filter_by(id=current_user["id"]).first()
    if not user: return jsonify({"error":"User not found"}), 404
    return jsonify({"id":user.id,"email":user.email,
        "firstName":user.firstName,"lastName":user.lastName}), 200


# ══════════════════════════════════════════════════════════════════════════════
# TRIPS
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/trips", methods=["GET"])
@token_required
def get_trips(current_user):
    db    = get_db()
    trips = db.query(Trip).filter_by(userId=current_user["id"]).order_by(Trip.createdAt.desc()).all()
    return jsonify([{
        "id":trip.id,"destination":trip.destination,"title":trip.title,
        "startDate":trip.startDate.isoformat() if trip.startDate else None,
        "endDate":trip.endDate.isoformat() if trip.endDate else None,
        "status":trip.status.value if trip.status else None,
        "totalBudget":trip.totalBudget,"numberOfTravelers":trip.numberOfTravelers,
        "conversationPhase":trip.conversationPhase or "gathering",
        "hasItinerary":trip.itinerary is not None,
        "coverImage":trip.coverImage,
    } for trip in trips]), 200


@app.route("/api/trips/<trip_id>", methods=["GET"])
@token_required
def get_trip(current_user, trip_id):
    db   = get_db()
    trip = db.query(Trip).filter_by(id=trip_id).first()
    if not trip or trip.userId != current_user["id"]:
        return jsonify({"error":"Trip not found"}), 404
    resp = {
        "id":trip.id,"destination":trip.destination,"title":trip.title,
        "description":trip.description,
        "startDate":trip.startDate.isoformat() if trip.startDate else None,
        "endDate":trip.endDate.isoformat() if trip.endDate else None,
        "duration":trip.duration,"status":trip.status.value if trip.status else None,
        "totalBudget":trip.totalBudget,"estimatedCost":trip.estimatedCost,
        "numberOfTravelers":trip.numberOfTravelers,"currency":trip.currency,
        "conversationPhase":trip.conversationPhase or "gathering",
        "tripContext":trip.tripContext,"itinerary":None,
    }
    if trip.itinerary:
        acts = sorted(trip.itinerary.activities, key=lambda x:(x.dayNumber,x.orderIndex))
        resp["itinerary"] = {
            "id":trip.itinerary.id,"summary":trip.itinerary.summary,
            "highlights":trip.itinerary.highlights or [],
            "fullItinerary":trip.itinerary.fullItinerary,
            "activities":[{"id":a.id,"dayNumber":a.dayNumber,"title":a.title,
                "description":a.description,"type":a.type.value if a.type else None,
                "status":a.status.value if a.status else None,
                "location":a.location,"latitude":a.latitude,"longitude":a.longitude,
                "startTime":a.startTime,"endTime":a.endTime,
                "estimatedCost":a.estimatedCost,"orderIndex":a.orderIndex} for a in acts],
        }
    return jsonify(resp), 200


@app.route("/api/trips", methods=["POST"])
@token_required
def create_trip(current_user):
    db   = get_db()
    data = request.json or {}
    trip = Trip(id=generate_uuid(), userId=current_user["id"],
                destination=data.get("destination","New Trip"),
                title=data.get("title"),
                numberOfTravelers=data.get("numberOfTravelers",1),
                status=TripStatus.PLANNING, conversationPhase="gathering")
    db.add(trip); db.commit()
    return jsonify({"id":trip.id,"destination":trip.destination,
        "status":trip.status.value,"conversationPhase":trip.conversationPhase}), 201


@app.route("/api/trips/create-with-form", methods=["POST"])
@token_required
def create_trip_with_form(current_user):
    db   = get_db()
    data = request.json or {}
    if not data.get("destination") or not data.get("startDate"):
        return jsonify({"error":"destination and startDate required"}), 400
    try:
        start_date = datetime.fromisoformat(data["startDate"].replace("Z",""))
        end_date   = None
        duration   = data.get("durationDays")
        if data.get("endDate"):
            end_date = datetime.fromisoformat(data["endDate"].replace("Z",""))
            if not duration: duration = (end_date - start_date).days + 1
        elif duration:
            end_date = start_date + timedelta(days=duration - 1)
    except Exception as e:
        return jsonify({"error":f"Invalid date: {e}"}), 400

    ctx = {
        "core_info": {"destination":data.get("destination"),"start_date":data.get("startDate"),
            "duration_days":duration,"number_of_travelers":data.get("numberOfTravelers",1),
            "budget":data.get("budget"),"currency":data.get("currency","INR")},
        "preferences": {"activity_types":data.get("activityTypes",[]),
            "interests":data.get("interests",[]),"must_visit":data.get("mustVisit",[]),
            "avoid":data.get("avoid",[]),"food_preferences":data.get("foodPreferences",[]),
            "pace":data.get("pace","moderate")},
    }
    trip = Trip(id=generate_uuid(), userId=current_user["id"],
        destination=data.get("destination"),
        title=data.get("title",f"Trip to {data.get('destination')}"),
        description=data.get("description"),
        startDate=start_date, endDate=end_date, duration=duration,
        numberOfTravelers=data.get("numberOfTravelers",1),
        totalBudget=data.get("budget"), currency=data.get("currency","INR"),
        status=TripStatus.PLANNING, conversationPhase="ready", tripContext=ctx)
    db.add(trip)
    welcome = ChatMessage(id=generate_uuid(), userId=current_user["id"], tripId=trip.id,
        role=MessageRole.ASSISTANT,
        content=f"Perfect! I have all the details for your trip to {data.get('destination')}. Say **yes** to generate your itinerary! 🗺️",
        intent="welcome")
    db.add(welcome); db.commit()
    return jsonify({"trip":{"id":trip.id,"destination":trip.destination,"title":trip.title,
        "conversationPhase":trip.conversationPhase,"status":trip.status.value,
        "startDate":trip.startDate.isoformat() if trip.startDate else None,
        "duration":trip.duration,"numberOfTravelers":trip.numberOfTravelers,
        "totalBudget":trip.totalBudget}}), 201


@app.route("/api/trips/<trip_id>", methods=["PUT"])
@token_required
def update_trip(current_user, trip_id):
    db   = get_db()
    trip = db.query(Trip).filter_by(id=trip_id).first()
    if not trip or trip.userId != current_user["id"]:
        return jsonify({"error":"Trip not found"}), 404
    data = request.json or {}

    for field, col in [("destination","destination"),("title","title"),("description","description"),
                        ("totalBudget","totalBudget"),("numberOfTravelers","numberOfTravelers"),
                        ("currency","currency")]:
        if field in data: setattr(trip, col, data[field])

    # Parse startDate
    if "startDate" in data:
        try:
            trip.startDate = datetime.fromisoformat(data["startDate"].replace("Z",""))
        except Exception: pass

    # Handle durationDays → duration + compute endDate
    if "durationDays" in data:
        trip.duration = int(data["durationDays"])
    if trip.startDate and trip.duration:
        trip.endDate = trip.startDate + timedelta(days=trip.duration - 1)

    # Sync tripContext.core_info so AI always has fresh data
    ctx = dict(trip.tripContext) if trip.tripContext else {}
    ci  = dict(ctx.get("core_info", {}))
    ci["destination"]         = trip.destination
    ci["start_date"]          = trip.startDate.isoformat() if trip.startDate else ci.get("start_date")
    ci["duration_days"]       = trip.duration       or ci.get("duration_days")
    ci["number_of_travelers"] = trip.numberOfTravelers or ci.get("number_of_travelers", 1)
    ci["budget"]              = trip.totalBudget    or ci.get("budget")
    ci["currency"]            = trip.currency       or ci.get("currency", "INR")
    # Save with both human-readable keys (core_info) AND the keys graph.py reads
    if "startTime" in data:
        ci["start_time"]     = data["startTime"]
        ctx["departure_time"] = data["startTime"]
    if "startLocation" in data:
        ci["start_location"] = data["startLocation"]
        ctx["origin"]         = data["startLocation"]
    if "travelMode" in data:
        ci["travel_mode"]    = data["travelMode"]
        ctx["travel_mode"]   = data["travelMode"]
    ctx["core_info"]   = ci
    trip.tripContext   = ctx

    # Promote to 'ready' when all required fields are present
    if trip.destination and trip.startDate and trip.duration:
        trip.conversationPhase = "ready"

    trip.updatedAt = datetime.utcnow()
    db.commit()
    return jsonify({"message": "Trip updated", "conversationPhase": trip.conversationPhase}), 200


@app.route("/api/trips/<trip_id>/status", methods=["PATCH"])
@token_required
def update_trip_status(current_user, trip_id):
    db   = get_db()
    trip = db.query(Trip).filter_by(id=trip_id).first()
    if not trip or trip.userId != current_user["id"]:
        return jsonify({"error":"Trip not found"}), 404
    trip.status = TripStatus(request.json.get("status")); trip.updatedAt = datetime.utcnow()
    db.commit()
    return jsonify({"status":trip.status.value}), 200


@app.route("/api/trips/<trip_id>", methods=["DELETE"])
@token_required
def delete_trip(current_user, trip_id):
    db   = get_db()
    trip = db.query(Trip).filter_by(id=trip_id).first()
    if not trip: return jsonify({"error":"Trip not found"}), 404
    if trip.userId != current_user["id"]: return jsonify({"error":"Unauthorized"}), 403
    try:
        db.delete(trip); db.commit()
        return jsonify({"message":"Trip deleted","tripId":trip_id}), 200
    except Exception as e:
        db.rollback(); return jsonify({"error":str(e)}), 500


@app.route("/api/trips/batch-delete", methods=["POST"])
@token_required
def batch_delete(current_user):
    db      = get_db()
    ids     = (request.json or {}).get("tripIds",[])
    deleted = 0; failed = []
    for tid in ids:
        t = db.query(Trip).filter_by(id=tid).first()
        if not t or t.userId != current_user["id"]: failed.append(tid); continue
        db.delete(t); deleted += 1
    db.commit()
    return jsonify({"deletedCount":deleted,"failedIds":failed}), 200


# ══════════════════════════════════════════════════════════════════════════════
# CHAT  (LangGraph)
# ══════════════════════════════════════════════════════════════════════════════
def _thread_id(user_id, trip_id): return f"{user_id}_{trip_id}"


@app.route("/api/chat/message", methods=["POST"])
@token_required
def send_message(current_user):
    db   = get_db()
    data = request.json or {}
    msg  = data.get("message","").strip()
    tid  = data.get("tripId","")
    if not msg:  return jsonify({"error":"message required"}), 400
    if not tid:  return jsonify({"error":"tripId required"}), 400

    trip = db.query(Trip).filter_by(id=tid).first()
    if not trip or trip.userId != current_user["id"]:
        return jsonify({"error":"Trip not found"}), 404

    thread_id = _thread_id(current_user["id"], tid)
    config    = {"configurable":{"thread_id":thread_id}}
    print(f"\n[CHAT] thread={thread_id} phase={trip.conversationPhase} msg={msg}")

    # Build input state
    input_state = {"user_message": msg, "user_id": current_user["id"], "trip_id": tid}

    # Pre-fill from form context
    if trip.tripContext and trip.conversationPhase in ("ready", "generated"):
        ci  = trip.tripContext.get("core_info", {})
        pr  = trip.tripContext.get("preferences", {})
        ctx = trip.tripContext
        input_state.update({
            "destination":    ci.get("destination"),
            "start_date":     ci.get("start_date"),
            "duration_days":  ci.get("duration_days"),
            "num_travelers":  ci.get("number_of_travelers"),
            "budget":         str(ci.get("budget")) if ci.get("budget") else "flexible",
            # These are stored at top-level ctx by the PUT endpoint
            "origin":         ctx.get("origin")         or ci.get("start_location"),
            "departure_time": ctx.get("departure_time") or ci.get("start_time"),
            "travel_mode":    ctx.get("travel_mode")    or ci.get("travel_mode"),
            "user_preferences": pr,
            "core_info_complete": True,
        })

    try:
        result = travel_graph.invoke(input_state, config)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error":"Graph error","details":str(e)}), 500

    print(f"[CHAT] phase={result.get('current_phase')} intent={result.get('detected_intent')} complete={result.get('core_info_complete')}")

    # ── Get a FRESH db session after graph runs (graph can take 60s — old session may be dead)
    db = get_db()

    # Save user message
    try:
        db.add(ChatMessage(id=generate_uuid(), userId=current_user["id"], tripId=tid,
            role=MessageRole.USER, content=msg)); db.commit()
    except Exception as e:
        print(f"[CHAT] User msg save failed (non-fatal): {e}")
        try: db.rollback()
        except: pass

    ai_reply         = result.get("final_response","I encountered an issue. Please try again.")
    itin_generated   = result.get("itinerary") is not None
    response_data    = {"message":ai_reply,"tripId":tid,
        "phase":result.get("current_phase"),"intent":result.get("detected_intent"),
        "itinerary_updated":itin_generated,"itinerary":None}

    is_modification = (result.get("detected_intent") == "itinerary_request"
                        and db.query(Trip).filter_by(id=tid).first() and
                        db.query(Trip).filter_by(id=tid).first().conversationPhase == "generated")

    if itin_generated:
        if not is_modification:
            ai_reply += "\n\n✨ **Your itinerary is ready!** Check the itinerary panel."
        response_data["message"] = ai_reply
        try:
            trip = db.query(Trip).filter_by(id=tid).first()
            if trip:
                trip.conversationPhase = "generated"
                db.commit()
        except Exception as e:
            print(f"[CHAT] Phase update failed (non-fatal): {e}")
            try: db.rollback()
            except: pass

        try:
            itin_data = result["itinerary"]
            # Update trip metadata
            if result.get("destination"):
                trip.destination = result["destination"]
                trip.title = f"Trip to {result['destination']}"
            if result.get("num_travelers"): trip.numberOfTravelers = result["num_travelers"]
            if result.get("start_date"):
                try:
                    ds = result["start_date"].strip().replace("Z","").split("+")[0]
                    trip.startDate = datetime.fromisoformat(ds)
                    if result.get("duration_days"):
                        trip.endDate = trip.startDate + timedelta(days=result["duration_days"])
                except: pass
            if result.get("budget"):
                try:
                    nums = re.findall(r"\d+", str(result["budget"]).replace(",",""))
                    if nums: trip.totalBudget = float(nums[0])
                except: pass
            trip.updatedAt = datetime.utcnow(); db.commit()

            # Upsert itinerary record
            notes    = itin_data.get("notes",{})
            summary  = next((v for v in notes.values() if v and isinstance(v,str)), "Your personalised itinerary")
            highlights = [v for v in notes.values() if v and isinstance(v,str)][:5]

            ex = db.query(Itinerary).filter_by(tripId=tid).first()
            if ex:
                db.query(ItineraryActivity).filter_by(itineraryId=ex.id).delete()
                ex.summary=summary; ex.highlights=highlights; ex.fullItinerary=itin_data
                ex.updatedAt=datetime.utcnow(); itin_obj=ex
            else:
                itin_obj = Itinerary(id=generate_uuid(), tripId=tid, summary=summary,
                    highlights=highlights, fullItinerary=itin_data)
                db.add(itin_obj)
            db.commit()

            acts_resp = []
            for dp in itin_data.get("daily_plans",[]):
                day_num = dp.get("day_number",1)
                for idx, act in enumerate(dp.get("activities",[])):
                    title = act.get("name") or act.get("title") or "Activity"
                    desc  = act.get("details") or act.get("description") or ""
                    atype = ActivityType.SIGHTSEEING
                    combo = f"{title} {desc}".lower()
                    if any(w in combo for w in ["restaurant","lunch","dinner","breakfast","food","cafe"]):
                        atype = ActivityType.FOOD
                    elif any(w in combo for w in ["hotel","stay","check-in","resort","hostel"]):
                        atype = ActivityType.ACCOMMODATION
                    elif any(w in combo for w in ["transport","drive","train","bus","taxi","flight"]):
                        atype = ActivityType.TRANSPORT
                    loc = act.get("location",{})
                    loc_str = (f"{loc.get('lat',0)},{loc.get('lon',0)}" if isinstance(loc,dict) else str(loc or title))
                    cost = None
                    for k in ["estimatedCost","cost","estimated_cost"]:
                        if act.get(k):
                            try: cost=float(str(act[k]).replace("$","").replace(",","").replace("₹","")); break
                            except: pass
                    a = ItineraryActivity(
                        id=generate_uuid(), itineraryId=itin_obj.id, dayNumber=day_num,
                        title=title, description=desc, type=atype, status=ActivityStatus.SUGGESTED,
                        location=loc_str,
                        latitude=loc.get("lat") if isinstance(loc,dict) else None,
                        longitude=loc.get("lon") if isinstance(loc,dict) else None,
                        startTime=act.get("start_time") or act.get("startTime"),
                        endTime=act.get("end_time") or act.get("endTime"),
                        orderIndex=idx, estimatedCost=cost)
                    db.add(a)
                    acts_resp.append({"id":a.id,"dayNumber":day_num,"title":title,
                        "description":desc,"type":atype.value,"status":ActivityStatus.SUGGESTED.value,
                        "location":loc_str,"startTime":a.startTime,"endTime":a.endTime,
                        "estimatedCost":cost,"orderIndex":idx})
            db.commit()
            response_data["itinerary"] = {"id":itin_obj.id,"summary":summary,
                "highlights":highlights,"fullItinerary":itin_data,"activities":acts_resp}
        except Exception as e:
            try: db.rollback()
            except: pass
            print(f"[CHAT] Itinerary DB error (non-fatal): {e}")
            # Get fresh session for AI message save
            db = get_db()

    # Save AI message
    try:
        db.add(ChatMessage(id=generate_uuid(), userId=current_user["id"], tripId=tid,
            role=MessageRole.ASSISTANT, content=response_data["message"],
            intent=result.get("detected_intent"))); db.commit()
    except Exception as e:
        print(f"[CHAT] AI msg save failed (non-fatal): {e}")
        try: db.rollback()
        except: pass

    # Trip status summary
    if result.get("core_info_complete"):
        response_data["tripStatus"] = {"complete":True,
            "destination":result.get("destination"),"startDate":result.get("start_date"),
            "duration":result.get("duration_days"),"travelers":result.get("num_travelers"),
            "budget":result.get("budget")}
    else:
        missing = [f for f in ["destination","start_date","duration_days","num_travelers","budget"] if not result.get(f)]
        response_data["tripStatus"] = {"complete":False,"missing":missing}

    print(f"[CHAT] itinerary_updated={itin_generated}")
    return jsonify(response_data), 200


@app.route("/api/chat/history/<trip_id>", methods=["GET"])
@token_required
def get_history(current_user, trip_id):
    db   = get_db()
    trip = db.query(Trip).filter_by(id=trip_id).first()
    if not trip: return jsonify({"error":"Trip not found"}), 404
    if trip.userId != current_user["id"]: return jsonify({"error":"Unauthorized"}), 403
    msgs = db.query(ChatMessage).filter_by(tripId=trip_id, userId=current_user["id"])\
             .order_by(ChatMessage.createdAt.asc()).all()
    return jsonify([{"id":m.id,"role":m.role.value,"content":m.content,
        "createdAt":m.createdAt.isoformat() if m.createdAt else None} for m in msgs]), 200


@app.route("/api/chat/reset/<trip_id>", methods=["POST"])
@token_required
def reset_chat(current_user, trip_id):
    db = get_db()
    db.query(ChatMessage).filter_by(userId=current_user["id"],tripId=trip_id).delete()
    db.commit()
    return jsonify({"message":"Chat reset"}), 200


@app.route("/api/chat/trip-status/<trip_id>", methods=["GET"])
@token_required
def trip_planning_status(current_user, trip_id):
    try:
        config = {"configurable":{"thread_id":_thread_id(current_user["id"],trip_id)}}
        state  = travel_graph.get_state(config)
        if state and state.values:
            v = state.values
            if v.get("core_info_complete"):
                return jsonify({"complete":True,"destination":v.get("destination"),
                    "startDate":v.get("start_date"),"duration":v.get("duration_days"),
                    "travelers":v.get("num_travelers"),"budget":v.get("budget"),
                    "hasItinerary":v.get("itinerary") is not None}), 200
            missing=[f for f in ["destination","start_date","duration_days","num_travelers","budget"] if not v.get(f)]
            return jsonify({"complete":False,"missing":missing}), 200
    except Exception: pass
    return jsonify({"complete":False,"missing":["destination","start_date","duration_days","num_travelers","budget"]}), 200


# ══════════════════════════════════════════════════════════════════════════════
# ITINERARY
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/itinerary/<trip_id>", methods=["GET"])
@token_required
def get_itinerary(current_user, trip_id):
    db   = get_db()
    trip = db.query(Trip).filter_by(id=trip_id).first()
    if not trip or trip.userId != current_user["id"]: return jsonify({"error":"Trip not found"}), 404
    if not trip.itinerary: return jsonify({"error":"No itinerary"}), 404
    acts = sorted(trip.itinerary.activities, key=lambda x:(x.dayNumber,x.orderIndex))
    return jsonify({"id":trip.itinerary.id,"summary":trip.itinerary.summary,
        "highlights":trip.itinerary.highlights or [],
        "fullItinerary":trip.itinerary.fullItinerary,
        "activities":[{"id":a.id,"dayNumber":a.dayNumber,"date":a.date.isoformat() if a.date else None,
            "title":a.title,"description":a.description,
            "type":a.type.value if a.type else None,"status":a.status.value if a.status else None,
            "location":a.location,"latitude":a.latitude,"longitude":a.longitude,
            "startTime":a.startTime,"endTime":a.endTime,"estimatedCost":a.estimatedCost,
            "orderIndex":a.orderIndex,"notes":a.notes} for a in acts]}), 200


@app.route("/api/itinerary/<trip_id>/activity/<activity_id>", methods=["PATCH"])
@token_required
def update_activity(current_user, trip_id, activity_id):
    db   = get_db()
    trip = db.query(Trip).filter_by(id=trip_id).first()
    if not trip or trip.userId != current_user["id"]: return jsonify({"error":"Trip not found"}), 404
    act  = db.query(ItineraryActivity).filter_by(id=activity_id).first()
    if not act: return jsonify({"error":"Activity not found"}), 404
    data = request.json or {}
    for f in ["title","description","startTime","endTime","notes","location","estimatedCost","status"]:
        if f in data:
            if f == "status":
                try: setattr(act, f, ActivityStatus(data[f]))
                except: pass
            else: setattr(act, f, data[f])
    act.updatedAt = datetime.utcnow(); db.commit()
    return jsonify({"message":"Activity updated"}), 200


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","service":"travel-copilot-backend"}), 200


@app.route("/api/debug/graph-state/<trip_id>", methods=["GET"])
@token_required
def graph_state(current_user, trip_id):
    try:
        config = {"configurable":{"thread_id":_thread_id(current_user["id"],trip_id)}}
        state  = travel_graph.get_state(config)
        return jsonify({"state": dict(state.values) if state and state.values else None}), 200
    except Exception as e:
        return jsonify({"error":str(e)}), 500


if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 5000))
    app.run(
        debug=os.getenv("FLASK_DEBUG", "False") == "True",
        port=port,
        host="0.0.0.0",
        use_reloader=False,   # ← stops double init, prevents engine state corruption
    )