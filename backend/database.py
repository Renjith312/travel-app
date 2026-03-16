"""
database.py  — thread-safe scoped session + convenience exports
"""
from sqlalchemy.orm import scoped_session
from database_models import (
    get_session_maker, init_database,
    User, Trip, Itinerary, ItineraryActivity, ChatMessage,
    TripStatus, ActivityType, ActivityStatus, MessageRole,
)
import uuid

SessionLocal = scoped_session(get_session_maker())

def get_db():
    return SessionLocal()

def init_db():
    init_database()
    print("✅ Database connected")

def close_db():
    SessionLocal.remove()

def generate_uuid():
    return str(uuid.uuid4())

__all__ = [
    'get_db', 'init_db', 'close_db', 'generate_uuid',
    'User', 'Trip', 'Itinerary', 'ItineraryActivity', 'ChatMessage',
    'TripStatus', 'ActivityType', 'ActivityStatus', 'MessageRole',
]
