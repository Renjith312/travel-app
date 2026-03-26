"""
database.py — DB session using NullPool for Supabase PgBouncer compatibility.

Why NullPool?
  Supabase uses PgBouncer in transaction mode which aggressively closes idle
  connections (< 5 min). SQLAlchemy's local connection pool fights with this
  and serves dead connections → "SSL connection closed unexpectedly".

  NullPool = no local pool. Every request opens a fresh connection to Supabase
  and closes it immediately after. PgBouncer handles the actual pooling on
  Supabase's side. Slightly more overhead per request but 100% reliable.
"""
import time, sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from database_models import (
    get_engine, init_database,
    User, Trip, Itinerary, ItineraryActivity, ChatMessage,
    TripStatus, ActivityType, ActivityStatus, MessageRole,
)
import uuid

def get_db():
    """Open a fresh DB session. Retries once on connection error."""
    for attempt in range(3):
        try:
            session = sessionmaker(
                bind=get_engine(),
                expire_on_commit=False,
            )()
            session.execute(sqlalchemy.text("SELECT 1"))
            return session
        except OperationalError as e:
            print(f"[DB] Attempt {attempt+1}/3 failed: {type(e).__name__}")
            time.sleep(1)
    raise RuntimeError("[DB] Could not connect after 3 retries")

def init_db():
    init_database()
    print("✅ Database connected")

def close_db():
    pass  # NullPool closes connections automatically

def generate_uuid():
    return str(uuid.uuid4())

__all__ = [
    'get_db', 'init_db', 'close_db', 'generate_uuid',
    'User', 'Trip', 'Itinerary', 'ItineraryActivity', 'ChatMessage',
    'TripStatus', 'ActivityType', 'ActivityStatus', 'MessageRole',
]