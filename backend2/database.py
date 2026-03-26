"""
database.py — DB session for local PostgreSQL.

Local Postgres uses a real QueuePool (pool_pre_ping handles stale
connections automatically). No NullPool, no hstore patches, no
SSL workarounds needed — those were all Supabase/PgBouncer specific.
"""
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from database_models import (
    get_engine, reset_engine, init_database,
    User, Trip, Itinerary, ItineraryActivity, ChatMessage,
    TripStatus, ActivityType, ActivityStatus, MessageRole,
)
import uuid


def get_db():
    """
    Return a DB session from the connection pool.
    pool_pre_ping=True handles stale connections automatically,
    so a single attempt is enough. Reset engine only on actual failure.
    """
    try:
        session = sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            autocommit=False,
            autoflush=True,
        )()
        return session
    except OperationalError as e:
        print(f"[DB] Connection failed, resetting engine: {e}")
        try:
            session = sessionmaker(
                bind=reset_engine(),
                expire_on_commit=False,
                autocommit=False,
                autoflush=True,
            )()
            return session
        except OperationalError as e2:
            print(f"[DB] Retry failed — returning None: {e2}")
            return None


def get_db_or_raise():
    """Use for routes where DB is strictly required (auth, trip lookup)."""
    db = get_db()
    if db is None:
        raise RuntimeError("[DB] Could not connect to PostgreSQL")
    return db


def init_db():
    init_database()  # raises on error for local — fix config before starting
    print("✅ Database connected")


def close_db():
    pass  # session closed by caller; pool manages physical connections


def generate_uuid():
    return str(uuid.uuid4())


__all__ = [
    'get_db', 'get_db_or_raise', 'init_db', 'close_db', 'generate_uuid',
    'User', 'Trip', 'Itinerary', 'ItineraryActivity', 'ChatMessage',
    'TripStatus', 'ActivityType', 'ActivityStatus', 'MessageRole',
]