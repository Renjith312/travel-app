"""
database_models.py
SQLAlchemy ORM models for local PostgreSQL.
"""
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime,
    Boolean, Text, ForeignKey, JSON, Index
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime, timezone
import enum, os
from dotenv import load_dotenv

load_dotenv()
Base = declarative_base()

def utc_now():
    return datetime.now(timezone.utc)

# ── Enums ──────────────────────────────────────────────────────────────────────
class TripStatus(str, enum.Enum):
    PLANNING = "PLANNING"
    PLANNED  = "PLANNED"
    ONGOING  = "ONGOING"
    FINISHED = "FINISHED"

class ActivityType(str, enum.Enum):
    TRANSPORT     = "TRANSPORT"
    ACCOMMODATION = "ACCOMMODATION"
    FOOD          = "FOOD"
    SIGHTSEEING   = "SIGHTSEEING"
    ACTIVITY      = "ACTIVITY"
    SHOPPING      = "SHOPPING"
    RELAXATION    = "RELAXATION"
    NIGHTLIFE     = "NIGHTLIFE"
    OTHER         = "OTHER"

class ActivityStatus(str, enum.Enum):
    SUGGESTED = "SUGGESTED"
    CONFIRMED = "CONFIRMED"
    BOOKED    = "BOOKED"
    COMPLETED = "COMPLETED"
    SKIPPED   = "SKIPPED"
    CANCELLED = "CANCELLED"

class MessageRole(str, enum.Enum):
    USER      = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM    = "SYSTEM"

class PlaceCategory(str, enum.Enum):
    ATTRACTION = "ATTRACTION"
    MUSEUM     = "MUSEUM"
    VIEWPOINT  = "VIEWPOINT"
    THEME_PARK = "THEME_PARK"
    ZOO        = "ZOO"
    HISTORICAL = "HISTORICAL"
    RELIGIOUS  = "RELIGIOUS"
    PARK       = "PARK"
    BEACH      = "BEACH"
    RESTAURANT = "RESTAURANT"
    SHOPPING   = "SHOPPING"
    OTHER      = "OTHER"

class GraphStatus(str, enum.Enum):
    BUILDING = "BUILDING"
    ACTIVE   = "ACTIVE"
    OUTDATED = "OUTDATED"
    ARCHIVED = "ARCHIVED"

# ── Models ─────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = 'users'
    id           = Column(String, primary_key=True)
    email        = Column(String, unique=True, nullable=False)
    passwordHash = Column('password_hash', String, nullable=False)
    firstName    = Column('first_name', String)
    lastName     = Column('last_name', String)
    phone        = Column(String)
    profileImage = Column('profile_image', String)
    createdAt    = Column('created_at', DateTime, default=utc_now)
    updatedAt    = Column('updated_at', DateTime, default=utc_now, onupdate=utc_now)
    trips        = relationship('Trip', back_populates='user', cascade='all, delete-orphan')
    chatMessages = relationship('ChatMessage', back_populates='user', cascade='all, delete-orphan')


class Trip(Base):
    __tablename__ = 'trips'
    id                = Column(String, primary_key=True)
    userId            = Column('user_id', String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    destination       = Column(String, nullable=False)
    title             = Column(String)
    description       = Column(String)
    coverImage        = Column('cover_image', String)
    startDate         = Column('start_date', DateTime)
    endDate           = Column('end_date', DateTime)
    duration          = Column(Integer)
    status            = Column(SQLEnum(TripStatus), default=TripStatus.PLANNING)
    totalBudget       = Column('total_budget', Float)
    estimatedCost     = Column('estimated_cost', Float)
    actualCost        = Column('actual_cost', Float)
    currency          = Column(String, default='INR')
    numberOfTravelers = Column('number_of_travelers', Integer, default=1)
    tripContext       = Column('trip_context', JSONB)
    conversationPhase = Column('conversation_phase', String, default='gathering')
    createdAt         = Column('created_at', DateTime, default=utc_now)
    updatedAt         = Column('updated_at', DateTime, default=utc_now, onupdate=utc_now)
    user         = relationship('User', back_populates='trips')
    itinerary    = relationship('Itinerary', back_populates='trip', uselist=False, cascade='all, delete-orphan')
    chatMessages = relationship('ChatMessage', back_populates='trip', cascade='all, delete-orphan')


class Itinerary(Base):
    __tablename__ = 'itineraries'
    id            = Column(String, primary_key=True)
    tripId        = Column('trip_id', String, ForeignKey('trips.id', ondelete='CASCADE'), unique=True, nullable=False)
    summary       = Column(Text)
    highlights    = Column(ARRAY(String))
    fullItinerary = Column('full_itinerary', JSONB)
    createdAt     = Column('created_at', DateTime, default=utc_now)
    updatedAt     = Column('updated_at', DateTime, default=utc_now, onupdate=utc_now)
    trip       = relationship('Trip', back_populates='itinerary')
    activities = relationship('ItineraryActivity', back_populates='itinerary', cascade='all, delete-orphan')


class ItineraryActivity(Base):
    __tablename__ = 'itinerary_activities'
    id              = Column(String, primary_key=True)
    itineraryId     = Column('itinerary_id', String, ForeignKey('itineraries.id', ondelete='CASCADE'), nullable=False)
    dayNumber       = Column('day_number', Integer, nullable=False)
    date            = Column(DateTime)
    title           = Column(String, nullable=False)
    description     = Column(Text)
    type            = Column(SQLEnum(ActivityType), nullable=False)
    status          = Column(SQLEnum(ActivityStatus), default=ActivityStatus.SUGGESTED)
    location        = Column(String, nullable=False)
    address         = Column(String)
    latitude        = Column(Float)
    longitude       = Column(Float)
    startTime       = Column('start_time', String)
    endTime         = Column('end_time', String)
    duration        = Column(Integer)
    orderIndex      = Column('order_index', Integer, nullable=False)
    estimatedCost   = Column('estimated_cost', Float)
    actualCost      = Column('actual_cost', Float)
    currency        = Column(String, default='INR')
    bookingRequired = Column('booking_required', Boolean, default=False)
    bookingUrl      = Column('booking_url', String)
    notes           = Column(Text)
    images          = Column(ARRAY(String))
    createdAt       = Column('created_at', DateTime, default=utc_now)
    updatedAt       = Column('updated_at', DateTime, default=utc_now, onupdate=utc_now)
    itinerary = relationship('Itinerary', back_populates='activities')


class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id           = Column(String, primary_key=True)
    userId       = Column('user_id', String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    tripId       = Column('trip_id', String, ForeignKey('trips.id', ondelete='CASCADE'))
    role         = Column(SQLEnum(MessageRole), nullable=False)
    content      = Column(Text, nullable=False)
    intent       = Column(String)
    chatMetadata = Column('metadata', JSON)
    createdAt    = Column('created_at', DateTime, default=utc_now)
    user = relationship('User', back_populates='chatMessages')
    trip = relationship('Trip', back_populates='chatMessages')


class Region(Base):
    __tablename__ = 'regions'
    id          = Column(String, primary_key=True)
    name        = Column(String, nullable=False)
    state       = Column(String)
    country     = Column(String, nullable=False)
    minLat      = Column('min_lat', Float)
    maxLat      = Column('max_lat', Float)
    minLon      = Column('min_lon', Float)
    maxLon      = Column('max_lon', Float)
    population  = Column(Integer)
    description = Column(Text)
    createdAt   = Column('created_at', DateTime, default=utc_now)
    updatedAt   = Column('updated_at', DateTime, default=utc_now, onupdate=utc_now)
    places = relationship('Place', back_populates='region', cascade='all, delete-orphan')
    graphs = relationship('PlaceGraph', back_populates='region', cascade='all, delete-orphan')


class Place(Base):
    __tablename__ = 'places'
    id                   = Column(String, primary_key=True)
    regionId             = Column('region_id', String, ForeignKey('regions.id', ondelete='CASCADE'), nullable=False)
    name                 = Column(String, nullable=False)
    category             = Column(SQLEnum(PlaceCategory), nullable=False)
    subcategory          = Column(String)
    latitude             = Column(Float, nullable=False)
    longitude            = Column(Float, nullable=False)
    address              = Column(Text)
    description          = Column(Text)
    rating               = Column(Float)
    popularityScore      = Column('popularity_score', Float)
    typicalVisitDuration = Column('typical_visit_duration', Integer)
    openingHours         = Column('opening_hours', JSON)
    bestTimeToVisit      = Column('best_time_to_visit', String)
    entryFee             = Column('entry_fee', Float)
    currency             = Column(String, default='INR')
    osmId                = Column('osm_id', String)
    googlePlaceId        = Column('google_place_id', String)
    images               = Column(ARRAY(String))
    tags                 = Column(ARRAY(String))
    createdAt            = Column('created_at', DateTime, default=utc_now)
    updatedAt            = Column('updated_at', DateTime, default=utc_now, onupdate=utc_now)
    region        = relationship('Region', back_populates='places')
    outgoingEdges = relationship('PlaceEdge', foreign_keys='PlaceEdge.fromPlaceId', back_populates='fromPlace', cascade='all, delete-orphan')
    incomingEdges = relationship('PlaceEdge', foreign_keys='PlaceEdge.toPlaceId',   back_populates='toPlace',   cascade='all, delete-orphan')


class PlaceEdge(Base):
    __tablename__ = 'place_edges'
    id                     = Column(String, primary_key=True)
    graphId                = Column('graph_id',      String, ForeignKey('place_graphs.id', ondelete='CASCADE'), nullable=False)
    fromPlaceId            = Column('from_place_id', String, ForeignKey('places.id',       ondelete='CASCADE'), nullable=False)
    toPlaceId              = Column('to_place_id',   String, ForeignKey('places.id',       ondelete='CASCADE'), nullable=False)
    roadDistanceKm         = Column('road_distance_km', Float, nullable=False)
    durationMin            = Column('duration_min',     Float, nullable=False)
    straightLineDistanceKm = Column('straight_line_distance_km', Float)
    transportMode          = Column('transport_mode', String, default='driving')
    travelCost             = Column('travel_cost', Float)
    edgeType               = Column('edge_type',   String, default='road')
    routePolyline          = Column('route_polyline', Text)
    createdAt              = Column('created_at', DateTime, default=utc_now)
    updatedAt              = Column('updated_at', DateTime, default=utc_now, onupdate=utc_now)
    graph     = relationship('PlaceGraph', back_populates='edges')
    fromPlace = relationship('Place', foreign_keys=[fromPlaceId], back_populates='outgoingEdges')
    toPlace   = relationship('Place', foreign_keys=[toPlaceId],   back_populates='incomingEdges')


class PlaceGraph(Base):
    __tablename__ = 'place_graphs'
    id                   = Column(String, primary_key=True)
    regionId             = Column('region_id', String, ForeignKey('regions.id', ondelete='CASCADE'), nullable=False)
    name                 = Column(String, nullable=False)
    version              = Column(Integer, default=1)
    status               = Column(SQLEnum(GraphStatus), default=GraphStatus.ACTIVE)
    kNeighbors           = Column('k_neighbors',        Integer, default=5)
    maxTravelTimeMin     = Column('max_travel_time_min', Integer, default=90)
    transportMode        = Column('transport_mode', String, default='driving')
    totalNodes           = Column('total_nodes',   Integer)
    totalEdges           = Column('total_edges',   Integer)
    avgDegree            = Column('avg_degree',    Float)
    graphmlData          = Column('graphml_data',  Text)
    buildStartedAt       = Column('build_started_at',   DateTime)
    buildCompletedAt     = Column('build_completed_at', DateTime)
    buildDurationSeconds = Column('build_duration_seconds', Integer)
    createdAt            = Column('created_at', DateTime, default=utc_now)
    updatedAt            = Column('updated_at', DateTime, default=utc_now, onupdate=utc_now)
    region = relationship('Region', back_populates='graphs')
    edges  = relationship('PlaceEdge', back_populates='graph', cascade='all, delete-orphan')


# ── Indexes ────────────────────────────────────────────────────────────────────
Index('idx_region_location', Region.country,      Region.state, Region.name)
Index('idx_place_location',  Place.latitude,      Place.longitude)
Index('idx_place_category',  Place.category)
Index('idx_place_region',    Place.regionId)
Index('idx_edge_graph',      PlaceEdge.graphId)
Index('idx_edge_from',       PlaceEdge.fromPlaceId)
Index('idx_edge_to',         PlaceEdge.toPlaceId)
Index('idx_graph_region',    PlaceGraph.regionId)
Index('idx_graph_status',    PlaceGraph.status)


# ── DB setup ───────────────────────────────────────────────────────────────────
def _build_db_url() -> str:
    load_dotenv()

    # If DATABASE_URL is explicitly set, use it directly
    url = os.getenv('DATABASE_URL')
    if url:
        return url

    # Otherwise build from individual vars (typical for local Postgres)
    host     = os.getenv('DB_HOST',     'localhost')
    port     = os.getenv('DB_PORT',     '5432')
    name     = os.getenv('DB_NAME',     'travel_copilot')
    user     = os.getenv('DB_USER',     'postgres')
    password = os.getenv('DB_PASSWORD', '')
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


_engine = None

def _create_engine_instance():
    from sqlalchemy.pool import QueuePool
    return create_engine(
        _build_db_url(),
        # Local Postgres: use a real connection pool (no PgBouncer fighting)
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,       # test connection before use
        pool_recycle=1800,        # recycle connections every 30 min
        connect_args={
            "connect_timeout": 10,
            "options":         "-c statement_timeout=30000",
        },
    )

def get_engine():
    global _engine
    if _engine is None:
        _engine = _create_engine_instance()
    return _engine

def reset_engine():
    global _engine
    if _engine is not None:
        try:
            _engine.dispose(close=True)
        except Exception:
            pass
    _engine = _create_engine_instance()
    return _engine

def init_database():
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        print("✅ Database tables ready")
    except Exception as e:
        print(f"⚠️  init_database error: {e}")
        raise   # local Postgres errors should be fatal — fix them before starting

def get_session_maker():
    from sqlalchemy.orm import sessionmaker as _sm
    return _sm(
        bind=get_engine(),
        expire_on_commit=False,
        autocommit=False,
        autoflush=True,
    )