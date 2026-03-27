import networkx as nx
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from database_models import (
    Region, Place, PlaceEdge, PlaceGraph,
    PlaceCategory, GraphStatus,
    get_engine, init_database
)
from sqlalchemy.orm import sessionmaker


# =====================================================
# DB SESSION (inline — no database.py needed)
# =====================================================

def get_db() -> Session:
    """Create and return a new database session."""
    engine = get_engine()
    SessionLocal = sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autocommit=False,
        autoflush=True,
    )
    return SessionLocal()


def generate_uuid():
    return str(uuid.uuid4())


# =====================================================
# GRAPH IMPORTER
# =====================================================

class GraphImporter:
    """Import NetworkX graphs into the database"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def import_graph_from_file(
        self,
        graphml_path: str,
        region_name: str,
        state: str = None,
        country: str = "India"
    ):
        """
        Import a graph from GraphML file into the database.

        Args:
            graphml_path: Path to .graphml file
            region_name:  Name of the region (e.g. "Goa")
            state:        State name (e.g. "Goa")
            country:      Country name

        Returns:
            PlaceGraph object
        """
        G = nx.read_graphml(graphml_path)
        print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        region    = self._get_or_create_region(region_name, state, country, G)
        graph     = self._create_graph_record(region, G)
        place_map = self._import_places(region, G)
        self._import_edges(graph, G, place_map)
        self._update_graph_stats(graph, G)

        self.db.commit()

        print("✅ Graph imported successfully!")
        print(f"   Region : {region.name}")
        print(f"   Places : {len(place_map)}")
        print(f"   Edges  : {graph.totalEdges}")

        return graph

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _get_or_create_region(
        self, name: str, state: str, country: str, G: nx.Graph
    ) -> Region:
        region = (
            self.db.query(Region)
            .filter_by(name=name, state=state, country=country)
            .first()
        )
        if region:
            print(f"Using existing region: {region.name}")
            return region

        lats = [d["lat"] for _, d in G.nodes(data=True)]
        lons = [d["lon"] for _, d in G.nodes(data=True)]

        region = Region(
            id=generate_uuid(),
            name=name,
            state=state,
            country=country,
            minLat=min(lats),
            maxLat=max(lats),
            minLon=min(lons),
            maxLon=max(lons),
        )
        self.db.add(region)
        self.db.flush()   # get the id before dependent records need it
        print(f"Created new region: {region.name}")
        return region

    def _create_graph_record(self, region: Region, G: nx.Graph) -> PlaceGraph:
        graph = PlaceGraph(
            id=generate_uuid(),
            regionId=region.id,
            name=f"{region.name} Tourist Graph",
            version=1,
            status=GraphStatus.BUILDING,
            buildStartedAt=datetime.utcnow(),
        )
        self.db.add(graph)
        self.db.flush()
        return graph

    def _import_places(self, region: Region, G: nx.Graph) -> dict:
        """Import graph nodes as Place rows. Returns {node_id: Place}."""
        place_map = {}

        for node_id, data in G.nodes(data=True):
            category = self._map_category(data.get("category", "attraction"))

            place = Place(
                id=generate_uuid(),
                regionId=region.id,
                name=data["name"],
                category=category,
                latitude=float(data["lat"]),
                longitude=float(data["lon"]),
                osmId=str(node_id),
            )
            self.db.add(place)
            place_map[node_id] = place

        self.db.flush()   # resolve all place IDs before edges reference them
        print(f"Imported {len(place_map)} places")
        return place_map

    def _import_edges(
        self, graph: PlaceGraph, G: nx.Graph, place_map: dict
    ) -> None:
        edge_count = 0

        for u, v, data in G.edges(data=True):
            from_place = place_map.get(u)
            to_place   = place_map.get(v)

            if from_place is None or to_place is None:
                print(f"  ⚠️  Skipping edge ({u}, {v}) — place not found in map")
                continue

            edge = PlaceEdge(
                id=generate_uuid(),
                graphId=graph.id,
                fromPlaceId=from_place.id,
                toPlaceId=to_place.id,
                roadDistanceKm=float(data.get("road_distance_km", 0)),
                durationMin=float(data.get("duration_min", 0)),
                transportMode="driving",
                edgeType=data.get("edge_type", "road"),
            )
            self.db.add(edge)
            edge_count += 1

        self.db.flush()
        print(f"Imported {edge_count} edges")

    def _update_graph_stats(self, graph: PlaceGraph, G: nx.Graph) -> None:
        graph.totalNodes = G.number_of_nodes()
        graph.totalEdges = G.number_of_edges()

        if graph.totalNodes > 0:
            graph.avgDegree = (2 * graph.totalEdges) / graph.totalNodes

        graph.buildCompletedAt = datetime.utcnow()
        graph.status = GraphStatus.ACTIVE

        if graph.buildStartedAt:
            delta = (graph.buildCompletedAt - graph.buildStartedAt).total_seconds()
            graph.buildDurationSeconds = int(delta)

    @staticmethod
    def _map_category(osm_category: str) -> PlaceCategory:
        mapping = {
            "attraction": PlaceCategory.ATTRACTION,
            "museum":     PlaceCategory.MUSEUM,
            "viewpoint":  PlaceCategory.VIEWPOINT,
            "theme_park": PlaceCategory.THEME_PARK,
            "zoo":        PlaceCategory.ZOO,
        }
        return mapping.get(osm_category, PlaceCategory.OTHER)


# =====================================================
# GRAPH QUERIER
# =====================================================

class GraphQuerier:
    """Query graph data from the database."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def get_graph_for_region(
        self, region_name: str, state: str = None
    ) -> PlaceGraph | None:
        query = (
            self.db.query(PlaceGraph)
            .join(Region)
            .filter(
                Region.name == region_name,
                PlaceGraph.status == GraphStatus.ACTIVE,
            )
        )
        if state:
            query = query.filter(Region.state == state)
        return query.order_by(PlaceGraph.version.desc()).first()

    def get_places_in_region(
        self, region_id: str, category: PlaceCategory = None
    ) -> list[Place]:
        query = self.db.query(Place).filter_by(regionId=region_id)
        if category:
            query = query.filter_by(category=category)
        return query.all()

    def get_nearby_places(
        self, place_id: str, max_distance_km: float = 20
    ) -> list[tuple]:
        edges = (
            self.db.query(PlaceEdge)
            .filter(
                PlaceEdge.fromPlaceId == place_id,
                PlaceEdge.roadDistanceKm <= max_distance_km,
            )
            .all()
        )
        return [
            (edge.toPlace, edge.roadDistanceKm, edge.durationMin)
            for edge in edges
        ]

    def reconstruct_networkx_graph(self, graph_id: str) -> nx.Graph | None:
        graph_record = (
            self.db.query(PlaceGraph).filter_by(id=graph_id).first()
        )
        if not graph_record:
            return None

        G = nx.Graph()
        places = (
            self.db.query(Place)
            .filter_by(regionId=graph_record.regionId)
            .all()
        )

        place_to_node = {}
        for i, place in enumerate(places):
            G.add_node(
                i,
                place_id=place.id,
                name=place.name,
                lat=place.latitude,
                lon=place.longitude,
                category=place.category.value,
            )
            place_to_node[place.id] = i

        edges = (
            self.db.query(PlaceEdge).filter_by(graphId=graph_id).all()
        )
        for edge in edges:
            u = place_to_node.get(edge.fromPlaceId)
            v = place_to_node.get(edge.toPlaceId)
            if u is not None and v is not None:
                G.add_edge(
                    u, v,
                    road_distance_km=edge.roadDistanceKm,
                    duration_min=edge.durationMin,
                    edge_type=edge.edgeType,
                )
        return G

    def find_shortest_path(
        self, from_place_id: str, to_place_id: str, graph_id: str
    ) -> dict | None:
        G = self.reconstruct_networkx_graph(graph_id)
        if not G:
            return None

        place_to_node = {
            data["place_id"]: node
            for node, data in G.nodes(data=True)
        }

        start = place_to_node.get(from_place_id)
        end   = place_to_node.get(to_place_id)

        if start is None or end is None:
            return None

        try:
            path = nx.shortest_path(G, start, end, weight="duration_min")

            path_info      = []
            total_distance = 0.0
            total_duration = 0.0

            for i in range(len(path) - 1):
                u, v      = path[i], path[i + 1]
                edge_data = G[u][v]

                path_info.append({
                    "from":         G.nodes[u]["name"],
                    "to":           G.nodes[v]["name"],
                    "distance_km":  edge_data["road_distance_km"],
                    "duration_min": edge_data["duration_min"],
                })
                total_distance += edge_data["road_distance_km"]
                total_duration += edge_data["duration_min"]

            return {
                "path":               path_info,
                "total_distance_km":  round(total_distance, 2),
                "total_duration_min": round(total_duration, 1),
            }

        except nx.NetworkXNoPath:
            return None


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    # Ensure all tables exist
    init_database()

    db = get_db()

    try:
        # ── Import ──────────────────────────────────────
        importer = GraphImporter(db)
        graph = importer.import_graph_from_file(
            graphml_path="tourist_place_graph.graphml",
            region_name="wayanad",
            state="kerala",
            country="India",
        )

        # ── Basic queries ────────────────────────────────
        querier = GraphQuerier(db)

        places = querier.get_places_in_region(graph.regionId)
        print(f"\nTotal places in region : {len(places)}")

        museums = querier.get_places_in_region(
            graph.regionId, category=PlaceCategory.MUSEUM
        )
        print(f"Museums                : {len(museums)}")

        if places:
            nearby = querier.get_nearby_places(places[0].id, max_distance_km=10)
            print(f"Places within 10 km of '{places[0].name}': {len(nearby)}")

        # ── Shortest path demo ───────────────────────────
        if len(places) >= 2:
            result = querier.find_shortest_path(
                from_place_id=places[0].id,
                to_place_id=places[-1].id,
                graph_id=graph.id,
            )
            if result:
                print(f"\nShortest path ({places[0].name} → {places[-1].name}):")
                print(f"  Distance : {result['total_distance_km']} km")
                print(f"  Duration : {result['total_duration_min']} min")
                for step in result["path"]:
                    print(f"  {step['from']} → {step['to']}  "
                          f"({step['distance_km']} km, {step['duration_min']} min)")
            else:
                print("\nNo path found between selected places.")

    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        db.close()