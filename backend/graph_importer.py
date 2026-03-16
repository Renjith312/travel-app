"""
graph_importer.py  — import a .graphml file into the database
Usage: python graph_importer.py --file tourist_place_graph.graphml --region "Kochi" --state "Kerala"
"""
import networkx as nx, uuid, argparse
from datetime import datetime
from database_models import Region, Place, PlaceEdge, PlaceGraph, PlaceCategory, GraphStatus, get_session_maker

_CAT_MAP = {
    "attraction":PlaceCategory.ATTRACTION,"museum":PlaceCategory.MUSEUM,
    "viewpoint":PlaceCategory.VIEWPOINT,"theme_park":PlaceCategory.THEME_PARK,
    "zoo":PlaceCategory.ZOO,"historic":PlaceCategory.HISTORICAL,
    "place_of_worship":PlaceCategory.RELIGIOUS,"park":PlaceCategory.PARK,
}

def import_graphml(path, region_name, state=None, country="India"):
    G = nx.read_graphml(path)
    print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    S = get_session_maker()
    with S() as s:
        region = s.query(Region).filter(Region.name.ilike(region_name)).first()
        if not region:
            lats=[float(d["lat"]) for _,d in G.nodes(data=True)]
            lons=[float(d["lon"]) for _,d in G.nodes(data=True)]
            region = Region(id=str(uuid.uuid4()), name=region_name, state=state, country=country,
                minLat=min(lats),maxLat=max(lats),minLon=min(lons),maxLon=max(lons),
                createdAt=datetime.utcnow(),updatedAt=datetime.utcnow())
            s.add(region); s.flush()
        gr = PlaceGraph(id=str(uuid.uuid4()),regionId=region.id,
            name=f"{region_name} Tourist Graph",version=1,status=GraphStatus.BUILDING,
            buildStartedAt=datetime.utcnow(),createdAt=datetime.utcnow(),updatedAt=datetime.utcnow())
        s.add(gr); s.flush()
        n2p = {}
        for ni, nd in G.nodes(data=True):
            p = Place(id=str(uuid.uuid4()),regionId=region.id,name=nd["name"],
                category=_CAT_MAP.get(str(nd.get("category","")).lower(),PlaceCategory.OTHER),
                latitude=float(nd["lat"]),longitude=float(nd["lon"]),
                osmId=str(ni),createdAt=datetime.utcnow(),updatedAt=datetime.utcnow())
            s.add(p); n2p[ni]=p.id
        s.flush()
        ec=0
        for u,v,ed in G.edges(data=True):
            for src,dst in [(n2p[u],n2p[v]),(n2p[v],n2p[u])]:
                s.add(PlaceEdge(id=str(uuid.uuid4()),graphId=gr.id,fromPlaceId=src,toPlaceId=dst,
                    roadDistanceKm=float(ed.get("road_distance_km",0)),
                    durationMin=float(ed.get("duration_min",0)),
                    transportMode="driving",edgeType=ed.get("edge_type","road"),
                    createdAt=datetime.utcnow(),updatedAt=datetime.utcnow()))
                ec+=1
        gr.totalNodes=G.number_of_nodes(); gr.totalEdges=ec
        gr.status=GraphStatus.ACTIVE; gr.buildCompletedAt=datetime.utcnow()
        gr.buildDurationSeconds=int((gr.buildCompletedAt-gr.buildStartedAt).total_seconds())
        gr.updatedAt=datetime.utcnow(); s.commit()
        print(f"✅ Imported: {G.number_of_nodes()} places, {ec} edges → region={region.name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file",    required=True)
    ap.add_argument("--region",  required=True)
    ap.add_argument("--state",   default=None)
    ap.add_argument("--country", default="India")
    a = ap.parse_args()
    import_graphml(a.file, a.region, a.state, a.country)
