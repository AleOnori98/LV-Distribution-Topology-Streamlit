from __future__ import annotations

import io
from typing import Dict, List, Tuple, Any

import geopandas as gpd
import networkx as nx
import pandas as pd
import math
from shapely.geometry import Point, LineString


# ---------------------------------------------------------------------
# Sampling & association algorithms
# ---------------------------------------------------------------------

def sample_points_along_line(line: LineString, sampling_distance: float) -> List[Point]:
    """
    Sample points along a line geometry at regular intervals.

    Parameters
    ----------
    line : shapely LineString
    sampling_distance : float
        Distance between successive sample points (in CRS units, e.g. meters).
    """
    points: List[Point] = []
    current_distance = 0.0
    while current_distance < line.length:
        points.append(line.interpolate(current_distance))
        current_distance += sampling_distance
    return points


def collect_sampled_points(gdf_roads: gpd.GeoDataFrame, sampling_distance: float) -> List[Point]:
    """
    Collect candidate pole locations from road geometries.

    Strategy
    --------
    - Take start and end point of each road segment.
    - If the segment is long, sample additional points every `sampling_distance`.
    """
    sampled_points: List[Point] = []
    for _, row in gdf_roads.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue

        # Always add endpoints
        coords = list(geometry.coords)
        sampled_points.append(Point(coords[0]))
        sampled_points.append(Point(coords[-1]))

        # Additional sampling along the line
        if geometry.length > sampling_distance:
            sampled_points.extend(sample_points_along_line(geometry, sampling_distance))

    return sampled_points


def associate_buildings_to_poles(
    gdf_buildings: gpd.GeoDataFrame,
    gdf_poles: gpd.GeoDataFrame,
    user_distance: float,
    max_associations: int,
) -> pd.DataFrame:
    """
    Associate buildings to poles based on spatial proximity.

    The algorithm loops over poles, builds a buffer of radius `user_distance`,
    finds currently unassociated buildings within that buffer, sorts by distance
    to the pole, and associates up to `max_associations` buildings.

    Returns
    -------
    DataFrame with columns ['pole_id', 'building_id'].
    """
    building_association: Dict[int, List[int]] = {idx: [] for idx in gdf_poles.index}
    associated_buildings = set()

    for pole in gdf_poles.itertuples():
        buffer = pole.geometry.buffer(user_distance)

        nearby_buildings = gdf_buildings[
            ~gdf_buildings.index.isin(associated_buildings)
            & gdf_buildings.geometry.within(buffer)
        ].copy()

        if nearby_buildings.empty:
            continue

        nearby_buildings["distance_to_pole"] = nearby_buildings.geometry.distance(
            pole.geometry
        )
        nearby_buildings = nearby_buildings.sort_values("distance_to_pole")

        num_associated = 0
        for building_id in nearby_buildings.index:
            if num_associated >= max_associations:
                break
            if building_id not in associated_buildings:
                building_association[pole.Index].append(building_id)
                associated_buildings.add(building_id)
                num_associated += 1

        if len(building_association[pole.Index]) > max_associations:
            building_association[pole.Index] = building_association[pole.Index][
                :max_associations
            ]

    records = [
        (pole_id, building_id)
        for pole_id, buildings in building_association.items()
        for building_id in buildings
    ]
    return pd.DataFrame(records, columns=["pole_id", "building_id"])


def place_poles_for_unassociated_buildings(
    gdf_unassociated_buildings: gpd.GeoDataFrame,
    user_distance: float,
    max_associations: int,
    *,
    allow_unserved_isolated: bool = False,
    min_cluster_size: int = 1,
) -> Tuple[gpd.GeoDataFrame, List[Tuple[Point, Point]], gpd.GeoDataFrame]:
    """
    Place new poles for unassociated buildings based on proximity.

    If allow_unserved_isolated is True, clusters smaller than min_cluster_size
    are left unserved and returned as 'remaining' buildings.
    """
    new_poles: List[Point] = []
    all_associations: List[Tuple[Point, Point]] = []

    gdf_remaining = gdf_unassociated_buildings.copy()

    while not gdf_remaining.empty:
        largest_cluster = None
        max_cluster_size = 0

        # Find the largest cluster of unassociated buildings
        for building in gdf_remaining.itertuples():
            buffer = building.geometry.buffer(user_distance)
            intersecting = gdf_remaining[gdf_remaining.geometry.intersects(buffer)]
            if len(intersecting) > max_cluster_size:
                max_cluster_size = len(intersecting)
                largest_cluster = intersecting

        if largest_cluster is None:
            break

        # If we allow unserved and even the largest cluster is too small,
        # stop placing poles and leave remaining buildings as unserved.
        if allow_unserved_isolated and max_cluster_size < min_cluster_size:
            break

        # Operate on the largest cluster
        buffers = [b.geometry.buffer(user_distance) for b in largest_cluster.itertuples()]
        merged_buffer = buffers[0]
        for buf in buffers[1:]:
            merged_buffer = merged_buffer.union(buf)

        pole_location: Point = merged_buffer.centroid
        new_poles.append(pole_location)

        # Sort buildings in this cluster by distance to the new pole
        buildings_in_cluster = [(b.Index, b.geometry) for b in largest_cluster.itertuples()]
        buildings_in_cluster.sort(key=lambda x: pole_location.distance(x[1]))

        # Associate up to max_associations buildings
        closest = buildings_in_cluster[:max_associations]
        for building_idx, building_geom in closest:
            all_associations.append((pole_location, building_geom))

        associated_indices = [b[0] for b in closest]
        gdf_remaining = gdf_remaining[~gdf_remaining.index.isin(associated_indices)]

    gdf_new_poles = gpd.GeoDataFrame({"geometry": new_poles}, crs=gdf_unassociated_buildings.crs)
    return gdf_new_poles, all_associations, gdf_remaining



# ---------------------------------------------------------------------
# Graph + MST + exports
# ---------------------------------------------------------------------
def create_graph_and_mst(gdf_poles: gpd.GeoDataFrame) -> nx.Graph:
    """
    Create a complete graph over poles using Euclidean distance as edge weight,
    then return its Minimum Spanning Tree (MST).
    """
    G = nx.Graph()
    coords = [(p.geometry.x, p.geometry.y) for p in gdf_poles.itertuples()]
    for i, (x1, y1) in enumerate(coords):
        for j, (x2, y2) in enumerate(coords):
            if i == j:
                continue
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            G.add_edge(i, j, weight=distance)
    return nx.minimum_spanning_tree(G)


def mst_edges_as_latlon(
    gdf_poles_4326: gpd.GeoDataFrame, mst: nx.Graph
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Convert MST edges to ((lat1, lon1), (lat2, lon2)) pairs for plotting.
    """
    out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for u, v in mst.edges():
        p1 = gdf_poles_4326.geometry.iloc[u]
        p2 = gdf_poles_4326.geometry.iloc[v]
        out.append(((p1.y, p1.x), (p2.y, p2.x)))
    return out


def save_mst_to_geojson(
    gdf_poles: gpd.GeoDataFrame, mst: nx.Graph
) -> Tuple[io.BytesIO, io.BytesIO]:
    """
    Export MST nodes/edges to GeoJSON. Nodes include:
      - pole_id
      - pole_type
      - pole_origin
    """
    poles = gdf_poles.copy().reset_index(drop=True)
    if "pole_id" not in poles.columns:
        poles["pole_id"] = poles.index.astype(int)
    if "pole_type" not in poles.columns:
        poles["pole_type"] = "base"
    if "pole_origin" not in poles.columns:
        poles["pole_origin"] = "base"

    nodes_records = []
    for idx, pole in enumerate(poles.itertuples()):
        nodes_records.append(
            {
                "node": int(idx),  # graph node id (row index)
                "pole_id": int(getattr(pole, "pole_id")),
                "pole_type": str(getattr(pole, "pole_type")),
                "pole_origin": str(getattr(pole, "pole_origin")),
                "geometry": pole.geometry,
            }
        )
    gdf_nodes = gpd.GeoDataFrame(nodes_records, crs=poles.crs)

    edges_records = []
    for u, v, data in mst.edges(data=True):
        start = poles.geometry.iloc[u]
        end = poles.geometry.iloc[v]
        edges_records.append(
            {
                "source": int(u),
                "target": int(v),
                "weight": float(data["weight"]),
                "geometry": LineString([start, end]),
            }
        )
    gdf_edges = gpd.GeoDataFrame(edges_records, crs=poles.crs)

    nodes_buf = io.BytesIO()
    gdf_nodes.to_file(nodes_buf, driver="GeoJSON")
    nodes_buf.seek(0)

    edges_buf = io.BytesIO()
    gdf_edges.to_file(edges_buf, driver="GeoJSON")
    edges_buf.seek(0)

    return nodes_buf, edges_buf

# ---------------------------------------------------------------------
# MST post-processing
# ---------------------------------------------------------------------
def densify_mst_edges(
    gdf_poles: gpd.GeoDataFrame,
    mst: nx.Graph,
    max_pole_span_m: float,
) -> Tuple[gpd.GeoDataFrame, nx.Graph]:
    """
    Split MST edges longer than `max_pole_span_m` by inserting intermediate poles.

    Output schema guarantees:
      - pole_id: stable unique integer id (preserved for base poles, new for inserted poles)
      - pole_type: preserved for base poles; inserted poles start as "support"
      - pole_origin: "base" for input poles, "inserted" for new poles
    """
    if max_pole_span_m is None or max_pole_span_m <= 0:
        out = gdf_poles.copy().reset_index(drop=True)
        if "pole_id" not in out.columns:
            out["pole_id"] = out.index.astype(int)
        if "pole_type" not in out.columns:
            out["pole_type"] = "base"
        if "pole_origin" not in out.columns:
            out["pole_origin"] = "base"
        return out, mst

    base = gdf_poles.copy().reset_index(drop=True)
    if "pole_id" not in base.columns:
        base["pole_id"] = base.index.astype(int)
    if "pole_type" not in base.columns:
        base["pole_type"] = "base"
    if "pole_origin" not in base.columns:
        base["pole_origin"] = "base"

    base_geoms: List[Point] = [row.geometry for row in base.itertuples()]
    n_base = len(base_geoms)

    if n_base != mst.number_of_nodes():
        raise ValueError(
            "Inconsistent MST: number of nodes in MST does not match number of poles in gdf_poles."
        )

    G2 = nx.Graph()
    for node_id in range(n_base):
        G2.add_node(node_id)

    new_rows: List[Dict[str, Any]] = []
    next_pole_id = int(base["pole_id"].max()) + 1
    next_node_id = n_base

    for u, v, data in mst.edges(data=True):
        p_u: Point = base_geoms[u]
        p_v: Point = base_geoms[v]
        d = float(data.get("weight", p_u.distance(p_v)))

        if d <= max_pole_span_m:
            G2.add_edge(u, v, weight=d)
            continue

        n_segments = max(2, int(math.ceil(d / max_pole_span_m)))
        line = LineString([p_u, p_v])

        prev_node = u
        prev_point = p_u

        for k in range(1, n_segments):
            t = k / n_segments
            pt_k: Point = line.interpolate(t, normalized=True)

            # Do not create a node for v itself
            if k == n_segments:
                break

            new_rows.append(
                {
                    "geometry": pt_k,
                    "pole_id": next_pole_id,
                    "pole_type": "support",
                    "pole_origin": "inserted",
                }
            )
            next_pole_id += 1

            new_node = next_node_id
            next_node_id += 1
            G2.add_node(new_node)

            seg_len = float(prev_point.distance(pt_k))
            G2.add_edge(prev_node, new_node, weight=seg_len)

            prev_node = new_node
            prev_point = pt_k

        seg_len_last = float(prev_point.distance(p_v))
        G2.add_edge(prev_node, v, weight=seg_len_last)

    gdf_support = gpd.GeoDataFrame(new_rows, crs=base.crs)
    densified = pd.concat([base, gdf_support], ignore_index=True)
    densified = gpd.GeoDataFrame(densified, crs=base.crs).reset_index(drop=True)

    return densified, G2