from __future__ import annotations

import io
from typing import Dict, List, Tuple, Any

import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
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

        if geometry.geom_type == "LineString":
            line_geoms = [geometry]
        elif geometry.geom_type == "MultiLineString":
            line_geoms = list(geometry.geoms)
        else:
            continue

        for line in line_geoms:
            if line is None or line.is_empty:
                continue
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            # Always add endpoints
            sampled_points.append(Point(coords[0]))
            sampled_points.append(Point(coords[-1]))

            # Additional sampling along the line
            if line.length > sampling_distance:
                sampled_points.extend(sample_points_along_line(line, sampling_distance))

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

    IMPORTANT:
    - Graph node IDs are *stable pole_id* (not row index).
    - This prevents downstream identity mismatches (PF, exports, promotion).
    """
    if gdf_poles.empty:
        raise ValueError("create_graph_and_mst: empty pole set.")

    poles = gdf_poles.copy().reset_index(drop=True)
    if "pole_id" not in poles.columns:
        poles["pole_id"] = poles.index.astype(int)

    poles["pole_id"] = pd.to_numeric(poles["pole_id"], errors="coerce")
    poles = poles.dropna(subset=["pole_id"]).copy()
    poles["pole_id"] = poles["pole_id"].astype(int)

    # must be projected CRS for Euclidean distance in meters
    if getattr(poles.crs, "is_geographic", False):
        raise ValueError("create_graph_and_mst expects projected CRS in meters, not geographic.")

    pids = poles["pole_id"].to_numpy(dtype=int)
    xs = poles.geometry.x.to_numpy(dtype=float)
    ys = poles.geometry.y.to_numpy(dtype=float)

    G = nx.Graph()
    for pid in pids:
        G.add_node(int(pid))

    # complete graph (OK for a few hundred)
    n = len(pids)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = float(np.sqrt(dx * dx + dy * dy))
            G.add_edge(int(pids[i]), int(pids[j]), weight=d)

    return nx.minimum_spanning_tree(G)


def mst_edges_as_latlon(
    gdf_poles_4326: gpd.GeoDataFrame, mst: nx.Graph
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Convert MST edges to ((lat1, lon1), (lat2, lon2)) pairs for plotting.

    MST nodes are stable pole_id values.
    """
    poles = gdf_poles_4326.copy()
    if "pole_id" not in poles.columns:
        poles["pole_id"] = poles.index.astype(int)

    poles["pole_id"] = pd.to_numeric(poles["pole_id"], errors="coerce")
    poles = poles.dropna(subset=["pole_id"]).copy()
    poles["pole_id"] = poles["pole_id"].astype(int)

    geom_by_pid = poles.set_index("pole_id").geometry.to_dict()

    out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for u_pid, v_pid in mst.edges():
        p1 = geom_by_pid.get(int(u_pid))
        p2 = geom_by_pid.get(int(v_pid))
        if p1 is None or p2 is None:
            continue
        out.append(((float(p1.y), float(p1.x)), (float(p2.y), float(p2.x))))
    return out


def save_mst_to_geojson(
    gdf_poles: gpd.GeoDataFrame, mst: nx.Graph
) -> Tuple[io.BytesIO, io.BytesIO]:
    """
    Export MST nodes/edges to GeoJSON. Nodes include:
      - pole_id
      - pole_type
      - pole_origin

    Edges include:
      - u_pole_id, v_pole_id (stable endpoints)
      - weight
    """
    poles = gdf_poles.copy().reset_index(drop=True)
    if "pole_id" not in poles.columns:
        poles["pole_id"] = poles.index.astype(int)
    if "pole_type" not in poles.columns:
        poles["pole_type"] = "base"
    if "pole_origin" not in poles.columns:
        poles["pole_origin"] = "base"

    poles["pole_id"] = pd.to_numeric(poles["pole_id"], errors="coerce")
    poles = poles.dropna(subset=["pole_id"]).copy()
    poles["pole_id"] = poles["pole_id"].astype(int)

    geom_by_pid = poles.set_index("pole_id").geometry.to_dict()

    nodes_records = []
    for pole in poles.itertuples():
        nodes_records.append(
            {
                "pole_id": int(getattr(pole, "pole_id")),
                "pole_type": str(getattr(pole, "pole_type")),
                "pole_origin": str(getattr(pole, "pole_origin")),
                "geometry": pole.geometry,
            }
        )
    gdf_nodes = gpd.GeoDataFrame(nodes_records, crs=poles.crs)

    edges_records = []
    for u_pid, v_pid, data in mst.edges(data=True):
        u_pid = int(u_pid)
        v_pid = int(v_pid)
        start = geom_by_pid.get(u_pid)
        end = geom_by_pid.get(v_pid)
        if start is None or end is None:
            continue

        edges_records.append(
            {
                "u_pole_id": u_pid,
                "v_pole_id": v_pid,
                "weight": float(data.get("weight", start.distance(end))),
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

    IMPORTANT:
    - Graph node IDs are stable pole_id (not row index).
    - Inserted poles get new unique pole_id values.
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

    base["pole_id"] = pd.to_numeric(base["pole_id"], errors="coerce")
    base = base.dropna(subset=["pole_id"]).copy()
    base["pole_id"] = base["pole_id"].astype(int)

    # Map pole_id -> geometry
    geom_by_pid: Dict[int, Point] = base.set_index("pole_id").geometry.to_dict()

    # Ensure MST nodes match pole_id set
    pole_ids_set = set(int(x) for x in base["pole_id"].tolist())
    mst_nodes_set = set(int(n) for n in mst.nodes())
    if mst_nodes_set != pole_ids_set:
        missing = sorted(list(pole_ids_set - mst_nodes_set))[:20]
        extra = sorted(list(mst_nodes_set - pole_ids_set))[:20]
        raise ValueError(
            "Inconsistent MST vs poles (node set mismatch). "
            f"Missing-in-mst (first 20): {missing} | Extra-in-mst (first 20): {extra}"
        )

    G2 = nx.Graph()
    for pid in pole_ids_set:
        G2.add_node(int(pid))

    new_rows: List[Dict[str, Any]] = []
    next_pole_id = int(base["pole_id"].max()) + 1

    for u_pid, v_pid, data in mst.edges(data=True):
        u_pid = int(u_pid)
        v_pid = int(v_pid)
        p_u: Point = geom_by_pid[u_pid]
        p_v: Point = geom_by_pid[v_pid]
        d = float(data.get("weight", p_u.distance(p_v)))

        if d <= max_pole_span_m:
            G2.add_edge(u_pid, v_pid, weight=d)
            continue

        n_segments = max(2, int(math.ceil(d / max_pole_span_m)))
        line = LineString([p_u, p_v])

        prev_pid = u_pid
        prev_point = p_u

        # insert intermediate poles (n_segments-1 inserted points)
        for k in range(1, n_segments):
            t = k / n_segments
            pt_k: Point = line.interpolate(t, normalized=True)

            # last segment ends at v_pid, so stop before creating a pole at v
            if k == n_segments:
                break

            new_pid = int(next_pole_id)
            next_pole_id += 1

            new_rows.append(
                {
                    "geometry": pt_k,
                    "pole_id": new_pid,
                    "pole_type": "support",
                    "pole_origin": "inserted",
                }
            )

            G2.add_node(new_pid)
            seg_len = float(prev_point.distance(pt_k))
            G2.add_edge(prev_pid, new_pid, weight=seg_len)

            prev_pid = new_pid
            prev_point = pt_k

        seg_len_last = float(prev_point.distance(p_v))
        G2.add_edge(prev_pid, v_pid, weight=seg_len_last)

    gdf_support = gpd.GeoDataFrame(new_rows, crs=base.crs)
    densified = pd.concat([base, gdf_support], ignore_index=True)
    densified = gpd.GeoDataFrame(densified, crs=base.crs).reset_index(drop=True)

    return densified, G2

def deduplicate_poles_with_remap(
    gdf_poles: gpd.GeoDataFrame,
    associations_df: pd.DataFrame,
    *,
    tol_m: float = 0.75,
    prefer_serving: bool = True,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[int, int]]:
    """
    Deduplicate poles that are within `tol_m` meters.

    Keeps ONE canonical pole per cluster, drops the rest, and returns:
      - cleaned poles GeoDataFrame
      - remapped associations_df (pole_id updated to canonical)
      - mapping old_pole_id -> canonical_pole_id

    Notes:
    - Requires projected CRS (meters). Call BEFORE reprojecting to 4326.
    - If prefer_serving=True, clusters keep a pole that already serves at least one building, if present.
      Otherwise keep the smallest pole_id.
    """
    if gdf_poles.empty:
        return gdf_poles.copy(), associations_df.copy(), {}

    poles = gdf_poles.copy().reset_index(drop=True)

    if "pole_id" not in poles.columns:
        poles["pole_id"] = poles.index.astype(int)

    # Defensive: must be projected CRS
    if getattr(poles.crs, "is_geographic", False):
        raise ValueError("deduplicate_poles_with_remap expects projected CRS in meters, not geographic.")

    poles["pole_id"] = pd.to_numeric(poles["pole_id"], errors="coerce")
    poles = poles.dropna(subset=["pole_id"]).copy()
    poles["pole_id"] = poles["pole_id"].astype(int)

    # Build serving set (optional preference)
    serving_ids: set[int] = set()
    if prefer_serving and associations_df is not None and not associations_df.empty and "pole_id" in associations_df.columns:
        serving_ids = set(pd.to_numeric(associations_df["pole_id"], errors="coerce").dropna().astype(int).unique())

    # Simple greedy clustering by distance threshold (O(N^2) but OK for ~hundreds poles)
    used = np.zeros(len(poles), dtype=bool)
    mapping: dict[int, int] = {}

    # Precompute coordinates for speed
    xs = poles.geometry.x.to_numpy(dtype=float)
    ys = poles.geometry.y.to_numpy(dtype=float)
    pids = poles["pole_id"].to_numpy(dtype=int)

    # For reproducibility: process poles ordered by pole_id
    order = np.argsort(pids)

    keep_rows: list[int] = []

    for idx in order:
        if used[idx]:
            continue

        # find all poles within tol of this seed
        dx = xs - xs[idx]
        dy = ys - ys[idx]
        d2 = dx * dx + dy * dy
        cluster_idx = np.where((~used) & (d2 <= float(tol_m) ** 2))[0]

        cluster_pids = [int(pids[i]) for i in cluster_idx]

        # choose canonical
        if prefer_serving:
            serving_in_cluster = [pid for pid in cluster_pids if pid in serving_ids]
            if serving_in_cluster:
                canonical = int(min(serving_in_cluster))
            else:
                canonical = int(min(cluster_pids))
        else:
            canonical = int(min(cluster_pids))

        # choose which *row* to keep for the canonical geometry (if canonical exists in cluster)
        if canonical in cluster_pids:
            keep_i = cluster_idx[cluster_pids.index(canonical)]
        else:
            keep_i = int(cluster_idx[0])

        keep_rows.append(int(keep_i))

        # map all cluster members -> canonical
        for pid in cluster_pids:
            mapping[int(pid)] = canonical

        used[cluster_idx] = True

    cleaned = poles.iloc[sorted(set(keep_rows))].copy().reset_index(drop=True)

    # Optionally "snap" canonical geometry to centroid of cluster (more realistic)
    # Here we keep the chosen geometry to avoid moving poles unexpectedly.

    # Remap associations
    assoc_out = associations_df.copy() if associations_df is not None else pd.DataFrame(columns=["pole_id", "building_id"])
    if not assoc_out.empty and "pole_id" in assoc_out.columns:
        assoc_out["pole_id"] = pd.to_numeric(assoc_out["pole_id"], errors="coerce")
        assoc_out = assoc_out.dropna(subset=["pole_id"]).copy()
        assoc_out["pole_id"] = assoc_out["pole_id"].astype(int).map(lambda pid: mapping.get(int(pid), int(pid)))
        assoc_out = assoc_out.dropna(subset=["pole_id"]).copy()
        assoc_out["pole_id"] = assoc_out["pole_id"].astype(int)
        assoc_out = assoc_out.drop_duplicates(subset=["pole_id", "building_id"])

    return cleaned, assoc_out, mapping


def merge_graph_nodes_with_remap(
    graph: nx.Graph,
    mapping: dict[int, int],
) -> nx.Graph:
    """
    Collapse graph nodes according to an old->canonical pole_id mapping.

    - Node IDs remain stable pole_id values.
    - Self-loops introduced by merges are dropped.
    - Parallel edges created by merges are collapsed, keeping the shortest weight.
    """
    if not mapping:
        return graph.copy()

    merged = nx.Graph()

    for node in graph.nodes():
        merged.add_node(int(mapping.get(int(node), int(node))))

    for u, v, data in graph.edges(data=True):
        u2 = int(mapping.get(int(u), int(u)))
        v2 = int(mapping.get(int(v), int(v)))
        if u2 == v2:
            continue

        weight = float(data.get("weight", 0.0))
        if merged.has_edge(u2, v2):
            existing = float(merged[u2][v2].get("weight", weight))
            merged[u2][v2]["weight"] = min(existing, weight)
        else:
            merged.add_edge(u2, v2, weight=weight)

    return merged
