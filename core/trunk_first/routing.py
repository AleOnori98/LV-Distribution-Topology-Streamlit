from __future__ import annotations

import math
from itertools import combinations

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from .config import RoutingConfig


def _is_angle_aligned(angle: float, reference: float, tolerance_radians: float) -> bool:
    """Return True when two line directions are parallel, ignoring sign."""
    diff = abs((angle - reference + math.pi) % (2 * math.pi) - math.pi)
    alt = abs((angle + math.pi - reference + math.pi) % (2 * math.pi) - math.pi)
    return min(diff, alt) <= tolerance_radians


def _edge_weight(
    pole_u,
    pole_v,
    distance: float,
    angle_width: float,
    angle_length: float,
    trunk_distance_u: float,
    trunk_distance_v: float,
    spacing: float,
    config: RoutingConfig,
) -> float:
    """Compute the directional bias for one candidate edge."""
    angle = math.atan2(pole_v["y"] - pole_u["y"], pole_v["x"] - pole_u["x"])
    tolerance = math.radians(config.angle_tolerance_degrees)
    orthogonal = _is_angle_aligned(angle, angle_width, tolerance) or _is_angle_aligned(angle, angle_length, tolerance)

    if pole_u["is_trunk"] and pole_v["is_trunk"]:
        return distance * config.trunk_trunk_factor

    if orthogonal and (pole_u["is_trunk"] or pole_v["is_trunk"]) and distance <= spacing * config.trunk_connection_radius_factor:
        return distance * config.orthogonal_trunk_bonus

    trunk_penalty = ((trunk_distance_u / spacing) + (trunk_distance_v / spacing)) * config.trunk_distance_penalty_scale
    if orthogonal:
        return distance * (1.0 + trunk_penalty)

    return distance * config.diagonal_penalty * (1.0 + trunk_penalty)


def _connect_required_components(
    graph: nx.Graph,
    poles_gdf: gpd.GeoDataFrame,
    required_pole_ids: set[int],
    angle_width: float,
    angle_length: float,
    spacing: float,
    trunk_distance_map: dict[int, float],
    config: RoutingConfig,
) -> None:
    """Connect disconnected required-node components with cheapest bridging edges."""
    while True:
        components = list(nx.connected_components(graph))
        required_components = [comp for comp in components if comp & required_pole_ids]
        if len(required_components) <= 1:
            return

        best = None
        for comp_a, comp_b in combinations(required_components, 2):
            a_rows = poles_gdf[poles_gdf["pole_id"].isin(comp_a)]
            b_rows = poles_gdf[poles_gdf["pole_id"].isin(comp_b)]
            for _, row_a in a_rows.iterrows():
                for _, row_b in b_rows.iterrows():
                    distance = row_a.geometry.distance(row_b.geometry)
                    weight = _edge_weight(
                        row_a,
                        row_b,
                        distance=distance,
                        angle_width=angle_width,
                        angle_length=angle_length,
                        trunk_distance_u=trunk_distance_map[int(row_a["pole_id"])],
                        trunk_distance_v=trunk_distance_map[int(row_b["pole_id"])],
                        spacing=spacing,
                        config=config,
                    )
                    if best is None or weight < best[0]:
                        best = (weight, int(row_a["pole_id"]), int(row_b["pole_id"]), distance)

        if best is None:
            raise ValueError("Could not reconnect required-node components in the routing graph.")

        _, pole_a, pole_b, distance = best
        row_a = poles_gdf.loc[poles_gdf["pole_id"] == pole_a].iloc[0]
        row_b = poles_gdf.loc[poles_gdf["pole_id"] == pole_b].iloc[0]
        graph.add_edge(
            pole_a,
            pole_b,
            weight=_edge_weight(
                row_a,
                row_b,
                distance=distance,
                angle_width=angle_width,
                angle_length=angle_length,
                trunk_distance_u=trunk_distance_map[pole_a],
                trunk_distance_v=trunk_distance_map[pole_b],
                spacing=spacing,
                config=config,
            ),
            length=distance,
        )


def build_biased_secondary_network(
    poles_gdf: gpd.GeoDataFrame,
    trunk_pole_ids: set[int],
    required_pole_ids: set[int],
    angle_width: float,
    angle_length: float,
    pole_spacing: float,
    config: RoutingConfig,
) -> tuple[gpd.GeoDataFrame, set[int]]:
    """
    Build a sparse, biased secondary network.

    The method uses:
    1. a sparse local graph (kNN + explicit nearest-trunk links),
    2. shortest-path distances between required poles,
    3. an MST over the metric closure of required poles,
    4. path expansion back into physical line segments.
    """
    if not required_pole_ids:
        empty = gpd.GeoDataFrame(columns=["from_pole", "to_pole", "Length", "geometry"], geometry="geometry", crs=poles_gdf.crs)
        return empty, set()

    if poles_gdf.empty:
        raise ValueError("Cannot route secondary lines without poles.")

    coords = np.column_stack([poles_gdf["x"].to_numpy(), poles_gdf["y"].to_numpy()])
    tree = cKDTree(coords)

    trunk_rows = poles_gdf[poles_gdf["pole_id"].isin(trunk_pole_ids)]
    if trunk_rows.empty:
        raise ValueError("At least one trunk pole is required for secondary routing.")

    trunk_coords = np.column_stack([trunk_rows["x"].to_numpy(), trunk_rows["y"].to_numpy()])
    trunk_tree = cKDTree(trunk_coords)
    trunk_local_index_to_id = trunk_rows["pole_id"].to_numpy()

    trunk_distance_map: dict[int, float] = {}
    for _, row in poles_gdf.iterrows():
        dist, _ = trunk_tree.query((row["x"], row["y"]), k=1)
        trunk_distance_map[int(row["pole_id"])] = float(dist)

    graph = nx.Graph()
    for _, row in poles_gdf.iterrows():
        pole_id = int(row["pole_id"])
        graph.add_node(pole_id, x=float(row["x"]), y=float(row["y"]), is_trunk=bool(row["is_trunk"]))

    k = min(len(poles_gdf), max(2, config.k_neighbors + 1))
    max_radius = pole_spacing * config.neighbor_radius_factor
    row_by_id = {int(row["pole_id"]): row for _, row in poles_gdf.iterrows()}

    for _, row in poles_gdf.iterrows():
        pole_id = int(row["pole_id"])
        distances, neighbor_idx = tree.query((row["x"], row["y"]), k=k)
        distances = np.atleast_1d(distances)
        neighbor_idx = np.atleast_1d(neighbor_idx)

        for distance, idx in zip(distances[1:], neighbor_idx[1:]):
            if np.isinf(distance):
                continue
            if float(distance) > max_radius:
                continue
            neighbor_row = poles_gdf.iloc[int(idx)]
            neighbor_id = int(neighbor_row["pole_id"])
            if graph.has_edge(pole_id, neighbor_id):
                continue

            graph.add_edge(
                pole_id,
                neighbor_id,
                weight=_edge_weight(
                    row,
                    neighbor_row,
                    distance=float(distance),
                    angle_width=angle_width,
                    angle_length=angle_length,
                    trunk_distance_u=trunk_distance_map[pole_id],
                    trunk_distance_v=trunk_distance_map[neighbor_id],
                    spacing=pole_spacing,
                    config=config,
                ),
                length=float(distance),
            )

        if not bool(row["is_trunk"]):
            dist, trunk_idx = trunk_tree.query((row["x"], row["y"]), k=1)
            trunk_id = int(trunk_local_index_to_id[int(trunk_idx)])
            if not graph.has_edge(pole_id, trunk_id):
                trunk_row = row_by_id[trunk_id]
                graph.add_edge(
                    pole_id,
                    trunk_id,
                    weight=_edge_weight(
                        row,
                        trunk_row,
                        distance=float(dist),
                        angle_width=angle_width,
                        angle_length=angle_length,
                        trunk_distance_u=trunk_distance_map[pole_id],
                        trunk_distance_v=trunk_distance_map[trunk_id],
                        spacing=pole_spacing,
                        config=config,
                    ),
                    length=float(dist),
                )

    _connect_required_components(
        graph=graph,
        poles_gdf=poles_gdf,
        required_pole_ids=required_pole_ids,
        angle_width=angle_width,
        angle_length=angle_length,
        spacing=pole_spacing,
        trunk_distance_map=trunk_distance_map,
        config=config,
    )

    metric_graph = nx.Graph()
    shortest_paths: dict[tuple[int, int], list[int]] = {}

    required_ids_sorted = sorted(required_pole_ids)
    for source in required_ids_sorted:
        lengths, paths = nx.single_source_dijkstra(graph, source=source, weight="weight")
        for target in required_ids_sorted:
            if target <= source:
                continue
            if target not in lengths:
                continue
            metric_graph.add_edge(source, target, weight=float(lengths[target]))
            shortest_paths[(source, target)] = paths[target]

    if metric_graph.number_of_edges() == 0 and len(required_pole_ids) > 1:
        raise ValueError("Required poles are not mutually reachable in the routing graph.")

    if len(required_pole_ids) == 1:
        empty = gpd.GeoDataFrame(columns=["from_pole", "to_pole", "Length", "geometry"], geometry="geometry", crs=poles_gdf.crs)
        return empty, set(required_pole_ids)

    mst_required = nx.minimum_spanning_tree(metric_graph, weight="weight")

    used_edges: set[tuple[int, int]] = set()
    used_poles: set[int] = set(required_pole_ids)

    for source, target in mst_required.edges():
        path = shortest_paths.get((min(source, target), max(source, target)))
        if path is None:
            raise ValueError("Metric closure path expansion failed.")
        for u, v in zip(path[:-1], path[1:]):
            used_edges.add(tuple(sorted((u, v))))
            used_poles.add(u)
            used_poles.add(v)

    records = []
    for u, v in sorted(used_edges):
        row_u = row_by_id[u]
        row_v = row_by_id[v]
        if (not config.include_trunk_trunk_in_secondary) and bool(row_u["is_trunk"]) and bool(row_v["is_trunk"]):
            continue
        line = LineString([row_u.geometry, row_v.geometry])
        records.append(
            {
                "from_pole": u,
                "to_pole": v,
                "Length": float(line.length),
                "geometry": line,
            }
        )

    secondary_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=poles_gdf.crs)
    return secondary_gdf, used_poles
