from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import geopandas as gpd
import networkx as nx
import pandas as pd

from .distribution_io import load_and_transform_data
from .distribution_algos import (
    collect_sampled_points,
    associate_buildings_to_poles,
    place_poles_for_unassociated_buildings,
    create_graph_and_mst,
    densify_mst_edges,          
    mst_edges_as_latlon,
    save_mst_to_geojson,
)


def run_low_voltage(
    users_file,
    roads_file,
    sampling_distance: float,
    user_distance: float,
    max_associations: int,
    cost_per_km: float,
    fixed_costs: float,
    centroid_hint: Tuple[float, float] | None = None,
    *,
    allow_unserved_isolated: bool = False,
    min_cluster_size: int = 1,
    max_pole_span_m: float | None = None,
) -> Dict[str, Any]:
    """
    High-level service to compute LV layout & costs.

    Backend-only (no Streamlit dependencies). It orchestrates:
      - reading the data,
      - running heuristics (association + clustering),
      - computing and densifying the MST,
      - assembling a results dictionary for the UI.

    Parameters
    ----------
    users_file : file-like
        Uploaded file handle (.gpkg or .xlsx) with user locations.
    roads_file : file-like or None
        Uploaded file handle (.gpkg) for roads, or None.
    sampling_distance : float
        Distance between candidate poles along roads [m].
    user_distance : float
        Maximum radius to associate users to a pole [m].
    max_associations : int
        Maximum number of users per pole.
    cost_per_km : float
        LV line cost per kilometer [USD/km].
    fixed_costs : float
        Additional fixed costs [USD].
    centroid_hint : (lat, lon), optional
        Optional map center override in EPSG:4326.
    allow_unserved_isolated : bool, optional
        If True, small / isolated building clusters (below min_cluster_size)
        can remain unserved (standalone candidates).
    min_cluster_size : int, optional
        Minimum cluster size for LV pole placement when
        allow_unserved_isolated=True.
    max_pole_span_m : float or None, optional
        If provided and > 0, long MST edges are post-processed:
        any edge > max_pole_span_m is subdivided with intermediate
        support poles. Connectivity is preserved (single tree).

    Returns
    -------
    dict
        {
            "metrics": {...},
            "gdf_buildings_4326": GeoDataFrame,
            "gdf_poles_4326": GeoDataFrame,
            "gdf_roads_4326": GeoDataFrame or None,
            "gdf_served_4326": GeoDataFrame,
            "gdf_unserved_4326": GeoDataFrame,
            "mst_edges_latlon": list[((lat1, lon1), (lat2, lon2))],
            "downloads": {
                "nodes_geojson": BytesIO,
                "edges_geojson": BytesIO,
            },
            "center": (lat, lon),
        }
    """
    # ------------------------------------------------------------------
    # 1) Load data in projected CRS (meters)
    # ------------------------------------------------------------------
    gdf_roads = load_and_transform_data(roads_file) if roads_file else None
    gdf_buildings = load_and_transform_data(users_file)

    if gdf_buildings is None or gdf_buildings.empty:
        raise ValueError("Users file could not be loaded or is empty.")

    # ------------------------------------------------------------------
    # 2) Candidate poles from roads (if any)
    # ------------------------------------------------------------------
    if gdf_roads is not None and not gdf_roads.empty:
        sampled_points = collect_sampled_points(gdf_roads, sampling_distance)
        gdf_associated_poles = (
            gpd.GeoDataFrame({"geometry": sampled_points}, crs=gdf_buildings.crs)
            if sampled_points
            else gpd.GeoDataFrame(geometry=[], crs=gdf_buildings.crs)
        )
    else:
        # Start with no poles; all will be created from unassociated buildings
        gdf_associated_poles = gpd.GeoDataFrame(geometry=[], crs=gdf_buildings.crs)

    # ------------------------------------------------------------------
    # 3) Associate buildings to existing (road-based) poles
    # ------------------------------------------------------------------
    if not gdf_associated_poles.empty:
        associations_df = associate_buildings_to_poles(
            gdf_buildings=gdf_buildings,
            gdf_poles=gdf_associated_poles,
            user_distance=user_distance,
            max_associations=max_associations,
        )
        # Keep only poles that actually got at least one association
        gdf_associated_poles = gdf_associated_poles[
            gdf_associated_poles.index.isin(associations_df["pole_id"])
        ].copy()
    else:
        associations_df = pd.DataFrame(columns=["pole_id", "building_id"])

    # ------------------------------------------------------------------
    # 4) New poles for remaining buildings (clustering heuristic)
    # ------------------------------------------------------------------
    gdf_unassociated = gdf_buildings[
        ~gdf_buildings.index.isin(associations_df.get("building_id", []))
    ]

    gdf_new_poles, new_associations, gdf_remaining = place_poles_for_unassociated_buildings(
        gdf_unassociated_buildings=gdf_unassociated,
        user_distance=user_distance,
        max_associations=max_associations,
        allow_unserved_isolated=allow_unserved_isolated,
        min_cluster_size=min_cluster_size,
    )

    # Final pole set (roads-based + newly placed)
    gdf_final_poles = pd.concat([gdf_associated_poles, gdf_new_poles])

    # Map new associations to final pole indices
    if len(new_associations) > 0:
        # new_associations is list of (pole_point, building_geom)
        new_df = pd.DataFrame(new_associations, columns=["pole", "building_geom"])

        # Map pole geometry -> index in gdf_final_poles
        new_df["pole_id"] = new_df["pole"].apply(
            lambda p: gdf_final_poles[gdf_final_poles.geometry == p].index[0]
        )

        # Map building geometry -> index in gdf_buildings
        new_df["building_id"] = new_df["building_geom"].apply(
            lambda g: gdf_buildings[gdf_buildings.geometry == g].index[0]
        )

        new_assoc_ids = new_df[["pole_id", "building_id"]]

        if "building_id" not in associations_df.columns:
            associations_df["building_id"] = pd.Series(dtype=int)

        associations_df = pd.concat(
            [associations_df[["pole_id", "building_id"]], new_assoc_ids],
            ignore_index=True,
        )

    # ------------------------------------------------------------------
    # 5) Served vs unserved buildings
    # ------------------------------------------------------------------
    if allow_unserved_isolated:
        gdf_unserved = gdf_remaining
    else:
        gdf_unserved = gdf_buildings.iloc[0:0]

    unserved_ids = set(gdf_unserved.index)
    served_ids = [idx for idx in gdf_buildings.index if idx not in unserved_ids]
    gdf_served = (
        gdf_buildings.loc[served_ids] if len(served_ids) > 0 else gdf_buildings.iloc[0:0]
    )

    num_buildings = len(gdf_buildings)
    num_served = len(gdf_served)
    num_unserved = len(gdf_unserved)

    # ------------------------------------------------------------------
    # 6) MST in projected CRS + densification of long spans
    # ------------------------------------------------------------------
    if gdf_final_poles.empty:
        raise ValueError("No poles could be placed; check input data and parameters.")

    mst_base: nx.Graph = create_graph_and_mst(gdf_final_poles)

    # Densify MST edges if requested (post-processing)
    gdf_poles_densified, mst = densify_mst_edges(
        gdf_poles=gdf_final_poles,
        mst=mst_base,
        max_pole_span_m=max_pole_span_m or 0.0,
    )

    # ------------------------------------------------------------------
    # 7) Length & costs (using densified MST)
    # ------------------------------------------------------------------
    length_m = sum(nx.get_edge_attributes(mst, "weight").values())
    length_km = length_m / 1000.0
    total_lv_cost = length_km * cost_per_km + float(fixed_costs)

    num_poles_code = len(gdf_poles_densified)
    num_poles_length = math.ceil((length_km * 1000.0) / sampling_distance)

    # ------------------------------------------------------------------
    # 8) Reproject outputs to EPSG:4326 for plotting
    # ------------------------------------------------------------------
    gdf_buildings_4326 = gdf_buildings.to_crs(epsg=4326)
    gdf_poles_4326 = gdf_poles_densified.to_crs(epsg=4326)
    gdf_roads_4326 = gdf_roads.to_crs(epsg=4326) if gdf_roads is not None else None

    gdf_served_4326 = (
        gdf_served.to_crs(epsg=4326) if num_served > 0 else gdf_buildings_4326.iloc[0:0]
    )
    gdf_unserved_4326 = (
        gdf_unserved.to_crs(epsg=4326) if num_unserved > 0 else gdf_buildings_4326.iloc[0:0]
    )

    # ------------------------------------------------------------------
    # 9) Map center
    # ------------------------------------------------------------------
    if centroid_hint and any(centroid_hint):
        center = centroid_hint
    else:
        if not gdf_poles_4326.empty:
            c = gdf_poles_4326.unary_union.centroid
            center = (c.y, c.x)
        else:
            c = gdf_buildings_4326.unary_union.centroid
            center = (c.y, c.x)

    # ------------------------------------------------------------------
    # 10) Edge list for plotting
    # ------------------------------------------------------------------
    mst_edges_latlon = mst_edges_as_latlon(gdf_poles_4326, mst)

    # ------------------------------------------------------------------
    # 11) Downloads (GeoJSON buffers)
    # ------------------------------------------------------------------
    nodes_geojson, edges_geojson = save_mst_to_geojson(gdf_poles_4326, mst)

    # ------------------------------------------------------------------
    # 12) Final result dict
    # ------------------------------------------------------------------
    return {
        "metrics": {
            "network_length_km": length_km,
            "total_lv_cost_usd": total_lv_cost,
            "num_buildings": num_buildings,
            "num_poles_code": num_poles_code,
            "num_poles_length": num_poles_length,
            "num_served": num_served,
            "num_unserved": num_unserved,
        },
        "gdf_buildings_4326": gdf_buildings_4326,
        "gdf_poles_4326": gdf_poles_4326,
        "gdf_roads_4326": gdf_roads_4326,
        "gdf_served_4326": gdf_served_4326,
        "gdf_unserved_4326": gdf_unserved_4326,
        "mst_edges_latlon": mst_edges_latlon,
        "downloads": {
            "nodes_geojson": nodes_geojson,
            "edges_geojson": edges_geojson,
        },
        "center": center,
    }
