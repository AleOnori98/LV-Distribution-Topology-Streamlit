from __future__ import annotations

from typing import Any, Dict, Tuple
import math

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point

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
    centroid_hint: Tuple[float, float] | None = None,
    *,
    allow_unserved_isolated: bool = False,
    min_cluster_size: int = 1,
    max_pole_span_m: float | None = None,
) -> Dict[str, Any]:
    """
    Robust LV topology pipeline:
      - loads buildings (+ optional roads) in projected CRS (meters),
      - generates candidate road poles (optional),
      - associates buildings to road poles (capacity + radius),
      - places new poles for remaining buildings (clustering heuristic),
      - computes service-drop length (building -> assigned serving pole),
      - computes MST backbone on poles and optionally densifies long spans,
      - returns metrics + plotting/export artifacts in EPSG:4326.

    Metrics include:
      - total length, backbone length, service-drop length
      - total poles, serving poles, support poles (added by densification)
      - buildings served/unserved (standalone candidates)
    """

    # ------------------------------------------------------------------
    # 1) Load data in projected CRS (meters)
    # ------------------------------------------------------------------
    gdf_roads = load_and_transform_data(roads_file) if roads_file else None
    gdf_buildings = load_and_transform_data(users_file)

    if gdf_buildings is None or gdf_buildings.empty:
        raise ValueError("Users file could not be loaded or is empty.")
    if gdf_buildings.geometry.isna().any():
        raise ValueError("Users file contains missing geometries.")
    if getattr(gdf_buildings.crs, "is_geographic", False):
        # load_and_transform_data should handle reprojection; this is just a safety belt
        raise ValueError("Buildings are still in a geographic CRS; expected projected CRS in meters.")

    # Ensure stable building IDs (do NOT reset index if you rely on input IDs elsewhere)
    # Here we assume index is a valid building_id universe.
    bldg_geom_by_id = gdf_buildings.geometry.to_dict()
    bldg_wkb_to_id = {geom.wkb: idx for idx, geom in bldg_geom_by_id.items()}

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
        gdf_associated_poles = gpd.GeoDataFrame(geometry=[], crs=gdf_buildings.crs)

    # ------------------------------------------------------------------
    # 3) Associate buildings to existing (road-based) poles
    #     IMPORTANT: enforce stable pole IDs = row positions 0..N-1
    # ------------------------------------------------------------------
    if not gdf_associated_poles.empty:
        gdf_associated_poles = gdf_associated_poles.reset_index(drop=True)

        associations_df = associate_buildings_to_poles(
            gdf_buildings=gdf_buildings,
            gdf_poles=gdf_associated_poles,
            user_distance=user_distance,
            max_associations=max_associations,
        )

        if not associations_df.empty:
            associations_df = associations_df[["pole_id", "building_id"]].dropna().drop_duplicates()

            # Keep only poles that got >=1 building (POSITION-BASED)
            kept_old_ids = sorted(associations_df["pole_id"].astype(int).unique().tolist())
            gdf_associated_poles = gdf_associated_poles.iloc[kept_old_ids].reset_index(drop=True)

            # Remap pole IDs to compact 0..(n_kept-1)
            remap = {old: new for new, old in enumerate(kept_old_ids)}
            associations_df["pole_id"] = associations_df["pole_id"].map(remap).astype(int)
            associations_df["building_id"] = associations_df["building_id"].astype(int)
        else:
            # No building attached to road candidates -> treat them as unused candidates
            gdf_associated_poles = gdf_associated_poles.iloc[0:0].copy()
            associations_df = pd.DataFrame(columns=["pole_id", "building_id"])
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

    # Ensure gdf_new_poles has a clean local RangeIndex for any fallback logic
    gdf_new_poles = gdf_new_poles.reset_index(drop=True)

    # Final pole set (roads-based + newly placed) with stable 0..N-1 IDs
    gdf_final_poles = pd.concat([gdf_associated_poles, gdf_new_poles], ignore_index=True).reset_index(
        drop=True
    )

    if gdf_final_poles.empty:
        raise ValueError("No poles could be placed; check input data and parameters.")

    # ------------------------------------------------------------------
    # 4b) Append new associations with robust (pole, building) -> ids mapping
    # ------------------------------------------------------------------
    if len(new_associations) > 0:
        # new_associations: list[(pole_point, building_geom)]
        new_df = pd.DataFrame(new_associations, columns=["pole", "building_geom"])

        # pole geom -> local index within gdf_new_poles
        new_pole_wkb_to_local = {geom.wkb: i for i, geom in enumerate(gdf_new_poles.geometry)}
        offset = len(gdf_associated_poles)  # final poles = [associated | new]

        def _safe_lookup_new_pole_local_id(p: Point) -> int:
            k = p.wkb
            if k in new_pole_wkb_to_local:
                return int(new_pole_wkb_to_local[k])
            # fallback: nearest match (defensive)
            if len(gdf_new_poles) == 0:
                raise ValueError("Internal error: new_associations provided but gdf_new_poles is empty.")
            dists = gdf_new_poles.geometry.distance(p)
            return int(dists.idxmin())

        def _safe_lookup_building_id(g: Point) -> int:
            k = g.wkb
            if k in bldg_wkb_to_id:
                return int(bldg_wkb_to_id[k])
            # fallback: nearest building (defensive)
            dists = gdf_buildings.geometry.distance(g)
            return int(dists.idxmin())

        new_df["pole_id"] = new_df["pole"].apply(lambda p: offset + _safe_lookup_new_pole_local_id(p))
        new_df["building_id"] = new_df["building_geom"].apply(_safe_lookup_building_id)

        new_assoc_ids = new_df[["pole_id", "building_id"]].dropna().drop_duplicates()
        new_assoc_ids["pole_id"] = new_assoc_ids["pole_id"].astype(int)
        new_assoc_ids["building_id"] = new_assoc_ids["building_id"].astype(int)

        associations_df = pd.concat(
            [associations_df[["pole_id", "building_id"]], new_assoc_ids],
            ignore_index=True,
        )

    # Final cleanup + validation of associations
    if not associations_df.empty:
        associations_df = associations_df[["pole_id", "building_id"]].dropna().drop_duplicates()
        associations_df["pole_id"] = associations_df["pole_id"].astype(int)
        associations_df["building_id"] = associations_df["building_id"].astype(int)

        max_pid = int(associations_df["pole_id"].max())
        if max_pid >= len(gdf_final_poles):
            raise ValueError(
                f"Pole ID mismatch: associations refer to pole_id={max_pid} "
                f"but only {len(gdf_final_poles)} poles exist."
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
    gdf_served = gdf_buildings.loc[served_ids] if served_ids else gdf_buildings.iloc[0:0]

    num_buildings = int(len(gdf_buildings))
    num_served = int(len(gdf_served))
    num_unserved = int(len(gdf_unserved))

    # ------------------------------------------------------------------
    # 5b) Service drops length (building -> assigned pole)
    # ------------------------------------------------------------------
    service_drop_length_m = 0.0
    if not associations_df.empty:
        pole_geom_by_id = gdf_final_poles.geometry.to_dict()  # pole_id -> Point

        assoc = associations_df[["pole_id", "building_id"]].dropna().drop_duplicates()
        for pole_id, building_id in assoc.itertuples(index=False):
            pole_geom = pole_geom_by_id.get(int(pole_id))
            bldg_geom = bldg_geom_by_id.get(int(building_id))
            if pole_geom is None or bldg_geom is None:
                continue
            service_drop_length_m += float(pole_geom.distance(bldg_geom))

    service_drop_length_km = service_drop_length_m / 1000.0
    serving_poles = int(associations_df["pole_id"].nunique()) if not associations_df.empty else 0

    # ------------------------------------------------------------------
    # 6) MST in projected CRS + densification of long spans
    # ------------------------------------------------------------------
    mst_base: nx.Graph = create_graph_and_mst(gdf_final_poles)

    gdf_poles_densified, mst = densify_mst_edges(
        gdf_poles=gdf_final_poles,
        mst=mst_base,
        max_pole_span_m=max_pole_span_m or 0.0,
    )

    # ------------------------------------------------------------------
    # 7) Backbone length + totals
    # ------------------------------------------------------------------
    backbone_length_m = float(sum(nx.get_edge_attributes(mst, "weight").values()))
    backbone_length_km = backbone_length_m / 1000.0
    total_network_length_km = backbone_length_km + service_drop_length_km

    total_poles = int(len(gdf_poles_densified))
    base_poles = int(len(gdf_final_poles))
    support_poles = int(max(0, total_poles - base_poles))

    # ------------------------------------------------------------------
    # 8) Reproject outputs to EPSG:4326 for plotting
    # ------------------------------------------------------------------
    gdf_buildings_4326 = gdf_buildings.to_crs(epsg=4326)
    gdf_poles_4326 = gdf_poles_densified.to_crs(epsg=4326)
    gdf_roads_4326 = gdf_roads.to_crs(epsg=4326) if gdf_roads is not None else None

    gdf_served_4326 = gdf_served.to_crs(epsg=4326) if num_served > 0 else gdf_buildings_4326.iloc[0:0]
    gdf_unserved_4326 = gdf_unserved.to_crs(epsg=4326) if num_unserved > 0 else gdf_buildings_4326.iloc[0:0]

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
            # Lengths
            "total_network_length_km": total_network_length_km,
            "backbone_length_km": backbone_length_km,
            "service_drop_length_km": service_drop_length_km,
            # Poles breakdown
            "num_poles_total": total_poles,
            "num_poles_serving": serving_poles,
            "num_poles_support": support_poles,
            # Buildings breakdown
            "num_buildings": num_buildings,
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
