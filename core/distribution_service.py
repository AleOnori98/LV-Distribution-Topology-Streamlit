from __future__ import annotations

from typing import Any, Dict, Tuple, List

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
    deduplicate_poles_with_remap,
    merge_graph_nodes_with_remap,
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
        raise ValueError("Buildings are still in a geographic CRS; expected projected CRS in meters.")

    bldg_geom_by_id = gdf_buildings.geometry.to_dict()
    bldg_wkb_to_id = {geom.wkb: idx for idx, geom in bldg_geom_by_id.items()}

    # ------------------------------------------------------------------
    # 2) Candidate poles from roads
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
    # 3) Associate buildings to road poles
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

            kept_old_ids = sorted(associations_df["pole_id"].astype(int).unique().tolist())
            gdf_associated_poles = gdf_associated_poles.iloc[kept_old_ids].reset_index(drop=True)

            remap = {old: new for new, old in enumerate(kept_old_ids)}
            associations_df["pole_id"] = associations_df["pole_id"].map(remap).astype(int)
            associations_df["building_id"] = associations_df["building_id"].astype(int)
        else:
            gdf_associated_poles = gdf_associated_poles.iloc[0:0].copy()
            associations_df = pd.DataFrame(columns=["pole_id", "building_id"])
    else:
        associations_df = pd.DataFrame(columns=["pole_id", "building_id"])

    # ------------------------------------------------------------------
    # 4) New poles for remaining buildings
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
    gdf_new_poles = gdf_new_poles.reset_index(drop=True)

    gdf_final_poles = pd.concat([gdf_associated_poles, gdf_new_poles], ignore_index=True).reset_index(drop=True)
    gdf_final_poles = gpd.GeoDataFrame(gdf_final_poles, crs=gdf_buildings.crs)

    if gdf_final_poles.empty:
        raise ValueError("No poles could be placed; check input data and parameters.")

    # stable ids + origin (create, then deduplicate in projected CRS)
    gdf_final_poles["pole_id"] = gdf_final_poles.index.astype(int)
    gdf_final_poles["pole_origin"] = "base"
    gdf_final_poles["pole_type"] = "base"  # temporary; serving/non-serving set after associations

    # ------------------------------------------------------------------
    # 4a) Deduplicate near-coincident poles (prevents degenerate MST edges)
    # ------------------------------------------------------------------
    # Recommended tolerance: 0.5–1.0 m depending on input cleanliness
    DEDUP_TOL_M = 0.75

    gdf_final_poles, associations_df, pole_id_remap = deduplicate_poles_with_remap(
        gdf_final_poles,
        associations_df,
        tol_m=DEDUP_TOL_M,
        prefer_serving=True,
    )

    # After dropping poles, ensure pole_id are still unique ints
    gdf_final_poles["pole_id"] = pd.to_numeric(gdf_final_poles["pole_id"], errors="coerce")
    gdf_final_poles = gdf_final_poles.dropna(subset=["pole_id"]).copy()
    gdf_final_poles["pole_id"] = gdf_final_poles["pole_id"].astype(int)

    if gdf_final_poles["pole_id"].duplicated().any():
        raise ValueError("Deduplication produced duplicate pole_id values; this should not happen.")


    # ------------------------------------------------------------------
    # 4b) Append new associations
    # ------------------------------------------------------------------
    if len(new_associations) > 0:
        new_df = pd.DataFrame(new_associations, columns=["pole", "building_geom"])

        new_pole_wkb_to_local = {geom.wkb: i for i, geom in enumerate(gdf_new_poles.geometry)}
        offset = len(gdf_associated_poles)

        def _safe_lookup_new_pole_local_id(p: Point) -> int:
            k = p.wkb
            if k in new_pole_wkb_to_local:
                return int(new_pole_wkb_to_local[k])
            if len(gdf_new_poles) == 0:
                raise ValueError("Internal error: new_associations provided but gdf_new_poles is empty.")
            dists = gdf_new_poles.geometry.distance(p)
            return int(dists.idxmin())

        def _safe_lookup_building_id(g: Point) -> int:
            k = g.wkb
            if k in bldg_wkb_to_id:
                return int(bldg_wkb_to_id[k])
            dists = gdf_buildings.geometry.distance(g)
            return int(dists.idxmin())

        new_df["pole_id"] = new_df["pole"].apply(lambda p: offset + _safe_lookup_new_pole_local_id(p))
        new_df["building_id"] = new_df["building_geom"].apply(_safe_lookup_building_id)

        new_assoc_ids = new_df[["pole_id", "building_id"]].dropna().drop_duplicates()
        new_assoc_ids["pole_id"] = new_assoc_ids["pole_id"].astype(int)
        new_assoc_ids["building_id"] = new_assoc_ids["building_id"].astype(int)

        associations_df = pd.concat([associations_df[["pole_id", "building_id"]], new_assoc_ids], ignore_index=True)

    # cleanup associations
    if not associations_df.empty:
        associations_df = associations_df[["pole_id", "building_id"]].dropna().drop_duplicates()
        associations_df["pole_id"] = associations_df["pole_id"].astype(int)
        associations_df["building_id"] = associations_df["building_id"].astype(int)

        valid_base_pole_ids = set(pd.to_numeric(gdf_final_poles["pole_id"], errors="coerce").dropna().astype(int).tolist())
        missing_base_poles = sorted(
            pid for pid in associations_df["pole_id"].astype(int).tolist() if int(pid) not in valid_base_pole_ids
        )
        if missing_base_poles:
            raise ValueError(
                "Pole ID mismatch in base associations vs base pole set. "
                f"Example missing pole_ids (up to 20): {missing_base_poles[:20]}"
            )

    # base serving ids and types
    base_serving_ids = set(associations_df["pole_id"].unique()) if not associations_df.empty else set()
    gdf_final_poles["pole_type"] = gdf_final_poles["pole_id"].apply(
        lambda pid: "serving" if int(pid) in base_serving_ids else "non_serving"
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

    # Ensure associations only contain served buildings (defensive)
    if not associations_df.empty:
        associations_df = associations_df[associations_df["building_id"].isin(served_ids)].copy()

    # ------------------------------------------------------------------
    # 6) MST + densification
    # ------------------------------------------------------------------
    mst_base: nx.Graph = create_graph_and_mst(gdf_final_poles)

    gdf_poles_densified, mst = densify_mst_edges(
        gdf_poles=gdf_final_poles,
        mst=mst_base,
        max_pole_span_m=float(max_pole_span_m or 0.0),
    )

    # ------------------------------------------------------------------
    # 6b) PROMOTION STEP: allow inserted poles to become serving poles
    # ------------------------------------------------------------------
    poles_all = gdf_poles_densified.copy().reset_index(drop=True)
    if "pole_id" not in poles_all.columns:
        poles_all["pole_id"] = poles_all.index.astype(int)
    if "pole_type" not in poles_all.columns:
        poles_all["pole_type"] = "base"
    if "pole_origin" not in poles_all.columns:
        poles_all["pole_origin"] = "base"

    # Build current assignment + capacities
    current_assign: Dict[int, int] = {}
    if not associations_df.empty:
        for pid, bid in associations_df[["pole_id", "building_id"]].itertuples(index=False):
            current_assign[int(bid)] = int(pid)

    # counts per pole
    pole_counts: Dict[int, int] = {}
    for pid in poles_all["pole_id"].astype(int).tolist():
        pole_counts[pid] = 0
    for bid, pid in current_assign.items():
        pole_counts[pid] = pole_counts.get(pid, 0) + 1

    pole_geom_by_id = poles_all.set_index("pole_id").geometry.to_dict()

    # Only consider served buildings for promotion/reassignment
    served_buildings = gdf_buildings.loc[served_ids]

    # Precompute building geometries for speed
    bldg_geom = served_buildings.geometry.to_dict()

    # Candidate inserted poles
    inserted = poles_all[poles_all["pole_origin"].astype(str) == "inserted"].copy()

    # Greedy promotion:
    # For each inserted pole, attach nearby buildings if:
    # - within user_distance
    # - pole has spare capacity
    # - inserted pole is closer than the building's currently assigned pole
    promoted_pole_ids: set[int] = set()

    for row in inserted.itertuples(index=False):
        pid = int(getattr(row, "pole_id"))
        pgeom = pole_geom_by_id.get(pid)
        if pgeom is None:
            continue

        # Find served buildings within radius
        # (no spatial index used here for simplicity; OK for modest N)
        candidates: List[Tuple[float, int]] = []
        for bid, g in bldg_geom.items():
            d = float(pgeom.distance(g))
            if d <= float(user_distance):
                candidates.append((d, int(bid)))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])

        for d_new, bid in candidates:
            if pole_counts.get(pid, 0) >= int(max_associations):
                break

            # current pole and distance
            cur_pid = current_assign.get(bid, None)
            if cur_pid is None:
                # if somehow unassigned but served, we can attach directly
                current_assign[bid] = pid
                pole_counts[pid] = pole_counts.get(pid, 0) + 1
                promoted_pole_ids.add(pid)
                continue

            cur_geom = pole_geom_by_id.get(int(cur_pid))
            if cur_geom is None:
                continue
            d_cur = float(cur_geom.distance(bldg_geom[bid]))

            # Only reassign if strictly closer (you can add a tolerance if you like)
            if d_new < d_cur:
                # free capacity on old pole
                pole_counts[cur_pid] = max(0, pole_counts.get(cur_pid, 0) - 1)
                # assign to promoted pole
                current_assign[bid] = pid
                pole_counts[pid] = pole_counts.get(pid, 0) + 1
                promoted_pole_ids.add(pid)

    # Rebuild associations_df from current_assign (served buildings only)
    if len(current_assign) > 0:
        associations_df = pd.DataFrame(
            [(pid, bid) for bid, pid in current_assign.items()],
            columns=["pole_id", "building_id"],
        ).drop_duplicates()
        associations_df["pole_id"] = associations_df["pole_id"].astype(int)
        associations_df["building_id"] = associations_df["building_id"].astype(int)
    else:
        associations_df = pd.DataFrame(columns=["pole_id", "building_id"])

    # Update pole_type after promotion
    final_serving_ids = set(associations_df["pole_id"].unique()) if not associations_df.empty else set()

    def _final_type(row) -> str:
        pid = int(row["pole_id"])
        origin = str(row.get("pole_origin", "base"))
        if pid in final_serving_ids:
            return "serving"
        if origin == "inserted":
            return "support"
        return "non_serving"

    poles_all["pole_type"] = poles_all.apply(_final_type, axis=1)

    gdf_poles_densified = gpd.GeoDataFrame(poles_all, crs=gdf_buildings.crs)

    # ------------------------------------------------------------------
    # 6c) Post-densification cleanup for near-coincident final poles.
    # Densification can insert support poles very close to existing poles,
    # especially where long spans terminate near intersections. If left as-is,
    # this creates ultra-short electrical segments that the PF step correctly
    # filters out as numerical near-shorts, which can orphan a load-bearing pole.
    #
    # We merge only very-close final poles here, then collapse the MST node IDs
    # with the same remap so topology and associations stay consistent.
    # ------------------------------------------------------------------
    FINAL_MERGE_TOL_M = 10.0
    gdf_poles_densified, associations_df, final_pole_id_remap = deduplicate_poles_with_remap(
        gdf_poles_densified,
        associations_df,
        tol_m=FINAL_MERGE_TOL_M,
        prefer_serving=True,
    )
    mst = merge_graph_nodes_with_remap(mst, final_pole_id_remap)

    final_serving_ids = set(associations_df["pole_id"].unique()) if not associations_df.empty else set()
    gdf_poles_densified["pole_type"] = gdf_poles_densified.apply(_final_type, axis=1)

    # Stable MST edges as pole_id pairs (for PF / validation).
    #
    # IMPORTANT:
    # `mst` already uses stable pole_id values as graph node IDs.
    # Do not remap through GeoDataFrame row indices here: after deduplication the
    # row index is no longer guaranteed to match pole_id, and that can corrupt the
    # exported PF topology (load-bearing poles appear disconnected even though the
    # internal MST is connected).
    mst_edges_pole_ids = [(int(u), int(v)) for u, v in mst.edges()]

    # ------------------------------------------------------------------
    # 6d) Consistency check: every served building must reference a graph pole,
    # and every serving pole must belong to the electrical tree.
    # ------------------------------------------------------------------
    if not associations_df.empty:
        graph_pole_ids = {int(n) for n in mst.nodes()}
        assoc_pole_ids = set(pd.to_numeric(associations_df["pole_id"], errors="coerce").dropna().astype(int).tolist())

        missing_poles = sorted(pid for pid in assoc_pole_ids if pid not in graph_pole_ids)
        if missing_poles:
            raise ValueError(
                "Topology consistency error: some buildings reference poles that are not present in the MST graph. "
                f"Example pole_ids (up to 20): {missing_poles[:20]}"
            )

        orphan_serving_poles = sorted(pid for pid in assoc_pole_ids if int(mst.degree(pid)) < 1)
        if orphan_serving_poles:
            raise ValueError(
                "Topology consistency error: some poles have assigned buildings but no electrical connection in the MST. "
                f"Example pole_ids (up to 20): {orphan_serving_poles[:20]}"
            )

        pole_geom_by_id_post = gdf_poles_densified.set_index("pole_id").geometry.to_dict()
        missing_geom_poles = sorted(pid for pid in assoc_pole_ids if int(pid) not in pole_geom_by_id_post)
        if missing_geom_poles:
            raise ValueError(
                "Topology consistency error: some buildings reference poles missing from the final pole table. "
                f"Example pole_ids (up to 20): {missing_geom_poles[:20]}"
            )

    short_final_edges = sorted(
        float(data.get("weight", 0.0))
        for _, _, data in mst.edges(data=True)
        if float(data.get("weight", 0.0)) < FINAL_MERGE_TOL_M
    )
    if short_final_edges:
        raise ValueError(
            "Topology consistency error: final MST still contains ultra-short pole-to-pole segments after cleanup. "
            f"Shortest remaining segment = {short_final_edges[0]:.2f} m; expected >= {FINAL_MERGE_TOL_M:.2f} m."
        )

    # ------------------------------------------------------------------
    # 7) Service drops length (recomputed using densified poles)
    # ------------------------------------------------------------------
    service_drop_length_m = 0.0
    if not associations_df.empty:
        pole_geom_by_id = gdf_poles_densified.set_index("pole_id").geometry.to_dict()
        for pole_id, building_id in associations_df[["pole_id", "building_id"]].itertuples(index=False):
            pole_geom = pole_geom_by_id.get(int(pole_id))
            bldg = bldg_geom_by_id.get(int(building_id))
            if pole_geom is None or bldg is None:
                continue
            service_drop_length_m += float(pole_geom.distance(bldg))

    service_drop_length_km = service_drop_length_m / 1000.0

    # ------------------------------------------------------------------
    # 8) Backbone length + totals
    # ------------------------------------------------------------------
    backbone_length_m = float(sum(nx.get_edge_attributes(mst, "weight").values()))
    backbone_length_km = backbone_length_m / 1000.0
    total_network_length_km = backbone_length_km + service_drop_length_km

    total_poles = int(len(gdf_poles_densified))
    serving_poles = int(associations_df["pole_id"].nunique()) if not associations_df.empty else 0
    support_poles = int((gdf_poles_densified["pole_type"] == "support").sum())

    # ------------------------------------------------------------------
    # 9) Reproject outputs to EPSG:4326 for plotting
    # ------------------------------------------------------------------
    gdf_buildings_4326 = gdf_buildings.to_crs(epsg=4326)
    gdf_poles_4326 = gdf_poles_densified.to_crs(epsg=4326)
    gdf_roads_4326 = gdf_roads.to_crs(epsg=4326) if gdf_roads is not None else None
    gdf_served_4326 = gdf_served.to_crs(epsg=4326) if num_served > 0 else gdf_buildings_4326.iloc[0:0]
    gdf_unserved_4326 = gdf_unserved.to_crs(epsg=4326) if num_unserved > 0 else gdf_buildings_4326.iloc[0:0]

    # ------------------------------------------------------------------
    # 10) Map center
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
    # 11) Edge list for plotting + downloads
    # ------------------------------------------------------------------
    mst_edges_latlon = mst_edges_as_latlon(gdf_poles_4326, mst)
    nodes_geojson, edges_geojson = save_mst_to_geojson(gdf_poles_4326, mst)
    associations_csv = associations_df.sort_values(["pole_id", "building_id"]).to_csv(index=False).encode("utf-8")

    return {
        "metrics": {
            "total_network_length_km": total_network_length_km,
            "backbone_length_km": backbone_length_km,
            "service_drop_length_km": service_drop_length_km,
            "num_poles_total": total_poles,
            "num_poles_serving": serving_poles,
            "num_poles_support": support_poles,
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
        "mst_edges_pole_ids": mst_edges_pole_ids,
        "associations_df": associations_df,
        "downloads": {
            "nodes_geojson": nodes_geojson,
            "edges_geojson": edges_geojson,
            "associations_csv": associations_csv,
        },
        "center": center,
    }
