from __future__ import annotations

from typing import Dict, Any, Tuple

import pandas as pd
import geopandas as gpd

from core.powerflow_io import find_column


# =============================================================================
# External topology validation
# =============================================================================
def validate_external_topology(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_edges: gpd.GeoDataFrame,
    assoc: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Validate and normalize an externally provided topology (nodes + edges + associations).

    Returns a dict that the PF page can consume in a consistent way.

    Guarantees in return:
      - mode="external"
      - gdf_nodes: unchanged GeoDataFrame (caller can reproject)
      - gdf_edges: unchanged GeoDataFrame (caller can reproject)
      - associations: DataFrame with columns ["pole_id"(int), "building_id"(any->str later)]
      - pole_col: node ID column name in nodes
      - u_col, v_col: endpoint column names in edges (integers)
      - topology_debug: lightweight diagnostics
    """
    if gdf_nodes is None or gdf_nodes.empty:
        raise ValueError("Nodes GeoDataFrame is empty.")
    if gdf_edges is None or gdf_edges.empty:
        raise ValueError("Edges GeoDataFrame is empty.")
    if assoc is None or assoc.empty:
        raise ValueError("Associations table is empty.")

    pole_col = find_column(gdf_nodes, "pole_id", "id")
    if pole_col is None:
        raise ValueError("Nodes must contain 'pole_id' or 'id'.")

    # edges endpoints columns (common patterns)
    u_col = find_column(gdf_edges, "source", "bus0", "from", "u")
    v_col = find_column(gdf_edges, "target", "bus1", "to", "v")
    if u_col is None or v_col is None:
        raise ValueError("Edges must contain valid endpoint columns (e.g., source/target or u/v).")

    pole_a = find_column(assoc, "pole_id", "pole")
    bld_a = find_column(assoc, "building_id", "building")
    if pole_a is None or bld_a is None:
        raise ValueError("Associations must contain pole_id and building_id.")

    # ---- normalize associations ----
    assoc_out = assoc[[pole_a, bld_a]].rename(columns={pole_a: "pole_id", bld_a: "building_id"}).copy()
    assoc_out["pole_id"] = pd.to_numeric(assoc_out["pole_id"], errors="coerce")
    if assoc_out["pole_id"].isna().any():
        bad = assoc_out.loc[assoc_out["pole_id"].isna(), "pole_id"]
        raise ValueError(f"Non-numeric pole_id found in associations (first examples): {bad.head(10).tolist()}")
    assoc_out["pole_id"] = assoc_out["pole_id"].astype(int)

    # ---- normalize node IDs to int set (for edge sanity) ----
    node_ids = pd.to_numeric(gdf_nodes[pole_col], errors="coerce")
    node_ids = node_ids.dropna().astype(int)
    node_id_set = set(node_ids.tolist())
    if not node_id_set:
        raise ValueError("No valid numeric pole IDs found in nodes file.")

    # ---- normalize edge endpoints to int, drop nonsense ----
    e = gdf_edges.copy()
    eu = pd.to_numeric(e[u_col], errors="coerce")
    ev = pd.to_numeric(e[v_col], errors="coerce")
    e = e.loc[eu.notna() & ev.notna()].copy()
    e[u_col] = eu.loc[eu.notna() & ev.notna()].astype(int)
    e[v_col] = ev.loc[eu.notna() & ev.notna()].astype(int)

    # drop self loops
    e = e.loc[e[u_col] != e[v_col]].copy()

    # keep only edges whose endpoints exist in nodes
    e = e.loc[e[u_col].isin(node_id_set) & e[v_col].isin(node_id_set)].copy()

    # optional: dedupe undirected duplicates (u-v and v-u)
    a = e[[u_col, v_col]].min(axis=1)
    b = e[[u_col, v_col]].max(axis=1)
    e["_a"] = a
    e["_b"] = b
    e = e.drop_duplicates(subset=["_a", "_b"]).drop(columns=["_a", "_b"]).copy()

    # (we return original gdfs, but we also return cleaned counts)
    topology_debug = {
        "nodes_n": int(len(gdf_nodes)),
        "edges_n_in": int(len(gdf_edges)),
        "edges_n_clean": int(len(e)),
        "assoc_n": int(len(assoc_out)),
        "pole_col": pole_col,
        "u_col": u_col,
        "v_col": v_col,
    }

    # IMPORTANT:
    # we return the CLEANED edges dataframe to prevent downstream surprises
    return {
        "mode": "external",
        "gdf_nodes": gdf_nodes,
        "gdf_edges": e,
        "associations": assoc_out,
        "pole_col": pole_col,
        "u_col": u_col,
        "v_col": v_col,
        "topology_debug": topology_debug,
    }


# =============================================================================
# Session results extraction (Page 1 -> PF page)
# =============================================================================
def extract_from_session_results(dist_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a PF-ready bundle from Page 1 session results.

    This function is intentionally strict about what exists in session mode:
      - nodes are expected as dist_results["gdf_poles_4326"]
      - associations are expected either as dist_results["associations_df"] or downloads["associations_csv"]
      - edges for PF should be stable pole-id pairs:
            dist_results["mst_edges_pole_ids"]  (RECOMMENDED)
        Visualization can still use:
            dist_results["mst_edges_latlon"]

    If mst_edges_pole_ids is missing, we keep it as None here and let the PF page
    raise a clear error that Page 1 needs re-run after update.
    """
    dist_results = dist_results or {}
    downloads = dist_results.get("downloads", {}) or {}

    return {
        "mode": "session",
        "dist_results": dist_results,

        # downloads (optional)
        "nodes_geojson": downloads.get("nodes_geojson"),
        "edges_geojson": downloads.get("edges_geojson"),
        "associations_csv": downloads.get("associations_csv"),

        # in-memory (preferred)
        "associations_df": dist_results.get("associations_df"),

        # nodes for PF + mapping
        "gdf_nodes_4326": dist_results.get("gdf_poles_4326"),

        # edges:
        # - PF should use stable pole-id pairs (NOT latlon mapping)
        "mst_edges_pole_ids": dist_results.get("mst_edges_pole_ids"),

        # - keep latlon for map overlay only
        "mst_edges_latlon": dist_results.get("mst_edges_latlon"),

        # optional extras (if you want to pass through)
        "center": dist_results.get("center"),
        "topology_debug": {
            "has_nodes": dist_results.get("gdf_poles_4326") is not None,
            "has_mst_edges_pole_ids": dist_results.get("mst_edges_pole_ids") is not None,
            "has_mst_edges_latlon": dist_results.get("mst_edges_latlon") is not None,
            "has_assoc_df": isinstance(dist_results.get("associations_df"), pd.DataFrame),
            "has_assoc_csv": downloads.get("associations_csv") is not None,
        },
    }


# =============================================================================
# Demand aggregation
# =============================================================================
def aggregate_pole_loads(
    associations: pd.DataFrame,
    building_meta: pd.DataFrame,
    category_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mode 2 aggregation:
      - associations: columns [building_id (any), pole_id (int)]
      - building_meta: columns [building_id (str), category (str), weight (float)]
      - category_profiles: index=hour (int), columns=categories, values=kW per building

    Returns:
      pole_loads_kW: index=hour, columns=pole_id, values=kW aggregated at each pole.
    """
    assoc = associations.copy()
    assoc["building_id"] = assoc["building_id"].astype(str)

    meta = building_meta.copy()
    meta["building_id"] = meta["building_id"].astype(str)
    meta["category"] = meta["category"].astype(str)

    # Join building -> pole_id
    bm = assoc.merge(meta, on="building_id", how="inner")

    if bm.empty:
        raise ValueError(
            "No buildings matched between associations.csv and building_metadata.csv. "
            "Check that building_id values are consistent across files."
        )

    # Validate categories exist in profiles
    cats_in_meta = sorted(bm["category"].unique().tolist())
    cats_in_profiles = set(map(str, category_profiles.columns.tolist()))
    missing = [c for c in cats_in_meta if c not in cats_in_profiles]
    if missing:
        raise ValueError(
            f"Some building categories are missing in category_profiles.csv columns. Missing: {missing[:20]}"
        )

    # Build weights matrix W[pole_id, category] = sum of weights of buildings at that pole in that category
    W = (
        bm.pivot_table(index="pole_id", columns="category", values="weight", aggfunc="sum", fill_value=0.0)
        .sort_index()
    )

    # Align profiles columns to W columns
    P = category_profiles.reindex(columns=W.columns).fillna(0.0)  # hours x categories

    # Matrix multiply: (hours x categories) dot (categories x poles) = hours x poles
    pole_loads = P.to_numpy() @ W.to_numpy().T
    pole_loads_kW = pd.DataFrame(pole_loads, index=P.index, columns=W.index)

    pole_loads_kW.index.name = "hour"
    return pole_loads_kW
