# core/powerflow_pypsa.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import pypsa

RUNNER_VERSION = "2026-02-28c"


# =============================================================================
# Data containers
# =============================================================================
@dataclass(frozen=True)
class PFScenarioParams:
    # PF controls
    slack_pole_id: int
    v_min_pu: float
    v_max_pu: float
    pf_load: float  # lagging, in [0..1]

    # Nominal voltage (kV) used by PyPSA to express v_mag_pu
    v_nom_kv: float = 0.4  # LV (line-to-line), e.g. 0.4 kV

    # Uniform line model (typical LV cable)
    r_ohm_per_km: float = 0.642
    x_ohm_per_km: float = 0.083
    s_nom_kva: float = 100.0  # thermal rating per segment

    # For the single-phase equivalent model, total 3-phase aggregated loads
    # should be converted to per-phase equivalent before injection.
    load_scale: float = 1.0


@dataclass(frozen=True)
class PFTopologyBundle:
    """
    Topology bundle always in EPSG:4326 (lat/lon) for consistency with UI.

    Exactly one edge source must be provided:
      - mst_edges_pole_ids  (preferred; stable)
      - gdf_edges_4326 + (edge_u_col, edge_v_col)
      - mst_edges_latlon    (legacy; avoid if possible)
    """
    gdf_nodes_4326: gpd.GeoDataFrame
    pole_id_col: str

    mst_edges_pole_ids: Optional[List[Tuple[int, int]]] = None  # <-- NEW preferred source
    mst_edges_latlon: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None  # legacy
    gdf_edges_4326: Optional[gpd.GeoDataFrame] = None
    edge_u_col: Optional[str] = None
    edge_v_col: Optional[str] = None


# =============================================================================
# Helpers
# =============================================================================
def _safe_latlon(geom) -> Tuple[float, float]:
    if geom is None or geom.is_empty:
        return (np.nan, np.nan)
    pt = geom if geom.geom_type == "Point" else geom.representative_point()
    return (float(pt.y), float(pt.x))


def _infer_q_from_pf(p_kw: float, pf: float) -> float:
    """
    Infer Q from P and power factor (lagging loads).
      P in kW, returns Q in kVAr.
    Convention: positive Q = reactive consumption (lagging).
    """
    p = float(p_kw)
    if abs(p) < 1e-12:
        return 0.0
    pf = float(np.clip(pf, 1e-6, 1.0))
    s = abs(p) / pf  # kVA
    q = np.sqrt(max(s * s - p * p, 0.0))  # kVAr (since P in kW, S in kVA)
    return float(q)


def _approx_km_from_latlon(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Approx distance in km from two lat/lon points.
    Adequate for village-scale networks; for robust engineering use a metric CRS.
    """
    dlat_km = (lat2 - lat1) * 111.0
    dlon_km = (lon2 - lon1) * 111.0 * np.cos(np.deg2rad((lat1 + lat2) / 2.0))
    return float(np.sqrt(dlat_km * dlat_km + dlon_km * dlon_km))


def _normalize_nodes(gdf_nodes_4326: gpd.GeoDataFrame, pole_id_col: str) -> pd.DataFrame:
    """
    Return a clean nodes table with columns: pole_id (int), lat (float), lon (float).
    Drops invalid IDs and invalid geometries.
    """
    if pole_id_col not in gdf_nodes_4326.columns:
        raise ValueError(f"pole_id_col='{pole_id_col}' not found in nodes GeoDataFrame.")

    df = gdf_nodes_4326.copy()

    pid = pd.to_numeric(df[pole_id_col], errors="coerce")
    df = df.loc[pid.notna()].copy()
    df["pole_id"] = pid.loc[pid.notna()].astype(int)

    latlon = np.array([_safe_latlon(g) for g in df.geometry])
    df["lat"] = latlon[:, 0]
    df["lon"] = latlon[:, 1]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon"]).copy()

    out = df[["pole_id", "lat", "lon"]].drop_duplicates(subset=["pole_id"]).sort_values("pole_id")
    if out.empty:
        raise ValueError("No valid poles found after cleaning (pole_id + geometry).")
    return out.reset_index(drop=True)


def _build_edges_from_gdf(gdf_edges_4326: gpd.GeoDataFrame, u_col: str, v_col: str) -> pd.DataFrame:
    """
    Return edges DataFrame with columns: line_id (str), u (int), v (int), length_km (float).
    Length derived from geometry endpoints (approx in km).
    """
    if u_col not in gdf_edges_4326.columns or v_col not in gdf_edges_4326.columns:
        raise ValueError(f"Edges must contain endpoint columns '{u_col}' and '{v_col}'.")

    df = gdf_edges_4326.copy()

    u = pd.to_numeric(df[u_col], errors="coerce")
    v = pd.to_numeric(df[v_col], errors="coerce")
    df = df.loc[u.notna() & v.notna()].copy()
    df["u"] = u.loc[u.notna() & v.notna()].astype(int)
    df["v"] = v.loc[u.notna() & v.notna()].astype(int)

    # derive length_km from geometry endpoints
    lengths_km: List[float] = []
    for geom in df.geometry:
        if geom is None or geom.is_empty:
            lengths_km.append(np.nan)
            continue
        try:
            coords = list(geom.coords)
            (lon1, lat1) = coords[0]
            (lon2, lat2) = coords[-1]
            lengths_km.append(_approx_km_from_latlon(lat1, lon1, lat2, lon2))
        except Exception:
            lengths_km.append(np.nan)

    line_id_col = "line_id" if "line_id" in df.columns else None
    if line_id_col is None:
        line_ids = [f"L{i}" for i in range(len(df))]
    else:
        line_ids = df[line_id_col].astype(str).tolist()

    out = pd.DataFrame(
        {"line_id": line_ids, "u": df["u"].values, "v": df["v"].values, "length_km": lengths_km}
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["u", "v", "length_km"]).copy()
    out["line_id"] = out["line_id"].astype(str)
    if out["line_id"].duplicated().any():
        dups = out.loc[out["line_id"].duplicated(), "line_id"].head(10).tolist()
        raise ValueError(f"Edges contain duplicated line_id values. Examples: {dups}")
    out["u"] = out["u"].astype(int)
    out["v"] = out["v"].astype(int)
    out["length_km"] = out["length_km"].astype(float)

    # drop self-loops
    out = out.loc[out["u"] != out["v"]].copy()

    return out.reset_index(drop=True)


def _build_edges_from_pole_id_pairs(
    mst_edges_pole_ids: List[Tuple[int, int]],
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build edges from stable pole_id pairs.

    - Verifies endpoints exist in nodes
    - Computes length_km from node coordinates (approx)
    - Drops self-loops and duplicates (undirected)
    """
    if nodes.empty:
        raise ValueError("Nodes are empty; cannot build edges.")

    coord = nodes.set_index("pole_id")[["lat", "lon"]].to_dict(orient="index")
    node_ids = set(coord.keys())

    rows: List[Tuple[str, int, int, float]] = []
    for idx, (u, v) in enumerate(mst_edges_pole_ids):
        u = int(u); v = int(v)
        if u == v:
            continue
        if u not in node_ids or v not in node_ids:
            # skip invalid edges but you may prefer raising
            continue

        lat1, lon1 = coord[u]["lat"], coord[u]["lon"]
        lat2, lon2 = coord[v]["lat"], coord[v]["lon"]
        L = _approx_km_from_latlon(float(lat1), float(lon1), float(lat2), float(lon2))
        rows.append((f"L{idx}", u, v, L))

    out = pd.DataFrame(rows, columns=["line_id", "u", "v", "length_km"])
    if out.empty:
        raise ValueError("No valid MST edges after checking endpoints against nodes.")

    # undirected dedup
    a = np.minimum(out["u"].to_numpy(), out["v"].to_numpy())
    b = np.maximum(out["u"].to_numpy(), out["v"].to_numpy())
    out["_a"] = a
    out["_b"] = b
    out = out.drop_duplicates(subset=["_a", "_b"]).drop(columns=["_a", "_b"]).reset_index(drop=True)
    return out

def _build_edges_from_pole_ids(
    mst_edges_pole_ids: List[Tuple[int, int]],
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build edges from stable (u_pole_id, v_pole_id).
    Length is computed from node coordinates.
    """
    if nodes.empty:
        raise ValueError("Nodes are empty; cannot build edges.")

    # map pole_id -> (lat, lon)
    coord = nodes.set_index("pole_id")[["lat", "lon"]].to_dict(orient="index")

    rows: List[Tuple[str, int, int, float]] = []
    for idx, (u, v) in enumerate(mst_edges_pole_ids):
        u = int(u); v = int(v)
        if u == v:
            continue
        if u not in coord or v not in coord:
            # ignore edges referencing missing poles (or raise if you prefer)
            continue
        lat1, lon1 = coord[u]["lat"], coord[u]["lon"]
        lat2, lon2 = coord[v]["lat"], coord[v]["lon"]
        L = _approx_km_from_latlon(float(lat1), float(lon1), float(lat2), float(lon2))
        rows.append((f"L{idx}", u, v, L))

    out = pd.DataFrame(rows, columns=["line_id", "u", "v", "length_km"])
    out = out.dropna(subset=["u", "v", "length_km"]).copy()
    out["u"] = out["u"].astype(int)
    out["v"] = out["v"].astype(int)
    out = out.loc[out["u"] != out["v"]].copy()

    # drop duplicates ignoring direction
    a = np.minimum(out["u"].values, out["v"].values)
    b = np.maximum(out["u"].values, out["v"].values)
    out["_a"] = a; out["_b"] = b
    out = out.drop_duplicates(subset=["_a", "_b"]).drop(columns=["_a", "_b"]).reset_index(drop=True)
    return out


def _validate_params(params: PFScenarioParams) -> None:
    if params.v_min_pu >= params.v_max_pu:
        raise ValueError("Invalid voltage bounds: v_min_pu must be < v_max_pu.")
    if not (0.0 < params.pf_load <= 1.0):
        raise ValueError("pf_load must be in (0, 1].")
    if params.v_nom_kv <= 0:
        raise ValueError("v_nom_kv must be > 0.")
    if params.r_ohm_per_km <= 0 or params.x_ohm_per_km < 0:
        raise ValueError("Line parameters must satisfy: r_ohm_per_km > 0 and x_ohm_per_km >= 0.")
    if params.s_nom_kva <= 0:
        raise ValueError("s_nom_kva must be > 0.")
    if params.load_scale <= 0:
        raise ValueError("load_scale must be > 0.")


def _validate_topology(topo: PFTopologyBundle) -> None:
    has_pid = topo.mst_edges_pole_ids is not None
    has_mst = topo.mst_edges_latlon is not None
    has_gdf = topo.gdf_edges_4326 is not None

    # exactly one
    if sum([has_pid, has_mst, has_gdf]) != 1:
        raise ValueError(
            "PFTopologyBundle must include exactly one of: "
            "mst_edges_pole_ids OR mst_edges_latlon OR gdf_edges_4326 (+ endpoint cols)."
        )
    if has_gdf and (topo.edge_u_col is None or topo.edge_v_col is None):
        raise ValueError("External edges require edge_u_col and edge_v_col.")


def _assert_single_island(n: pypsa.Network, slack_bus: str) -> None:
    """
    PyPSA PF is only well-posed if each electrical island has its own slack.
    Here we enforce a single island (typical for LV radial microgrids).
    """
    n.determine_network_topology()
    if "sub_network" not in n.buses.columns:
        return  # version-robust
    sn = n.buses["sub_network"]
    if sn.nunique(dropna=False) <= 1:
        return

    sizes = sn.value_counts(dropna=False).to_dict()
    slack_sn = sn.loc[slack_bus]
    raise ValueError(
        "Topology has multiple electrical islands (sub_networks). "
        "This tool assumes ONE island with ONE slack. "
        f"Slack bus '{slack_bus}' is in sub_network={slack_sn}. Sizes: {sizes}"
    )


# =============================================================================
# Runner
# =============================================================================
class PyPSAPowerFlowRunner:
    """
    Minimal PF runner (single snapshot) with guardrails + compact diagnostics.

    Key fixes vs failing version:
    - DO NOT scale `sn_mva` from line rating (bad conditioning). Use default or a sane constant.
    - Filter out very short edges (GIS segmentation can create near-shorts -> numerical blow-up).
    - Emit a small `debug` dict so Streamlit can show what happened step-by-step.
    """

    # practical minimum segment length (km) to avoid near-short numerical issues
    MIN_LEN_KM_DEFAULT = 0.005  # 5 m
    LOAD_RAMP_STEPS = (0.25, 0.50, 0.75, 1.00)

    def __init__(self, topology: PFTopologyBundle):
        _validate_topology(topology)
        self.topo = topology
        self._nodes = _normalize_nodes(topology.gdf_nodes_4326, topology.pole_id_col)


    def _build_edges(self, *, min_len_km: float) -> pd.DataFrame:
        topo = self.topo
        if topo.mst_edges_pole_ids is not None:
            edges = _build_edges_from_pole_ids(topo.mst_edges_pole_ids, self._nodes)
        elif topo.mst_edges_latlon is not None:
            edges = _build_edges_from_pole_id_pairs(topo.mst_edges_latlon, self._nodes)  # legacy
        else:
            edges = _build_edges_from_gdf(
                topo.gdf_edges_4326, topo.edge_u_col, topo.edge_v_col  # type: ignore[arg-type]
            )

        # sanitize
        edges = edges.replace([np.inf, -np.inf], np.nan).dropna(subset=["u", "v", "length_km"]).copy()
        edges["length_km"] = pd.to_numeric(edges["length_km"], errors="coerce").fillna(0.0)

        # ensure int endpoints
        edges["u"] = pd.to_numeric(edges["u"], errors="coerce")
        edges["v"] = pd.to_numeric(edges["v"], errors="coerce")
        edges = edges.dropna(subset=["u", "v"]).copy()
        edges["u"] = edges["u"].astype(int)
        edges["v"] = edges["v"].astype(int)

        # drop self-loops
        edges = edges.loc[edges["u"] != edges["v"]].copy()

        # drop tiny edges (near-short / unstable)
        edges = edges.loc[edges["length_km"] >= float(min_len_km)].copy()

        # drop duplicates ignoring direction (radial lines shouldn't appear twice)
        uv_min = np.minimum(edges["u"].values, edges["v"].values)
        uv_max = np.maximum(edges["u"].values, edges["v"].values)
        edges["_a"] = uv_min
        edges["_b"] = uv_max
        edges = edges.drop_duplicates(subset=["_a", "_b"]).drop(columns=["_a", "_b"]).reset_index(drop=True)

        if edges.empty:
            raise ValueError(
                "No valid edges after cleaning. "
                f"Try lowering min_len_km (current {min_len_km}) or check edge geometries/endpoints."
            )
        return edges

    @staticmethod
    def _len_stats_km(lengths: np.ndarray) -> Dict[str, Any]:
        lengths = np.asarray(lengths, dtype=float)
        lengths = lengths[np.isfinite(lengths)]
        if lengths.size == 0:
            return {"n": 0}
        q = np.quantile(lengths, [0.01, 0.50, 0.99])
        return {
            "n": int(lengths.size),
            "min": float(np.min(lengths)),
            "p01": float(q[0]),
            "median": float(q[1]),
            "p99": float(q[2]),
            "max": float(np.max(lengths)),
            "lt_1m": int(np.sum(lengths < 0.001)),
            "lt_5m": int(np.sum(lengths < 0.005)),
            "lt_10m": int(np.sum(lengths < 0.010)),
        }

    @staticmethod
    def _set_scaled_loads(
        n: pypsa.Network,
        load_targets: Dict[str, tuple[float, float]],
        scale: float,
    ) -> None:
        for name, (p_mw, q_mvar) in load_targets.items():
            n.loads.at[name, "p_set"] = float(p_mw) * float(scale)
            n.loads.at[name, "q_set"] = float(q_mvar) * float(scale)

    def _run_pf_with_ramp(
        self,
        n: pypsa.Network,
        load_targets: Dict[str, tuple[float, float]],
        *,
        debug: bool,
        dbg: Dict[str, Any],
    ) -> None:
        steps = list(self.LOAD_RAMP_STEPS)
        if debug:
            dbg["load_ramp_steps"] = steps

        for i, scale in enumerate(steps):
            self._set_scaled_loads(n, load_targets, scale)
            try:
                n.pf(use_seed=(i > 0))
            except Exception as e:
                if debug:
                    dbg["pf_exception"] = repr(e)
                    dbg["pf_failed_at_scale"] = float(scale)
                raise RuntimeError(f"PyPSA power flow did not converge during load ramp at scale={scale}: {repr(e)}") from e

    @staticmethod
    def _run_radial_distflow_fallback(
        *,
        slack: int,
        bus_ids: list[int],
        line_meta: pd.DataFrame,
        bus_load_targets: Dict[int, tuple[float, float]],
        sn_mva_eff: float,
        params: PFScenarioParams,
        debug: bool,
        dbg: Dict[str, Any],
    ) -> Dict[str, Any]:
        adjacency: Dict[int, list[tuple[int, dict[str, Any]]]] = {int(pid): [] for pid in bus_ids}
        for row in line_meta.to_dict(orient="records"):
            u = int(row["u"])
            v = int(row["v"])
            adjacency.setdefault(u, []).append((v, row))
            adjacency.setdefault(v, []).append((u, row))

        parent: Dict[int, int | None] = {int(slack): None}
        parent_edge: Dict[int, dict[str, Any]] = {}
        order: list[int] = []
        stack = [int(slack)]

        while stack:
            node = int(stack.pop())
            order.append(node)
            for nbr, edge in adjacency.get(node, []):
                nbr = int(nbr)
                if nbr in parent:
                    continue
                parent[nbr] = node
                parent_edge[nbr] = edge
                stack.append(nbr)

        if len(parent) != len(bus_ids):
            missing = sorted(pid for pid in bus_ids if int(pid) not in parent)
            raise RuntimeError(
                "Radial fallback failed: topology is not fully reachable from the slack bus. "
                f"Missing buses (up to 20): {missing[:20]}"
            )

        p_load_pu: Dict[int, float] = {}
        q_load_pu: Dict[int, float] = {}
        for pid in bus_ids:
            p_mw, q_mvar = bus_load_targets.get(int(pid), (0.0, 0.0))
            p_load_pu[int(pid)] = float(p_mw) / float(sn_mva_eff)
            q_load_pu[int(pid)] = float(q_mvar) / float(sn_mva_eff)

        p_flow_pu: Dict[int, float] = {}
        q_flow_pu: Dict[int, float] = {}

        for node in reversed(order):
            p_total = float(p_load_pu.get(int(node), 0.0))
            q_total = float(q_load_pu.get(int(node), 0.0))
            for nbr, _ in adjacency.get(int(node), []):
                if parent.get(int(nbr)) == int(node):
                    p_total += float(p_flow_pu.get(int(nbr), 0.0))
                    q_total += float(q_flow_pu.get(int(nbr), 0.0))
            p_flow_pu[int(node)] = p_total
            q_flow_pu[int(node)] = q_total

        v2_pu: Dict[int, float] = {int(slack): 1.0}
        for node in order:
            node = int(node)
            for nbr, _ in adjacency.get(node, []):
                nbr = int(nbr)
                if parent.get(nbr) != node:
                    continue
                edge = parent_edge[nbr]
                drop = 2.0 * (
                    float(edge["r_pu"]) * float(p_flow_pu.get(nbr, 0.0))
                    + float(edge["x_pu"]) * float(q_flow_pu.get(nbr, 0.0))
                )
                v2_pu[nbr] = max(1e-9, float(v2_pu[node]) - drop)

        v_pu = {pid: float(np.sqrt(v2)) for pid, v2 in v2_pu.items()}
        df_bus = pd.DataFrame(
            {"bus": sorted(v_pu.keys()), "v_pu": [v_pu[pid] for pid in sorted(v_pu.keys())]}
        )
        df_bus["violates_limits"] = (df_bus["v_pu"] < float(params.v_min_pu)) | (df_bus["v_pu"] > float(params.v_max_pu))

        line_rows: list[Dict[str, Any]] = []
        for row in line_meta.to_dict(orient="records"):
            u = int(row["u"])
            v = int(row["v"])
            child = v if parent.get(v) == u else u
            p_mw = float(p_flow_pu.get(child, 0.0)) * float(sn_mva_eff)
            q_mvar = float(q_flow_pu.get(child, 0.0)) * float(sn_mva_eff)
            s_mva = float(np.sqrt(p_mw * p_mw + q_mvar * q_mvar))
            s_nom_mva = float(row["s_nom_MVA"])
            loading_pu = (s_mva / s_nom_mva) if s_nom_mva > 0 else np.nan
            line_rows.append(
                {
                    "line_id": str(row.get("line_id", row["line"])),
                    "line": str(row["line"]),
                    "bus0": str(u),
                    "bus1": str(v),
                    "line_type": row.get("line_type"),
                    "length_km": float(row["length_km"]),
                    "r_ohm_per_km": float(row.get("r_ohm_per_km", np.nan)),
                    "x_ohm_per_km": float(row.get("x_ohm_per_km", np.nan)),
                    "s_nom_kva": float(row.get("s_nom_kva", np.nan)),
                    "r_ohm": float(row["r_ohm"]),
                    "x_ohm": float(row["x_ohm"]),
                    "s_nom_MVA": s_nom_mva,
                    "p0_MW": p_mw,
                    "q0_MVAr": q_mvar,
                    "s0_MVA": s_mva,
                    "loading_pu": loading_pu,
                }
            )
        df_line = pd.DataFrame(line_rows)

        if debug:
            dbg["solver"] = "radial_distflow_fallback"
            dbg["fallback_v_pu_min"] = float(df_bus["v_pu"].min()) if len(df_bus) else None
            dbg["fallback_v_pu_max"] = float(df_bus["v_pu"].max()) if len(df_bus) else None

        summary: Dict[str, Any] = {
            "slack_bus": int(slack),
            "num_buses": int(len(df_bus)),
            "num_lines": int(len(df_line)),
            "num_voltage_violations": int(df_bus["violates_limits"].sum()),
            "v_min_pu_observed": float(df_bus["v_pu"].min()) if len(df_bus) else None,
            "v_max_pu_observed": float(df_bus["v_pu"].max()) if len(df_bus) else None,
            "max_line_loading_pu": (
                float(np.nanmax(df_line["loading_pu"].to_numpy(dtype=float)))
                if len(df_line) and df_line["loading_pu"].notna().any()
                else None
            ),
        }

        return {
            "summary": summary,
            "bus_results": df_bus,
            "line_results": df_line,
            "debug": dbg if debug else None,
            "network": None,
        }

    def run_snapshot(
        self,
        *,
        pole_p_kw: Dict[int, float],
        params: PFScenarioParams,
        line_params_df: Optional[pd.DataFrame] = None,
        debug: bool = True,
        min_len_km: float | None = None,
        sn_mva: float | None = 1.0,   # set None to use PyPSA default; 1.0 is usually stable for LV microgrids
        check_nonsense: bool = True,  # fail fast if PF returns non-physical voltages
    ) -> Dict[str, Any]:
        _validate_params(params)

        dbg: Dict[str, Any] = {}
        if debug:
            dbg["params"] = params.__dict__.copy()

        nodes = self._nodes.copy()
        slack = int(params.slack_pole_id)
        if slack not in set(nodes["pole_id"].tolist()):
            raise ValueError(f"slack_pole_id={slack} not present in nodes pole_id list.")

        # ---- edges (with min length filter) ----
        min_len_km = float(min_len_km if min_len_km is not None else self.MIN_LEN_KM_DEFAULT)
        edges_raw = self._build_edges(min_len_km=min_len_km)

        if debug:
            dbg["min_len_km"] = min_len_km
            dbg["edges_length_stats_km"] = self._len_stats_km(edges_raw["length_km"].to_numpy(dtype=float))
            dbg["edges_n"] = int(len(edges_raw))

        # ---- used poles: edge endpoints + poles with nonzero load + slack ----
        edge_poles = set(pd.concat([edges_raw["u"], edges_raw["v"]], ignore_index=True).astype(int).tolist())
        load_poles = set(int(pid) for pid, p in (pole_p_kw or {}).items() if float(p) > 0.0)
        used_poles = edge_poles | load_poles | {slack}

        if debug:
            dbg["edge_poles_n"] = int(len(edge_poles))
            dbg["load_poles_posP_n"] = int(len(load_poles))
            dbg["used_poles_n"] = int(len(used_poles))

        # orphan loads (load pole not in edge endpoints, unless it's slack itself)
        orphan = sorted([pid for pid in load_poles if pid not in edge_poles and pid != slack])
        if orphan:
            raise ValueError(
                "Some poles have load but are not connected by any edge (electrical islands). "
                f"Example pole_ids (up to 20): {orphan[:20]}"
            )

        # ---- build PyPSA network ----
        n = pypsa.Network()
        n.set_snapshots([0])

        # IMPORTANT FIX: do not tie sn_mva to line rating; keep default or constant
        if sn_mva is not None:
            n.sn_mva = float(sn_mva)
        if debug:
            dbg["sn_mva"] = float(getattr(n, "sn_mva", np.nan))

        # buses (only used poles)
        buses_df = nodes.loc[nodes["pole_id"].isin(sorted(used_poles))].copy()
        if buses_df.empty:
            raise ValueError("No buses to build after filtering used poles.")
        buses_df = buses_df.sort_values("pole_id").reset_index(drop=True)

        for _, r in buses_df.iterrows():
            pid = int(r["pole_id"])
            n.add(
                "Bus",
                name=str(pid),
                v_nom=float(params.v_nom_kv),  # kV
                v_mag_pu_set=(1.0 if pid == slack else np.nan),
                x=float(r["lon"]),             # lon
                y=float(r["lat"]),             # lat
                carrier="AC",
            )

        # Ensure v_nom exists for all buses (guards PyPSA v_mag_pu)
        n.buses["v_nom"] = pd.to_numeric(n.buses["v_nom"], errors="coerce").fillna(float(params.v_nom_kv))

        # Lines: effective parameters may be global (legacy behavior) or resolved
        # per-line upstream from catalog/override inputs.
        sn_mva_eff = float(getattr(n, "sn_mva", 1.0))
        z_base_ohm = (float(params.v_nom_kv) ** 2) / sn_mva_eff
        if z_base_ohm <= 0:
            raise ValueError(
                f"Invalid impedance base derived from v_nom_kv={params.v_nom_kv} and sn_mva={sn_mva_eff}."
            )

        # only keep edges with endpoints in used_poles
        edges = edges_raw.loc[edges_raw["u"].isin(used_poles) & edges_raw["v"].isin(used_poles)].copy()
        if edges.empty:
            raise ValueError("All edges were filtered out by used_poles. Cannot run PF.")

        if line_params_df is not None:
            lp = line_params_df.copy()
            if "line_id" not in lp.columns:
                raise ValueError("line_params_df must contain 'line_id'.")
            lp["line_id"] = lp["line_id"].astype(str)
            keep_cols = [c for c in ["line_id", "line_type", "r_ohm_per_km", "x_ohm_per_km", "s_nom_kva"] if c in lp.columns]
            lp = lp[keep_cols].drop_duplicates(subset=["line_id"])
            edges = edges.merge(lp, on="line_id", how="left")
            missing_lp = edges["r_ohm_per_km"].isna() | edges["x_ohm_per_km"].isna() | edges["s_nom_kva"].isna()
            if missing_lp.any():
                bad = edges.loc[missing_lp, "line_id"].head(20).tolist()
                raise ValueError(
                    "Resolved line parameters are missing values for some edges. "
                    f"Example line_id values: {bad}"
                )
        else:
            edges["line_type"] = "global_default"
            edges["r_ohm_per_km"] = float(params.r_ohm_per_km)
            edges["x_ohm_per_km"] = float(params.x_ohm_per_km)
            edges["s_nom_kva"] = float(params.s_nom_kva)

        edges["r_ohm_per_km"] = pd.to_numeric(edges["r_ohm_per_km"], errors="coerce").astype(float)
        edges["x_ohm_per_km"] = pd.to_numeric(edges["x_ohm_per_km"], errors="coerce").astype(float)
        edges["s_nom_kva"] = pd.to_numeric(edges["s_nom_kva"], errors="coerce").astype(float)
        edges["s_nom_MVA"] = edges["s_nom_kva"] / 1000.0
        edges["r_ohm"] = pd.to_numeric(edges["length_km"], errors="coerce").astype(float) * edges["r_ohm_per_km"]
        edges["x_ohm"] = pd.to_numeric(edges["length_km"], errors="coerce").astype(float) * edges["x_ohm_per_km"]
        edges["r_pu"] = edges["r_ohm"] / float(z_base_ohm)
        edges["x_pu"] = edges["x_ohm"] / float(z_base_ohm)

        line_meta_records: list[Dict[str, Any]] = []
        for k, r in edges.reset_index(drop=True).iterrows():
            line_id = str(r["line_id"])
            u = int(r["u"])
            v = int(r["v"])
            L_km = float(r["length_km"])
            r_ohm = float(r["r_ohm"])
            x_ohm = float(r["x_ohm"])
            r_pu = float(r["r_pu"])
            x_pu = float(r["x_pu"])
            s_nom_mva = float(r["s_nom_MVA"])
            line_name = line_id if line_id else f"L{k}"

            n.add(
                "Line",
                name=line_name,
                bus0=str(u),
                bus1=str(v),
                r=float(r_ohm),
                x=float(x_ohm),
                length=float(L_km),
                s_nom=float(s_nom_mva),
                carrier="AC",
            )
            line_meta_records.append(
                {
                    "line_id": line_id,
                    "line": line_name,
                    "u": int(u),
                    "v": int(v),
                    "line_type": None if pd.isna(r.get("line_type")) else str(r.get("line_type")),
                    "length_km": float(L_km),
                    "r_ohm_per_km": float(r["r_ohm_per_km"]),
                    "x_ohm_per_km": float(r["x_ohm_per_km"]),
                    "s_nom_kva": float(r["s_nom_kva"]),
                    "r_ohm": float(r_ohm),
                    "x_ohm": float(x_ohm),
                    "r_pu": float(r_pu),
                    "x_pu": float(x_pu),
                    "s_nom_MVA": float(s_nom_mva),
                }
            )

        if len(n.lines) == 0:
            raise ValueError("No lines were added. Cannot run PF.")

        if debug:
            rvals = pd.to_numeric(n.lines["r"], errors="coerce").to_numpy(dtype=float)
            xvals = pd.to_numeric(n.lines["x"], errors="coerce").to_numpy(dtype=float)
            dbg["z_base_ohm"] = float(z_base_ohm)
            dbg["line_r_stats_pu"] = {
                "min": float(np.nanmin(rvals)),
                "p01": float(np.nanquantile(rvals, 0.01)),
                "median": float(np.nanmedian(rvals)),
                "max": float(np.nanmax(rvals)),
                "lt_1e-4": int(np.sum(rvals < 1e-4)),
                "lt_1e-3": int(np.sum(rvals < 1e-3)),
            }
            dbg["line_x_stats_pu"] = {
                "min": float(np.nanmin(xvals)),
                "p01": float(np.nanquantile(xvals, 0.01)),
                "median": float(np.nanmedian(xvals)),
                "max": float(np.nanmax(xvals)),
                "lt_1e-4": int(np.sum(xvals < 1e-4)),
                "lt_1e-3": int(np.sum(xvals < 1e-3)),
            }
            dbg["num_lines"] = int(len(n.lines))
            dbg["num_buses"] = int(len(n.buses))

        # loads (ONLY where P>0)
        P_total_MW = 0.0
        Q_total_MVAr = 0.0
        n_loads = 0
        load_targets: Dict[str, tuple[float, float]] = {}
        bus_load_targets: Dict[int, tuple[float, float]] = {}

        for pid, p_kw in (pole_p_kw or {}).items():
            pid = int(pid)
            if pid not in used_poles:
                continue
            p_kw = float(p_kw)
            if p_kw <= 0.0:
                continue

            p_kw_eff = p_kw / float(params.load_scale)
            q_kvar = _infer_q_from_pf(p_kw_eff, params.pf_load)
            p_MW = p_kw_eff / 1000.0
            q_MVAr = q_kvar / 1000.0

            n.add(
                "Load",
                name=f"load_{pid}",
                bus=str(pid),
                p_set=p_MW,
                q_set=q_MVAr,
            )
            load_targets[f"load_{pid}"] = (float(p_MW), float(q_MVAr))
            bus_load_targets[int(pid)] = (float(p_MW), float(q_MVAr))

            P_total_MW += p_MW
            Q_total_MVAr += q_MVAr
            n_loads += 1

        if debug:
            dbg["loads_n"] = int(n_loads)
            dbg["P_total_MW"] = float(P_total_MW)
            dbg["Q_total_MVAr"] = float(Q_total_MVAr)

        # slack generator
        slack_bus = str(slack)
        n.add(
            "Generator",
            name="Slack",
            bus=slack_bus,
            control="Slack",
            p_nom=1e3,
            p_set=0.0,
        )

        # topology sanity (single island assumption)
        if debug:
            try:
                n.determine_network_topology()
                if "sub_network" in n.buses.columns:
                    vc = n.buses["sub_network"].value_counts(dropna=False)
                    dbg["subnet_count"] = int(vc.shape[0])
                    dbg["subnet_sizes_top10"] = vc.head(10).to_dict()
            except Exception as e:
                dbg["topology_check_error"] = repr(e)

        _assert_single_island(n, slack_bus=slack_bus)

        # ---- run PF ----
        self._run_pf_with_ramp(n, load_targets, debug=debug, dbg=dbg)

        # ---- post PF sanity: detect nonsense voltages early ----
        v_pu_series = n.buses_t.v_mag_pu.loc[0].astype(float)
        v_arr = v_pu_series.to_numpy(dtype=float)
        if debug:
            dbg["v_pu_min"] = float(np.nanmin(v_arr)) if v_arr.size else None
            dbg["v_pu_max"] = float(np.nanmax(v_arr)) if v_arr.size else None
            dbg["v_pu_any_nan"] = bool(np.isnan(v_arr).any()) if v_arr.size else False
            dbg["v_pu_any_inf"] = bool(np.isinf(v_arr).any()) if v_arr.size else False

        if check_nonsense:
            bad = (
                (v_arr.size == 0)
                or (np.isnan(v_arr).any())
                or (np.isinf(v_arr).any())
                or (np.nanmin(v_arr) < 0.0)
                or (np.nanmax(v_arr) > 2.0)
            )
            if bad:
                if debug:
                    dbg["pypsa_bad_result"] = True
                return self._run_radial_distflow_fallback(
                    slack=slack,
                    bus_ids=sorted(used_poles),
                    line_meta=pd.DataFrame(line_meta_records),
                    bus_load_targets=bus_load_targets,
                    sn_mva_eff=sn_mva_eff,
                    params=params,
                    debug=debug,
                    dbg=dbg,
                )

        # =============================================================================
        # Results
        # =============================================================================
        df_bus = pd.DataFrame({"bus": [int(b) for b in v_pu_series.index], "v_pu": v_pu_series.values})
        df_bus["violates_limits"] = (df_bus["v_pu"] < float(params.v_min_pu)) | (df_bus["v_pu"] > float(params.v_max_pu))
        df_bus = df_bus.sort_values("bus").reset_index(drop=True)

        df_line = pd.DataFrame(line_meta_records)[
            [
                "line_id",
                "line",
                "u",
                "v",
                "line_type",
                "length_km",
                "r_ohm_per_km",
                "x_ohm_per_km",
                "s_nom_kva",
                "r_ohm",
                "x_ohm",
                "s_nom_MVA",
            ]
        ].rename(columns={"u": "bus0", "v": "bus1"})
        df_line["bus0"] = df_line["bus0"].astype(str)
        df_line["bus1"] = df_line["bus1"].astype(str)

        if hasattr(n, "lines_t") and hasattr(n.lines_t, "p0") and hasattr(n.lines_t, "q0"):
            p0 = n.lines_t.p0.loc[0].astype(float)
            q0 = n.lines_t.q0.loc[0].astype(float)
            s0 = np.sqrt(p0**2 + q0**2)

            df_line["p0_MW"] = p0.values
            df_line["q0_MVAr"] = q0.values
            df_line["s0_MVA"] = s0.values

            denom = df_line["s_nom_MVA"].replace(0.0, np.nan).astype(float)
            df_line["loading_pu"] = df_line["s0_MVA"].astype(float) / denom
        else:
            df_line["loading_pu"] = np.nan

        summary: Dict[str, Any] = {
            "slack_bus": int(slack),
            "num_buses": int(len(n.buses)),
            "num_lines": int(len(n.lines)),
            "num_voltage_violations": int(df_bus["violates_limits"].sum()),
            "v_min_pu_observed": float(df_bus["v_pu"].min()) if len(df_bus) else None,
            "v_max_pu_observed": float(df_bus["v_pu"].max()) if len(df_bus) else None,
            "max_line_loading_pu": (
                float(np.nanmax(df_line["loading_pu"].to_numpy(dtype=float)))
                if df_line["loading_pu"].notna().any()
                else None
            ),
        }

        return {
            "summary": summary,
            "bus_results": df_bus,
            "line_results": df_line,
            "debug": dbg if debug else None,
            "network": n,  # keep for dev; remove in prod
        }
