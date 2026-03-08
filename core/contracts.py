from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import geopandas as gpd
import pandas as pd


SESSION_SCHEMA_VERSION = 1


class DomainEnvelope(TypedDict, total=False):
    version: int


class ProjectDomain(DomainEnvelope, total=False):
    topology_request: Dict[str, Any]
    validation_request: Dict[str, Any]


class TopologyDomain(DomainEnvelope, total=False):
    result: Optional["TopologyResult"]
    solve_seconds: Optional[float]


class ValidationDomain(DomainEnvelope, total=False):
    inputs: Optional["ValidationInputs"]
    result: Optional["ValidationResult"]
    reinforcement: Optional["ReinforcementResult"]
    runner: Any
    topo_fingerprint: Optional[Tuple[Any, ...]]


class UIDomain(DomainEnvelope, total=False):
    flags: Dict[str, Any]


@dataclass
class TopologyResult:
    schema_version: int
    metrics: Dict[str, float]
    gdf_buildings_4326: gpd.GeoDataFrame
    gdf_poles_4326: gpd.GeoDataFrame
    gdf_edges_4326: Optional[gpd.GeoDataFrame]
    gdf_roads_4326: Optional[gpd.GeoDataFrame]
    gdf_served_4326: gpd.GeoDataFrame
    gdf_unserved_4326: gpd.GeoDataFrame
    mst_edges_latlon: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    mst_edges_pole_ids: List[Tuple[int, int]]
    associations_df: pd.DataFrame
    center: Tuple[float, float]


@dataclass
class ValidationInputs:
    schema_version: int
    mode: str
    gdf_nodes_4326: gpd.GeoDataFrame
    associations_df: pd.DataFrame
    pole_id_col: str
    center: Tuple[float, float]
    mst_edges_latlon: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None
    mst_edges_pole_ids: Optional[List[Tuple[int, int]]] = None
    gdf_edges_4326: Optional[gpd.GeoDataFrame] = None
    gdf_roads_4326: Optional[gpd.GeoDataFrame] = None
    edge_u_col: Optional[str] = None
    edge_v_col: Optional[str] = None
    pole_loads_kW: Optional[pd.DataFrame] = None
    selected_hour: Optional[int] = None
    pole_load_dict: Dict[int, float] = field(default_factory=dict)
    scaling_mode: str = "Absolute (fixed over the year)"
    pmax_ref_kW: Optional[float] = None
    year_max_pole_kW: float = 0.0
    slack_pole_id: Optional[int] = None
    v_min_pu: float = 0.90
    v_max_pu: float = 1.10
    pf_load: float = 0.95
    v_nom_kv: float = 0.4
    v_base_mode: str = "3-phase LV (0.4 kV line-to-line)"
    r_ohm_per_km: float = 0.642
    x_ohm_per_km: float = 0.083
    s_nom_kva: float = 100.0
    line_params_mode: str = "global"
    default_line_type: Optional[str] = None
    line_types_df: Optional[pd.DataFrame] = None
    lines_meta_df: Optional[pd.DataFrame] = None
    resolved_line_params_df: Optional[pd.DataFrame] = None


@dataclass
class ValidationResult:
    schema_version: int
    hour: int
    params: Dict[str, Any]
    summary: Dict[str, Any]
    bus_results: pd.DataFrame
    line_results: pd.DataFrame
    debug: Optional[Dict[str, Any]] = None


@dataclass
class ReinforcementResult:
    schema_version: int
    hour: int
    settings: Dict[str, Any]
    optimization_status: str
    objective_value: Optional[float]
    pre_summary: Dict[str, Any]
    post_summary: Dict[str, Any]
    reinforced_lines: pd.DataFrame
    post_bus_results: pd.DataFrame
    post_line_results: pd.DataFrame
    total_added_capacity_kva: float
    total_reinforcement_cost: float
    upgraded_line_params_df: pd.DataFrame
    optimize_debug: Optional[Dict[str, Any]] = None
