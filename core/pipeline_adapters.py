from __future__ import annotations

from typing import Any, Dict, Optional

import geopandas as gpd
from shapely.geometry import LineString

from .contracts import (
    ReinforcementResult,
    SESSION_SCHEMA_VERSION,
    TopologyResult,
    ValidationInputs,
    ValidationResult,
)


def topology_result_from_legacy_payload(payload: Dict[str, Any]) -> TopologyResult:
    gdf_poles_4326 = payload["gdf_poles_4326"]
    gdf_edges_4326 = payload.get("gdf_edges_4326")
    if gdf_edges_4326 is None:
        temp_topology = TopologyResult(
            schema_version=SESSION_SCHEMA_VERSION,
            metrics=dict(payload["metrics"]),
            gdf_buildings_4326=payload["gdf_buildings_4326"],
            gdf_poles_4326=gdf_poles_4326,
            gdf_edges_4326=None,
            gdf_roads_4326=payload.get("gdf_roads_4326"),
            gdf_served_4326=payload["gdf_served_4326"],
            gdf_unserved_4326=payload["gdf_unserved_4326"],
            mst_edges_latlon=list(payload.get("mst_edges_latlon", [])),
            mst_edges_pole_ids=[(int(u), int(v)) for (u, v) in payload.get("mst_edges_pole_ids", [])],
            associations_df=payload["associations_df"].copy(),
            center=(float(payload["center"][0]), float(payload["center"][1])),
        )
        gdf_edges_4326 = _build_edges_gdf(temp_topology)

    return TopologyResult(
        schema_version=SESSION_SCHEMA_VERSION,
        metrics=dict(payload["metrics"]),
        gdf_buildings_4326=payload["gdf_buildings_4326"],
        gdf_poles_4326=gdf_poles_4326,
        gdf_edges_4326=gdf_edges_4326,
        gdf_roads_4326=payload.get("gdf_roads_4326"),
        gdf_served_4326=payload["gdf_served_4326"],
        gdf_unserved_4326=payload["gdf_unserved_4326"],
        mst_edges_latlon=list(payload.get("mst_edges_latlon", [])),
        mst_edges_pole_ids=[(int(u), int(v)) for (u, v) in payload.get("mst_edges_pole_ids", [])],
        associations_df=payload["associations_df"].copy(),
        center=(float(payload["center"][0]), float(payload["center"][1])),
    )


def _build_edges_gdf(topology: TopologyResult) -> gpd.GeoDataFrame:
    geom_by_pid = topology.gdf_poles_4326.set_index("pole_id").geometry.to_dict()
    rows: list[dict[str, Any]] = []
    for idx, (u, v) in enumerate(topology.mst_edges_pole_ids):
        p1 = geom_by_pid.get(int(u))
        p2 = geom_by_pid.get(int(v))
        if p1 is None or p2 is None:
            continue
        rows.append(
            {
                "line_id": f"L{idx}",
                "u": int(u),
                "v": int(v),
                "geometry": LineString([p1, p2]),
            }
        )
    return gpd.GeoDataFrame(rows, crs=topology.gdf_poles_4326.crs)


def build_topology_exports(topology: TopologyResult) -> Dict[str, bytes]:
    nodes_geojson = topology.gdf_poles_4326.to_json().encode("utf-8")
    edges_gdf = _build_edges_gdf(topology)
    edges_geojson = edges_gdf.to_json().encode("utf-8")
    associations_csv = (
        topology.associations_df.sort_values(["pole_id", "building_id"]).to_csv(index=False).encode("utf-8")
    )
    return {
        "nodes_geojson": nodes_geojson,
        "edges_geojson": edges_geojson,
        "associations_csv": associations_csv,
    }


def topology_result_to_view_payload(topology: Optional[TopologyResult]) -> Optional[Dict[str, Any]]:
    if topology is None:
        return None

    return {
        "metrics": topology.metrics,
        "gdf_buildings_4326": topology.gdf_buildings_4326,
        "gdf_poles_4326": topology.gdf_poles_4326,
        "gdf_edges_4326": topology.gdf_edges_4326,
        "gdf_roads_4326": topology.gdf_roads_4326,
        "gdf_served_4326": topology.gdf_served_4326,
        "gdf_unserved_4326": topology.gdf_unserved_4326,
        "mst_edges_latlon": topology.mst_edges_latlon,
        "mst_edges_pole_ids": topology.mst_edges_pole_ids,
        "associations_df": topology.associations_df,
        "downloads": build_topology_exports(topology),
        "center": topology.center,
    }


def validation_inputs_from_topology_result(topology: TopologyResult) -> ValidationInputs:
    pole_id_col = "pole_id" if "pole_id" in topology.gdf_poles_4326.columns else "id"
    return ValidationInputs(
        schema_version=SESSION_SCHEMA_VERSION,
        mode="session",
        gdf_nodes_4326=topology.gdf_poles_4326,
        associations_df=topology.associations_df.copy(),
        pole_id_col=pole_id_col,
        center=(float(topology.center[0]), float(topology.center[1])),
        mst_edges_latlon=list(topology.mst_edges_latlon),
        mst_edges_pole_ids=list(topology.mst_edges_pole_ids),
        gdf_edges_4326=topology.gdf_edges_4326,
        gdf_roads_4326=topology.gdf_roads_4326,
    )


def validation_inputs_from_external_payload(payload: Dict[str, Any]) -> ValidationInputs:
    gdf_nodes = payload["gdf_nodes"]
    if gdf_nodes.crs is None:
        raise ValueError("Nodes file has no CRS. Please export with a CRS.")
    gdf_nodes_4326 = gdf_nodes.to_crs(epsg=4326)

    gdf_edges = payload.get("gdf_edges")
    gdf_edges_4326 = None
    if gdf_edges is not None:
        if gdf_edges.crs is None:
            gdf_edges_4326 = gdf_edges.set_crs(epsg=4326, allow_override=True)
        else:
            gdf_edges_4326 = gdf_edges.to_crs(epsg=4326)

    c = gdf_nodes_4326.unary_union.centroid
    return ValidationInputs(
        schema_version=SESSION_SCHEMA_VERSION,
        mode="external",
        gdf_nodes_4326=gdf_nodes_4326,
        associations_df=payload["associations"].copy(),
        pole_id_col=str(payload["pole_col"]),
        center=(float(c.y), float(c.x)),
        gdf_edges_4326=gdf_edges_4326,
        edge_u_col=payload.get("u_col"),
        edge_v_col=payload.get("v_col"),
    )


def validation_inputs_to_map_view(inputs: ValidationInputs) -> Optional[Dict[str, Any]]:
    if inputs.selected_hour is None:
        return None
    return {
        "hour": int(inputs.selected_hour),
        "pole_load_dict": dict(inputs.pole_load_dict),
        "center": inputs.center,
        "pole_col": inputs.pole_id_col,
        "mst_edges_latlon": inputs.mst_edges_latlon,
        "gdf_edges_4326": inputs.gdf_edges_4326,
        "gdf_roads_4326": inputs.gdf_roads_4326,
        "scaling_mode": inputs.scaling_mode,
        "pmax_ref_kW": inputs.pmax_ref_kW,
        "year_max_pole_kW": inputs.year_max_pole_kW,
        "resolved_line_params_df": inputs.resolved_line_params_df,
    }


def validation_result_from_runner_output(*, hour: int, params: Dict[str, Any], out: Dict[str, Any]) -> ValidationResult:
    return ValidationResult(
        schema_version=SESSION_SCHEMA_VERSION,
        hour=int(hour),
        params=dict(params),
        summary=dict(out["summary"]),
        bus_results=out["bus_results"],
        line_results=out["line_results"],
        debug=out.get("debug"),
    )


def validation_result_to_view_payload(result: Optional[ValidationResult]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return {
        "hour": result.hour,
        "params": result.params,
        "summary": result.summary,
        "bus_results": result.bus_results,
        "line_results": result.line_results,
        "debug": result.debug,
    }


def reinforcement_result_to_view_payload(result: Optional[ReinforcementResult]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return {
        "hour": result.hour,
        "settings": result.settings,
        "optimization_status": result.optimization_status,
        "objective_value": result.objective_value,
        "pre_summary": result.pre_summary,
        "post_summary": result.post_summary,
        "reinforced_lines": result.reinforced_lines,
        "post_bus_results": result.post_bus_results,
        "post_line_results": result.post_line_results,
        "total_added_capacity_kva": result.total_added_capacity_kva,
        "total_reinforcement_cost": result.total_reinforcement_cost,
        "optimize_debug": result.optimize_debug,
    }
