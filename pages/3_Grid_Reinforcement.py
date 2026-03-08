from __future__ import annotations

import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely.geometry import LineString

from core.pipeline_adapters import reinforcement_result_to_view_payload, validation_result_to_view_payload
from core.pipeline_state import (
    ensure_session_domains,
    get_validation_inputs,
    get_validation_reinforcement_result,
    get_validation_result,
    get_validation_runner_cache,
    set_project_request,
    set_validation_reinforcement_result,
    set_validation_runner_cache,
)
from core.powerflow_network import PFScenarioParams, PFTopologyBundle, PyPSAPowerFlowRunner
from core.powerflow_reinforcement import ReinforcementSettings, run_reinforcement_optimization
from pages.ui_sections.reinforcement_sections import (
    render_infeasibility_guidance,
    render_page_header,
    render_reinforcement_params,
    render_reinforcement_results,
)


def _get_nodes_gdf_4326(validation_inputs) -> gpd.GeoDataFrame:
    gdf_nodes_4326 = validation_inputs.gdf_nodes_4326
    if gdf_nodes_4326 is None:
        raise ValueError("Validation inputs do not include nodes.")
    if gdf_nodes_4326.crs is None:
        gdf_nodes_4326 = gdf_nodes_4326.set_crs(epsg=4326, allow_override=True)
    return gdf_nodes_4326.to_crs(epsg=4326)


def _infer_pole_id_col(validation_inputs, gdf_nodes_4326: gpd.GeoDataFrame) -> str:
    if validation_inputs.pole_id_col in gdf_nodes_4326.columns:
        return validation_inputs.pole_id_col
    if "pole_id" in gdf_nodes_4326.columns:
        return "pole_id"
    if "id" in gdf_nodes_4326.columns:
        return "id"
    raise ValueError("Cannot find pole id column in nodes. Expected 'pole_id' (preferred) or 'id'.")


def _build_session_edges_for_pf_map(
    *,
    gdf_nodes_4326: gpd.GeoDataFrame,
    pole_id_col: str,
    mst_edges_pole_ids: list[tuple[int, int]] | None,
) -> gpd.GeoDataFrame | None:
    if not mst_edges_pole_ids:
        return None

    gdfp = gdf_nodes_4326.copy()
    gdfp["_pid"] = pd.to_numeric(gdfp[pole_id_col], errors="coerce")
    gdfp = gdfp.dropna(subset=["_pid"]).copy()
    if gdfp.empty:
        return None
    gdfp["_pid"] = gdfp["_pid"].astype(int)
    geom_by_pid = gdfp.set_index("_pid").geometry.to_dict()

    rows: list[dict[str, object]] = []
    for u, v in mst_edges_pole_ids:
        p1 = geom_by_pid.get(int(u))
        p2 = geom_by_pid.get(int(v))
        if p1 is None or p2 is None:
            continue
        rows.append({"u": int(u), "v": int(v), "geometry": LineString([p1, p2])})

    if not rows:
        return None
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf_nodes_4326.crs)


def _build_runner(validation_inputs) -> tuple[PyPSAPowerFlowRunner, str, gpd.GeoDataFrame, str]:
    gdf_nodes_4326 = _get_nodes_gdf_4326(validation_inputs)
    pole_col = _infer_pole_id_col(validation_inputs, gdf_nodes_4326)

    if validation_inputs.mode == "session":
        mst_edges_pole_ids = validation_inputs.mst_edges_pole_ids
        if mst_edges_pole_ids is None:
            raise ValueError("Session topology is missing mst_edges_pole_ids.")
        gdf_edges_for_pf = None
        edge_u_col = None
        edge_v_col = None
    elif validation_inputs.mode == "external":
        mst_edges_pole_ids = None
        gdf_edges_for_pf = validation_inputs.gdf_edges_4326
        edge_u_col = validation_inputs.edge_u_col
        edge_v_col = validation_inputs.edge_v_col
        if gdf_edges_for_pf is None:
            raise ValueError("External mode requires edges data.")
        if edge_u_col is None or edge_v_col is None:
            raise ValueError("External mode requires edge endpoint columns.")
    else:
        raise ValueError(f"Unknown validation input mode: {validation_inputs.mode}")

    topo = PFTopologyBundle(
        gdf_nodes_4326=gdf_nodes_4326,
        pole_id_col=pole_col,
        mst_edges_pole_ids=mst_edges_pole_ids,
        gdf_edges_4326=gdf_edges_for_pf,
        edge_u_col=edge_u_col,
        edge_v_col=edge_v_col,
    )

    from core.powerflow_network import RUNNER_VERSION  # local import to mirror Page 2

    topo_fingerprint = (
        RUNNER_VERSION,
        validation_inputs.mode,
        len(gdf_nodes_4326),
        None if validation_inputs.mst_edges_latlon is None else len(validation_inputs.mst_edges_latlon),
        None if validation_inputs.gdf_edges_4326 is None else len(validation_inputs.gdf_edges_4326),
        pole_col,
        edge_u_col,
        edge_v_col,
    )

    runner, current_fingerprint = get_validation_runner_cache(st.session_state)
    if current_fingerprint != topo_fingerprint or runner is None:
        runner = PyPSAPowerFlowRunner(topo)
        set_validation_runner_cache(st.session_state, runner, topo_fingerprint)

    return runner, pole_col, gdf_nodes_4326, RUNNER_VERSION


st.set_page_config(
    page_title="3) Grid Reinforcement",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_page_header()
ensure_session_domains(st.session_state)
ui_flags = st.session_state["ui"]["flags"]

validation_inputs = get_validation_inputs(st.session_state)
if validation_inputs is None:
    st.info("No validation inputs found. Run **Grid Validation** first.")
    st.stop()

pf_result = get_validation_result(st.session_state)
pf_result_view = validation_result_to_view_payload(pf_result)
if pf_result_view is None:
    st.info("No power-flow result found. Run **Grid Validation** first.")
    st.stop()

if validation_inputs.pole_loads_kW is None:
    st.info("No aggregated demand found. Run demand aggregation in **Grid Validation** first.")
    st.stop()
if validation_inputs.slack_pole_id is None:
    st.info("No slack pole configured. Complete PF setup in **Grid Validation** first.")
    st.stop()

runner, pole_col, gdf_nodes_4326, _ = _build_runner(validation_inputs)

params = PFScenarioParams(
    slack_pole_id=int(validation_inputs.slack_pole_id),
    v_min_pu=float(validation_inputs.v_min_pu),
    v_max_pu=float(validation_inputs.v_max_pu),
    pf_load=float(validation_inputs.pf_load),
    v_nom_kv=float(validation_inputs.v_nom_kv),
    r_ohm_per_km=float(validation_inputs.r_ohm_per_km),
    x_ohm_per_km=float(validation_inputs.x_ohm_per_km),
    s_nom_kva=float(validation_inputs.s_nom_kva),
    load_scale=(3.0 if float(validation_inputs.v_nom_kv) < 0.3 else 1.0),
)

has_violations = bool(
    (pf_result_view["summary"].get("num_voltage_violations", 0) or 0) > 0
    or (
        pd.to_numeric(pf_result_view["line_results"].get("loading_pu"), errors="coerce").fillna(0.0) > 1.0
    ).sum()
    > 0
)
rf_controls = render_reinforcement_params(default_enabled=has_violations)
set_project_request(
    st.session_state,
    reinforcement_request={
        "has_validation_inputs": validation_inputs is not None,
        "has_pf_result": pf_result_view is not None,
        "controls": dict(rf_controls),
    },
)
ui_flags["reinforcement_last_controls"] = dict(rf_controls)

if rf_controls["run_clicked"]:
    try:
        if (not has_violations) and (not rf_controls["run_even_if_clean"]):
            st.info("No current PF violations detected. Enable 'run even if clean' to force optimization.")
        else:
            rf_hour = int(pf_result_view["hour"])
            rf_loads = validation_inputs.pole_loads_kW.loc[rf_hour]
            rf_pole_load_dict = {int(k): float(v) for k, v in rf_loads.to_dict().items()}

            last_pf_ctx = ui_flags.get("last_pf_run_context", {}) or {}
            pf_min_len_m = float(last_pf_ctx.get("pf_min_len_m", st.session_state.get("pf_min_len_m", 5.0)))
            pf_sn_mva = float(last_pf_ctx.get("pf_sn_mva", st.session_state.get("pf_sn_mva", 0.1)))
            pf_fail_on_nonsense = bool(
                last_pf_ctx.get("pf_fail_on_nonsense", st.session_state.get("pf_fail_on_nonsense", True))
            )
            rf_line_params_df = last_pf_ctx.get("resolved_line_params_df", validation_inputs.resolved_line_params_df)
            if "last_pf_run_context" not in ui_flags:
                st.warning(
                    "PF run context was not found in session state. Reinforcement will use current PF control defaults, "
                    "which may differ from the last validation run."
                )

            with st.spinner("Optimizing reinforcement plan..."):
                rf_result = run_reinforcement_optimization(
                    runner=runner,
                    hour=rf_hour,
                    pole_load_dict=rf_pole_load_dict,
                    params=params,
                    line_params_df=rf_line_params_df,
                    settings=ReinforcementSettings(
                        selection_mode=str(rf_controls["selection_mode"]),
                        cost_per_km_per_kva=float(rf_controls["cost_per_km_per_kva"]),
                        max_upgrade_factor=float(rf_controls["max_upgrade_factor"]),
                        allow_emergency_load_shedding=bool(rf_controls["allow_emergency_load_shedding"]),
                        shedding_penalty_per_mwh=float(rf_controls["shedding_penalty_per_mwh"]),
                        solver_name=rf_controls["solver_name"],
                        min_len_km=pf_min_len_m / 1000.0,
                        sn_mva=pf_sn_mva,
                        check_nonsense=pf_fail_on_nonsense,
                    ),
                    pre_summary=dict(pf_result_view["summary"]),
                )
            set_validation_reinforcement_result(st.session_state, rf_result)
            st.success("Reinforcement optimization completed.")
    except Exception as e:
        st.error(f"Reinforcement optimization failed: {repr(e)}")
        err_txt = str(e).lower()
        if ("infeasible" in err_txt) or ("unbounded" in err_txt) or ("did not converge" in err_txt):
            render_infeasibility_guidance(prefix="Reinforcement optimization failed and appears infeasible.")
        st.exception(e)

reinforcement_result = get_validation_reinforcement_result(st.session_state)
reinforcement_result_view = reinforcement_result_to_view_payload(reinforcement_result)

reinforced_line_pairs: set[tuple[int, int]] = set()
if reinforcement_result_view is not None and not reinforcement_result_view["reinforced_lines"].empty:
    for row in reinforcement_result_view["reinforced_lines"].to_dict(orient="records"):
        u = pd.to_numeric(row.get("from_bus"), errors="coerce")
        v = pd.to_numeric(row.get("to_bus"), errors="coerce")
        if pd.notna(u) and pd.notna(v):
            reinforced_line_pairs.add((min(int(u), int(v)), max(int(u), int(v))))

rf_map = None
if reinforcement_result_view is not None:
    post_bus_v_pu: dict[int, float] = {}
    for row in reinforcement_result_view["post_bus_results"].to_dict(orient="records"):
        pid = pd.to_numeric(row.get("bus"), errors="coerce")
        vpu = pd.to_numeric(row.get("v_pu"), errors="coerce")
        if pd.notna(pid) and pd.notna(vpu):
            post_bus_v_pu[int(pid)] = float(vpu)

    post_line_loading_pu: dict[tuple[int, int], float] = {}
    for row in reinforcement_result_view["post_line_results"].to_dict(orient="records"):
        u = pd.to_numeric(row.get("bus0"), errors="coerce")
        v = pd.to_numeric(row.get("bus1"), errors="coerce")
        loading = pd.to_numeric(row.get("loading_pu"), errors="coerce")
        if pd.notna(u) and pd.notna(v) and pd.notna(loading):
            post_line_loading_pu[(int(u), int(v))] = float(loading)

    edges_for_map = validation_inputs.gdf_edges_4326
    if edges_for_map is None:
        edges_for_map = _build_session_edges_for_pf_map(
            gdf_nodes_4326=gdf_nodes_4326,
            pole_id_col=pole_col,
            mst_edges_pole_ids=validation_inputs.mst_edges_pole_ids,
        )

    rf_map = {
        "center": validation_inputs.center,
        "gdf_poles_4326": gdf_nodes_4326,
        "pole_id_col": pole_col,
        "gdf_edges_4326": edges_for_map,
        "mst_edges_latlon": validation_inputs.mst_edges_latlon,
        "gdf_roads_4326": validation_inputs.gdf_roads_4326,
        "slack_pole_id": int(validation_inputs.slack_pole_id),
        "bus_v_pu": post_bus_v_pu,
        "line_loading_pu": post_line_loading_pu,
        "reinforced_line_pairs": reinforced_line_pairs,
        "v_min_pu": float(validation_inputs.v_min_pu),
        "v_max_pu": float(validation_inputs.v_max_pu),
    }
    ui_flags["reinforcement_last_map_payload"] = rf_map
else:
    rf_map = ui_flags.get("reinforcement_last_map_payload")

render_reinforcement_results(reinforcement_result_view, pf_map=rf_map)
