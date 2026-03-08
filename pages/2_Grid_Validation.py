from __future__ import annotations

from dataclasses import replace

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import LineString

from core.contracts import ValidationInputs
from core.line_params import (
    build_line_params_for_edges,
    read_line_types_csv,
    read_lines_metadata_csv,
)
from core.pipeline_adapters import (
    validation_inputs_from_external_payload,
    validation_inputs_from_topology_result,
    validation_inputs_to_map_view,
    validation_result_from_runner_output,
    validation_result_to_view_payload,
)
from core.pipeline_state import (
    ensure_session_domains,
    get_topology_result,
    get_validation_inputs,
    get_validation_result,
    get_validation_runner_cache,
    set_project_request,
    set_validation_inputs,
    set_validation_result,
    set_validation_runner_cache,
    update_validation_line_params_state,
    update_validation_load_state,
    update_validation_pf_settings,
)
from core.powerflow_io import (
    read_building_metadata_csv,
    read_category_profiles_csv,
    read_vector,
)
from core.powerflow_network import PFScenarioParams, PFTopologyBundle, PyPSAPowerFlowRunner
from core.powerflow_validation import aggregate_pole_loads, validate_external_topology
from pages.ui_sections.validation_sections import (
    render_demand_controls,
    render_demand_upload_section,
    render_line_params_section,
    render_load_visualization,
    render_page_header,
    render_pf_results,
    render_pf_run_controls,
    render_pf_setup_section,
    render_topology_source_section,
)


def _merge_existing_runtime_state(
    base_inputs: ValidationInputs,
    existing_inputs: ValidationInputs | None,
) -> ValidationInputs:
    if existing_inputs is None or existing_inputs.mode != base_inputs.mode:
        return base_inputs

    return replace(
        base_inputs,
        pole_loads_kW=existing_inputs.pole_loads_kW,
        selected_hour=existing_inputs.selected_hour,
        pole_load_dict=dict(existing_inputs.pole_load_dict),
        scaling_mode=existing_inputs.scaling_mode,
        pmax_ref_kW=existing_inputs.pmax_ref_kW,
        year_max_pole_kW=existing_inputs.year_max_pole_kW,
        slack_pole_id=existing_inputs.slack_pole_id,
        v_min_pu=existing_inputs.v_min_pu,
        v_max_pu=existing_inputs.v_max_pu,
        pf_load=existing_inputs.pf_load,
        v_nom_kv=existing_inputs.v_nom_kv,
        v_base_mode=existing_inputs.v_base_mode,
        r_ohm_per_km=existing_inputs.r_ohm_per_km,
        x_ohm_per_km=existing_inputs.x_ohm_per_km,
        s_nom_kva=existing_inputs.s_nom_kva,
        line_params_mode=existing_inputs.line_params_mode,
        default_line_type=existing_inputs.default_line_type,
        line_types_df=existing_inputs.line_types_df,
        lines_meta_df=existing_inputs.lines_meta_df,
        resolved_line_params_df=existing_inputs.resolved_line_params_df,
    )


def _validation_topology_signature(inputs: ValidationInputs | None) -> tuple[object, ...] | None:
    if inputs is None:
        return None
    return (
        inputs.mode,
        len(inputs.gdf_nodes_4326),
        None if inputs.gdf_edges_4326 is None else len(inputs.gdf_edges_4326),
        None if inputs.mst_edges_pole_ids is None else len(inputs.mst_edges_pole_ids),
        inputs.pole_id_col,
        inputs.edge_u_col,
        inputs.edge_v_col,
    )


def _get_nodes_gdf_4326(inputs: ValidationInputs) -> gpd.GeoDataFrame:
    gdf_nodes_4326 = inputs.gdf_nodes_4326
    if gdf_nodes_4326 is None:
        raise ValueError("Validation inputs do not include nodes.")
    if gdf_nodes_4326.crs is None:
        gdf_nodes_4326 = gdf_nodes_4326.set_crs(epsg=4326, allow_override=True)
    return gdf_nodes_4326.to_crs(epsg=4326)


def _get_associations_from_validation_inputs(inputs: ValidationInputs) -> pd.DataFrame:
    out = inputs.associations_df.copy()
    cols = {c.lower(): c for c in out.columns}
    bcol = cols.get("building_id", cols.get("building", None))
    pcol = cols.get("pole_id", cols.get("pole", None))
    if bcol is None or pcol is None:
        raise ValueError("Associations must contain building_id and pole_id columns.")

    out = out[[bcol, pcol]].rename(columns={bcol: "building_id", pcol: "pole_id"}).copy()
    out["building_id"] = out["building_id"].astype(str)
    out["pole_id"] = pd.to_numeric(out["pole_id"], errors="coerce")
    out = out.dropna(subset=["pole_id"]).copy()
    out["pole_id"] = out["pole_id"].astype(int)
    return out


def _infer_pole_id_col(inputs: ValidationInputs, gdf_nodes_4326: gpd.GeoDataFrame) -> str:
    if inputs.pole_id_col in gdf_nodes_4326.columns:
        return inputs.pole_id_col
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


def _edges_have_explicit_line_id(inputs: ValidationInputs) -> bool:
    if inputs.mode == "session":
        return True
    return inputs.gdf_edges_4326 is not None and "line_id" in inputs.gdf_edges_4326.columns




st.set_page_config(
    page_title="2) Grid Validation",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_page_header()
ensure_session_domains(st.session_state)

PF_VBASE_3PH = "3-phase LV (0.4 kV line-to-line)"
PF_VBASE_PHASE = "Per-phase equivalent (0.230 kV L-N)"

# One-time migration of legacy defaults.
# Important: do not overwrite user choices on reruns.
PF_WIDGET_MIGRATION_FLAG = "pf_widget_defaults_migrated_v1"
if not st.session_state.get(PF_WIDGET_MIGRATION_FLAG, False):
    if "pf_vbase_mode" not in st.session_state:
        st.session_state["pf_vbase_mode"] = PF_VBASE_PHASE
    if "pf_sn_mva" not in st.session_state:
        st.session_state["pf_sn_mva"] = 0.1
    st.session_state[PF_WIDGET_MIGRATION_FLAG] = True

topology_source, nodes_file, edges_file, assoc_file = render_topology_source_section()
set_project_request(
    st.session_state,
    validation_request={
        "topology_source": topology_source,
        "has_nodes_file": nodes_file is not None,
        "has_edges_file": edges_file is not None,
        "has_assoc_file": assoc_file is not None,
    },
)

existing_validation_inputs = get_validation_inputs(st.session_state)

if topology_source.startswith("Use results"):
    topology_result = get_topology_result(st.session_state)
    if topology_result is None:
        st.info("No topology results found in session. Go to **1) Grid Topology** and run the LV design first.")
        set_validation_inputs(st.session_state, None)
    else:
        merged_inputs = _merge_existing_runtime_state(
            validation_inputs_from_topology_result(topology_result),
            existing_validation_inputs,
        )
        set_validation_inputs(
            st.session_state,
            merged_inputs,
            reset_runtime=_validation_topology_signature(existing_validation_inputs)
            != _validation_topology_signature(merged_inputs),
        )
        st.success("Using topology from session (Page 1).")
else:
    if nodes_file and edges_file and assoc_file:
        try:
            gdf_nodes = read_vector(nodes_file)
            gdf_edges = read_vector(edges_file)
            assoc = pd.read_csv(assoc_file)

            external_payload = validate_external_topology(gdf_nodes, gdf_edges, assoc)
            merged_inputs = _merge_existing_runtime_state(
                validation_inputs_from_external_payload(external_payload),
                existing_validation_inputs,
            )
            set_validation_inputs(
                st.session_state,
                merged_inputs,
                reset_runtime=_validation_topology_signature(existing_validation_inputs)
                != _validation_topology_signature(merged_inputs),
            )
            st.success("External topology loaded and validated.")
            with st.expander("Detected schema", expanded=False):
                st.write(
                    {
                        "pole_id_column": external_payload["pole_col"],
                        "edge_u_column": external_payload["u_col"],
                        "edge_v_column": external_payload["v_col"],
                        "num_nodes": int(len(gdf_nodes)),
                        "num_edges": int(len(gdf_edges)),
                        "num_associations": int(len(external_payload["associations"])),
                    }
                )
        except Exception as e:
            set_validation_inputs(st.session_state, None)
            st.error(f"External topology input error: {repr(e)}")
            st.stop()
    else:
        set_validation_inputs(st.session_state, None)
        st.info("Upload nodes, edges, and associations to proceed.")

st.divider()
st.subheader("Demand inputs")

validation_inputs = get_validation_inputs(st.session_state)
if validation_inputs is None:
    st.info("Select a topology source above first.")
else:
    try:
        associations = _get_associations_from_validation_inputs(validation_inputs)
        st.caption(f"Associations loaded: {len(associations)} served buildings mapped to poles.")
    except Exception as e:
        st.error(f"Cannot load associations from selected topology source: {e}")
        st.stop()

    meta_file, profiles_file = render_demand_upload_section()

    if meta_file and profiles_file:
        try:
            building_meta = read_building_metadata_csv(meta_file)
            category_profiles = read_category_profiles_csv(profiles_file)

            pole_loads_kW = aggregate_pole_loads(
                associations=associations,
                building_meta=building_meta,
                category_profiles=category_profiles,
            )
            pole_loads_kW.index = pole_loads_kW.index.astype(int)

            st.success(
                f"Pole loads aggregated. Hours: {len(pole_loads_kW)} | Poles with load: {pole_loads_kW.shape[1]}"
            )

            demand_controls = render_demand_controls(pole_loads_kW)
            pole_load_at_hour = pole_loads_kW.loc[int(demand_controls["hour"])]
            pole_load_dict = {int(k): float(v) for k, v in pole_load_at_hour.to_dict().items()}

            validation_inputs = update_validation_load_state(
                st.session_state,
                pole_loads_kW=pole_loads_kW,
                selected_hour=int(demand_controls["hour"]),
                pole_load_dict=pole_load_dict,
                scaling_mode=str(demand_controls["scaling_mode"]),
                pmax_ref_kW=demand_controls["pmax_ref_kW"],
                year_max_pole_kW=float(demand_controls["year_max_pole_kW"]),
            )

            st.caption(
                f"Selected hour: {demand_controls['hour']} | Total load: {float(pole_load_at_hour.sum()):.2f} kW "
                f"| Max pole this hour: {float(pole_load_at_hour.max()):.2f} kW "
                + (
                    f"| Year max pole: {demand_controls['year_max_pole_kW']:.2f} kW"
                    if demand_controls["year_max_pole_kW"] > 0
                    else ""
                )
            )
        except Exception as e:
            st.error(f"Demand aggregation failed: {repr(e)}")
            st.exception(e)

    validation_inputs = get_validation_inputs(st.session_state)
    vis = None if validation_inputs is None else validation_inputs_to_map_view(validation_inputs)
    slack_pid = None
    if validation_inputs is not None and validation_inputs.slack_pole_id is not None:
        slack_pid = int(validation_inputs.slack_pole_id)
    render_load_visualization(
        vis=vis,
        gdf_nodes_4326=None if validation_inputs is None else _get_nodes_gdf_4326(validation_inputs),
        slack_pole_id=slack_pid,
    )

validation_inputs = get_validation_inputs(st.session_state)
if validation_inputs is None or validation_inputs.selected_hour is None:
    st.divider()
    st.subheader("Power flow setup")
    st.markdown(
        "Select the **plant / slack pole** and the **minimum electrical assumptions** used to build the PyPSA network "
        "(single snapshot, one generator injection point)."
    )
    st.info("Run demand aggregation first (so the map and pole list are available).")
else:
    gdf_nodes_4326 = _get_nodes_gdf_4326(validation_inputs)
    pole_col = _infer_pole_id_col(validation_inputs, gdf_nodes_4326)

    pole_ids = (
        pd.to_numeric(gdf_nodes_4326[pole_col], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    if not pole_ids:
        st.error("No valid pole IDs found in nodes file.")
        st.stop()

    load_dict = validation_inputs.pole_load_dict or {}

    gdfp = gdf_nodes_4326.copy()
    gdfp["_pid"] = pd.to_numeric(gdfp[pole_col], errors="coerce")
    gdfp = gdfp.dropna(subset=["_pid"]).copy()
    gdfp["_pid"] = gdfp["_pid"].astype(int)

    pts = gdfp.geometry.apply(lambda geom: geom if geom.geom_type == "Point" else geom.representative_point())
    gdfp["_lat"] = pts.y.astype(float)
    gdfp["_lon"] = pts.x.astype(float)
    gdfp["_p"] = gdfp["_pid"].map({int(k): float(v) for k, v in load_dict.items()}).fillna(0.0).astype(float)

    def _suggest_slack_pole_id() -> int:
        w = gdfp["_p"].to_numpy(dtype=float)
        if np.nanmax(w) <= 0:
            lat0 = float(np.nanmean(gdfp["_lat"]))
            lon0 = float(np.nanmean(gdfp["_lon"]))
        else:
            lat0 = float(np.average(gdfp["_lat"], weights=w))
            lon0 = float(np.average(gdfp["_lon"], weights=w))

        d2 = (gdfp["_lat"] - lat0) ** 2 + (gdfp["_lon"] - lon0) ** 2
        best_idx = int(d2.idxmin())
        return int(gdfp.loc[best_idx, "_pid"])

    suggested_slack = _suggest_slack_pole_id()
    pf_setup = render_pf_setup_section(
        pole_ids=pole_ids,
        suggested_slack=suggested_slack,
    )

    if pf_setup["v_min_pu"] >= pf_setup["v_max_pu"]:
        st.error("Voltage limits invalid: Min voltage must be < Max voltage.")
        st.stop()

    v_base_mode = str(pf_setup["v_base_mode"])
    v_nom_kv = 0.4 if v_base_mode.startswith("3-phase") else (0.4 / np.sqrt(3))

    validation_inputs = update_validation_pf_settings(
        st.session_state,
        slack_pole_id=int(pf_setup["slack_pole_id"]),
        v_min_pu=float(pf_setup["v_min_pu"]),
        v_max_pu=float(pf_setup["v_max_pu"]),
        pf_load=float(pf_setup["pf_load"]),
        v_nom_kv=float(v_nom_kv),
        v_base_mode=v_base_mode,
        r_ohm_per_km=float(validation_inputs.r_ohm_per_km),
        x_ohm_per_km=float(validation_inputs.x_ohm_per_km),
        s_nom_kva=float(validation_inputs.s_nom_kva),
    )

validation_inputs = get_validation_inputs(st.session_state)
if validation_inputs is None or validation_inputs.selected_hour is None:
    st.divider()
    st.subheader("Run Power Flow (PyPSA)")
    st.info("Load topology + demand first.")
    st.stop()

if validation_inputs.slack_pole_id is None:
    st.divider()
    st.subheader("Run Power Flow (PyPSA)")
    st.info("Set PF settings + line params first.")
    st.stop()

gdf_nodes_4326 = _get_nodes_gdf_4326(validation_inputs)
pole_col = _infer_pole_id_col(validation_inputs, gdf_nodes_4326)

if validation_inputs.mode == "session":
    mst_edges_pole_ids = validation_inputs.mst_edges_pole_ids
    if mst_edges_pole_ids is None:
        raise ValueError(
            "Session topology missing 'mst_edges_pole_ids'. "
            "Update Page 1 run_low_voltage() to return it, then rerun Page 1."
        )
    edge_u_col = None
    edge_v_col = None
    gdf_edges_for_pf = None
elif validation_inputs.mode == "external":
    mst_edges_pole_ids = None
    gdf_edges_for_pf = validation_inputs.gdf_edges_4326
    edge_u_col = validation_inputs.edge_u_col
    edge_v_col = validation_inputs.edge_v_col
    if gdf_edges_for_pf is None:
        raise ValueError("External mode requires edges file; gdf_edges_4326 is None.")
    if edge_u_col is None or edge_v_col is None:
        raise ValueError("External mode requires detected edge endpoint columns (u_col, v_col).")
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

from core.powerflow_network import RUNNER_VERSION  # noqa: E402

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

line_params_ui = render_line_params_section()
line_types_df = None
lines_meta_df = None
default_line_type = validation_inputs.default_line_type
resolved_line_params_df = None

try:
    preview_edges_df = runner._build_edges(min_len_km=0.0).copy()
except Exception as e:
    preview_edges_df = None
    st.warning(f"Could not build line-parameter preview from the current topology: {e}")

if preview_edges_df is not None:
    if line_params_ui["mode"] != "global" and validation_inputs.mode == "external" and not _edges_have_explicit_line_id(validation_inputs):
        st.error(
            "Catalog-based line parameters require an external edges file with a 'line_id' column. "
            "Either add line_id to the edges input or use Global mode."
        )
    else:
        try:
            if line_params_ui["mode"] != "global":
                if line_params_ui["line_types_file"] is None:
                    raise ValueError("line_types.csv is required in catalog-based modes.")
                line_types_df = read_line_types_csv(line_params_ui["line_types_file"])
                line_type_options = line_types_df["line_type"].astype(str).tolist()
                if not line_type_options:
                    raise ValueError("line_types.csv does not contain any valid line types.")
                default_idx = 0
                if default_line_type in line_type_options:
                    default_idx = line_type_options.index(str(default_line_type))
                default_line_type = st.selectbox(
                    "Default line type (used when a line has no metadata row)",
                    options=line_type_options,
                    index=default_idx,
                    key="pf_default_line_type",
                )
                if line_params_ui["lines_meta_file"] is not None:
                    lines_meta_df = read_lines_metadata_csv(line_params_ui["lines_meta_file"])

            resolved_line_params_df = build_line_params_for_edges(
                preview_edges_df,
                mode=str(line_params_ui["mode"]),
                default_params={
                    "r_ohm_per_km": float(line_params_ui["r_ohm_per_km"]),
                    "x_ohm_per_km": float(line_params_ui["x_ohm_per_km"]),
                    "s_nom_kva": float(line_params_ui["s_nom_kva"]),
                },
                default_line_type=default_line_type,
                line_types_df=line_types_df,
                lines_meta_df=lines_meta_df,
            )
            with st.expander("Preview final merged line parameters (top 20)", expanded=False):
                preview_cols = [
                    c
                    for c in [
                        "line_id",
                        "u",
                        "v",
                        "line_type",
                        "length_km",
                        "r_ohm_per_km",
                        "x_ohm_per_km",
                        "s_nom_kva",
                    ]
                    if c in resolved_line_params_df.columns
                ]
                preview_df = resolved_line_params_df.sort_values("length_km", ascending=False).head(20)
                st.dataframe(preview_df[preview_cols], use_container_width=True)
        except Exception as e:
            st.error(f"Line parameter setup failed: {repr(e)}")

validation_inputs = update_validation_pf_settings(
    st.session_state,
    slack_pole_id=int(validation_inputs.slack_pole_id),
    v_min_pu=float(validation_inputs.v_min_pu),
    v_max_pu=float(validation_inputs.v_max_pu),
    pf_load=float(validation_inputs.pf_load),
    v_nom_kv=float(validation_inputs.v_nom_kv),
    v_base_mode=str(validation_inputs.v_base_mode),
    r_ohm_per_km=float(line_params_ui["r_ohm_per_km"]),
    x_ohm_per_km=float(line_params_ui["x_ohm_per_km"]),
    s_nom_kva=float(line_params_ui["s_nom_kva"]),
)
validation_inputs = update_validation_line_params_state(
    st.session_state,
    line_params_mode=str(line_params_ui["mode"]),
    default_line_type=default_line_type,
    line_types_df=line_types_df,
    lines_meta_df=lines_meta_df,
    resolved_line_params_df=resolved_line_params_df,
)
validation_inputs = get_validation_inputs(st.session_state)

if validation_inputs.line_params_mode != "global" and validation_inputs.resolved_line_params_df is None:
    st.info("Provide a valid line catalog configuration before running power flow.")
    st.stop()

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

run_controls = render_pf_run_controls(
    runner=runner,
    runner_version=RUNNER_VERSION,
    hour_min=int(validation_inputs.pole_loads_kW.index.min()),
    hour_max=int(validation_inputs.pole_loads_kW.index.max()),
    selected_hour=int(validation_inputs.pole_loads_kW.sum(axis=1).idxmax()),
)

if run_controls["run_clicked"]:
    try:
        pf_hour = int(run_controls["pf_hour"])
        pole_load_at_hour = validation_inputs.pole_loads_kW.loc[pf_hour]
        pole_load_dict = {int(k): float(v) for k, v in pole_load_at_hour.to_dict().items()}
        with st.spinner(f"Running power flow for hour {pf_hour}..."):
            out = runner.run_snapshot(
                pole_p_kw=pole_load_dict,
                params=params,
                line_params_df=validation_inputs.resolved_line_params_df,
                debug=True,
                check_nonsense=bool(run_controls["pf_fail_on_nonsense"]),
                min_len_km=float(run_controls["pf_min_len_m"]) / 1000.0,
                sn_mva=float(run_controls["pf_sn_mva"]),
            )

        set_validation_result(
            st.session_state,
            validation_result_from_runner_output(
                hour=pf_hour,
                params=params.__dict__,
                out=out,
            ),
        )
        st.session_state["ui"]["flags"]["last_pf_run_context"] = {
            "pf_hour": int(pf_hour),
            "pf_min_len_m": float(run_controls["pf_min_len_m"]),
            "pf_sn_mva": float(run_controls["pf_sn_mva"]),
            "pf_fail_on_nonsense": bool(run_controls["pf_fail_on_nonsense"]),
            "resolved_line_params_df": (
                None
                if validation_inputs.resolved_line_params_df is None
                else validation_inputs.resolved_line_params_df.copy()
            ),
        }

        st.success("Power flow completed.")
    except Exception as e:
        st.error(f"Power flow failed: {repr(e)}")
        st.exception(e)

pf_result = get_validation_result(st.session_state)
pf_result_view = validation_result_to_view_payload(pf_result)
pf_map = None
if pf_result_view is not None:
    bus_v_pu: dict[int, float] = {}
    for row in pf_result_view["bus_results"].to_dict(orient="records"):
        pid = pd.to_numeric(row.get("bus"), errors="coerce")
        vpu = pd.to_numeric(row.get("v_pu"), errors="coerce")
        if pd.notna(pid) and pd.notna(vpu):
            bus_v_pu[int(pid)] = float(vpu)

    line_loading_pu: dict[tuple[int, int], float] = {}
    for row in pf_result_view["line_results"].to_dict(orient="records"):
        u = pd.to_numeric(row.get("bus0"), errors="coerce")
        v = pd.to_numeric(row.get("bus1"), errors="coerce")
        loading = pd.to_numeric(row.get("loading_pu"), errors="coerce")
        if pd.notna(u) and pd.notna(v) and pd.notna(loading):
            line_loading_pu[(int(u), int(v))] = float(loading)

    edges_for_map = validation_inputs.gdf_edges_4326
    if edges_for_map is None:
        edges_for_map = _build_session_edges_for_pf_map(
            gdf_nodes_4326=gdf_nodes_4326,
            pole_id_col=pole_col,
            mst_edges_pole_ids=validation_inputs.mst_edges_pole_ids,
        )

    pf_map = {
        "center": validation_inputs.center,
        "gdf_poles_4326": gdf_nodes_4326,
        "pole_id_col": pole_col,
        "gdf_edges_4326": edges_for_map,
        "mst_edges_latlon": validation_inputs.mst_edges_latlon,
        "gdf_roads_4326": validation_inputs.gdf_roads_4326,
        "slack_pole_id": int(validation_inputs.slack_pole_id),
        "bus_v_pu": bus_v_pu,
        "line_loading_pu": line_loading_pu,
        "v_min_pu": float(validation_inputs.v_min_pu),
        "v_max_pu": float(validation_inputs.v_max_pu),
    }

render_pf_results(
    pf_result_view,
    pf_map=pf_map,
)

st.divider()
st.info(
    "If line overloads occur, the reinforcement optimization can increase line capacities while keeping the topology fixed.\n\n"
    "Go to the **Grid Reinforcement** page to run the optimization."
)
