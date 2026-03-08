from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.powerflow_io import make_map_lv_with_load_bubbles, make_map_lv_with_pf_violations


def render_page_header() -> None:
    with st.sidebar:
        st.markdown("**Example data**: see the `examples/` folder in this project.")

    st.title("Grid Validation (Power Flow)")
    st.markdown(
        """
This page implements the **minimum electrical validation step** for an LV topology:

1) select a topology source (reuse Page 1 results *or* load external files)  
2) load/assign hourly building demands and aggregate to poles  
3) choose line-parameter assumptions (global or catalog-driven)  
4) run a single-snapshot power flow with a single slack/generation bus  
"""
    )
    st.divider()


def render_topology_source_section() -> tuple[str, Optional[Any], Optional[Any], Optional[Any]]:
    st.subheader("Grid Topology")

    topology_source = st.radio(
        "Choose how to provide topology data",
        options=("Use results from Grid Topology (Page 1)", "Load external files (stand-alone)"),
        index=1,
        help="Both options are supported. External files let you use this page as a stand-alone PF tool.",
    )

    nodes_file: Optional[Any] = None
    edges_file: Optional[Any] = None
    assoc_file: Optional[Any] = None

    if not topology_source.startswith("Use results"):
        st.markdown("Upload the **topology outputs** (nodes + edges + associations).")

        c1, c2 = st.columns(2)
        with c1:
            nodes_file = st.file_uploader(
                "Nodes (poles) file (.geojson/.gpkg)",
                type=["geojson", "json", "gpkg"],
                key="pf_nodes_file",
            )
        with c2:
            edges_file = st.file_uploader(
                "Edges (LV network) file (.geojson/.gpkg)",
                type=["geojson", "json", "gpkg"],
                key="pf_edges_file",
            )

        assoc_file = st.file_uploader(
            "Associations CSV (building_id, pole_id)",
            type=["csv"],
            key="pf_assoc_file",
        )

    return topology_source, nodes_file, edges_file, assoc_file


def render_demand_upload_section() -> tuple[Optional[Any], Optional[Any]]:
    c1, c2 = st.columns(2)
    with c1:
        meta_file = st.file_uploader(
            "Building metadata CSV (building_id, category, weight optional)",
            type=["csv"],
            key="pf_building_meta_file",
        )
    with c2:
        profiles_file = st.file_uploader(
            "Category profiles CSV (hour + one column per category, kW per building)",
            type=["csv"],
            key="pf_category_profiles_file",
        )

    return meta_file, profiles_file


def render_demand_controls(pole_loads_kW: pd.DataFrame) -> Dict[str, Any]:
    col_1, col_2 = st.columns([2.5, 1])
    with col_2:
        scaling_mode = st.selectbox(
            "Bubble scaling mode",
            options=("Absolute (fixed over the year)", "Relative (rescaled each hour)"),
            index=0,
            key="pf_scaling_mode_dropdown",
        )

    year_max_pole_kW = float(np.nanmax(pole_loads_kW.to_numpy())) if pole_loads_kW.size else 0.0
    pmax_ref_kW = year_max_pole_kW if scaling_mode.startswith("Absolute") else None

    with col_1:
        total_load = pole_loads_kW.sum(axis=1)
        peak_hour = int(total_load.idxmax())
        hour = st.slider(
            "Select hour for visualization",
            min_value=int(pole_loads_kW.index.min()),
            max_value=int(pole_loads_kW.index.max()),
            value=int(peak_hour),
            step=1,
            key="pf_vis_hour_slider",
        )

    return {
        "scaling_mode": scaling_mode,
        "year_max_pole_kW": year_max_pole_kW,
        "pmax_ref_kW": pmax_ref_kW,
        "hour": int(hour),
    }


def render_load_visualization(
    *,
    vis: Optional[Dict[str, Any]],
    gdf_nodes_4326,
    slack_pole_id: Optional[int],
) -> None:
    if vis is None:
        st.info("No saved visualization yet. Upload building metadata + category profiles to create it.")
        return

    st.markdown(f"**Aggregated pole load mapping** (hour {vis['hour']})")

    pole_col = str(vis["pole_col"])
    pole_ids = (
        pd.to_numeric(gdf_nodes_4326[pole_col], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )

    load_dict = vis.get("pole_load_dict", {}) or {}
    default_pid = None
    if load_dict:
        try:
            default_pid = int(max(load_dict.items(), key=lambda kv: float(kv[1]))[0])
        except Exception:
            default_pid = None

    highlight_enabled = st.checkbox(
        "Highlight a pole on the map",
        value=False,
        key="pf_highlight_enabled",
    )
    highlight_pole_id = st.selectbox(
        "Pole to highlight",
        options=[None] + pole_ids,
        index=(0 if default_pid is None else (pole_ids.index(default_pid) + 1)) if pole_ids else 0,
        format_func=lambda x: "None" if x is None else f"Pole {x}",
        disabled=not highlight_enabled,
        key="pf_highlight_pole_id",
    )

    m = make_map_lv_with_load_bubbles(
        center=tuple(vis["center"]),
        gdf_poles_4326=gdf_nodes_4326,
        pole_id_col=pole_col,
        pole_load_kW_at_hour=vis["pole_load_dict"],
        mst_edges_latlon=vis.get("mst_edges_latlon"),
        gdf_edges_4326=vis.get("gdf_edges_4326"),
        gdf_roads_4326=vis.get("gdf_roads_4326"),
        zoom_start=15,
        pmax_ref_kW=vis.get("pmax_ref_kW"),
        show_legend=True,
        slack_pole_id=slack_pole_id,
        highlight_pole_id=(int(highlight_pole_id) if highlight_pole_id is not None else None),
        zoom_to_highlight=bool(highlight_enabled and highlight_pole_id is not None),
    )

    map_key = (
        f"pf_map_{int(vis['hour'])}_"
        f"{'abs' if vis.get('pmax_ref_kW') is not None else 'rel'}_"
        f"hl_{highlight_pole_id}"
    )
    st_folium(m, height=650, use_container_width=True, key=map_key)

    with st.expander("Preview aggregated pole loads (top 20)", expanded=False):
        df_preview = (
            pd.Series(vis["pole_load_dict"])
            .sort_values(ascending=False)
            .head(20)
            .reset_index()
        )
        df_preview.columns = ["pole_id", "p_kW"]
        st.dataframe(df_preview, use_container_width=True)

    if vis.get("resolved_line_params_df") is not None:
        with st.expander("Preview final merged line parameters (top 20)", expanded=False):
            st.dataframe(vis["resolved_line_params_df"].head(20), use_container_width=True)


def render_pf_setup_section(*, pole_ids: list[int], suggested_slack: int) -> Dict[str, Any]:
    st.divider()
    st.subheader("Power flow setup")
    st.markdown(
        "Select the **plant / slack pole** and the **minimum electrical assumptions** used to build the network."
    )

    slack_pole_id = st.selectbox(
        "Slack / plant connection pole (pole_id)",
        options=pole_ids,
        index=pole_ids.index(suggested_slack),
        key="pf_slack_pole_dropdown",
    )
    st.caption(f"Suggested based on load-weighted centroid (current map hour): pole_id = {suggested_slack}")

    p1, p2, p3 = st.columns(3)
    with p1:
        v_min_pu = st.number_input("Min voltage limit (p.u.)", 0.70, 1.00, 0.90, 0.01, format="%.2f", key="pf_v_min_pu")
    with p2:
        v_max_pu = st.number_input("Max voltage limit (p.u.)", 1.00, 1.30, 1.10, 0.01, format="%.2f", key="pf_v_max_pu")
    with p3:
        pf_load = st.number_input("Assumed load power factor (lagging)", 0.50, 1.00, 0.95, 0.01, format="%.2f", key="pf_pf_load")

    v_base_mode = st.selectbox(
        "Voltage base convention",
        options=("3-phase LV (0.4 kV line-to-line)", "Per-phase equivalent (0.230 kV L-N)"),
        index=0,
        key="pf_vbase_mode",
    )

    return {
        "slack_pole_id": int(slack_pole_id),
        "v_min_pu": float(v_min_pu),
        "v_max_pu": float(v_max_pu),
        "pf_load": float(pf_load),
        "v_base_mode": v_base_mode,
    }


def render_line_params_section() -> Dict[str, Any]:
    with st.expander("Line parameters", expanded=True):
        st.markdown(
            "Choose how line electrical characteristics are assigned. Global mode preserves the current behavior."
        )

        mode_label = st.radio(
            "Line parameter mode",
            options=(
                "Global default cable type",
                "Catalog-based assignment",
                "Catalog + per-line overrides",
            ),
            key="pf_line_params_mode",
        )
        mode = {
            "Global default cable type": "global",
            "Catalog-based assignment": "catalog",
            "Catalog + per-line overrides": "catalog_overrides",
        }[mode_label]

        col1, col2, col3 = st.columns(3)
        with col1:
            r_ohm_per_km = st.number_input(
                "Line resistance R (ohm/km)",
                min_value=0.0001,
                max_value=5.0,
                value=0.642,
                step=0.01,
                format="%.4f",
                key="pf_line_r_ohm_per_km",
            )
        with col2:
            x_ohm_per_km = st.number_input(
                "Line reactance X (ohm/km)",
                min_value=0.0001,
                max_value=5.0,
                value=0.083,
                step=0.01,
                format="%.4f",
                key="pf_line_x_ohm_per_km",
            )
        with col3:
            s_nom_kva = st.number_input(
                "Thermal capacity S_nom (kVA)",
                min_value=1.0,
                max_value=1000.0,
                value=100.0,
                step=5.0,
                key="pf_line_s_nom_kva",
            )

        line_types_file = None
        lines_meta_file = None
        if mode != "global":
            c1, c2 = st.columns(2)
            with c1:
                line_types_file = st.file_uploader("line_types.csv", type=["csv"], key="pf_line_types_file")
            with c2:
                lines_meta_file = st.file_uploader(
                    "lines_metadata.csv (optional)",
                    type=["csv"],
                    key="pf_lines_meta_file",
                )
            st.caption("Missing line metadata rows fall back to the selected default line type.")

        return {
            "mode": mode,
            "r_ohm_per_km": float(r_ohm_per_km),
            "x_ohm_per_km": float(x_ohm_per_km),
            "s_nom_kva": float(s_nom_kva),
            "line_types_file": line_types_file,
            "lines_meta_file": lines_meta_file,
        }


def render_pf_run_controls(
    *,
    runner,
    runner_version: str,
    hour_min: int,
    hour_max: int,
    selected_hour: int,
) -> Dict[str, Any]:
    st.divider()
    st.subheader("Run Power Flow (PyPSA)")

    cdbg1, cdbg2, cdbg3 = st.columns(3)
    with cdbg1:
        pf_min_len_m = st.number_input("PF: min edge length filter (m)", 0.0, 50.0, 5.0, 1.0, key="pf_min_len_m")
    with cdbg2:
        pf_sn_mva = st.number_input("PF: system base (sn_mva)", 0.01, 100.0, 0.1, 0.1, key="pf_sn_mva")
    with cdbg3:
        pf_fail_on_nonsense = st.checkbox(
            "PF: fail if voltages non-physical",
            value=True,
            key="pf_fail_on_nonsense",
        )

    with st.expander("Runner identity (sanity check)", expanded=False):
        st.write("RUNNER_VERSION:", runner_version)
        st.write("runner type:", type(runner))
        st.write("runner module:", getattr(type(runner), "__module__", None))
        st.write("run_snapshot defaults:", getattr(runner.run_snapshot, "__defaults__", None))

    st.markdown(
        "Choose the exact demand hour used for the electrical check. This is independent from the load-bubble map hour."
    )
    st.caption(f"Recommended hour: {int(selected_hour)} (highest total aggregated load).")
    pf_hour = st.slider(
        "Hour to run power flow",
        min_value=int(hour_min),
        max_value=int(hour_max),
        value=int(selected_hour),
        step=1,
        key="pf_run_hour_slider",
    )

    run_clicked = st.button("Run power flow for selected hour", type="primary", key="pf_run_pf_btn")

    return {
        "pf_min_len_m": float(pf_min_len_m),
        "pf_sn_mva": float(pf_sn_mva),
        "pf_fail_on_nonsense": bool(pf_fail_on_nonsense),
        "pf_hour": int(pf_hour),
        "run_clicked": bool(run_clicked),
    }


def render_pf_results(
    res: Optional[Dict[str, Any]],
    pf_map: Optional[Dict[str, Any]] = None,
) -> None:
    if res is None:
        return

    st.markdown(f"**Snapshot hour:** `{res['hour']}`")
    st.json(res["summary"])

    with st.expander("PF debug (quick checks)", expanded=False):
        st.json(res.get("debug", {}) or {})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Bus voltages**")
        st.dataframe(res["bus_results"], use_container_width=True)
    with c2:
        st.markdown("**Line loading / flows**")
        st.dataframe(res["line_results"], use_container_width=True)

    if pf_map is not None:
        st.markdown("**Network status map**")
        m = make_map_lv_with_pf_violations(
            center=tuple(pf_map["center"]),
            gdf_poles_4326=pf_map["gdf_poles_4326"],
            pole_id_col=str(pf_map["pole_id_col"]),
            gdf_edges_4326=pf_map.get("gdf_edges_4326"),
            mst_edges_latlon=pf_map.get("mst_edges_latlon"),
            gdf_roads_4326=pf_map.get("gdf_roads_4326"),
            zoom_start=15,
            slack_pole_id=pf_map.get("slack_pole_id"),
            bus_v_pu=pf_map.get("bus_v_pu"),
            line_loading_pu=pf_map.get("line_loading_pu"),
            v_min_pu=float(pf_map["v_min_pu"]),
            v_max_pu=float(pf_map["v_max_pu"]),
            line_loading_limit_pu=1.0,
            show_legend=True,
        )
        st_folium(m, height=650, use_container_width=True, key=f"pf_results_map_{int(res['hour'])}")

    if res["summary"]["num_voltage_violations"] > 0:
        st.warning("Voltage violations detected.")
    else:
        st.success("No voltage violations for this snapshot.")
