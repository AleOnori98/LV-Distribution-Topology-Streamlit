from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.powerflow_io import make_map_lv_with_pf_violations


def render_page_header() -> None:
    with st.sidebar:
        st.markdown("**Example data**: see the `examples/` folder in this project.")

    st.title("Grid Reinforcement")
    st.markdown(
        """
This page runs **fixed-topology reinforcement optimization** after power-flow validation.

- Topology, load allocation, and slack bus are kept fixed.
- Optimization can increase line thermal capacities on existing edges.
- Results include pre/post PF comparison, reinforced lines, map overlays, and CSV export.
"""
    )
    st.divider()


def render_infeasibility_guidance(*, prefix: str = "Reinforcement optimization may be infeasible.") -> None:
    st.warning(
        prefix
        + "\n\nPossible reasons:\n"
        "- **Max upgrade factor is too low** for one or more critical lines.\n"
        "- **Very strict electrical assumptions** (high R/X, low nominal voltage base, low PF, tight limits).\n"
        "- **Edge filtering removed key segments** (high PF min-edge-length setting).\n"
        "- **Cost/constraints combination** prevents required upgrades.\n"
        "- **Voltage issues persist** because this step upgrades thermal capacity only (impedance unchanged).\n\n"
        "Try increasing max upgrade factor, reducing PF min edge filter, using a larger sn_mva, "
        "or temporarily enabling emergency load shedding for diagnostics."
    )


def render_reinforcement_params(*, default_enabled: bool) -> Dict[str, Any]:
    st.subheader("Reinforcement parameters")

    run_even_if_clean = st.checkbox(
        "Allow optimization even when PF has no violations",
        value=bool(default_enabled),
        key="p3_rf_run_even_if_clean",
    )

    mode_label = st.selectbox(
        "Reinforcement scope",
        options=(
            "All lines",
            "Only currently overloaded lines",
            "Only feeder paths to voltage-violating buses",
        ),
        index=0,
        key="p3_rf_selection_mode",
    )
    selection_mode = {
        "All lines": "all_lines",
        "Only currently overloaded lines": "overloaded_only",
        "Only feeder paths to voltage-violating buses": "violating_feeder_path",
    }[mode_label]

    c1, c2 = st.columns(2)
    with c1:
        cost_per_km_per_kva = st.number_input(
            "Cost fallback [currency / (km*kVA)]",
            min_value=0.0001,
            max_value=100.0,
            value=0.08,
            step=0.01,
            key="p3_rf_cost_per_km_per_kva",
        )
    with c2:
        max_upgrade_factor = st.number_input(
            "Max upgrade factor (x current capacity)",
            min_value=1.0,
            max_value=50.0,
            value=4.0,
            step=0.5,
            key="p3_rf_max_upgrade_factor",
        )

    emergency_shed = st.checkbox(
        "Enable emergency load shedding (very high penalty)",
        value=False,
        key="p3_rf_allow_shedding",
    )
    shedding_penalty = st.number_input(
        "Shedding penalty [currency/MWh]",
        min_value=1000.0,
        max_value=1_000_000_000.0,
        value=100000.0,
        step=1000.0,
        key="p3_rf_shedding_penalty",
        disabled=not emergency_shed,
    )

    solver_name = st.text_input(
        "Solver name (optional, leave blank for PyPSA default)",
        value="",
        key="p3_rf_solver_name",
    ).strip()

    run_clicked = st.button("Optimize grid reinforcement", type="primary", key="p3_rf_run_btn")

    return {
        "run_even_if_clean": bool(run_even_if_clean),
        "selection_mode": selection_mode,
        "cost_per_km_per_kva": float(cost_per_km_per_kva),
        "max_upgrade_factor": float(max_upgrade_factor),
        "allow_emergency_load_shedding": bool(emergency_shed),
        "shedding_penalty_per_mwh": float(shedding_penalty),
        "solver_name": (solver_name if solver_name else None),
        "run_clicked": bool(run_clicked),
    }


def render_reinforcement_results(res: Optional[Dict[str, Any]], *, pf_map: Optional[Dict[str, Any]] = None) -> None:
    if res is None:
        st.info("No reinforcement result yet. Configure parameters and run optimization.")
        return

    st.divider()
    st.subheader("Results")

    status_txt = str(res.get("optimization_status", "")).lower()
    if ("infeasible" in status_txt) or ("unbounded" in status_txt):
        render_infeasibility_guidance(prefix=f"Solver status: `{res['optimization_status']}`.")
        st.markdown("**Optimization debug**")
        st.json(res.get("optimize_debug", {}) or {})
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimization status", str(res["optimization_status"]))
    c2.metric("Added capacity [kVA]", f"{float(res['total_added_capacity_kva']):.2f}")
    c3.metric("Estimated reinforcement cost", f"{float(res['total_reinforcement_cost']):.2f}")

    pre = dict(res["pre_summary"])
    post = dict(res["post_summary"])
    cmp_df = pd.DataFrame(
        [
            {"metric": "PF buses", "pre": pre.get("num_buses"), "post": post.get("num_buses")},
            {"metric": "PF lines", "pre": pre.get("num_lines"), "post": post.get("num_lines")},
            {"metric": "Voltage violations", "pre": pre.get("num_voltage_violations"), "post": post.get("num_voltage_violations")},
            {"metric": "Min voltage [p.u.]", "pre": pre.get("v_min_pu_observed"), "post": post.get("v_min_pu_observed")},
            {"metric": "Max voltage [p.u.]", "pre": pre.get("v_max_pu_observed"), "post": post.get("v_max_pu_observed")},
            {"metric": "Worst line loading [p.u.]", "pre": pre.get("max_line_loading_pu"), "post": post.get("max_line_loading_pu")},
        ]
    )
    st.dataframe(cmp_df, use_container_width=True)

    st.markdown("**Reinforced lines**")
    st.dataframe(res["reinforced_lines"], use_container_width=True)

    csv_bytes = res["reinforced_lines"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download reinforced lines CSV",
        data=csv_bytes,
        file_name=f"reinforced_lines_hour_{int(res['hour'])}.csv",
        mime="text/csv",
        key="p3_rf_download_csv",
    )

    st.markdown("**Optimization debug**")
    st.json(res.get("optimize_debug", {}) or {})

    if pf_map is not None:
        st.markdown("**Post-optimization network status**")
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
            reinforced_line_pairs=pf_map.get("reinforced_line_pairs"),
            v_min_pu=float(pf_map["v_min_pu"]),
            v_max_pu=float(pf_map["v_max_pu"]),
            line_loading_limit_pu=1.0,
            show_legend=True,
        )
        st_folium(m, height=650, use_container_width=True, key=f"p3_rf_results_map_{int(res['hour'])}")
