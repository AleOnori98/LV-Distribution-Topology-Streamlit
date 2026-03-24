from __future__ import annotations

import time

import streamlit as st

from config.settings import (
    PathManager,
    DEFAULT_SAMPLING_DISTANCE_M,
    DEFAULT_USER_DISTANCE_M,
    DEFAULT_MAX_ASSOCIATIONS,
)
from core.pipeline_adapters import topology_result_from_legacy_payload, topology_result_to_view_payload
from core.pipeline_state import (
    clear_topology,
    ensure_session_domains,
    get_topology_result,
    get_topology_solve_seconds,
    set_project_request,
    set_topology_result,
)
from core.distribution_service import run_low_voltage
from pages.ui_sections.topology_sections import (
    render_action_buttons,
    render_intro_section,
    render_params_section,
    render_results_section,
    render_sidebar,
    render_upload_section,
)


render_sidebar()


def main() -> None:
    ensure_session_domains(st.session_state)

    methodology_img = PathManager.ASSETS / "distribution_methodology.png"
    render_intro_section(methodology_img)

    users_file, follow_roads_mode, roads_file = render_upload_section()
    params = render_params_section(
        default_sampling_distance_m=int(DEFAULT_SAMPLING_DISTANCE_M),
        default_user_distance_m=int(DEFAULT_USER_DISTANCE_M),
        default_max_associations=int(DEFAULT_MAX_ASSOCIATIONS),
    )
    run_clicked, clear_clicked = render_action_buttons()

    set_project_request(
        st.session_state,
        topology_request={
            "has_users_file": users_file is not None,
            "has_roads_file": roads_file is not None,
            "follow_roads_mode": follow_roads_mode,
            "params": dict(params),
        },
    )

    if clear_clicked:
        clear_topology(st.session_state, clear_validation=True)
        st.success("Previous results cleared.")

    if run_clicked:
        if users_file is None:
            st.error("Please upload a **users file** before running the LV design.")
        elif follow_roads_mode.startswith("Follow roads") and roads_file is None:
            st.error(
                "You selected **Follow roads** but didn't upload a roads file. "
                "Either upload a `.gpkg` or switch to **Free placement**."
            )
        else:
            with st.spinner("Running LV distribution design..."):
                t0 = time.perf_counter()
                try:
                    results = run_low_voltage(
                        users_file=users_file,
                        roads_file=roads_file,
                        sampling_distance=float(params["road_pole_spacing_m"]),
                        user_distance=float(params["max_user_connection_radius_m"]),
                        max_associations=int(params["max_users_per_pole"]),
                        allow_unserved_isolated=bool(params["allow_unserved_isolated"]),
                        min_cluster_size=int(params["min_cluster_size"]),
                        max_pole_span_m=float(params["max_pole_span_m"]),
                    )
                except Exception as exc:
                    st.error(f"LV design failed: {exc}")
                    return

                elapsed = time.perf_counter() - t0
                topology_result = topology_result_from_legacy_payload(results)
                set_topology_result(st.session_state, topology_result, elapsed)

            st.success(f"Computation completed in {elapsed:.2f} seconds.")

    render_results_section(
        topology_result_to_view_payload(get_topology_result(st.session_state)),
        get_topology_solve_seconds(st.session_state),
    )


if __name__ == "__main__":
    main()
