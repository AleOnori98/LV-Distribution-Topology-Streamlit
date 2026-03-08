from __future__ import annotations

import streamlit as st

from core.pipeline_state import ensure_session_domains


st.set_page_config(
    page_title="Mini-Grid LV Toolkit",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_session_domains(st.session_state)

st.title("Mini-Grid LV Toolkit")
st.markdown(
    """This app provides a streamlined workflow for low-voltage (LV) mini-grid design and screening, built
    for rapid iteration and early-stage exploration. The app is organized into three main steps: topology generation,
    electrical validation, and grid reinforcement optimization. The workflow is designed to be flexible, allowing users
    to move back and forth between steps while maintaining session state as the internal source of truth.
    The app is built on a heuristic MST-based layout for fast topology generation, with an integrated power-flow validation step that supports both global line assumptions and catalog-driven per-line characteristics. The reinforcement optimization
    focuses on minimum-cost line capacity upgrades without rerouting or load shifting, providing a practical tool for early-stage mini-grid design and screening.
    """
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
<div class="card">
    <div class="pill">Step 1</div>
    <h4>Grid Topology</h4>
    <p>
        Generate an LV pole-and-line layout from building points and optional roads using the current
        MST-based heuristic pipeline.
    </p>
    <p>
        The topology step places candidate poles, associates buildings, connects poles through a
        Minimum Spanning Tree, and prepares exportable GeoJSON and CSV outputs.
    </p>
    <p>
        Main outputs: poles, lines, building-to-pole associations, and summary metrics.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
<div class="card">
    <div class="pill">Step 2</div>
    <h4>Grid Validation</h4>
    <p>
        Reuse topology directly from session state or load external topology files, then aggregate
        hourly building demand to poles and run an electrical validation snapshot.
    </p>
    <p>
        Validation supports global line parameters or catalog-driven per-line electrical characteristics,
        including optional line-level overrides.
    </p>
    <p>
        Main outputs: bus voltages, line loading, result maps, and expanded line-flow exports.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
<div class="card">
    <div class="pill">Step 3</div>
    <h4>Grid Reinforcement</h4>
    <p>
        Keep topology and load snapshot fixed, then optimize line-capacity reinforcement on existing edges
        to reduce overloads at minimum upgrade cost.
    </p>
    <p>
        The optimization does not reroute the network, does not move loads, and does not add decentralized generation.
    </p>
    <p>
        Main outputs: reinforced lines, added capacity, estimated reinforcement cost, and post-upgrade PF checks.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("### Workflow")
st.markdown(
    """
1. Open **Grid Topology** in the sidebar and run the LV layout from buildings and optional roads.
2. Open **Grid Validation** to aggregate hourly loads, choose the slack bus, set line assumptions, and run the electrical check.
3. In **Grid Validation**, run **Step 3: Grid reinforcement optimization** to compute minimum-cost line capacity upgrades on fixed topology.
4. Export GeoJSON and CSV outputs when needed. Session state remains the internal source of truth while you move across pages.
"""
)

st.markdown("### What Is Included")
f1, f2, f3 = st.columns(3)
f1.info("Topology generation with MST-based LV layout and downloadable network outputs.")
f2.info("Demand aggregation from `building_metadata.csv` and `category_profiles.csv` into hourly pole loads.")
f3.info("Power-flow validation with voltage checks, line loading, map overlays, and optional per-line cable catalogs.")

st.markdown(
    """
<div class="note">
    Use the sidebar to move through the workflow. For a quick start, baseline sample inputs are available in <code>examples/</code>.
</div>
""",
    unsafe_allow_html=True,
)
