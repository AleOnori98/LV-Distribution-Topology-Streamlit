from __future__ import annotations

import time
from typing import Any, Dict, Optional

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import Map, PolyLine, CircleMarker

from config.settings import (
    PathManager,
    DEFAULT_COST_PER_KM_LV,
    DEFAULT_FIXED_COSTS_LV,
    DEFAULT_SAMPLING_DISTANCE_M,
    DEFAULT_USER_DISTANCE_M,
    DEFAULT_MAX_ASSOCIATIONS,
)
from core.distribution_service import run_low_voltage


# ---------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="LV Distribution Network",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# UI helper functions
# ---------------------------------------------------------------------
def _metric_row(metrics: Dict[str, float]) -> None:

    # ---- Row 1: Network lengths -------------------------------------
    st.text(
        "Estimated total line length of the LV system, split between the main backbone "
        "(pole-to-pole feeder network) and the final connections from poles to individual buildings."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total network length [km]", f"{metrics.get('total_network_length_km', 0.0):.2f}")
    c2.metric("LV backbone length [km]", f"{metrics.get('backbone_length_km', 0.0):.2f}")
    c3.metric("Service drop length [km]", f"{metrics.get('service_drop_length_km', 0.0):.2f}")

    # ---- Row 2: Poles breakdown -------------------------------------
    st.text(
        "Number of poles required for the LV network. Serving poles supply buildings directly, "
        "while support poles are added only to limit span lengths and do not host connections."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total poles", int(metrics.get("num_poles_total", 0)))
    c2.metric("Serving poles", int(metrics.get("num_poles_serving", 0)))
    c3.metric("Support poles", int(metrics.get("num_poles_support", 0)))

    # ---- Row 3: Buildings breakdown ---------------------------------
    st.text(
        "Coverage of the settlement by the LV network. Standalone candidates are buildings "
        "left unconnected due to isolation or clustering constraints."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total buildings", int(metrics.get("num_buildings", 0)))
    c2.metric("Grid-served buildings", int(metrics.get("num_served", 0)))
    c3.metric("Standalone candidates", int(metrics.get("num_unserved", 0)))

def _geom_to_latlon(g):
    """Return (lat, lon) for any shapely geometry."""
    if g is None or g.is_empty:
        return None
    if g.geom_type == "Point":
        p = g
    else:
        # Works for Polygon, MultiPolygon, LineString, MultiLineString, etc.
        p = g.representative_point()
    return (p.y, p.x)  # (lat, lon)

def _make_map_lv(
    center: tuple[float, float],
    gdf_served,
    gdf_unserved,
    gdf_poles,
    mst_edges_latlon,
    gdf_roads=None,
) -> Map:
    """
    Build the interactive LV map.

    - Roads: grey lines
    - MST edges: blue-ish lines
    - Poles: black markers
    - Grid-served buildings: green markers
    - Standalone candidates: red markers
    """
    m = folium.Map(location=[center[0], center[1]], zoom_start=15)

    # Roads
    if gdf_roads is not None and not gdf_roads.empty:
        for _, row in gdf_roads.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in geom.coords]
                PolyLine(locations=coords, color="gray", weight=3, opacity=0.7).add_to(m)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = [(lat, lon) for lon, lat in line.coords]
                    PolyLine(locations=coords, color="gray", weight=3, opacity=0.7).add_to(m)

    # MST edges
    for (lat1, lon1), (lat2, lon2) in mst_edges_latlon:
        PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color="blue",
            weight=2,
            opacity=0.9,
        ).add_to(m)

    # Poles
    if gdf_poles is not None and not gdf_poles.empty:
        for _, row in gdf_poles.iterrows():
            latlon = _geom_to_latlon(row.geometry)
            if latlon is None:
                continue
            y, x = latlon
            CircleMarker(
                location=[y, x],
                radius=3,
                color="black",
                fill=True,
                fill_color="black",
                fill_opacity=1.0,
            ).add_to(m)

    # Grid-served buildings (green)
    if gdf_served is not None and not gdf_served.empty:
        for _, row in gdf_served.iterrows():
            latlon = _geom_to_latlon(row.geometry)
            if latlon is None:
                continue
            y, x = latlon
            CircleMarker(
                location=[y, x],
                radius=2,
                color="green",
                fill=True,
                fill_color="green",
                fill_opacity=0.8,
            ).add_to(m)

    # Standalone candidates / unserved (red)
    if gdf_unserved is not None and not gdf_unserved.empty:
        for _, row in gdf_unserved.iterrows():
            latlon = _geom_to_latlon(row.geometry)
            if latlon is None:
                continue
            y, x = latlon
            CircleMarker(
                location=[y, x],
                radius=3,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.9,
            ).add_to(m)

    return m


def _render_downloads(downloads: Dict[str, Any]) -> None:
    if not downloads:
        return

    nodes = downloads.get("nodes_geojson")
    edges = downloads.get("edges_geojson")

    if not nodes and not edges:
        return

    st.subheader("Download outputs")

    st.text(
        "Export the estimated LV backbone topology as GeoJSON files for use in GIS software. "
        "Nodes represent pole locations (including support poles), while edges represent the "
        "pole-to-pole feeder network. Service drops to individual buildings are not included."
    )

    if nodes:
        st.download_button(
            "Download poles (nodes) GeoJSON",
            data=nodes,
            file_name="lv_poles.geojson",
            mime="application/geo+json",
        )

    if edges:
        st.download_button(
            "Download LV network (edges) GeoJSON",
            data=edges,
            file_name="lv_network.geojson",
            mime="application/geo+json",
        )



def _show_lv_results(results: Dict[str, Any]) -> None:
    _metric_row(results["metrics"])

    st.subheader("Distribution map")
    st.caption(
        "Green = grid-served buildings; Red = standalone candidates; "
        "Black = poles; Blue = LV network (MST)."
    )

    m = _make_map_lv(
        center=results["center"],
        gdf_served=results.get("gdf_served_4326"),
        gdf_unserved=results.get("gdf_unserved_4326"),
        gdf_poles=results.get("gdf_poles_4326"),
        mst_edges_latlon=results.get("mst_edges_latlon", []),
        gdf_roads=results.get("gdf_roads_4326"),
    )
    st_folium(m, height=600, width=900)

    _render_downloads(results.get("downloads", {}))


# ---------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------

def main() -> None:
    # Session state init
    if "dist_results" not in st.session_state:
        st.session_state["dist_results"] = None
        st.session_state["dist_solve_seconds"] = None

    # Sidebar instructions
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            """
            1. **Upload users file** (required)  
            2. (Optional) upload **roads file** if you want poles to follow roads  
            3. Adjust **cost** and **heuristic** parameters  
            4. Choose whether to **allow isolated buildings to remain unserved**  
            5. Click **Run LV design**  
            6. Inspect the map and download GeoJSON outputs
            """
        )
        st.markdown("---")
        st.markdown("**Example data**: see the `examples/` folder in this project.")

    # Title + description
    st.title("LV Distribution Network - Topology Assessment")
    st.markdown(
        """
        This app provides a **first-order topology assessment** for Low-Voltage (LV) distribution networks, starting from
        building connection points and, optionally, a road layout. The goal is not detailed engineering design, but to
        quickly explore how **settlement geometry and simple planning rules** translate into pole count, network length,
        and cost.

        The workflow has two main steps. First, **pole locations are generated and buildings are associated to poles**
        using a heuristic approach. When roads are available, candidate poles are sampled along road segments; otherwise,
        poles are placed inside the settlement based on clustering of nearby buildings. This step is guided by a few
        intuitive parameters:

        - **Max user–pole connection radius**, limiting how far a building can be from a pole  
        - **Max users per pole**, limiting how many buildings share the same pole  
        - **Road pole spacing**, controlling how dense candidate poles are along roads  

        Once poles are defined, they are connected into a **single radial LV network** by computing a
        **Minimum Spanning Tree (MST)**, which minimizes total cable length while avoiding loops.

        An optional **engineering post-processing** step can then be applied through the **Max LV span between poles**
        parameter. If any LV segment is longer than this threshold, it is subdivided by inserting intermediate
        support poles, preserving one connected network while avoiding unrealistically long spans.
    """
    )


    # Optional header image
    methodology_img = PathManager.ASSETS / "distribution_methodology.png"
    if methodology_img.exists():
        st.image(
            str(methodology_img),
            use_container_width=True,
            caption="Process flow for LV distribution network design",
        )

    st.markdown("---")

    # ---------------------------- Uploads ----------------------------
    st.subheader("Upload connection data")

    users_file = st.file_uploader(
        "Users file (GeoPackage `.gpkg` or Excel `.xlsx`) – **required**",
        type=["gpkg", "xlsx"],
        key="dist_users_file",
    )
    if users_file:
        st.success("Users file uploaded.")

    follow_roads_mode = st.radio(
        "Pole placement strategy",
        options=("Follow roads (requires roads file)", "Free placement"),
        index=0,
    )
    roads_file: Optional[Any] = None
    if follow_roads_mode.startswith("Follow roads"):
        roads_file = st.file_uploader(
            "Roads file (GeoPackage `.gpkg`)",
            type=["gpkg"],
            key="dist_roads_file",
        )
        if roads_file:
            st.success("Roads file uploaded.")
        else:
            st.info("If you don't have a roads file, switch to **Free placement**.")

    st.markdown("---")

    # ---------------------------- Parameters -------------------------
    st.subheader("Heuristic pole placement and customer association")

    col1, col2, col3 = st.columns(3)

    with col1:
        road_pole_spacing_m = st.slider(
            "Road pole spacing (candidate sampling) [m]",
            min_value=10,
            max_value=200,
            value=int(DEFAULT_SAMPLING_DISTANCE_M),
            step=5,
            help=(
                "Used ONLY when 'Follow roads' is selected. Candidate pole locations are "
                "generated by sampling points along road segments at this spacing. "
                "Smaller values → more candidate poles (denser), potentially shorter user–pole distances."
            ),
        )

    with col2:
        max_user_connection_radius_m = st.slider(
            "Max user–pole connection radius [m]",
            min_value=10,
            max_value=200,
            value=int(DEFAULT_USER_DISTANCE_M),
            step=5,
            help=(
                "Maximum distance allowed for connecting a building to an existing pole during association. "
                "Smaller values → more unassociated buildings, which may require additional poles (or become standalone candidates)."
            ),
        )

    with col3:
        max_users_per_pole = st.slider(
            "Max users per pole [#]",
            min_value=1,
            max_value=100,
            value=int(DEFAULT_MAX_ASSOCIATIONS),
            step=1,
            help=(
                "Upper limit on how many buildings can be assigned to the same pole. "
                "Smaller values → more poles to serve the same number of users."
            ),
        )

    st.subheader("Network span control (post-processing)")

    max_pole_span_m = st.slider(
        "Max LV span between poles (engineering cap) [m]",
        min_value=20,
        max_value=150,
        value=80,
        step=5,
        help=(
            "Applied AFTER the MST is computed. If an MST edge is longer than this value, "
            "the tool inserts intermediate support poles to subdivide the span. "
            "This keeps a single connected tree while limiting long unsupported LV segments."
        ),
    )

    st.markdown("---")

    # ------------------ Coverage behaviour (new toggle) --------------
    st.subheader("Coverage behaviour")

    allow_unserved_isolated = st.checkbox(
        "Allow very isolated buildings to remain unserved (standalone candidates)",
        value=False,
        help=(
            "If enabled, very small or isolated clusters of buildings will not be "
            "connected by LV. They will be shown as red points and can be treated "
            "as standalone system candidates."
        ),
    )

    if allow_unserved_isolated:
        min_cluster_size = st.slider(
            "Minimum cluster size for LV connection",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help=(
                "Clusters of unserved buildings smaller than this size will not get "
                "a new pole and will remain as standalone candidates."
            ),
        )
        st.info(
            "Selective coverage mode: only clusters with at least the chosen size "
            "are connected by LV. Smaller, isolated clusters are flagged as "
            "standalone candidates (red points)."
        )
    else:
        # In full-coverage mode we still pass a value, but it has no effect.
        min_cluster_size = 1
        st.info(
            "Full coverage mode: the heuristic will try to connect all buildings "
            "by LV. In very sparse areas this may lead to long LV extensions even "
            "if they exceed your intuitive distance thresholds."
        )

    st.markdown("---")

    # ---------------------------- Actions ----------------------------
    left, right, _ = st.columns([1, 1, 1])
    with left:
        run_clicked = st.button("Run LV design", type="primary", use_container_width=True)
    with right:
        clear_clicked = st.button("Clear results", use_container_width=True)

    if clear_clicked:
        st.session_state["dist_results"] = None
        st.session_state["dist_solve_seconds"] = None
        st.success("Previous results cleared.")

    # ---------------------- Run computation -------------------------
    if run_clicked:
        if users_file is None:
            st.error("Please upload a **users file** before running the LV design.")
        elif follow_roads_mode.startswith("Follow roads") and roads_file is None:
            st.error(
                "You selected **Follow roads** but didn't upload a roads file. "
                "Either upload a `.gpkg` or switch to **Free placement**."
            )
        else:
            with st.spinner("Running LV distribution design…"):
                t0 = time.perf_counter()
                try:
                    results = run_low_voltage(
                        users_file=users_file,
                        roads_file=roads_file,
                        sampling_distance=float(road_pole_spacing_m),
                        user_distance=float(max_user_connection_radius_m),
                        max_associations=int(max_users_per_pole),
                        allow_unserved_isolated=allow_unserved_isolated,
                        min_cluster_size=int(min_cluster_size),
                        max_pole_span_m=float(max_pole_span_m),
                    )
                except Exception as exc:
                    st.error(f"LV design failed: {exc}")
                    return

                elapsed = time.perf_counter() - t0
                st.session_state["dist_results"] = results
                st.session_state["dist_solve_seconds"] = elapsed

            st.success(f"Computation completed in {elapsed:.2f} seconds.")

    # ---------------------------- Results ---------------------------
    if st.session_state.get("dist_results") is not None:
        st.divider()
        st.header("Results")

        solve_time = st.session_state.get("dist_solve_seconds")
        if solve_time is not None:
            st.caption(f"Model run time: {solve_time:.2f} s")

        _show_lv_results(st.session_state["dist_results"])
    else:
        st.info("No results yet. Configure inputs and click **Run LV design**.")


if __name__ == "__main__":
    main()
