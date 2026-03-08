from __future__ import annotations
import io
import os
import tempfile
from typing import Optional, Tuple, Iterable, Dict

import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium import Map, PolyLine, CircleMarker
from branca.element import MacroElement, Template


def read_vector(uploaded_file) -> gpd.GeoDataFrame:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    data = uploaded_file.getvalue()

    if suffix in [".geojson", ".json"]:
        return gpd.read_file(io.BytesIO(data))

    if suffix == ".gpkg":
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return gpd.read_file(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    raise ValueError("Unsupported file format.")


def find_column(df, *candidates: str) -> Optional[str]:
    cols_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        k = cand.strip().lower()
        if k in cols_map:
            return cols_map[k]
    return None


def read_building_metadata_csv(uploaded_file) -> pd.DataFrame:
    """
    Expected columns:
      - building_id
      - category
      - weight (optional, default=1.0)
    """
    df = pd.read_csv(uploaded_file)

    bcol = find_column(df, "building_id", "building")
    ccol = find_column(df, "category", "user_category", "type")
    wcol = find_column(df, "weight", "multiplier", "scale")

    if bcol is None or ccol is None:
        raise ValueError("building_metadata.csv must include columns: building_id and category (weight optional).")

    cols = {bcol: "building_id", ccol: "category"}
    if wcol is not None:
        cols[wcol] = "weight"
    out = df[[bcol, ccol] + ([wcol] if wcol else [])].rename(columns=cols)
    if wcol is None:
        out["weight"] = 1.0

    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0)
    out["building_id"] = out["building_id"].astype(str)
    out["category"] = out["category"].astype(str)

    return out


def read_category_profiles_csv(uploaded_file) -> pd.DataFrame:
    """
    Expected format (wide):
      hour, CAT_A, CAT_B, ...  (values in kW per building of that category)

    Returns a DataFrame indexed by hour (int), columns = categories (str), values = kW (float).
    """
    df = pd.read_csv(uploaded_file)

    hcol = find_column(df, "hour", "t", "time", "snapshot")
    if hcol is None:
        raise ValueError("category_profiles.csv must include an 'hour' column.")

    hours = pd.to_numeric(df[hcol], errors="coerce")
    bad = hours.isna()
    if bad.any():
        raise ValueError(f"category_profiles.csv has non-numeric hour entries (showing up to 10 rows): {df.loc[bad].head(10)}")

    df = df.copy()
    df[hcol] = hours.astype(int)

    # Drop hour column -> all remaining are categories
    cats = [c for c in df.columns if c != hcol]
    if len(cats) == 0:
        raise ValueError("category_profiles.csv must have at least one category column besides 'hour'.")

    out = df.set_index(hcol)[cats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out.index.name = "hour"

    # Optional: basic sanity
    if out.index.duplicated().any():
        raise ValueError("category_profiles.csv contains duplicated hour values. Hours must be unique.")

    return out


def make_map_lv_with_load_bubbles(
    *,
    center: Tuple[float, float],
    gdf_poles_4326: gpd.GeoDataFrame,
    pole_id_col: str,
    pole_load_kW_at_hour,  # dict[int,float] preferred (or Series)
    mst_edges_latlon: Optional[Iterable[tuple[tuple[float, float], tuple[float, float]]]] = None,
    gdf_edges_4326: Optional[gpd.GeoDataFrame] = None,
    gdf_roads_4326: Optional[gpd.GeoDataFrame] = None,
    zoom_start: int = 15,
    max_bubble_radius: float = 18.0,
    min_bubble_radius: float = 3.0,
    pmax_ref_kW: Optional[float] = None,
    show_legend: bool = True,
    slack_pole_id: Optional[int] = None,
    highlight_pole_id: Optional[int] = None,          # NEW
    zoom_to_highlight: bool = False,                  # NEW
) -> Map:
    """
    Folium map with:
      - OSM basemap
      - optional roads layer (gray)
      - LV network edges (blue)
      - poles as black dots (ALWAYS with pole_id tooltip)
      - load bubbles (orange) at poles (tooltip includes pole_id + load)

    Scaling:
      - If pmax_ref_kW is None -> RELATIVE scaling (per hour): pmax = max(p_kW at this hour)
      - If pmax_ref_kW is provided -> ABSOLUTE scaling (fixed): pmax = pmax_ref_kW
    """

    # -------------------------------------------------------
    # 1) Normalize loads to dict[int, float]
    # -------------------------------------------------------
    load_dict: Dict[int, float] = {}
    if pole_load_kW_at_hour is not None:
        if isinstance(pole_load_kW_at_hour, pd.Series):
            load_dict = {int(k): float(v) for k, v in pole_load_kW_at_hour.to_dict().items() if v is not None}
        else:
            load_dict = {int(k): float(v) for k, v in dict(pole_load_kW_at_hour).items() if v is not None}

    # -------------------------------------------------------
    # 2) Base map
    # -------------------------------------------------------
    m = folium.Map(
        location=[float(center[0]), float(center[1])],
        zoom_start=int(zoom_start),
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # -------------------------------------------------------
    # 3) Roads layer
    # -------------------------------------------------------
    if gdf_roads_4326 is not None and not gdf_roads_4326.empty:
        for _, row in gdf_roads_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in geom.coords]
                PolyLine(coords, color="gray", weight=3, opacity=0.6).add_to(m)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = [(lat, lon) for lon, lat in line.coords]
                    PolyLine(coords, color="gray", weight=3, opacity=0.6).add_to(m)

    # -------------------------------------------------------
    # 4) LV edges
    # -------------------------------------------------------
    edge_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []

    if mst_edges_latlon is not None:
        edge_pairs = list(mst_edges_latlon)
    elif gdf_edges_4326 is not None and not gdf_edges_4326.empty:
        for _, row in gdf_edges_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in geom.coords]
                for i in range(len(coords) - 1):
                    edge_pairs.append((coords[i], coords[i + 1]))
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = [(lat, lon) for lon, lat in line.coords]
                    for i in range(len(coords) - 1):
                        edge_pairs.append((coords[i], coords[i + 1]))

    for (lat1, lon1), (lat2, lon2) in edge_pairs:
        PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color="blue",
            weight=2,
            opacity=0.9,
        ).add_to(m)

    # -------------------------------------------------------
    # 5) Prepare poles (pole_id + coords + load)
    # -------------------------------------------------------
    if pole_id_col not in gdf_poles_4326.columns:
        raise ValueError(f"Pole id column '{pole_id_col}' not found in poles GeoDataFrame.")

    gdfp = gdf_poles_4326.copy()
    gdfp["pole_id"] = pd.to_numeric(gdfp[pole_id_col], errors="coerce")
    gdfp = gdfp.dropna(subset=["pole_id"]).copy()
    gdfp["pole_id"] = gdfp["pole_id"].astype(int)

    # Load (0 if missing)
    gdfp["p_kW"] = gdfp["pole_id"].map(load_dict).fillna(0.0).astype(float)

    def _safe_latlon(geom):
        if geom is None or geom.is_empty:
            return None
        pt = geom if geom.geom_type == "Point" else geom.representative_point()
        return (float(pt.y), float(pt.x))

    latlons = gdfp.geometry.apply(_safe_latlon)
    gdfp["lat"] = [x[0] if x else np.nan for x in latlons]
    gdfp["lon"] = [x[1] if x else np.nan for x in latlons]
    gdfp = gdfp.dropna(subset=["lat", "lon"]).copy()

    # Optional: re-center map on highlighted pole
    if zoom_to_highlight and highlight_pole_id is not None:
        sel = gdfp.loc[gdfp["pole_id"] == int(highlight_pole_id)]
        if not sel.empty:
            center = (float(sel.iloc[0]["lat"]), float(sel.iloc[0]["lon"]))
            # Re-create map centered here
            m.location = [center[0], center[1]]

    # -------------------------------------------------------
    # 6) Scaling reference
    # -------------------------------------------------------
    p_values = gdfp["p_kW"].to_numpy(dtype=float)
    pmax_hour = float(np.nanmax(p_values)) if len(p_values) else 0.0
    pmax = float(pmax_ref_kW) if pmax_ref_kW is not None else float(pmax_hour)

    # -------------------------------------------------------
    # 7) Draw poles (ALWAYS with pole_id tooltip)
    #    Also highlight slack pole and an optional "highlight pole".
    # -------------------------------------------------------
    for _, r in gdfp.iterrows():
        pid = int(r["pole_id"])
        is_slack = (slack_pole_id is not None and pid == int(slack_pole_id))
        is_hl = (highlight_pole_id is not None and pid == int(highlight_pole_id))

        # base style
        base_radius = 2.5
        base_color = "black"

        # slack style
        if is_slack:
            base_radius = 4.5
            base_color = "purple"

        # draw base pole marker
        CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=base_radius,
            color=base_color,
            fill=True,
            fill_color=base_color,
            fill_opacity=1.0,
            tooltip=f"Pole {pid}" + (" (SLACK / PLANT)" if is_slack else ""),
        ).add_to(m)

        # highlight overlay (high-contrast ring) if requested
        if is_hl:
            # outer ring (cyan)
            CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=11.0,
                color="#00FFFF",      # cyan
                weight=5,
                fill=False,
                opacity=1.0,
                tooltip=f"Pole {pid} (HIGHLIGHTED)",
            ).add_to(m)

            # inner dot (white) to avoid blending with orange bubble
            CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=3.2,
                color="white",
                weight=2,
                fill=True,
                fill_color="white",
                fill_opacity=1.0,
                opacity=1.0,
            ).add_to(m)

    # -------------------------------------------------------
    # 8) Load bubbles (tooltip includes pole_id + load)
    # -------------------------------------------------------
    if pmax > 0:
        for _, r in gdfp.iterrows():
            pkW = float(r["p_kW"])
            if pkW <= 0:
                continue

            radius = min_bubble_radius + (pkW / pmax) * (max_bubble_radius - min_bubble_radius)
            radius = float(np.clip(radius, min_bubble_radius, max_bubble_radius))

            CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=radius,
                color="#ff7f0e",
                fill=True,
                fill_color="#ff7f0e",
                fill_opacity=0.35,
                opacity=0.8,
                tooltip=f"Pole {int(r['pole_id'])} — {pkW:.2f} kW",
            ).add_to(m)

    # -------------------------------------------------------
    # 9) Small fixed legend (pure HTML)
    # -------------------------------------------------------
    if show_legend:
        scale_txt = "ABS (year-fixed)" if pmax_ref_kW is not None else "REL (per-hour)"
        pmax_txt = f"{pmax:.2f} kW" if pmax > 0 else "0 kW"

        html = f"""
        {{% macro html(this, kwargs) %}}
        <div style="
            position: fixed;
            bottom: 30px; left: 30px;
            z-index: 9999;
            background: white;
            border: 1px solid #999;
            border-radius: 6px;
            padding: 10px 12px;
            font-size: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        ">
          <div style="font-weight: 600; margin-bottom: 6px;">Map legend</div>
          <div><span style="display:inline-block;width:10px;height:10px;background:black;border-radius:50%;margin-right:6px;"></span>Pole (hover to see ID)</div>
          <div><span style="display:inline-block;width:10px;height:10px;background:purple;border-radius:50%;margin-right:6px;"></span>Slack / plant pole</div>
          <div><span style="display:inline-block;width:10px;height:10px;background:#ff7f0e;border-radius:50%;margin-right:6px;opacity:0.7;"></span>Load bubble</div>
          <div style="margin-top:6px;">Scaling: <b>{scale_txt}</b></div>
          <div>Reference max: <b>{pmax_txt}</b></div>
        </div>
        {{% endmacro %}}
        """
        macro = MacroElement()
        macro._template = Template(html)
        m.get_root().add_child(macro)

    return m

def _infer_edge_uv_cols(gdf_edges: gpd.GeoDataFrame) -> tuple[str, str]:
    """Try common edge endpoint column names."""
    cols = {c.lower(): c for c in gdf_edges.columns}
    u = cols.get("u") or cols.get("from") or cols.get("bus0") or cols.get("source")
    v = cols.get("v") or cols.get("to")   or cols.get("bus1") or cols.get("target")
    if u is None or v is None:
        raise ValueError(
            "Cannot infer edge endpoint columns. Expected one of: "
            "u/v, from/to, bus0/bus1, source/target."
        )
    return u, v


def make_map_lv_with_pf_violations(
    *,
    center: Tuple[float, float],
    gdf_poles_4326: gpd.GeoDataFrame,
    pole_id_col: str,
    gdf_edges_4326: Optional[gpd.GeoDataFrame] = None,
    mst_edges_latlon: Optional[Iterable[tuple[tuple[float, float], tuple[float, float]]]] = None,
    gdf_roads_4326: Optional[gpd.GeoDataFrame] = None,
    zoom_start: int = 15,
    slack_pole_id: Optional[int] = None,
    # PF results (recommended)
    bus_v_pu: Optional[Dict[int, float]] = None,
    line_loading_pu: Optional[Dict[Tuple[int, int], float]] = None,  # (u,v) -> loading in p.u.
    reinforced_line_pairs: Optional[set[Tuple[int, int]]] = None,
    v_min_pu: float = 0.90,
    v_max_pu: float = 1.10,
    line_loading_limit_pu: float = 1.00,  # 1.0 = 100%
    show_legend: bool = True,
) -> Map:
    """
    Map:
      - OSM basemap
      - roads (gray, optional)
      - edges colored by loading violation (red if > limit, else blue/gray)
      - poles colored by voltage violation (red if outside [v_min,v_max], else black)
      - tooltips ALWAYS show pole_id; if voltage available, show v_pu as well
    """

    bus_v_pu = bus_v_pu or {}
    line_loading_pu = line_loading_pu or {}
    reinforced_line_pairs = reinforced_line_pairs or set()

    m = folium.Map(
        location=[float(center[0]), float(center[1])],
        zoom_start=int(zoom_start),
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # -----------------------------
    # Roads layer
    # -----------------------------
    if gdf_roads_4326 is not None and not gdf_roads_4326.empty:
        for _, row in gdf_roads_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in geom.coords]
                PolyLine(coords, color="gray", weight=3, opacity=0.6).add_to(m)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = [(lat, lon) for lon, lat in line.coords]
                    PolyLine(coords, color="gray", weight=3, opacity=0.6).add_to(m)

    # -----------------------------
    # Poles dataframe with coords
    # -----------------------------
    if pole_id_col not in gdf_poles_4326.columns:
        raise ValueError(f"Pole id column '{pole_id_col}' not found in poles GeoDataFrame.")

    gdfp = gdf_poles_4326.copy()
    gdfp["_pid"] = pd.to_numeric(gdfp[pole_id_col], errors="coerce")
    gdfp = gdfp.dropna(subset=["_pid"]).copy()
    gdfp["_pid"] = gdfp["_pid"].astype(int)

    # representative point coords
    pts = gdfp.geometry.apply(lambda geom: geom if geom.geom_type == "Point" else geom.representative_point())
    gdfp["_lat"] = pts.y.astype(float)
    gdfp["_lon"] = pts.x.astype(float)

    # voltage attach
    gdfp["_vpu"] = gdfp["_pid"].map(bus_v_pu)

    # -----------------------------
    # Edges layer (prefer gdf_edges_4326 if available, else mst_edges_latlon)
    # -----------------------------
    if gdf_edges_4326 is not None and not gdf_edges_4326.empty:
        u_col, v_col = _infer_edge_uv_cols(gdf_edges_4326)

        def _loading_color(loading: float) -> str:
            if not np.isfinite(loading):
                return "#5B6B7A"
            ratio = float(np.clip(loading / max(line_loading_limit_pu, 1e-9), 0.0, 2.0))
            if ratio <= 1.0:
                g = int(round(166 + (1.0 - ratio) * 54))
                r = int(round(40 + ratio * 180))
                b = int(round(74 - ratio * 34))
                return f"#{r:02X}{g:02X}{b:02X}"
            excess = min(1.0, ratio - 1.0)
            r = 220
            g = int(round(166 - excess * 116))
            b = int(round(40 - excess * 24))
            return f"#{r:02X}{g:02X}{b:02X}"

        for _, row in gdf_edges_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            u = int(pd.to_numeric(row[u_col], errors="coerce"))
            v = int(pd.to_numeric(row[v_col], errors="coerce"))
            key = (u, v) if (u, v) in line_loading_pu else (v, u)
            is_reinforced = (min(u, v), max(u, v)) in reinforced_line_pairs

            loading = float(line_loading_pu.get(key, np.nan))
            is_viol = (np.isfinite(loading) and loading > line_loading_limit_pu)

            if np.isfinite(loading):
                ratio = max(0.0, loading / max(line_loading_limit_pu, 1e-9))
                weight = 2.5 + 2.0 * min(1.0, ratio) + 4.0 * max(0.0, min(1.0, ratio - 1.0))
            else:
                weight = 2.5

            color = "#FF8C00" if is_reinforced else _loading_color(loading)
            opacity = 0.95 if is_viol else 0.85
            dash_array = None if not is_viol else "10, 8"

            def _draw_linestring(ls):
                coords = [(lat, lon) for lon, lat in ls.coords]
                tip = f"Line {u}-{v}"
                if np.isfinite(loading):
                    tip += f" — loading: {100*loading:.1f}%"
                if is_viol:
                    tip += " (OVERLOADED)"
                if is_reinforced:
                    tip += " (REINFORCED)"
                PolyLine(
                    coords,
                    color=color,
                    weight=weight,
                    opacity=opacity,
                    tooltip=tip,
                    dash_array=dash_array,
                ).add_to(m)

            if geom.geom_type == "LineString":
                _draw_linestring(geom)
            elif geom.geom_type == "MultiLineString":
                for ls in geom.geoms:
                    _draw_linestring(ls)

    elif mst_edges_latlon is not None:
        # fallback: no IDs, so just draw in blue (no violation logic possible)
        for (lat1, lon1), (lat2, lon2) in list(mst_edges_latlon):
            PolyLine([(lat1, lon1), (lat2, lon2)], color="blue", weight=2, opacity=0.8).add_to(m)

    # -----------------------------
    # Poles layer: color by voltage violation
    # -----------------------------
    for _, r in gdfp.iterrows():
        pid = int(r["_pid"])
        vpu = r["_vpu"]
        is_slack = slack_pole_id is not None and pid == int(slack_pole_id)

        violates_v = False
        if vpu is not None and np.isfinite(vpu):
            violates_v = (float(vpu) < float(v_min_pu)) or (float(vpu) > float(v_max_pu))

        # priority: slack highlight > violation > normal
        if is_slack:
            color = "purple"
            rad = 5.0
        elif violates_v:
            color = "red"
            rad = 4.5
        else:
            color = "black"
            rad = 2.8

        tip = f"Pole {pid}"
        if vpu is not None and np.isfinite(vpu):
            tip += f" — V={float(vpu):.3f} p.u."

        CircleMarker(
            location=[float(r["_lat"]), float(r["_lon"])],
            radius=rad,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            tooltip=tip,
        ).add_to(m)

    # -----------------------------
    # Legend
    # -----------------------------
    if show_legend:
        html = f"""
        {{% macro html(this, kwargs) %}}
        <div style="
            position: fixed;
            bottom: 30px; left: 30px;
            z-index: 9999;
            background: white;
            border: 1px solid #999;
            border-radius: 6px;
            padding: 10px 12px;
            font-size: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        ">
          <div style="font-weight: 600; margin-bottom: 6px;">PF violations legend</div>
          <div><span style="display:inline-block;width:10px;height:10px;background:black;border-radius:50%;margin-right:6px;"></span>Bus OK</div>
          <div><span style="display:inline-block;width:10px;height:10px;background:red;border-radius:50%;margin-right:6px;"></span>Bus voltage violated</div>
          <div><span style="display:inline-block;width:10px;height:10px;background:purple;border-radius:50%;margin-right:6px;"></span>Slack / plant bus</div>
          <div style="margin-top:6px;"><span style="display:inline-block;width:14px;height:3px;background:#28A745;margin-right:6px;"></span>Lightly loaded</div>
          <div><span style="display:inline-block;width:14px;height:3px;background:#DCA628;margin-right:6px;"></span>Near limit</div>
          <div><span style="display:inline-block;width:14px;height:3px;background:#DC3228;margin-right:6px;"></span>Overloaded (thicker + dashed = worse)</div>
          <div style="margin-top:6px;">
            Voltage limits: <b>[{v_min_pu:.2f}, {v_max_pu:.2f}]</b> p.u. &nbsp;|&nbsp;
            Loading limit: <b>{100*line_loading_limit_pu:.0f}%</b>
          </div>
        </div>
        {{% endmacro %}}
        """
        macro = MacroElement()
        macro._template = Template(html)
        m.get_root().add_child(macro)

    return m
