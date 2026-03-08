from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .config import EngineConfig
from .partition import build_subareas_gdf, subdivide_by_trunk_voronoi
from .poles import assign_households_to_poles, create_candidate_poles
from .routing import build_biased_secondary_network
from .trunk import build_trunks_gdf, create_trunk_segments, simplify_trunk_segments
from .types import EngineResult
from .validation import prepare_inputs


def _empty_lines_gdf(target_crs: object, line_type: str) -> gpd.GeoDataFrame:
    """Build an empty line GeoDataFrame with a standard schema."""
    return gpd.GeoDataFrame(
        columns=["segment_id", "Type", "Length", "geometry"],
        geometry="geometry",
        crs=target_crs,
    )


def _empty_poles_gdf(target_crs: object) -> gpd.GeoDataFrame:
    """Build an empty pole GeoDataFrame with a standard schema."""
    return gpd.GeoDataFrame(
        columns=["segment_id", "pole_uid", "pole_id", "is_trunk", "geometry"],
        geometry="geometry",
        crs=target_crs,
    )


def run_trunk_first_pipeline(
    community_geometry,
    households_gdf: gpd.GeoDataFrame,
    source_crs: object | None,
    target_crs: object,
    config: EngineConfig | None = None,
) -> EngineResult:
    """
    Run the full trunk-first mini-grid distribution layout workflow.

    Parameters
    ----------
    community_geometry:
        Settlement polygon (Polygon or MultiPolygon).
    households_gdf:
        Household points or polygons.
    source_crs:
        CRS of `community_geometry` if it is a bare Shapely geometry.
    target_crs:
        Projected CRS in meters.
    config:
        Engine configuration.

    Returns
    -------
    EngineResult
        Trunk, secondary, service, poles, total lines, and subareas.
    """
    config = config or EngineConfig()

    prepared = prepare_inputs(
        community_geometry=community_geometry,
        households_gdf=households_gdf,
        source_crs=source_crs,
        target_crs=target_crs,
        config=config.validation,
    )

    raw_trunks = create_trunk_segments(prepared.polygon, config=config.trunk)
    trunks, _ = simplify_trunk_segments(raw_trunks, config=config.trunk)
    trunks_gdf = build_trunks_gdf(trunks, target_crs=prepared.target_crs)

    subareas = subdivide_by_trunk_voronoi(
        trunks=trunks,
        polygon=prepared.polygon,
        spacing=config.trunk.subarea_spacing,
    )
    if len(subareas) != len(trunks):
        raise ValueError("Subarea count does not match trunk segment count.")

    subareas_gdf = build_subareas_gdf(subareas, target_crs=prepared.target_crs)

    secondary_parts: list[gpd.GeoDataFrame] = []
    service_parts: list[gpd.GeoDataFrame] = []
    selected_poles_parts: list[gpd.GeoDataFrame] = []

    for segment_id, (trunk_line, subarea_polygon) in enumerate(zip(trunks, subareas)):
        subarea_mask = gpd.GeoDataFrame({"geometry": [subarea_polygon]}, crs=prepared.target_crs)
        households_in_subarea = gpd.clip(prepared.households_gdf, subarea_mask)

        pole_set = create_candidate_poles(
            subarea_polygon=subarea_polygon,
            trunk_line=trunk_line,
            config=config.poles,
        )
        pole_set.poles_gdf.set_crs(prepared.target_crs, inplace=True)

        assigned_pole_ids, service_gdf = assign_households_to_poles(
            poles_gdf=pole_set.poles_gdf,
            households_gdf=households_in_subarea,
            point_precision=config.validation.point_precision,
        )

        required_pole_ids = set(assigned_pole_ids) | set(pole_set.trunk_pole_ids)

        secondary_gdf, used_pole_ids = build_biased_secondary_network(
            poles_gdf=pole_set.poles_gdf,
            trunk_pole_ids=pole_set.trunk_pole_ids,
            required_pole_ids=required_pole_ids,
            angle_width=pole_set.angle_width,
            angle_length=pole_set.angle_length,
            pole_spacing=config.poles.pole_spacing,
            config=config.routing,
        )

        if not service_gdf.empty:
            service_gdf = service_gdf.copy()
            service_gdf["segment_id"] = segment_id
            service_gdf["Type"] = "Service Line"
            service_parts.append(service_gdf)

        if not secondary_gdf.empty:
            secondary_gdf = secondary_gdf.copy()
            secondary_gdf["segment_id"] = segment_id
            secondary_gdf["Type"] = "Secondary Line"
            secondary_parts.append(secondary_gdf)

        keep_ids = set(used_pole_ids) | set(assigned_pole_ids) | set(pole_set.trunk_pole_ids)
        poles_kept = pole_set.poles_gdf[pole_set.poles_gdf["pole_id"].isin(keep_ids)].copy()
        poles_kept["segment_id"] = segment_id
        poles_kept["pole_uid"] = poles_kept.apply(lambda row: f"{segment_id}:{int(row['pole_id'])}", axis=1)
        selected_poles_parts.append(poles_kept[["segment_id", "pole_uid", "pole_id", "is_trunk", "geometry"]])

    if secondary_parts:
        secondary_gdf = gpd.GeoDataFrame(pd.concat(secondary_parts, ignore_index=True), geometry="geometry", crs=prepared.target_crs)
        secondary_gdf["Length"] = secondary_gdf.geometry.length
    else:
        secondary_gdf = _empty_lines_gdf(prepared.target_crs, "Secondary Line")

    if service_parts:
        service_gdf = gpd.GeoDataFrame(pd.concat(service_parts, ignore_index=True), geometry="geometry", crs=prepared.target_crs)
        service_gdf["Length"] = service_gdf.geometry.length
    else:
        service_gdf = _empty_lines_gdf(prepared.target_crs, "Service Line")

    trunks_gdf = trunks_gdf.copy()
    trunks_gdf["segment_id"] = trunks_gdf["segment_id"].astype(int)

    if selected_poles_parts:
        poles_gdf = gpd.GeoDataFrame(pd.concat(selected_poles_parts, ignore_index=True), geometry="geometry", crs=prepared.target_crs)
    else:
        poles_gdf = _empty_poles_gdf(prepared.target_crs)

    total_lines_gdf = gpd.GeoDataFrame(
        pd.concat(
            [
                trunks_gdf[["segment_id", "Type", "Length", "geometry"]],
                secondary_gdf[["segment_id", "Type", "Length", "geometry"]],
                service_gdf[["segment_id", "Type", "Length", "geometry"]],
            ],
            ignore_index=True,
        ),
        geometry="geometry",
        crs=prepared.target_crs,
    )

    return EngineResult(
        trunks_gdf=trunks_gdf,
        secondary_gdf=secondary_gdf,
        service_gdf=service_gdf,
        poles_gdf=poles_gdf,
        total_lines_gdf=total_lines_gdf,
        subareas_gdf=subareas_gdf,
    )
