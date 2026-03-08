from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
from shapely.geometry import LineString, MultiPolygon, Polygon

PolygonLike = Polygon | MultiPolygon


@dataclass
class PreparedInputs:
    """Validated, projected, point-normalized inputs."""

    community_gdf: gpd.GeoDataFrame
    households_gdf: gpd.GeoDataFrame
    polygon: PolygonLike
    target_crs: object


@dataclass
class CandidatePoleSet:
    """Candidate poles for one sub-area and one trunk segment."""

    poles_gdf: gpd.GeoDataFrame
    trunk_pole_ids: set[int]
    angle_width: float
    angle_length: float
    trunk_line: LineString
    subarea_polygon: PolygonLike


@dataclass
class EngineResult:
    """Top-level in-memory outputs of the trunk-first pipeline."""

    trunks_gdf: gpd.GeoDataFrame
    secondary_gdf: gpd.GeoDataFrame
    service_gdf: gpd.GeoDataFrame
    poles_gdf: gpd.GeoDataFrame
    total_lines_gdf: gpd.GeoDataFrame
    subareas_gdf: gpd.GeoDataFrame
