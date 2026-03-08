from __future__ import annotations

import warnings
from typing import Iterable

import geopandas as gpd
from shapely.geometry import MultiPolygon, Point, Polygon

from .config import ValidationConfig
from .types import PolygonLike, PreparedInputs


def _fix_geometry(geometry: PolygonLike, enabled: bool) -> PolygonLike:
    """Fix invalid polygonal geometry with a conservative buffer(0) fallback."""
    if not enabled:
        return geometry
    if geometry.is_valid:
        return geometry
    fixed = geometry.buffer(0)
    if fixed.is_empty:
        raise ValueError("Geometry became empty after attempted validity fix.")
    if not isinstance(fixed, (Polygon, MultiPolygon)):
        raise ValueError("Fixed geometry is not polygonal.")
    warnings.warn("Input polygon was invalid; applied buffer(0) fix.", stacklevel=2)
    return fixed


def _ensure_household_crs(households_gdf: gpd.GeoDataFrame, source_crs: object | None) -> gpd.GeoDataFrame:
    """Ensure the household layer has a CRS."""
    if households_gdf.crs is not None:
        return households_gdf.copy()
    if source_crs is None:
        raise ValueError("households_gdf has no CRS and source_crs was not provided.")
    out = households_gdf.copy()
    out.set_crs(source_crs, inplace=True)
    return out


def _convert_geometries_to_points(
    gdf: gpd.GeoDataFrame,
    use_representative_points: bool,
) -> gpd.GeoDataFrame:
    """Convert non-point geometries to points after projection."""
    out = gdf.copy()
    if out.empty:
        return out

    geom_types = set(out.geometry.geom_type.unique())
    if geom_types.issubset({"Point"}):
        return out

    if use_representative_points:
        out["geometry"] = out.geometry.apply(
            lambda geom: geom if isinstance(geom, Point) else geom.representative_point()
        )
    else:
        out["geometry"] = out.geometry.apply(
            lambda geom: geom if isinstance(geom, Point) else geom.centroid
        )
    return out


def prepare_inputs(
    community_geometry: PolygonLike,
    households_gdf: gpd.GeoDataFrame,
    source_crs: object | None,
    target_crs: object,
    config: ValidationConfig,
) -> PreparedInputs:
    """
    Validate and project the settlement polygon and household points.

    Parameters
    ----------
    community_geometry:
        Input settlement geometry in source CRS.
    households_gdf:
        Household layer, as points or polygons.
    source_crs:
        CRS of `community_geometry` when it is a bare Shapely geometry.
    target_crs:
        Projected CRS in meters.
    config:
        Validation options.

    Returns
    -------
    PreparedInputs
        Projected polygon and point-normalized households.
    """
    if not isinstance(community_geometry, (Polygon, MultiPolygon)):
        raise TypeError("community_geometry must be a Polygon or MultiPolygon.")

    fixed_geometry = _fix_geometry(community_geometry, enabled=config.fix_invalid_geometries)
    households = _ensure_household_crs(households_gdf, source_crs=source_crs)

    polygon_crs = source_crs if source_crs is not None else households.crs
    if polygon_crs is None:
        raise ValueError("A source CRS is required for the community polygon.")

    community_gdf = gpd.GeoDataFrame({"geometry": [fixed_geometry]}, crs=polygon_crs)
    community_gdf = community_gdf.to_crs(target_crs)
    households = households.to_crs(target_crs)

    households = _convert_geometries_to_points(
        households,
        use_representative_points=config.use_representative_points,
    )

    households = households.loc[~households.geometry.is_empty & households.geometry.notna()].copy()
    if households.empty:
        warnings.warn("No valid household points remain after projection and conversion.", stacklevel=2)

    if not set(households.geometry.geom_type.unique()).issubset({"Point"}):
        raise ValueError("Household geometries must be Points after normalization.")

    polygon = community_gdf.geometry.iloc[0]
    if polygon.is_empty:
        raise ValueError("Projected community polygon is empty.")

    return PreparedInputs(
        community_gdf=community_gdf,
        households_gdf=households,
        polygon=polygon,
        target_crs=target_crs,
    )
