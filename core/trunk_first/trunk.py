from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import geopandas as gpd
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon
from shapely.ops import linemerge, split, substring, unary_union

from .config import TrunkConfig
from .types import PolygonLike


def _iter_polygons(geometry: PolygonLike) -> list[Polygon]:
    """Return polygon parts as a flat list."""
    if isinstance(geometry, Polygon):
        return [geometry]
    return list(geometry.geoms)


def _iter_rings(geometry: PolygonLike):
    """Yield exterior and interior rings from all polygon parts."""
    for poly in _iter_polygons(geometry):
        yield poly.exterior
        for ring in poly.interiors:
            yield ring


def _coord_key(x: float, y: float, precision: int = 6) -> tuple[float, float]:
    """Stable rounded coordinate key."""
    return (round(float(x), precision), round(float(y), precision))


def _segment_key(line: LineString, precision: int = 6) -> tuple[tuple[float, float], tuple[float, float]]:
    """Order-independent key for de-duplicating line segments."""
    start = _coord_key(*line.coords[0], precision=precision)
    end = _coord_key(*line.coords[-1], precision=precision)
    return tuple(sorted((start, end)))


def _sample_ring_points(ring: LineString, spacing: float) -> list[Point]:
    """Sample evenly spaced points along one boundary ring."""
    if spacing <= 0:
        raise ValueError("boundary_spacing must be > 0.")
    if ring.length == 0:
        return []

    distances = np.arange(0.0, ring.length, spacing)
    points = [ring.interpolate(float(d)) for d in distances]
    if not points:
        points = [ring.interpolate(0.0)]
    return points


def _boundary_points(polygon: PolygonLike, spacing: float) -> list[Point]:
    """Sample evenly spaced points across all boundary rings."""
    points: list[Point] = []
    for ring in _iter_rings(polygon):
        points.extend(_sample_ring_points(ring, spacing=spacing))
    return points


def create_trunk_segments(polygon: PolygonLike, config: TrunkConfig) -> list[LineString]:
    """
    Create interior trunk candidates from a boundary-point Voronoi construction.

    Returns a list of interior Voronoi edge segments.
    """
    points = _boundary_points(polygon, spacing=config.boundary_spacing)
    if len(points) < 4:
        raise ValueError("At least four boundary sample points are required to build the trunk.")

    point_array = np.array([[pt.x, pt.y] for pt in points])
    try:
        vor = Voronoi(point_array, furthest_site=False)
    except Exception as exc:  # pragma: no cover - depends on QHull failure mode
        raise ValueError(f"Voronoi construction failed for boundary points: {exc}") from exc

    clipped_cells: list[PolygonLike] = []
    for region in vor.regions:
        if not region or (-1 in region):
            continue
        vertices = [vor.vertices[idx] for idx in region]
        candidate = Polygon(vertices)
        if candidate.is_empty:
            continue
        clipped = candidate.intersection(polygon)
        if clipped.is_empty:
            continue
        if clipped.geom_type in {"Polygon", "MultiPolygon"}:
            clipped_cells.append(clipped)

    if not clipped_cells:
        raise ValueError("No interior Voronoi cells intersected the settlement polygon.")

    boundary = polygon.boundary
    boundary_buffer = config.resolved_boundary_buffer()
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    trunk_segments: list[LineString] = []

    for cell in clipped_cells:
        for ring in _iter_rings(cell):
            coords = list(ring.coords)
            for idx in range(len(coords) - 1):
                line = LineString([coords[idx], coords[idx + 1]])
                if line.length == 0:
                    continue
                # Keep only edges sufficiently interior to the settlement boundary.
                if line.distance(boundary) < boundary_buffer:
                    continue
                key = _segment_key(line)
                if key in seen:
                    continue
                seen.add(key)
                trunk_segments.append(line)

    if not trunk_segments:
        raise ValueError("No trunk segments survived the boundary exclusion filter.")

    return trunk_segments


def _split_long_line(line: LineString, split_distance: float) -> list[LineString]:
    """Split one long line into shorter substrings."""
    if split_distance <= 0:
        raise ValueError("split_distance must be > 0.")
    if line.length <= split_distance * 1.5:
        return [line]

    segment_count = max(1, math.floor(line.length / split_distance))
    actual_step = line.length / (segment_count + 1)
    pieces: list[LineString] = []
    for idx in range(segment_count + 1):
        start = idx * actual_step
        end = min((idx + 1) * actual_step, line.length)
        piece = substring(line, start, end)
        if piece.length > 0:
            pieces.append(piece)
    return pieces


def simplify_trunk_segments(
    trunks: list[LineString],
    config: TrunkConfig,
) -> tuple[list[LineString], list[Point]]:
    """
    Remove short spurs, preserve branch-to-branch segments, and split long trunks.
    """
    if not trunks:
        raise ValueError("No trunk segments were provided for simplification.")

    endpoint_keys = []
    endpoint_points: dict[tuple[float, float], Point] = {}
    for line in trunks:
        start = Point(line.coords[0])
        end = Point(line.coords[-1])
        for pt in (start, end):
            key = _coord_key(pt.x, pt.y)
            endpoint_keys.append(key)
            endpoint_points[key] = pt

    degree = Counter(endpoint_keys)
    branch_keys = {key for key, count in degree.items() if count > 2}
    branch_points = [endpoint_points[key] for key in sorted(branch_keys)]

    if branch_points:
        branch_multipoint = MultiPoint(branch_points)
        split_lines = []
        for line in trunks:
            parts = split(line, branch_multipoint)
            split_lines.extend([geom for geom in parts.geoms if isinstance(geom, LineString)])
    else:
        split_lines = list(trunks)

    kept: list[LineString] = []
    for line in split_lines:
        start_key = _coord_key(*line.coords[0])
        end_key = _coord_key(*line.coords[-1])
        both_branches = start_key in branch_keys and end_key in branch_keys
        if both_branches or (line.length >= config.length_removal) or not branch_points:
            kept.append(line)

    if not kept:
        kept = list(split_lines)

    merged = linemerge(kept)
    if isinstance(merged, LineString):
        merged_lines = [merged]
    else:
        merged_lines = [geom for geom in merged.geoms if isinstance(geom, LineString)]

    final_lines: list[LineString] = []
    for line in merged_lines:
        final_lines.extend(_split_long_line(line, split_distance=config.split_distance))

    if not final_lines:
        raise ValueError("No trunk segments remain after simplification.")

    return final_lines, branch_points


def build_trunks_gdf(trunks: list[LineString], target_crs: object) -> gpd.GeoDataFrame:
    """Package trunk segments into a GeoDataFrame."""
    gdf = gpd.GeoDataFrame(
        {
            "segment_id": list(range(len(trunks))),
            "Type": ["Trunk Line"] * len(trunks),
            "Length": [line.length for line in trunks],
            "geometry": trunks,
        },
        crs=target_crs,
    )
    return gdf
