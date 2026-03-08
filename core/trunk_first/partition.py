from __future__ import annotations

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union, voronoi_diagram

from .types import PolygonLike


def _sample_line_points(line, spacing: float) -> list[Point]:
    """Sample one line at regular spacing, including the endpoint."""
    if spacing <= 0:
        raise ValueError("subarea spacing must be > 0.")
    if line.length == 0:
        return [Point(line.coords[0])]

    distances = np.arange(0.0, line.length, spacing)
    points = [line.interpolate(float(d)) for d in distances]
    end_point = Point(line.coords[-1])
    if not points or points[-1].distance(end_point) > 1e-9:
        points.append(end_point)
    return points


def subdivide_by_trunk_voronoi(
    trunks: list,
    polygon: PolygonLike,
    spacing: float,
) -> list[PolygonLike]:
    """
    Partition the settlement into one sub-area per trunk segment using a Voronoi
    tessellation of sampled trunk points.

    Each Voronoi cell is assigned to the nearest trunk sample (KD-tree), then
    dissolved by trunk segment.
    """
    if not trunks:
        raise ValueError("At least one trunk segment is required.")
    if len(trunks) == 1:
        return [polygon]

    all_points: list[Point] = []
    point_segment_ids: list[int] = []

    for segment_id, line in enumerate(trunks):
        sampled = _sample_line_points(line, spacing=spacing)
        all_points.extend(sampled)
        point_segment_ids.extend([segment_id] * len(sampled))

    point_array = np.array([[pt.x, pt.y] for pt in all_points])
    if len(point_array) < 2:
        return [polygon]

    tree = cKDTree(point_array)
    regions_geom = voronoi_diagram(MultiPoint(all_points), envelope=polygon)
    regions = list(regions_geom.geoms)

    grouped_regions: dict[int, list] = {segment_id: [] for segment_id in range(len(trunks))}
    for region in regions:
        rep = region.representative_point()
        _, nearest_idx = tree.query((rep.x, rep.y), k=1)
        segment_id = point_segment_ids[int(nearest_idx)]
        clipped = region.intersection(polygon)
        if clipped.is_empty:
            continue
        grouped_regions[segment_id].append(clipped)

    dissolved: list[PolygonLike] = []
    for segment_id in range(len(trunks)):
        items = grouped_regions.get(segment_id, [])
        if not items:
            dissolved.append(trunks[segment_id].buffer(spacing).intersection(polygon))
            continue
        merged = unary_union(items).intersection(polygon)
        dissolved.append(merged)

    return dissolved


def build_subareas_gdf(subareas: list[PolygonLike], target_crs: object) -> gpd.GeoDataFrame:
    """Package sub-area polygons into a GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "segment_id": list(range(len(subareas))),
            "Type": ["Subarea"] * len(subareas),
            "geometry": subareas,
        },
        crs=target_crs,
    )
