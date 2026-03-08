from __future__ import annotations

import math
from typing import Iterable

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely import minimum_rotated_rectangle
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point
from shapely.ops import unary_union

from .config import PoleConfig
from .types import CandidatePoleSet, PolygonLike


def _coord_key(x: float, y: float, precision: int) -> tuple[float, float]:
    """Stable rounded coordinate key."""
    return (round(float(x), precision), round(float(y), precision))


def _extract_point_geometries(geometry) -> list[Point]:
    """Flatten mixed intersection results into points."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [geometry]
    if isinstance(geometry, MultiPoint):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        out: list[Point] = []
        for geom in geometry.geoms:
            out.extend(_extract_point_geometries(geom))
        return out
    return []


def polygon_rotation(polygon: PolygonLike) -> tuple[float, float, float, float, float, float]:
    """
    Compute the axes and extents of the minimum rotated rectangle.
    """
    box = minimum_rotated_rectangle(polygon)
    coords = np.array(box.exterior.coords)
    edges = np.diff(coords, axis=0)
    side_lengths = np.linalg.norm(edges, axis=1)

    max_length = float(np.max(side_lengths))
    min_length = float(np.min(side_lengths))

    angle_width = float(np.arctan2(edges[1, 1], edges[1, 0]))
    angle_length = float(np.arctan2(edges[2, 1], edges[2, 0]))

    x_start = float(coords[0][0])
    y_start = float(coords[0][1])

    return angle_width, angle_length, max_length, min_length, x_start, y_start


def create_candidate_poles(
    subarea_polygon: PolygonLike,
    trunk_line: LineString,
    config: PoleConfig,
) -> CandidatePoleSet:
    """
    Create a rotated lattice of candidate poles plus trunk-intersection poles.
    """
    if config.pole_spacing <= 0:
        raise ValueError("pole_spacing must be > 0.")

    buffered_trunk = trunk_line.buffer(config.trunk_buffer)
    angle_width, angle_length, poly_length, poly_width, start_x, start_y = polygon_rotation(subarea_polygon)

    width_steps = int(math.ceil(poly_width / config.pole_spacing)) + 1
    length_steps = int(math.ceil(poly_length / config.pole_spacing)) + 1

    lattice_points: dict[tuple[float, float], dict] = {}
    mesh_lines_w: list[LineString] = []
    mesh_lines_l: list[LineString] = []

    for w_idx in range(width_steps):
        for l_idx in range(length_steps):
            x = start_x + w_idx * config.pole_spacing * math.cos(angle_width) - l_idx * config.pole_spacing * math.cos(angle_length)
            y = start_y + w_idx * config.pole_spacing * math.sin(angle_width) - l_idx * config.pole_spacing * math.sin(angle_length)
            point = Point(x, y)

            if subarea_polygon.covers(point) and not point.within(buffered_trunk):
                key = _coord_key(point.x, point.y, config.point_precision)
                lattice_points[key] = {
                    "geometry": point,
                    "is_trunk": False,
                    "grid_w": w_idx,
                    "grid_l": l_idx,
                }

            x_next_w = start_x + (w_idx + 1) * config.pole_spacing * math.cos(angle_width) - l_idx * config.pole_spacing * math.cos(angle_length)
            y_next_w = start_y + (w_idx + 1) * config.pole_spacing * math.sin(angle_width) - l_idx * config.pole_spacing * math.sin(angle_length)

            x_next_l = start_x + w_idx * config.pole_spacing * math.cos(angle_width) - (l_idx + 1) * config.pole_spacing * math.cos(angle_length)
            y_next_l = start_y + w_idx * config.pole_spacing * math.sin(angle_width) - (l_idx + 1) * config.pole_spacing * math.sin(angle_length)

            mesh_lines_w.append(LineString([point, Point(x_next_w, y_next_w)]))
            mesh_lines_l.append(LineString([point, Point(x_next_l, y_next_l)]))

    trunk_points: dict[tuple[float, float], dict] = {}
    trunk_union = unary_union([trunk_line])

    for mesh_line in mesh_lines_w + mesh_lines_l:
        if not mesh_line.intersects(trunk_union):
            continue
        intersections = _extract_point_geometries(mesh_line.intersection(trunk_union))
        for pt in intersections:
            key = _coord_key(pt.x, pt.y, config.point_precision)
            trunk_points[key] = {
                "geometry": pt,
                "is_trunk": True,
                "grid_w": None,
                "grid_l": None,
            }

    for endpoint in (Point(trunk_line.coords[0]), Point(trunk_line.coords[-1])):
        key = _coord_key(endpoint.x, endpoint.y, config.point_precision)
        trunk_points[key] = {
            "geometry": endpoint,
            "is_trunk": True,
            "grid_w": None,
            "grid_l": None,
        }

    records = []
    pole_id = 0
    trunk_pole_ids: set[int] = set()

    for item in lattice_points.values():
        records.append(
            {
                "pole_id": pole_id,
                "is_trunk": item["is_trunk"],
                "grid_w": item["grid_w"],
                "grid_l": item["grid_l"],
                "geometry": item["geometry"],
            }
        )
        pole_id += 1

    for item in trunk_points.values():
        records.append(
            {
                "pole_id": pole_id,
                "is_trunk": item["is_trunk"],
                "grid_w": item["grid_w"],
                "grid_l": item["grid_l"],
                "geometry": item["geometry"],
            }
        )
        trunk_pole_ids.add(pole_id)
        pole_id += 1

    if not records:
        raise ValueError("No candidate poles were created for the sub-area.")

    poles_gdf = gpd.GeoDataFrame(records, geometry="geometry")
    poles_gdf["x"] = poles_gdf.geometry.x
    poles_gdf["y"] = poles_gdf.geometry.y

    return CandidatePoleSet(
        poles_gdf=poles_gdf,
        trunk_pole_ids=trunk_pole_ids,
        angle_width=angle_width,
        angle_length=angle_length,
        trunk_line=trunk_line,
        subarea_polygon=subarea_polygon,
    )


def assign_households_to_poles(
    poles_gdf: gpd.GeoDataFrame,
    households_gdf: gpd.GeoDataFrame,
    point_precision: int,
) -> tuple[set[int], gpd.GeoDataFrame]:
    """
    Assign each household to its nearest pole and create service-drop lines.
    """
    if poles_gdf.empty:
        raise ValueError("Cannot assign households without candidate poles.")
    if households_gdf.empty:
        service_gdf = gpd.GeoDataFrame(
            columns=["household_id", "pole_id", "Length", "geometry"],
            geometry="geometry",
            crs=households_gdf.crs,
        )
        return set(), service_gdf

    pole_coords = np.column_stack([poles_gdf["x"].to_numpy(), poles_gdf["y"].to_numpy()])
    tree = cKDTree(pole_coords)

    assigned_ids: set[int] = set()
    records = []

    for household_id, row in households_gdf.reset_index(drop=True).iterrows():
        point = row.geometry
        distance, local_idx = tree.query((point.x, point.y), k=1)
        pole_row = poles_gdf.iloc[int(local_idx)]
        pole_id = int(pole_row["pole_id"])
        assigned_ids.add(pole_id)

        records.append(
            {
                "household_id": household_id,
                "pole_id": pole_id,
                "Length": float(distance),
                "geometry": LineString([pole_row.geometry, point]),
            }
        )

    service_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=households_gdf.crs)
    return assigned_ids, service_gdf
