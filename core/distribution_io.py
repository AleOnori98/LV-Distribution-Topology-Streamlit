from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from config.settings import TARGET_CRS


def load_and_transform_data(file, target_crs: int = TARGET_CRS) -> Optional[gpd.GeoDataFrame]:
    """
    Load and transform spatial data from an uploaded file.

    Supported formats
    -----------------
    - GeoPackage (.gpkg): assumed to already be in a projected CRS, reprojected to `target_crs`.
    - Excel (.xlsx): expects 'latitude' and 'longitude' columns in WGS84 (EPSG:4326).

    Returns
    -------
    GeoDataFrame in `target_crs` or None if the file cannot be parsed.
    """
    if file is None:
        return None

    name = getattr(file, "name", "")
    if name.endswith(".gpkg"):
        gdf = gpd.read_file(file)
        if gdf.crs is None:
            # assume already in target CRS if missing
            gdf.set_crs(epsg=target_crs, inplace=True)
        if gdf.crs.to_epsg() != target_crs:
            gdf = gdf.to_crs(epsg=target_crs)
        return gdf[gdf.is_valid]

    if name.endswith(".xlsx"):
        df = pd.read_excel(file)
        if "latitude" in df.columns and "longitude" in df.columns:
            geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry)
            gdf.set_crs(epsg=4326, inplace=True)
            gdf = gdf.to_crs(epsg=target_crs)
            return gdf[gdf.is_valid]

    # unsupported format
    return None
