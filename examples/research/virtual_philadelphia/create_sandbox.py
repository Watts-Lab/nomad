"""
Create a sandbox dataset for Old City Philadelphia.

This script downloads OSM buildings and streets for a small bounding box
(Old City Philadelphia), rotates them for grid alignment, and saves them
as a self-contained GeoPackage for testing and development.

Usage (from this directory or as Jupyter notebook):
  python create_sandbox.py
"""

from pathlib import Path
import time
import geopandas as gpd
from shapely.geometry import box

import nomad.map_utils as nm

# Old City Philadelphia bounding box (EPSG:4326)
OLD_CITY_BBOX = box(-75.1662060, 39.9411582, -75.1456557, 39.9557201)

# Output directory (relative path for Jupyter compatibility)
OUTPUT_DIR = Path("sandbox")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SANDBOX_GPKG = OUTPUT_DIR / "sandbox_data.gpkg"

print("Downloading OSM data for Old City Philadelphia...")

# Pass bbox in EPSG:4326 (OSMnx expects geographic coordinates)
# The download functions will convert to the target CRS internally

cache_mode = "persistent"

print("Downloading buildings (Web Mercator)...")
start_time = time.time()
buildings = nm.download_osm_buildings(
    OLD_CITY_BBOX,  # Pass in EPSG:4326, function converts to target CRS
    crs="EPSG:3857",
    schema="garden_city",
    clip=True,
    infer_building_types=True,
    explode=True,
    by_chunks=False,
    cache_mode=cache_mode,
)
elapsed = time.time() - start_time
print(f"Downloaded {len(buildings):,} buildings in {elapsed:.1f}s")

buildings = nm.remove_overlaps(buildings).reset_index(drop=True)

print("Downloading streets (Web Mercator)...")
start_time = time.time()
streets = nm.download_osm_streets(
    OLD_CITY_BBOX,  # Pass in EPSG:4326, function converts to target CRS
    crs="EPSG:3857",
    clip=True,
    explode=True,
    by_chunks=False,
    cache_mode=cache_mode,
)
elapsed = time.time() - start_time
print(f"Downloaded {len(streets):,} streets in {elapsed:.1f}s")

streets = streets.reset_index(drop=True)

# Convert to Web Mercator for rotation (both are already in EPSG:3857 from download)
old_city_polygon = gpd.GeoDataFrame(geometry=[OLD_CITY_BBOX], crs="EPSG:4326").to_crs("EPSG:3857").geometry.iloc[0]

print("Estimating optimal rotation for grid alignment...")
rotation_start = time.time()
rotated_streets, rotation_deg = nm.rotate_streets_to_align(streets, k=200)
rotation_elapsed = time.time() - rotation_start
print(f"Rotation estimated at {rotation_deg:.2f} degrees (computed in {rotation_elapsed:.1f}s)")

rotated_buildings = nm.rotate(buildings, rotation_deg=rotation_deg)
rotated_boundary = nm.rotate(
    gpd.GeoDataFrame(geometry=[old_city_polygon], crs="EPSG:3857"),
    rotation_deg=rotation_deg
)

print(f"Sandbox: {len(rotated_buildings):,} buildings, {len(rotated_streets):,} streets")

if SANDBOX_GPKG.exists():
    SANDBOX_GPKG.unlink()

print(f"Saving sandbox data to {SANDBOX_GPKG}...")
rotated_buildings.to_file(SANDBOX_GPKG, layer="buildings", driver="GPKG")
rotated_streets.to_file(SANDBOX_GPKG, layer="streets", driver="GPKG", mode="a")
rotated_boundary.to_file(SANDBOX_GPKG, layer="boundary", driver="GPKG", mode="a")

print(f"Saved sandbox data to {SANDBOX_GPKG}")
