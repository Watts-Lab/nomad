"""
Generate minimal fixture data for city generation tests.

Downloads OSM buildings and streets for a tiny bounding box in Philadelphia,
rotates for grid alignment, and saves as a GeoPackage with a visualization PNG.

Usage:
  python nomad/data/generate_fixture.py
"""

from pathlib import Path
import time
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import contextily as cx

import nomad.map_utils as nm

# Small bounding box in Old City Philadelphia (~6-9 blocks)
FIXTURE_BBOX = box(-75.1545, 39.9460, -75.1425, 39.9535)

OUTPUT_DIR = Path(__file__).parent
OUTPUT_GPKG = OUTPUT_DIR / "city_fixture.gpkg"
OUTPUT_PNG = OUTPUT_DIR / "city_fixture.png"

print("Downloading OSM data for fixture (Old City Philadelphia, ~2-3 blocks)...")

print("Downloading buildings...")
start_time = time.time()
buildings = nm.download_osm_buildings(
    FIXTURE_BBOX,
    crs="EPSG:3857",
    schema="garden_city",
    clip=True,
    infer_building_types=True,
    explode=True,
)
elapsed = time.time() - start_time
print(f"Downloaded {len(buildings):,} buildings in {elapsed:.1f}s")

boundary_polygon = gpd.GeoDataFrame(geometry=[FIXTURE_BBOX], crs="EPSG:4326").to_crs("EPSG:3857").geometry.iloc[0]
outside_mask = ~buildings.geometry.within(boundary_polygon)
if outside_mask.any():
    n_out = int(outside_mask.sum())
    print(f"Clipping {n_out} geometries to boundary.")
    buildings = gpd.clip(buildings, gpd.GeoDataFrame(geometry=[boundary_polygon], crs="EPSG:3857"))

buildings = nm.remove_overlaps(buildings).reset_index(drop=True)

print("Downloading streets...")
start_time = time.time()
streets = nm.download_osm_streets(
    FIXTURE_BBOX,
    crs="EPSG:3857",
    clip=True,
    explode=True,
)
elapsed = time.time() - start_time
print(f"Downloaded {len(streets):,} streets in {elapsed:.1f}s")

streets = streets.reset_index(drop=True)

print("Rotating for grid alignment...")
rotation_start = time.time()
rotated_streets, rotation_deg = nm.rotate_streets_to_align(streets, k=50)
rotation_elapsed = time.time() - rotation_start
print(f"Rotation: {rotation_deg:.2f} degrees ({rotation_elapsed:.1f}s)")

rotated_buildings = nm.rotate(buildings, rotation_deg=rotation_deg)
rotated_boundary = nm.rotate(
    gpd.GeoDataFrame(geometry=[boundary_polygon], crs="EPSG:3857"),
    rotation_deg=rotation_deg
)

print(f"Fixture: {len(rotated_buildings):,} buildings, {len(rotated_streets):,} streets")

if OUTPUT_GPKG.exists():
    OUTPUT_GPKG.unlink()

print(f"Saving to {OUTPUT_GPKG}...")
rotated_buildings.to_file(OUTPUT_GPKG, layer="buildings", driver="GPKG")
rotated_streets.to_file(OUTPUT_GPKG, layer="streets", driver="GPKG", mode="a")
rotated_boundary.to_file(OUTPUT_GPKG, layer="boundary", driver="GPKG", mode="a")

print("\nBuilding type summary (Garden City categories):")
if 'building_type' in rotated_buildings.columns:
    type_counts = rotated_buildings['building_type'].value_counts()
    for btype, count in type_counts.items():
        print(f"  {btype}: {count}")
else:
    print("  No building_type column available")

print(f"\nCreating visualization at {OUTPUT_PNG}...")
fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=150)
rotated_boundary.boundary.plot(ax=ax, color='red', linewidth=2, label='Boundary')
rotated_streets.plot(ax=ax, color='gray', linewidth=1, alpha=0.7, label='Streets')
if 'building_type' in rotated_buildings.columns:
    rotated_buildings.plot(ax=ax, column='building_type', categorical=True, legend=True, alpha=0.8, edgecolor='black', linewidth=0.5, legend_kwds={'title': 'Building Type', 'loc': 'upper right', 'fontsize': 10})
else:
    rotated_buildings.plot(ax=ax, alpha=0.8, edgecolor='black', linewidth=0.5, color='steelblue', label='Buildings')
cx.add_basemap(ax, crs=rotated_buildings.crs, source=cx.providers.CartoDB.Positron, alpha=0.5)
ax.set_title(f"City Fixture: {len(rotated_buildings)} buildings, {len(rotated_streets)} streets\nRotation: {rotation_deg:.2f}°", fontsize=14)
ax.set_xlabel("X (Web Mercator)")
ax.set_ylabel("Y (Web Mercator)")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, bbox_inches='tight')
plt.close()

print("\nCity fixture generated successfully")
print(f"  GeoPackage: {OUTPUT_GPKG}")
print(f"  Visualization: {OUTPUT_PNG}")
print(f"  Buildings: {len(rotated_buildings):,}")
print(f"  Streets: {len(rotated_streets):,}")

