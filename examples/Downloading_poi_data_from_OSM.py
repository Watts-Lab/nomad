# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Downloading OSM Data
#
# Download buildings and streets from OpenStreetMap, visualize them, and optionally rotate geometries for alignment.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from nomad.map_utils import (
    download_osm_buildings, download_osm_streets, rotate_and_explode, remove_overlaps,
    get_category_summary, get_subtype_summary, get_osm_type_summary
)

# %%
# Larger bounding box for better demonstration
bbox = (
    -75.19747721789525,    # west
    39.931392279878246,    # south
    -75.14652246706544,    # east
    39.96336810441389      # north
)

# Download buildings and streets
buildings = download_osm_buildings(bbox, schema='garden_city', clip=True)
streets = download_osm_streets(bbox, clip=True)

# Explode MultiPolygons/MultiLineStrings
buildings_exploded = buildings.explode(ignore_index=True)
streets_exploded = streets.explode(ignore_index=True)

# Remove overlaps separately for buildings and parks
buildings_only = buildings_exploded[buildings_exploded['category'] != 'park']
parks_only = buildings_exploded[buildings_exploded['category'] == 'park']

# Deduplicate parks by geometry (remove identical polygons)
parks_deduplicated = parks_only.drop_duplicates(subset=['geometry'])

buildings_clean = remove_overlaps(buildings_only)
parks_clean = remove_overlaps(parks_deduplicated)

# Combine cleaned buildings and parks
buildings_clean = gpd.GeoDataFrame(
    pd.concat([buildings_clean, parks_clean], ignore_index=True),
    crs=buildings_exploded.crs
)

# Rotate everything around a common centroid to align with N-S E-W
rotation_angle = 10
buildings_final = rotate_and_explode(buildings_clean, rotation_deg=rotation_angle)
streets_final = rotate_and_explode(streets_exploded, rotation_deg=rotation_angle)

# Show summary of final results
print("Final dataset summary:")
print(f"Buildings: {len(buildings_final)} features")
print(f"Streets: {len(streets_final)} segments")
print(f"\nBuilding categories: {get_category_summary(buildings_final)}")
print(f"Building subtypes: {get_subtype_summary(buildings_final)}")

# %%
# Visualize final result
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Get bounds of rotated data for proper plot scaling
all_geometries = list(buildings_final.geometry) + list(streets_final.geometry)
bounds = gpd.GeoSeries(all_geometries).total_bounds

ax1 = axes[0]

# Plot parks first (bottom layer), then other buildings on top
parks_only = buildings_final[buildings_final['category'] == 'park']
other_buildings = buildings_final[buildings_final['category'] != 'park']

# Plot parks first (bottom layer)
if len(parks_only) > 0:
    parks_only.plot(ax=ax1, color='green', edgecolor='black', linewidth=0.5, alpha=0.7)

# Plot other buildings on top
if len(other_buildings) > 0:
    other_buildings.plot(ax=ax1, column='category', legend=True, 
                        cmap='tab10', edgecolor='black', linewidth=0.5)

ax1.set_xlim(bounds[0], bounds[2])
ax1.set_ylim(bounds[1], bounds[3])
ax1.set_title(f'Buildings by Category (parks at bottom, rotated {rotation_angle}°)')

ax2 = axes[1]
buildings_final.plot(ax=ax2, color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.7)
streets_final.plot(ax=ax2, color='navy', linewidth=1.5, alpha=0.8)
ax2.set_xlim(bounds[0], bounds[2])
ax2.set_ylim(bounds[1], bounds[3])
ax2.set_title(f'Streets and Buildings (final, rotated {rotation_angle}°)')

# Third plot: Streets only with color coding
ax3 = axes[2]

# Check what columns are available for street coloring
street_columns = ['highway', 'surface', 'tunnel', 'bridge', 'oneway', 'lanes']
available_columns = [col for col in street_columns if col in streets_final.columns]

if available_columns:
    # Use the first available column for coloring
    color_column = available_columns[0]
    streets_final.plot(ax=ax3, column=color_column, legend=True, 
                      cmap='tab10', linewidth=1.5, alpha=0.8)
    ax3.set_title(f'Streets by {color_column.title()} (rotated {rotation_angle}°)')
else:
    # Fallback: color by geometry type or just use single color
    streets_final.plot(ax=ax3, color='navy', linewidth=1.5, alpha=0.8)
    ax3.set_title(f'Streets Only (rotated {rotation_angle}°)')

ax3.set_xlim(bounds[0], bounds[2])
ax3.set_ylim(bounds[1], bounds[3])

plt.tight_layout()
plt.show()

# %%
# Save datasets to files
print("Saving datasets...")

# Save as GeoJSON (human-readable, widely supported)
buildings_final.to_file("philadelphia_buildings.geojson", driver="GeoJSON")
streets_final.to_file("philadelphia_streets.geojson", driver="GeoJSON")

# Note: Parquet format has issues with mixed data types in OSM data
# GeoJSON is more reliable for complex OSM datasets
print(f"Saved {len(buildings_final)} buildings and {len(streets_final)} streets")
print("Files saved:")
print("- philadelphia_buildings.geojson")
print("- philadelphia_streets.geojson")
print("\nNote: GeoJSON format chosen for reliability with OSM data types")

# %%
