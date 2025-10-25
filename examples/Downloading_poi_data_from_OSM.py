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
bbox = (-75.19747721789525, 39.931392279878246, -75.14652246706544, 39.96336810441389)

# Download and process
buildings = download_osm_buildings(bbox, schema='garden_city', clip=True, explode=True)
streets = download_osm_streets(bbox, clip=True, explode=True)

# Remove overlaps and rotate
buildings = remove_overlaps(buildings)
streets = remove_overlaps(streets)

rotation_angle = 10
buildings = rotate_and_explode(buildings, rotation_deg=rotation_angle)
streets = rotate_and_explode(streets, rotation_deg=rotation_angle)

# Save data
buildings.to_file("philadelphia_buildings.geojson", driver="GeoJSON")
streets.to_file("philadelphia_streets.geojson", driver="GeoJSON")

print(f"Downloaded {len(buildings)} buildings, {len(streets)} streets")
print(f"Categories: {get_category_summary(buildings)}")

# %%
# Visualize final result
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Get bounds for proper scaling
bounds = gpd.GeoSeries(list(buildings.geometry) + list(streets.geometry)).total_bounds

# Plot 1: Buildings by category
ax1 = axes[0]
parks = buildings[buildings['category'] == 'park']
other = buildings[buildings['category'] != 'park']

if len(parks) > 0:
    parks.plot(ax=ax1, color='green', edgecolor='black', linewidth=0.5, alpha=0.7)
if len(other) > 0:
    other.plot(ax=ax1, column='category', legend=True, cmap='tab10', edgecolor='black', linewidth=0.5)

ax1.set_xlim(bounds[0], bounds[2])
ax1.set_ylim(bounds[1], bounds[3])
ax1.set_title(f'Buildings by Category (rotated {rotation_angle}°)')

# Plot 2: Buildings + Streets
ax2 = axes[1]
buildings.plot(ax=ax2, color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.7)
streets.plot(ax=ax2, color='navy', linewidth=1.5, alpha=0.8)
ax2.set_xlim(bounds[0], bounds[2])
ax2.set_ylim(bounds[1], bounds[3])
ax2.set_title(f'Streets and Buildings (rotated {rotation_angle}°)')

# Plot 3: Streets only
ax3 = axes[2]
street_cols = ['highway', 'surface', 'tunnel', 'bridge', 'oneway', 'lanes']
available_cols = [col for col in street_cols if col in streets.columns]

if available_cols:
    streets.plot(ax=ax3, column=available_cols[0], legend=True, cmap='tab10', linewidth=1.5, alpha=0.8)
    ax3.set_title(f'Streets by {available_cols[0].title()} (rotated {rotation_angle}°)')
else:
    streets.plot(ax=ax3, color='navy', linewidth=1.5, alpha=0.8)
    ax3.set_title(f'Streets Only (rotated {rotation_angle}°)')

ax3.set_xlim(bounds[0], bounds[2])
ax3.set_ylim(bounds[1], bounds[3])

plt.tight_layout()
plt.show()

# %%
