# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Downloading places of interest (POI) Data from OSM
#
# Useful open street maps overpass API wrappers to download buildings and streets.

# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from nomad.map_utils import download_osm_buildings, download_osm_streets, remove_overlaps

## Download by Bounding Box
bbox = (-75.19747721789525, 39.931392279878246, -75.14652246706544, 39.96336810441389)

# Download, process, and save
buildings = download_osm_buildings(bbox, schema='garden_city', clip=True, explode=True, infer_building_types=True)
streets = download_osm_streets(bbox, clip=True, explode=True)

buildings = remove_overlaps(buildings)

buildings.to_file("philadelphia_buildings.geojson", driver="GeoJSON")
streets.to_file("philadelphia_streets.geojson", driver="GeoJSON")

print(f"Downloaded {len(buildings)} buildings, {len(streets)} streets")
print(f"Building categories: {buildings['building_type'].value_counts().to_dict()}")

# Show sample of downloaded data
print("\nSample buildings data:")
print(buildings.head())

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

colors = {
    'park': 'green',
    'home': 'blue', 
    'retail': 'orange',
    'workplace': 'purple',
    'other': 'grey'
}

# Buildings by category with proper colors
for category, color in colors.items():
    subset = buildings[buildings['building_type'] == category]
    if len(subset) > 0:
        subset.plot(ax=axes[0], color=color, edgecolor='black', linewidth=0.1, alpha=0.7)
axes[0].set_title('Buildings by Category')
axes[0].set_aspect('equal')

# Buildings + Streets
for category, color in colors.items():
    subset = buildings[buildings['building_type'] == category]
    if len(subset) > 0:
        subset.plot(ax=axes[1], color=color, edgecolor='black', linewidth=0.08, alpha=0.7)
streets.plot(ax=axes[1], color='black', linewidth=0.4)
axes[1].set_title('Buildings and Streets')
axes[1].set_aspect('equal')

# Streets only
streets.plot(ax=axes[2], color='black', linewidth=0.5)
axes[2].set_title('Streets Only')
axes[2].set_aspect('equal')

plt.tight_layout()
plt.show(block=False)

# %%
## Download geometries by City Name
city_name = 'Salem, New Jersey'

salem_buildings = download_osm_buildings(city_name, schema='garden_city', explode=True, infer_building_types=True)
salem_streets = download_osm_streets(city_name, explode=True)

# Remove overlaps
salem_buildings = remove_overlaps(salem_buildings)

# Save data
salem_buildings.to_file("salem_buildings.geojson", driver="GeoJSON")
salem_streets.to_file("salem_streets.geojson", driver="GeoJSON")

print(f"Downloaded {len(salem_buildings)} buildings, {len(salem_streets)} streets")
print(f"Building categories: {salem_buildings['building_type'].value_counts().to_dict()}")
# Show sample of downloaded data
print("\nSample Salem buildings data:")
print(salem_buildings.head())

print("\nSample Salem streets data:")
print(salem_streets.head())

# Plot Salem results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

colors = {
    'park': 'green',
    'home': 'blue', 
    'retail': 'orange',
    'workplace': 'purple',
    'other': 'grey'
}

# Buildings by category
for category, color in colors.items():
    subset = salem_buildings[salem_buildings['building_type'] == category]
    if len(subset) > 0:
        subset.plot(ax=axes[0], color=color, edgecolor='black', linewidth=0.1)
axes[0].set_title('Salem Buildings by Category')
axes[0].set_aspect('equal')

# Buildings + Streets
for category, color in colors.items():
    subset = salem_buildings[salem_buildings['building_type'] == category]
    if len(subset) > 0:
        subset.plot(ax=axes[1], color=color, edgecolor='black', linewidth=0.08, alpha=0.7)
salem_streets.plot(ax=axes[1], color='black', linewidth=0.5)
axes[1].set_title('Salem Buildings and Streets')
axes[1].set_aspect('equal')

# Streets only
salem_streets.plot(ax=axes[2], color='black', linewidth=0.5)
axes[2].set_title('Salem Streets Only')
axes[2].set_aspect('equal')

plt.tight_layout()
plt.show(block=False)
