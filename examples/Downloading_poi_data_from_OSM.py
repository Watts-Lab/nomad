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
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from nomad.map_utils import download_osm_buildings, download_osm_streets, rotate, remove_overlaps, get_category_summary, get_subtype_summary, get_osm_type_summary

# %%
bbox = (-75.19747721789525, 39.931392279878246, -75.14652246706544, 39.96336810441389)

# Download, process, and save
buildings = download_osm_buildings(bbox, schema='garden_city', clip=True, explode=True, infer_building_types=True)
streets = download_osm_streets(bbox, clip=True, explode=True)

buildings = remove_overlaps(buildings)
streets = remove_overlaps(streets)

buildings = rotate(buildings, rotation_deg=10)
streets = rotate(streets, rotation_deg=10)

buildings.to_file("philadelphia_buildings.geojson", driver="GeoJSON")
streets.to_file("philadelphia_streets.geojson", driver="GeoJSON")

print(f"Downloaded {len(buildings)} buildings, {len(streets)} streets")
print(f"Categories: {get_category_summary(buildings)}")

# %%
# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Color scheme from virtual_philly.ipynb
colors = {
    'park': 'green',
    'residential': 'blue', 
    'retail': 'orange',
    'workplace': 'purple',
    'other': 'grey'
}

# Buildings by category with proper colors
for category, color in colors.items():
    subset = buildings[buildings['category'] == category]
    if len(subset) > 0:
        subset.plot(ax=axes[0], color=color, edgecolor='black', linewidth=0.2)
axes[0].set_title('Buildings by Category')

# Buildings + Streets
for category, color in colors.items():
    subset = buildings[buildings['category'] == category]
    if len(subset) > 0:
        subset.plot(ax=axes[1], color=color, edgecolor='black', linewidth=0.2)
streets.plot(ax=axes[1], color='black', linewidth=0.5)
axes[1].set_title('Buildings and Streets')

# Streets only
streets.plot(ax=axes[2], color='black', linewidth=0.5)
axes[2].set_title('Streets Only')

plt.tight_layout()
plt.show()
