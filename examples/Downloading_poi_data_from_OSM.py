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
# # Downloading OSM Buildings
#
# Download and categorize buildings from OpenStreetMap. This demo shows how to:
# - Download buildings and parks from OSM
# - Categorize them using configurable schemas
# - Remove overlaps and explode MultiPolygons
# - Visualize results with proper color coding
#
# ## Category Mapping (Garden City Schema)
#
# | OSM Tags | Garden City Category | Description |
# |----------|---------------------|-------------|
# | `building=house`, `building=residential` | residential | Homes, apartments |
# | `building=commercial`, `shop=*` | retail | Stores, restaurants |
# | `building=office`, `amenity=*` | workplace | Offices, civic buildings |
# | `leisure=park`, `natural=*` | park | Parks, green spaces |
# | `building=yes` (with inference) | residential/retail | Inferred from height/amenity |
# | Other buildings | other | Unclassified structures |
#
# *Note: Other schemas (e.g., `geolife_plus`) can be used by changing the `schema` parameter.*

# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from nomad.map_utils import download_osm_buildings, remove_overlaps, get_category_summary, get_subtype_summary, get_osm_type_summary

# %%
bbox = (-75.19747721789525, 39.931392279878246, -75.14652246706544, 39.96336810441389)

# Download and process buildings
buildings = download_osm_buildings(bbox, schema='garden_city', clip=True, explode=True, infer_building_types=True)

# Remove overlaps
buildings = remove_overlaps(buildings)

# Save data
buildings.to_file("philadelphia_buildings.geojson", driver="GeoJSON")

print(f"Downloaded {len(buildings)} buildings")
print(f"Categories: {get_category_summary(buildings)}")

# %%
# Plot results
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

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
axes[0].set_aspect('equal')

# All buildings overview
buildings.plot(ax=axes[1], color='lightgray', edgecolor='black', linewidth=0.2)
axes[1].set_title('All Buildings')
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()
