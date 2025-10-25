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
from nomad.map_utils import download_osm_buildings, download_osm_streets, rotate_and_explode, remove_overlaps

# %%
# Old City, Philadelphia (matching virtual_philly.ipynb)
bbox = (
    -75.16620602283949,   # west
    39.94115823455355,    # south
    -75.14565573634475,   # east
    39.955720193879245    # north
)

# Step 1: Download buildings and streets
buildings = download_osm_buildings(bbox, schema='garden_city', clip=True)
streets = download_osm_streets(bbox, clip=True)

print(f"Step 1 - Downloaded:")
print(f"  Buildings: {len(buildings)} features")
print(f"  Streets:   {len(streets)} segments")

# Step 2: Explode MultiPolygons/MultiLineStrings
buildings_exploded = buildings.explode(ignore_index=True)
streets_exploded = streets.explode(ignore_index=True)

print(f"\nStep 2 - After exploding:")
print(f"  Buildings: {len(buildings_exploded)} features")
print(f"  Streets:   {len(streets_exploded)} segments")

# Step 3: Remove overlaps from buildings only (streets are meant to intersect)
buildings_clean = remove_overlaps(buildings_exploded)

print(f"\nStep 3 - After removing building overlaps:")
print(f"  Buildings: {len(buildings_clean)} features")
print(f"  Streets:   {len(streets_exploded)} segments")

# Step 4: Rotate everything around a common centroid to align with N-S E-W
rotation_angle = 10
buildings_final = rotate_and_explode(buildings_clean, rotation_deg=rotation_angle)
streets_final = rotate_and_explode(streets_exploded, rotation_deg=rotation_angle)

print(f"\nStep 4 - After rotating {rotation_angle}° around common centroid:")
print(f"  Buildings: {len(buildings_final)} features")
print(f"  Streets:   {len(streets_final)} segments")

# %%
# Visualize final result
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

ax1 = axes[0]
buildings_final.plot(ax=ax1, column='category', legend=True, 
                     cmap='tab10', edgecolor='black', linewidth=0.5)
ax1.set_title(f'Buildings by Category (clipped, cleaned, rotated {rotation_angle}°)')

ax2 = axes[1]
buildings_final.plot(ax=ax2, color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.7)
streets_final.plot(ax=ax2, color='navy', linewidth=1.5, alpha=0.8)
ax2.set_title(f'Streets and Buildings (final, rotated {rotation_angle}°)')

plt.tight_layout()
plt.show()

# %%
