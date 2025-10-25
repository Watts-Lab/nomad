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
# # Street Alignment Demo
#
# Download streets from OSM, select prominent streets, and demonstrate optimal rotation
# to align them with N-S/E-W axes using the mathematical approach.

# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from nomad.map_utils import download_osm_streets, rotate_streets_to_align
from shapely.affinity import rotate as shapely_rotate

bbox = (-75.210, 39.920, -75.130, 39.970)
streets = download_osm_streets(bbox, clip=True, explode=True)

# Get the random sample for highlighting
non_highway = streets[~streets['highway'].isin(['motorway', 'trunk', 'primary'])]
sample_streets = non_highway.sample(n=400) if len(non_highway) > 400 else non_highway

rotated_streets, rotation_deg = rotate_streets_to_align(streets, k=400)

print(f"Downloaded {len(streets)} streets, rotated by {rotation_deg:.2f}°")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

streets.plot(ax=axes[0], color='gray', linewidth=0.5)
sample_streets.plot(ax=axes[0], color='red', linewidth=2)
axes[0].set_title('Original Streets')

rotated_streets.plot(ax=axes[1], color='gray', linewidth=0.5)
# Rotate the sample streets for highlighting
rotated_sample = sample_streets.copy()
all_geoms = streets.geometry.union_all()
origin_coords = (all_geoms.centroid.x, all_geoms.centroid.y)
rotated_sample.geometry = rotated_sample.geometry.apply(
    lambda geom: shapely_rotate(geom, rotation_deg, origin=origin_coords)
)
rotated_sample.plot(ax=axes[1], color='red', linewidth=2)
axes[1].set_title(f'Rotated Streets ({rotation_deg:.1f}°)')

plt.tight_layout()
plt.show()

# %%
