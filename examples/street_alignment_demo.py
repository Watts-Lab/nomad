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
import math
from nomad.map_utils import (
    download_osm_streets, get_prominent_streets, rotate_streets_to_align,
    segment_angles_and_lengths, optimal_rotation
)

# %%
# Smaller bounding box in Philadelphia for demo
bbox = (-75.15573773790878, 39.94379645112187, -75.15035318331617, 39.94908717099987)

# Download streets
streets = download_osm_streets(bbox, clip=True, explode=True)

print(f"Downloaded {len(streets)} street segments")

# %%
# Get prominent streets
prominent_streets = get_prominent_streets(streets, k=10)

print(f"Selected {len(prominent_streets)} prominent streets")
if 'highway' in prominent_streets.columns:
    print("Highway types:", prominent_streets['highway'].value_counts().to_dict())

# %%
# Calculate optimal rotation
rotation_rad = optimal_rotation(prominent_streets.geometry.tolist())
rotation_deg = rotation_rad * 180 / 3.14159

print(f"Optimal rotation angle: {rotation_deg:.2f}°")

# %%
# Rotate all streets
rotated_streets, actual_rotation = rotate_streets_to_align(streets, prominent_k=10)

print(f"Applied rotation: {actual_rotation:.2f}°")

# %%
# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original streets
streets.plot(ax=axes[0], color='lightgray', linewidth=0.5)
prominent_streets.plot(ax=axes[0], color='red', linewidth=2)
axes[0].set_title('Original Streets\n(Red = Prominent)')
axes[0].set_aspect('equal')

# Rotated streets
rotated_streets.plot(ax=axes[1], color='lightgray', linewidth=0.5)
rotated_prominent = get_prominent_streets(rotated_streets, k=10)
rotated_prominent.plot(ax=axes[1], color='red', linewidth=2)
axes[1].set_title(f'Rotated Streets\n(Rotation: {actual_rotation:.1f}°)')
axes[1].set_aspect('equal')

# Comparison overlay
streets.plot(ax=axes[2], color='lightblue', linewidth=0.5, alpha=0.7, label='Original')
rotated_streets.plot(ax=axes[2], color='darkblue', linewidth=0.5, alpha=0.7, label='Rotated')
axes[2].set_title('Overlay Comparison')
axes[2].set_aspect('equal')
axes[2].legend()

plt.tight_layout()
plt.show()

# %%
# Analyze segment angles for prominent streets
print("\nSegment analysis for prominent streets:")
print("Angle (rad) | Length | cos(4θ) | sin(4θ)")
print("-" * 45)

A_total = 0.0
B_total = 0.0

for geom in prominent_streets.geometry:
    for theta, length in segment_angles_and_lengths(geom):
        cos4theta = math.cos(4 * theta)
        sin4theta = math.sin(4 * theta)
        A_total += length * cos4theta
        B_total += length * sin4theta
        
        print(f"{theta:8.3f}   | {length:6.1f} | {cos4theta:7.3f} | {sin4theta:7.3f}")

print("-" * 45)
print(f"A = Σ(w·cos(4θ)) = {A_total:.3f}")
print(f"B = Σ(w·sin(4θ)) = {B_total:.3f}")
print(f"Optimal rotation = -¼·atan2(B,A) = {rotation_rad:.3f} rad = {rotation_deg:.2f}°")

# %%
