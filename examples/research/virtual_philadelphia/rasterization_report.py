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
#     display_name: Python (nomad repo venv)
#     language: python
#     name: nomad-repo-venv
# ---

# %% [markdown]
# # Rasterization Performance Report
#
# ## What is Rasterization?
#
# The rasterization pipeline converts real-world OpenStreetMap (OSM) vector data into discrete
# grid-based `City` objects compatible with trajectory generation. OSM provides continuous 
# geometries (building polygons, street linestrings), but our simulation framework operates 
# on discrete grid cells where each cell is either a street block or a building block.
#
# ## Key Implementation Details
#
# **RasterCityGenerator** handles the conversion:
# - Generates a grid aligned to the city boundary (not a full bounding box, which would waste memory)
# - Assigns block types based on intersection priority (streets > parks > other buildings)
# - Splits buildings that span disconnected components
# - Verifies street connectivity (keeps only the largest connected component)
# - Ensures all buildings have accessible doors adjacent to street blocks
#
# **Efficient Routing via Hub Network:**
# - Computing all-pairs shortest paths is O(n²) memory, infeasible for city-scale graphs
# - Instead, we select a sparse set of hub nodes (grid-distributed) and precompute hub-to-hub distances
# - Routing between arbitrary nodes uses: node → nearest_hub + hub_to_hub + nearest_hub → node
# - Memory is O(n) for per-node hub assignments + O(H²) for hub distances (H ≪ n)
# - Enables fast on-demand pathfinding without storing all-pairs distances
#
# **Vectorized Gravity Computation:**
# - Computes building-to-building gravity using Manhattan distances + hub shortcuts
# - Uses numpy broadcasting for pairwise distances (no loops)
# - Fancy indexing expands hub-to-hub distances to full door-to-door matrices
# - Result stored as DataFrame indexed by building IDs for O(1) lookup in trajectory generation

# %%
from pathlib import Path
import time
import geopandas as gpd
import pandas as pd
import numpy as np

from nomad.city_gen import RasterCity

SANDBOX_PATH = Path("sandbox/sandbox_data.gpkg")
buildings = gpd.read_file(SANDBOX_PATH, layer="buildings")
streets = gpd.read_file(SANDBOX_PATH, layer="streets")
boundary = gpd.read_file(SANDBOX_PATH, layer="boundary")

# %% [markdown]
# ## Benchmark: Sequential Pipeline Timing
#
# Times each step of the city generation pipeline sequentially

# %%
hub_size = 100

t0 = time.time()
city = RasterCity(boundary.geometry.iloc[0], streets, buildings, block_side_length=15.0)
gen_time = time.time() - t0

t1 = time.time()
G = city.get_street_graph()
graph_time = time.time() - t1

t2 = time.time()
city._build_hub_network(hub_size=hub_size)
hub_time = time.time() - t2

t3 = time.time()
city.compute_gravity(exponent=2.0)
grav_time = time.time() - t3

total_time = gen_time + graph_time + hub_time + grav_time

print("\n" + "="*50)
print("TIMING SUMMARY")
print("="*50)
print(f"City generation:    {gen_time:>6.2f}s")
print(f"Street graph:       {graph_time:>6.2f}s")
print(f"Hub network:        {hub_time:>6.2f}s")
print(f"Gravity matrix:     {grav_time:>6.2f}s")
print("-"*50)
print(f"Total:              {total_time:>6.2f}s")
print("="*50)

# %% [markdown]
# ## Debugging: Structure and Diagnostics
#
# Print all structural info and validation checks

# %%
print("\nCITY STRUCTURE")
print(f"  Blocks: {len(city.blocks_gdf):,}")
print(f"  Streets: {len(city.streets_gdf):,}")
print(f"  Buildings: {len(city.buildings_gdf):,}")

print("\nSTREET GRAPH")
print(f"  Nodes: {len(G.nodes):,}")
print(f"  Edges: {len(G.edges):,}")

print(f"\nHUB NETWORK ({hub_size} hubs)")
print(f"  Matrix shape: {city.hub_df.shape}")
print(f"  Sample (first 5x5):")
print(city.hub_df.iloc[:5, :5])

print("\nGRAVITY MATRIX")
print(f"  Shape: {city.grav.shape}")

# Diagonal should be all zeros (no self-gravity)
diag = np.diag(city.grav.values)
diag_zeros = (diag == 0).all()
print(f"  Diagonal all zeros: {diag_zeros}")
if not diag_zeros:
    nonzero_diag = np.where(diag != 0)[0]
    print(f"    WARNING: Non-zero diagonal entries: {len(nonzero_diag)}")
    print(f"    Example diagonal values: {diag[nonzero_diag[:5]]}")
    print(f"    Corresponding building IDs: {[city.grav.index[i] for i in nonzero_diag[:5]]}")

# Off-diagonal should all be positive (gravity between distinct buildings)
mask = ~np.eye(len(city.grav), dtype=bool)
offdiag = city.grav.values[mask]
offdiag_positive = (offdiag > 0).all()
print(f"  Off-diagonal all positive: {offdiag_positive}")
if not offdiag_positive:
    nonpositive = np.where(offdiag <= 0)[0]
    print(f"    WARNING: Non-positive off-diagonal entries: {len(nonpositive)} / {len(offdiag)}")
    print(f"    Min off-diagonal value: {offdiag.min()}")
    print(f"    Max off-diagonal value: {offdiag.max()}")
    rows, cols = np.where((city.grav.values <= 0) & ~np.eye(len(city.grav), dtype=bool))
    if len(rows) > 0:
        print(f"    Example: grav[{city.grav.index[rows[0]]}][{city.grav.columns[cols[0]]}] = {city.grav.iloc[rows[0], cols[0]]}")

print(f"  Sample (first building to 5 others):")
print(city.grav.iloc[0, :5])
