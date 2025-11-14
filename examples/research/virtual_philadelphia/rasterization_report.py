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
# **RasterCity** handles the conversion:
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
from shapely.geometry import box

from nomad.city_gen import RasterCity
import nomad.map_utils as nm

# %% [markdown]
# ## Configuration

# %%
SMALL_BOX = box(-75.1545, 39.946, -75.1425, 39.9535)
MEDIUM_BOX = box(-75.1665, 39.9385, -75.1425, 39.9535)
LARGE_BOX  = box(-75.1905, 39.9235, -75.1425, 39.9535)

# PHILLY_BOX = SMALL_BOX
# PHILLY_BOX = MEDIUM_BOX
PHILLY_BOX = SMALL_BOX

BLOCK_SIDE_LENGTH = 10.0  # meters

OUTPUT_DIR = Path("sandbox")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Determine which box we're using and set appropriate filename
if PHILLY_BOX == SMALL_BOX:
    BOX_NAME = "small"
elif PHILLY_BOX == MEDIUM_BOX:
    BOX_NAME = "medium"
else:
    BOX_NAME = "large"

SANDBOX_GPKG = OUTPUT_DIR / f"sandbox_data_{BOX_NAME}.gpkg"

REGENERATE_DATA = False  # Set to True to re-download OSM data

# %% [markdown]
# ## Benchmark: Data Generation (OSM Download + Rotation)

# %%
if REGENERATE_DATA or not SANDBOX_GPKG.exists():
    print("="*50)
    print("DATA GENERATION")
    print("="*50)
    
    t0 = time.time()
    buildings = nm.download_osm_buildings(
        PHILLY_BOX,
        crs="EPSG:3857",
        schema="garden_city",
        clip=True,
        infer_building_types=True,
        explode=True,
    )
    download_buildings_time = time.time() - t0
    print(f"Buildings download: {download_buildings_time:>6.2f}s ({len(buildings):,} buildings)")
    
    boundary_polygon = gpd.GeoDataFrame(geometry=[PHILLY_BOX], crs="EPSG:4326").to_crs("EPSG:3857").geometry.iloc[0]
    outside_mask = ~buildings.geometry.within(boundary_polygon)
    if outside_mask.any():
        buildings = gpd.clip(buildings, gpd.GeoDataFrame(geometry=[boundary_polygon], crs="EPSG:3857"))
    buildings = nm.remove_overlaps(buildings).reset_index(drop=True)
    
    t1 = time.time()
    streets = nm.download_osm_streets(
        PHILLY_BOX,
        crs="EPSG:3857",
        clip=True,
        explode=True,
        graphml_path=OUTPUT_DIR / "streets_consolidated.graphml",
    )
    download_streets_time = time.time() - t1
    print(f"Streets download:   {download_streets_time:>6.2f}s ({len(streets):,} streets)")
    
    streets = streets.reset_index(drop=True)
    
    t2 = time.time()
    rotated_streets, rotation_deg = nm.rotate_streets_to_align(streets, k=200)
    rotation_time = time.time() - t2
    print(f"Grid rotation:      {rotation_time:>6.2f}s ({rotation_deg:.2f}°)")
    
    rotated_buildings = nm.rotate(buildings, rotation_deg=rotation_deg)
    rotated_boundary = nm.rotate(
        gpd.GeoDataFrame(geometry=[boundary_polygon], crs="EPSG:3857"),
        rotation_deg=rotation_deg
    )
    
    if SANDBOX_GPKG.exists():
        SANDBOX_GPKG.unlink()
    
    rotated_buildings.to_file(SANDBOX_GPKG, layer="buildings", driver="GPKG")
    rotated_streets.to_file(SANDBOX_GPKG, layer="streets", driver="GPKG", mode="a")
    rotated_boundary.to_file(SANDBOX_GPKG, layer="boundary", driver="GPKG", mode="a")
    
    data_gen_time = download_buildings_time + download_streets_time + rotation_time
    print("-"*50)
    print(f"Data generation:    {data_gen_time:>6.2f}s")
    print("="*50 + "\n")
else:
    print(f"Loading existing data from {SANDBOX_GPKG}")
    data_gen_time = 0.0

buildings = gpd.read_file(SANDBOX_GPKG, layer="buildings")
streets = gpd.read_file(SANDBOX_GPKG, layer="streets")
boundary = gpd.read_file(SANDBOX_GPKG, layer="boundary")

# %% [markdown]
# ## Benchmark: Rasterization Pipeline

# %%
print("="*50)
print("RASTERIZATION PIPELINE")
print("="*50)

hub_size = 100
resolve_overlaps = True

t0 = time.time()
city = RasterCity(boundary.geometry.iloc[0], streets, buildings, block_side_length=BLOCK_SIDE_LENGTH, resolve_overlaps=resolve_overlaps)
gen_time = time.time() - t0
print(f"City generation:    {gen_time:>6.2f}s")

t1 = time.time()
G = city.get_street_graph()
graph_time = time.time() - t1
print(f"Street graph:       {graph_time:>6.2f}s")

t2 = time.time()
city._build_hub_network(hub_size=hub_size)
hub_time = time.time() - t2
print(f"Hub network:        {hub_time:>6.2f}s")

t3 = time.time()
city.compute_gravity(exponent=2.0, callable_only=True)
grav_time = time.time() - t3
print(f"Gravity computation: {grav_time:>6.2f}s")

raster_time = gen_time + graph_time + hub_time + grav_time
print("-"*50)
print(f"Rasterization:      {raster_time:>6.2f}s")
print("="*50)

if data_gen_time > 0:
    total_time = data_gen_time + raster_time
    print(f"\nTotal (with data):  {total_time:>6.2f}s")

# %%
# Verification: Check geometry columns and building_type/building_id consistency
print("\n" + "="*50)
print("VERIFICATION CHECKS")
print("="*50)

# Check that geometry columns are populated
blocks_null_geom = city.blocks_gdf.geometry.isna().sum()
streets_null_geom = city.streets_gdf.geometry.isna().sum()
print(f"Blocks with null geometry: {blocks_null_geom}/{len(city.blocks_gdf)}")
print(f"Streets with null geometry: {streets_null_geom}/{len(city.streets_gdf)}")

# Check geometry containment within city_boundary
if blocks_null_geom == 0:
    blocks_outside = ~city.blocks_gdf.geometry.within(city.city_boundary)
    blocks_outside_count = blocks_outside.sum()
    print(f"Blocks outside city_boundary: {blocks_outside_count}/{len(city.blocks_gdf)}")
    if blocks_outside_count > 0:
        print(f"  Example coordinates: {city.blocks_gdf[blocks_outside].index.tolist()[:5]}")

if streets_null_geom == 0:
    streets_outside = ~city.streets_gdf.geometry.within(city.city_boundary)
    streets_outside_count = streets_outside.sum()
    print(f"Streets outside city_boundary: {streets_outside_count}/{len(city.streets_gdf)}")
    if streets_outside_count > 0:
        print(f"  Example coordinates: {city.streets_gdf[streets_outside].index.tolist()[:5]}")

# Check building_type/building_id consistency
has_type_no_id = city.blocks_gdf['building_type'].notna() & city.blocks_gdf['building_id'].isna() & ~(city.blocks_gdf['building_type'] == 'street')
has_id_no_type = city.blocks_gdf['building_id'].notna() & city.blocks_gdf['building_type'].isna()
street_blocks_with_id = (city.blocks_gdf['building_type'] == 'street') & city.blocks_gdf['building_id'].notna()

print(f"\nBuilding type/ID consistency:")
print(f"  Blocks with building_type but no building_id: {has_type_no_id.sum()}")
if has_type_no_id.any():
    print(f"    Example coordinates: {city.blocks_gdf[has_type_no_id].index.tolist()[:5]}")
print(f"  Blocks with building_id but no building_type: {has_id_no_type.sum()}")
if has_id_no_type.any():
    print(f"    Example coordinates: {city.blocks_gdf[has_id_no_type].index.tolist()[:5]}")
print(f"  Street blocks with building_id (should be 0): {street_blocks_with_id.sum()}")
if street_blocks_with_id.any():
    print(f"    Example coordinates: {city.blocks_gdf[street_blocks_with_id].index.tolist()[:5]}")

print("="*50 + "\n")

# %%
city.city_boundary.bounds


# %% [markdown]
# ## Summary: City Structure

# %%
def get_size_mb(obj):
    """Estimate memory size in MB for common objects."""
    if isinstance(obj, (pd.DataFrame, gpd.GeoDataFrame)):
        return obj.memory_usage(deep=True).sum() / 1024**2
    elif hasattr(obj, 'nodes') and hasattr(obj, 'edges'):  # NetworkX graph
        # Approximate: 64 bytes per node + 96 bytes per edge
        return (len(obj.nodes) * 64 + len(obj.edges) * 96) / 1024**2
    else:
        return 0.0

summary_df = pd.DataFrame({
    'Component': ['Blocks', 'Streets', 'Buildings', 'Graph Nodes', 'Graph Edges', 'Hub Network', 'Hub Info', 'Nearby Doors', 'Gravity (callable)'],
    'Count/Shape': [
        f"{len(city.blocks_gdf):,}",
        f"{len(city.streets_gdf):,}",
        f"{len(city.buildings_gdf):,}",
        f"{len(G.nodes):,}",
        f"{len(G.edges):,}",
        f"{city.hub_df.shape[0]}×{city.hub_df.shape[1]}",
        f"{city.grav_hub_info.shape[0]}×{city.grav_hub_info.shape[1]}",
        f"{len(city.mh_dist_nearby_doors):,} pairs",
        "function"
    ],
    'Memory (MB)': [
        f"{get_size_mb(city.blocks_gdf):.1f}",
        f"{get_size_mb(city.streets_gdf):.1f}",
        f"{get_size_mb(city.buildings_gdf):.1f}",
        f"{get_size_mb(G):.1f}",
        "-",
        f"{get_size_mb(city.hub_df):.1f}",
        f"{get_size_mb(city.grav_hub_info):.1f}",
        f"{get_size_mb(city.mh_dist_nearby_doors):.1f}",
        "<0.1"
    ]
})
print("\n" + summary_df.to_string(index=False))
