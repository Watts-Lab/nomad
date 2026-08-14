# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Synthetic Philadelphia - Production Pipeline
#
# Full rasterization pipeline with EPR destination diary generation.

# %%
from pathlib import Path
import time
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import nomad.map_utils as nm
from nomad.city_gen import RasterCity
from nomad.traj_gen import Population
from tqdm import tqdm

# %% [markdown]
# ## Configuration

# %%
LARGE_BOX = box(-75.212193, 39.940800, -75.136933, 39.962847)

USE_FULL_CITY = False
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if USE_FULL_CITY:
    BOX_NAME = "full"
    POLY = "Philadelphia, Pennsylvania, USA"
else:
    BOX_NAME = "large"
    POLY = LARGE_BOX

SANDBOX_GPKG = OUTPUT_DIR / f"spatial_data_{BOX_NAME}.gpkg"
REGENERATE_DATA = False  # Set to True to regenerate data with rotation metadata

config = {
    "box_name": BOX_NAME,
    "block_side_length": 15.0,
    "hub_size": 100,
    "N": 1000,
    "name_seed": 42,
    "name_count": 2,
    "epr_params": {
        "datetime": "2025-05-23 00:00-05:00",
        "end_time": "2025-07-01 00:00-05:00",
        "epr_time_res": 15,
        "rho": 0.4,
        "gamma": 0.3,
        "seed_base": 100
    },
    "traj_params": {
        "dt": 0.5,
        "seed_base": 200
    },
    "sampling_params": {
        "beta_ping": 7,
        "beta_start": 300,
        "beta_durations": 55,
        "ha": 11.5 / 15,
        "seed_base": 1
    }
}


# %% [markdown]
# ## Data Generation (OSM Download + Rotation)

# %%
if REGENERATE_DATA or not SANDBOX_GPKG.exists():
    data_start = time.time()
    buildings = nm.download_osm_buildings(
        POLY,
        crs="EPSG:3857",
        schema="garden_city",
        clip=True,
        infer_building_types=True,
        explode=True,
    )
    download_buildings_time = time.time() - data_start
    print(f"Buildings download: {download_buildings_time:>6.2f}s ({len(buildings):,} buildings)")
    
    if USE_FULL_CITY:
        boundary_polygon = nm.get_city_boundary_osm(POLY, simplify=True)[0]
        boundary_polygon = gpd.GeoSeries([boundary_polygon], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    else:
        boundary_polygon = gpd.GeoDataFrame(geometry=[POLY], crs="EPSG:4326").to_crs("EPSG:3857").geometry.iloc[0]
    
    if (~buildings.geometry.within(boundary_polygon)).any():
        buildings = gpd.clip(buildings, gpd.GeoDataFrame(geometry=[boundary_polygon], crs="EPSG:3857"))
    buildings = nm.remove_overlaps(buildings).reset_index(drop=True)
    
    t1 = time.time()
    streets = nm.download_osm_streets(
        POLY,
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
    
    # Get rotation origin (centroid of original streets before rotation)
    all_streets = streets.geometry.union_all()
    rotation_origin = (all_streets.centroid.x, all_streets.centroid.y)
    
    rotated_buildings = nm.rotate(buildings, rotation_deg=rotation_deg, origin=rotation_origin)
    rotated_boundary = nm.rotate(
        gpd.GeoDataFrame(geometry=[boundary_polygon], crs="EPSG:3857"),
        rotation_deg=rotation_deg,
        origin=rotation_origin
    )
    
    if SANDBOX_GPKG.exists():
        SANDBOX_GPKG.unlink()
    
    rotated_buildings.to_file(SANDBOX_GPKG, layer="buildings", driver="GPKG")
    rotated_streets.to_file(SANDBOX_GPKG, layer="streets", driver="GPKG", mode="a")
    rotated_boundary.to_file(SANDBOX_GPKG, layer="boundary", driver="GPKG", mode="a")
    
    # Store rotation_deg and rotation_origin in metadata JSON for later retrieval
    with open(OUTPUT_DIR / f"rotation_metadata_{BOX_NAME}.json", 'w') as f:
        json.dump({
            'rotation_deg': rotation_deg,
            'rotation_origin': rotation_origin
        }, f)
    
    data_gen_time = time.time() - data_start
    print(f"Data generation:    {data_gen_time:>6.2f}s")
else:
    print(f"Loading existing data from {SANDBOX_GPKG}")
    data_gen_time = 0.0

buildings = gpd.read_file(SANDBOX_GPKG, layer="buildings")
streets = gpd.read_file(SANDBOX_GPKG, layer="streets")
boundary = gpd.read_file(SANDBOX_GPKG, layer="boundary")

with open(OUTPUT_DIR / f"rotation_metadata_{BOX_NAME}.json", 'r') as f:
    rotation_metadata = json.load(f)
rotation_deg = rotation_metadata['rotation_deg']
rotation_origin = rotation_metadata['rotation_origin']

# %% [markdown]
# ## Rasterization Pipeline

# %%
t0 = time.time()
city = RasterCity(
    boundary.geometry.iloc[0],
    streets,
    buildings,
    block_side_length=config["block_side_length"],
    resolve_overlaps=True,
    other_building_behavior="filter",
    rotation_deg=rotation_deg,
    rotation_origin=rotation_origin
)
gen_time = time.time() - t0
print(f"City generation:    {gen_time:>6.2f}s")

t1 = time.time()
G = city.get_street_graph()
graph_time = time.time() - t1
print(f"Street graph:       {graph_time:>6.2f}s")

t2 = time.time()
city._build_hub_network(hub_size=config["hub_size"])
hub_time = time.time() - t2
print(f"Hub network:        {hub_time:>6.2f}s")

t3 = time.time()
city.compute_gravity(exponent=2.0, callable_only=True)
grav_time = time.time() - t3
print(f"Gravity computation: {grav_time:>6.2f}s")

t4 = time.time()
city.compute_shortest_paths(callable_only=True)
paths_time = time.time() - t4
print(f"Shortest paths:     {paths_time:>6.2f}s")

raster_time = gen_time + graph_time + hub_time + grav_time + paths_time
print(f"Rasterization:      {raster_time:>6.2f}s")

if data_gen_time > 0:
    print(f"\nTotal (with data):  {data_gen_time + raster_time:>6.2f}s")

# %% [markdown]
# ## Summary: City Structure

# %%
print(pd.Series({
    'Blocks': len(city.blocks_gdf),
    'Streets': len(city.streets_gdf),
    'Buildings': len(city.buildings_gdf),
    'Graph nodes': len(G.nodes),
    'Graph edges': len(G.edges),
    'Hubs': len(city.hubs),
    'Nearby door pairs': len(city.mh_dist_nearby_doors)
}, name='Count').to_string())
print(city.buildings_gdf.building_type.value_counts())

# %% [markdown]
# ## Generate Population and Destination Diaries

# %%
config_path = OUTPUT_DIR / f"config_{BOX_NAME}.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

population = Population(city)
population.generate_agents(
    N=config["N"],
    seed=config["name_seed"],
    name_count=config["name_count"],
    datetimes=config["epr_params"]["datetime"]
)

end_time = pd.Timestamp(config["epr_params"]["end_time"])

t1 = time.time()
for i, agent in tqdm(enumerate(population.roster.values()), total=config["N"]):
    agent.generate_dest_diary(
        end_time=end_time,
        epr_time_res=config["epr_params"]["epr_time_res"],
        rho=config["epr_params"]["rho"],
        gamma=config["epr_params"]["gamma"],
        seed=config["epr_params"]["seed_base"] + i
    )

diary_gen_time = time.time() - t1
print(f"Diary generation:   {diary_gen_time:>6.2f}s")

print(f"Total entries:      {sum(len(agent.destination_diary) for agent in population.roster.values()):,}")

dest_diaries_path = OUTPUT_DIR / f"dest_diaries_{BOX_NAME}"
t2 = time.time()
population.save_pop(
    dest_diaries_path=dest_diaries_path,
    partition_cols=["date"],
    traj_cols={'user_id': 'identifier', 'geohash': 'location'}
)
persist_time = time.time() - t2
print(f"Persistence:        {persist_time:>6.2f}s")

print(f"\nConfig saved to {config_path}")
print(f"Destination diaries saved to {dest_diaries_path}")

# %% [markdown]
# ## Generate and Sample Trajectories

# %%
t1 = time.time()
total_points = 0
total_sparse_points = 0
for i, agent in tqdm(enumerate(population.roster.values()), total=config["N"]):
    agent.generate_trajectory(
        dt=config["traj_params"]["dt"],
        seed=config["traj_params"]["seed_base"] + i
    )
    total_points += len(agent.trajectory)
    agent.set_beta_params(
        beta_ping=config["sampling_params"]["beta_ping"],
        beta_durations=config["sampling_params"]["beta_durations"],
        beta_start=config["sampling_params"]["beta_start"]
    )
    agent.sample_trajectory(
        ha=config["sampling_params"]["ha"],
        seed=config["sampling_params"]["seed_base"] + i,
        flush_traj_cache=True
    )
    total_sparse_points += len(agent.sparse_traj)

generation_time = time.time() - t1
print(f"Generation:         {generation_time:>6.2f}s")
print(f"Dense points:       {total_points:,}")
print(f"Points per second:  {total_points/generation_time:.1f}")
print(f"Total sparse points: {total_sparse_points:,}")
print(f"Sparsity ratio: {total_sparse_points/total_points:.2%}")

# %% [markdown]
# ## Reproject to Mercator and Persist

# %%
print("Reprojecting sparse trajectories to Web Mercator...")
population.reproject_to_mercator(diaries=True)

print("Saving sparse trajectories and diaries...")
population.save_pop(
    sparse_path=OUTPUT_DIR / "device_level",
    diaries_path=OUTPUT_DIR / "travel_diaries",
    homes_path=OUTPUT_DIR / f"homes_{BOX_NAME}",
    partition_cols=["date"]
)
