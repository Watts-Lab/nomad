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
#     display_name: nomad-kernel
#     language: python
#     name: nomad-kernel
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
import matplotlib.pyplot as plt
import contextily as cx

import nomad.map_utils as nm
from nomad.city_gen import RasterCity
from nomad.traj_gen import Population
from nomad.io.base import from_file
from tqdm import tqdm

# %% [markdown]
# ## Configuration

# %%
LARGE_BOX = box(-75.1905, 39.9235, -75.1425, 39.9535)
MEDIUM_BOX = box(-75.1665, 39.9385, -75.1425, 39.9535)

USE_FULL_CITY = False
OUTPUT_DIR = Path("sandbox")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if USE_FULL_CITY:
    BOX_NAME = "full"
    POLY = "Philadelphia, Pennsylvania, USA"
else:
    BOX_NAME = "medium"
    POLY = MEDIUM_BOX

SANDBOX_GPKG = OUTPUT_DIR / f"sandbox_data_{BOX_NAME}.gpkg"
REGENERATE_DATA = False

config = {
    "box_name": BOX_NAME,
    "block_side_length": 10.0,
    "hub_size": 100,
    "N": 10,
    "name_seed": 42,
    "name_count": 2,
    "epr_params": {
        "datetime": "2024-01-01 00:00-05:00",
        "end_time": "2024-01-08 00:00-05:00",  # 7 days
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
        "beta_ping": 5,
        "beta_durations": None,
        "beta_start": None,
        "ha": 11.5/15,
        "seed_base": 300
    }
}


# %% [markdown]
# ## Data Generation (OSM Download + Rotation)

# %%
if REGENERATE_DATA or not SANDBOX_GPKG.exists():
    print("="*50)
    print("DATA GENERATION")
    print("="*50)
    
    t0 = time.time()
    buildings = nm.download_osm_buildings(
        POLY,
        crs="EPSG:3857",
        schema="garden_city",
        clip=True,
        infer_building_types=True,
        explode=True,
    )
    download_buildings_time = time.time() - t0
    print(f"Buildings download: {download_buildings_time:>6.2f}s ({len(buildings):,} buildings)")
    
    if USE_FULL_CITY:
        boundary_polygon = nm.get_city_boundary_osm(POLY, simplify=True)[0]
        boundary_polygon = gpd.GeoSeries([boundary_polygon], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    else:
        boundary_polygon = gpd.GeoDataFrame(geometry=[POLY], crs="EPSG:4326").to_crs("EPSG:3857").geometry.iloc[0]
    
    outside_mask = ~buildings.geometry.within(boundary_polygon)
    if outside_mask.any():
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
# ## Rasterization Pipeline

# %%
print("="*50)
print("RASTERIZATION PIPELINE")
print("="*50)

t0 = time.time()
city = RasterCity(
    boundary.geometry.iloc[0],
    streets,
    buildings,
    block_side_length=config["block_side_length"],
    resolve_overlaps=True,
    other_building_behavior="filter"
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
print("-"*50)
print(f"Rasterization:      {raster_time:>6.2f}s")
print("="*50)

if data_gen_time > 0:
    total_time = data_gen_time + raster_time
    print(f"\nTotal (with data):  {total_time:>6.2f}s")

# %% [markdown]
# ## Summary: City Structure

# %%
def get_size_mb(obj):
    if isinstance(obj, (pd.DataFrame, gpd.GeoDataFrame)):
        return obj.memory_usage(deep=True).sum() / 1024**2
    elif hasattr(obj, 'nodes') and hasattr(obj, 'edges'):
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
print(city.buildings_gdf.building_type.value_counts())

# %% [markdown]
# ## Generate Population and Destination Diaries

# %%
print("\n" + "="*50)
print("DESTINATION DIARY GENERATION")
print("="*50)

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

total_entries = sum(len(agent.destination_diary) for agent in population.roster.values())
print(f"Total entries:      {total_entries:,}")

dest_diaries_path = OUTPUT_DIR / f"dest_diaries_{BOX_NAME}"
t2 = time.time()
population.save_pop(
    dest_diaries_path=dest_diaries_path,
    partition_cols=["date"],
    fmt='parquet',
    traj_cols={'geohash': 'location'}
)
persist_time = time.time() - t2
print(f"Persistence:        {persist_time:>6.2f}s")
print("-"*50)
print(f"Total EPR:          {diary_gen_time:>6.2f}s")
print("="*50)

print(f"\nConfig saved to {config_path}")
print(f"Destination diaries saved to {dest_diaries_path}")

# %% [markdown]
# ## Generate Full Trajectories from Destination Diaries

# %%
print("\n" + "="*50)
print("TRAJECTORY GENERATION")
print("="*50)

t1 = time.time()
failed_agents = []
for i, agent in tqdm(enumerate(population.roster.values()), total=config["N"], desc="Generating trajectories"):
    try:
        agent.generate_trajectory(
            dt=config["traj_params"]["dt"],
            seed=config["traj_params"]["seed_base"] + i
        )
    except ValueError as e:
        failed_agents.append((agent.identifier, str(e)))
        continue

traj_gen_time = time.time() - t1
print(f"Trajectory generation: {traj_gen_time:>6.2f}s")
if failed_agents:
    print(f"Warning: {len(failed_agents)} agents failed trajectory generation")

total_points = sum(len(agent.trajectory) for agent in population.roster.values() if agent.trajectory is not None)
print(f"Total trajectory points: {total_points:,}")
print(f"Points per second: {total_points/traj_gen_time:.1f}")
print("-"*50)
print(f"Total trajectory:   {traj_gen_time:>6.2f}s")
print("="*50)

# %% [markdown]
# ## Sample Sparse Trajectories

# %%
print("\n" + "="*50)
print("SPARSE TRAJECTORY SAMPLING")
print("="*50)

t1 = time.time()
for i, agent in tqdm(enumerate(population.roster.values()), total=config["N"], desc="Sampling trajectories"):
    if agent.trajectory is None:
        continue
    agent.sample_trajectory(
        beta_ping=config["sampling_params"]["beta_ping"],
        beta_durations=config["sampling_params"]["beta_durations"],
        beta_start=config["sampling_params"]["beta_start"],
        ha=config["sampling_params"]["ha"],
        seed=config["sampling_params"]["seed_base"] + i,
        replace_sparse_traj=True
    )

sampling_time = time.time() - t1
print(f"Sparse sampling:    {sampling_time:>6.2f}s")

total_sparse_points = sum(len(agent.sparse_traj) for agent in population.roster.values() if agent.sparse_traj is not None)
print(f"Total sparse points: {total_sparse_points:,}")
print(f"Sparsity ratio: {total_sparse_points/total_points:.2%}")
print("-"*50)
print(f"Total sampling:     {sampling_time:>6.2f}s")
print("="*50)

# %% [markdown]
# ## Reproject to Mercator and Persist

# %%
print("\n" + "="*50)
print("REPROJECTION AND PERSISTENCE")
print("="*50)

# Build POI data for diary reprojection
cent = city.buildings_gdf['door_point'] if 'door_point' in city.buildings_gdf.columns else city.buildings_gdf.geometry.centroid
poi_data = pd.DataFrame({
    'building_id': city.buildings_gdf['id'].values,
    'x': (city.buildings_gdf['door_cell_x'].astype(float) + 0.5).values if 'door_cell_x' in city.buildings_gdf.columns else cent.x.values,
    'y': (city.buildings_gdf['door_cell_y'].astype(float) + 0.5).values if 'door_cell_y' in city.buildings_gdf.columns else cent.y.values
})

print("Reprojecting sparse trajectories to Web Mercator...")
population.reproject_to_mercator(sparse_traj=True, full_traj=False, diaries=True, poi_data=poi_data)

print("Saving sparse trajectories and diaries...")
population.save_pop(
    sparse_path=OUTPUT_DIR / f"sparse_traj_{BOX_NAME}",
    diaries_path=OUTPUT_DIR / f"diaries_{BOX_NAME}",
    partition_cols=["date"],
    fmt='parquet'
)
print("-"*50)
print("="*50)

# %% [markdown]
# ## Visualize Sparse Trajectories

# %%
print("\n" + "="*50)
print("VISUALIZATION")
print("="*50)

# Read sparse trajectories
sparse_traj_df = from_file(OUTPUT_DIR / f"sparse_traj_{BOX_NAME}", format="parquet")
print(f"Loaded {len(sparse_traj_df):,} sparse trajectory points for {config['N']} agents")

# Plot with contextily basemap
fig, ax = plt.subplots(figsize=(12, 10))

# Plot each agent with different color
for agent_id in sparse_traj_df['user_id'].unique():
    agent_traj = sparse_traj_df[sparse_traj_df['user_id'] == agent_id]
    ax.scatter(agent_traj['x'], agent_traj['y'], s=1, alpha=0.5, label=agent_id)

# Add basemap
cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron)

ax.set_xlabel('X (Web Mercator)')
ax.set_ylabel('Y (Web Mercator)')
ax.set_title(f'Sparse Trajectories - {config["N"]} Agents, 7 Days')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"sparse_trajectories_{BOX_NAME}.png", dpi=150, bbox_inches='tight')
print(f"Saved plot to {OUTPUT_DIR / f'sparse_trajectories_{BOX_NAME}.png'}")
plt.show()

# %%
