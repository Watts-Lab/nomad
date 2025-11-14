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

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Benchmark: `_sample_step` Performance Analysis
#
# ## Goals
# 1. Benchmark repeated `get_shortest_path()` calls (callable mode)
# 2. Profile trajectory generation to identify bottlenecks
# 3. Test optimization strategies

# %%
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

from nomad.city_gen import RasterCity
from nomad.traj_gen import Population

# %%
# Configuration
BOX_SIZE = 'small'  # 'small', 'medium', or 'large'
BLOCK_SIDE_LENGTH = 10.0
HUB_SIZE = 100
MAX_MANHATTAN_DIST = 50
NUM_PATH_QUERIES = 250
SIMULATION_HOURS = 48
DT = 0.5  # minutes
EPR_TIME_RES = 15
RHO = 0.4
GAMMA = 0.3
SEED = 42

# %%
print("="*60)
print(f"SETUP: {BOX_SIZE.upper()} BOX")
print("="*60)

# Load OSM data and rasterize
data_dir = Path("sandbox")
osm_path = data_dir / f"sandbox_data_{BOX_SIZE}.gpkg"

if not osm_path.exists():
    raise FileNotFoundError(f"OSM data not found at {osm_path}. Run rasterization_report.py first.")

print(f"Loading OSM data...")
buildings = gpd.read_file(osm_path, layer="buildings")
streets = gpd.read_file(osm_path, layer="streets")
boundary = gpd.read_file(osm_path, layer="boundary")
print(f"  Buildings: {len(buildings):,}")
print(f"  Streets: {len(streets):,}")

print(f"\nRasterizing city...")
t0 = time.time()
city = RasterCity(
    boundary.geometry.iloc[0],
    streets,
    buildings,
    block_side_length=BLOCK_SIDE_LENGTH,
    resolve_overlaps=True,
    verbose=False
)
print(f"  Rasterization: {time.time()-t0:.2f}s")
print(f"  Buildings added: {len(city.buildings_gdf):,}")
print(f"  Street blocks: {len(city.streets_gdf):,}")

# %%
print("\nBuilding street graph...")
t0 = time.time()
G = city.get_street_graph()
print(f"  Street graph: {time.time()-t0:.2f}s")
print(f"  Nodes: {G.number_of_nodes():,}")
print(f"  Edges: {G.number_of_edges():,}")

# %%
print("\nBuilding hub network...")
t0 = time.time()
city._build_hub_network(hub_size=HUB_SIZE)
print(f"  Hub network: {time.time()-t0:.2f}s")
print(f"  Hubs: {len(city.hubs):,}")

# %%
print("\nComputing gravity (callable mode)...")
t0 = time.time()
city.compute_gravity(exponent=2.0, callable_only=True)
print(f"  Gravity computation: {time.time()-t0:.2f}s")

# %%
print("\nComputing shortest paths (callable mode)...")
t0 = time.time()
city.compute_shortest_paths(callable_only=True)
print(f"  Shortest paths: {time.time()-t0:.2f}s")

# %% [markdown]
# ## Benchmark 1: Repeated `get_shortest_path()` Calls

# %%
print("\n" + "="*60)
print("BENCHMARK 1: get_shortest_path() Performance")
print("="*60)

# Find valid street block pairs with manhattan distance < MAX_MANHATTAN_DIST
streets_list = list(city.streets_gdf.index)
valid_pairs = []

print(f"Finding {NUM_PATH_QUERIES} pairs with Manhattan distance < {MAX_MANHATTAN_DIST}...")
rng = np.random.default_rng(42)

while len(valid_pairs) < NUM_PATH_QUERIES:
    i = rng.integers(0, len(streets_list))
    j = rng.integers(0, len(streets_list))
    if i == j:
        continue
    
    start = streets_list[i]
    end = streets_list[j]
    manhattan_dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    if manhattan_dist < MAX_MANHATTAN_DIST:
        valid_pairs.append((start, end))

print(f"Found {len(valid_pairs)} valid pairs")

# Benchmark
times = []
path_lengths = []

print(f"\nBenchmarking {NUM_PATH_QUERIES} get_shortest_path() calls...")
for start, end in valid_pairs:
    t0 = time.time()
    path = city.get_shortest_path(start, end)
    elapsed = time.time() - t0
    times.append(elapsed)
    path_lengths.append(len(path) if path else 0)

times = np.array(times) * 1000  # Convert to milliseconds
path_lengths = np.array(path_lengths)

print("\nResults:")
print(f"  Total queries: {len(times)}")
print(f"  Mean time: {times.mean():.2f} ms")
print(f"  Median time: {np.median(times):.2f} ms")
print(f"  Std time: {times.std():.2f} ms")
print(f"  Min time: {times.min():.2f} ms")
print(f"  Max time: {times.max():.2f} ms")
print(f"  Mean path length: {path_lengths.mean():.1f} blocks")
print(f"  Median path length: {np.median(path_lengths):.1f} blocks")

# %% [markdown]
# ## Benchmark 2: Destination Diary Generation

# %%
print("\n" + "="*60)
print("BENCHMARK 2: Destination Diary Generation")
print("="*60)

population = Population(city)
population.generate_agents(
    N=1,
    seed=SEED,
    name_count=1,
    datetimes="2024-01-01 00:00-05:00"
)

agent = list(population.roster.values())[0]
end_time = pd.Timestamp("2024-01-01 00:00-05:00") + pd.Timedelta(hours=SIMULATION_HOURS)

print(f"\nGenerating {SIMULATION_HOURS}-hour destination diary...")
print(f"  Start: {agent.last_ping['datetime']}")
print(f"  End: {end_time}")

t0 = time.time()
agent.generate_dest_diary(
    end_time=end_time,
    epr_time_res=EPR_TIME_RES,
    rho=RHO,
    gamma=GAMMA,
    seed=SEED
)
elapsed = time.time() - t0

print(f"\nResults:")
print(f"  Time: {elapsed:.2f}s")
print(f"  Diary entries: {len(agent.destination_diary):,}")

# %% [markdown]
# ## Benchmark 3: Trajectory Generation from Diary

# %%
print("\n" + "="*60)
print("BENCHMARK 3: Trajectory Generation from Diary")
print("="*60)

print(f"\nGenerating trajectory from destination diary...")
print(f"  Diary entries: {len(agent.destination_diary):,}")
print(f"  Time step (dt): {DT} minutes")

t0 = time.time()
agent.generate_trajectory(
    destination_diary=agent.destination_diary,
    dt=DT,
    seed=SEED
)
elapsed = time.time() - t0

trajectory = agent.trajectory

print(f"\nResults:")
print(f"  Time: {elapsed:.2f}s")
print(f"  Trajectory points: {len(trajectory):,}")
print(f"  Points per second: {len(trajectory)/elapsed:.1f}")
print(f"  Time per point: {1000*elapsed/len(trajectory):.2f} ms")
