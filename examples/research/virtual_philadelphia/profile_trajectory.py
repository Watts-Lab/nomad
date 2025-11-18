from pathlib import Path
import cProfile
import pstats
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from nomad.city_gen import RasterCity
from nomad.traj_gen import Population

LARGE_BOX = box(-75.1905, 39.9235, -75.1425, 39.9535)
buildings = gpd.read_file(Path("output/spatial_data_large.gpkg"), layer="buildings")
streets = gpd.read_file(Path("output/spatial_data_large.gpkg"), layer="streets")
boundary = gpd.read_file(Path("output/spatial_data_large.gpkg"), layer="boundary")

city = RasterCity(
    boundary.geometry.iloc[0],
    streets,
    buildings,
    block_side_length=15.0,
    resolve_overlaps=True,
    other_building_behavior="filter"
)
city.get_street_graph()
city._build_hub_network(hub_size=100)
city.compute_gravity(exponent=2.0, callable_only=True)
city.compute_shortest_paths(callable_only=True)

population = Population(city)
population.generate_agents(
    N=5,
    seed=42,
    name_count=2,
    datetimes="2025-05-23 00:00-05:00"
)

NUM_AGENTS = 5
NUM_DAYS = 14

# Profile destination diary generation
profiler = cProfile.Profile()
profiler.enable()
for i, agent in enumerate(population.roster.values()):
    agent.generate_dest_diary(
        end_time=pd.Timestamp("2025-06-06 00:00-05:00"),
        epr_time_res=15,
        rho=0.4,
        gamma=0.3,
        seed=100 + i
    )
profiler.disable()
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
print("\n=== DESTINATION DIARY GENERATION ===")
diary_time = stats.total_tt
print(f"Total time: {diary_time:.3f}s")
print(f"Time per agent per day: {diary_time / (NUM_AGENTS * NUM_DAYS):.3f}s")
print()
stats.print_stats(30)

# Profile trajectory generation
profiler = cProfile.Profile()
profiler.enable()
for i, agent in enumerate(population.roster.values()):
    agent.generate_trajectory(dt=0.5, seed=200 + i)
profiler.disable()
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
print("\n=== TRAJECTORY GENERATION ===")
traj_time = stats.total_tt
print(f"Total time: {traj_time:.3f}s")
print(f"Time per agent per day: {traj_time / (NUM_AGENTS * NUM_DAYS):.3f}s")
print()
stats.print_stats(30)

