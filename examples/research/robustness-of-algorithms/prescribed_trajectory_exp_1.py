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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
import json
from tqdm import tqdm
import os

# %%
import nomad.io.base as loader
import nomad.city_gen as cg
import nomad.traj_gen as tg
from nomad.traj_gen import Agent, Population
from nomad.city_gen import City

# %% [markdown]
# ## Load city and configure destination diaries

# %%
# Load the city
city_file = '../../../nomad/data/garden-city.gpkg'
city = cg.City.from_geopackage(city_file)
start = '2024-06-01 00:00-04:00'

#option 1: symmetric
start_time = pd.date_range(start=start, periods=4, freq='60min')
unix_timestamp = [int(t.timestamp()) for t in start_time]
duration = [60]*4  # in minutes
location = ['h-x13-y11'] * 1 + ['h-x13-y9'] * 1 + ['w-x18-y10'] * 1 + ['w-x18-y8'] * 1

destinations = pd.DataFrame(
    {
        "datetime":start_time,
         "timestamp":unix_timestamp,
         "duration":duration,
         "location":location
    }
)
destinations.to_csv("exp_1_destinations_balanced.csv", index=False)

#option 2: we break symmetries to produce more density heterogeneity
start_time = pd.date_range(start=start, periods=8, freq='30min')
unix_timestamp = [int(t.timestamp()) for t in start_time]
duration = [30]*8 
location = ['h-x13-y11'] * 2 + ['h-x13-y9'] * 1 + ['w-x18-y10'] * 2 + ['w-x18-y8'] * 3

destinations = pd.DataFrame(
    {
        "datetime":start_time,
         "timestamp":unix_timestamp,
         "duration":duration,
         "location":location
    }
)
destinations.to_csv("exp_1_destinations_unbalanced.csv", index=False)

# %% [markdown]
# ## Config files for simulations
# This could be in a json or yaml and should be passable to a Population object.

# %%
# option 1 (reduced for quick demo run)
N_reps = 2
sparsity_samples = 3
config = dict(
    dt = 0.5,
    N = N_reps*sparsity_samples,
    name_count=2,
    name_seed=2025,
    city_file='./../../../nomad/data/garden-city.gpkg',
    destination_diary_file='exp_1_destinations_balanced.csv',
    output_files = dict(
        sparse_path='./sparse_traj_1',
        diaries_path='./diaries_1',
        homes_path='./homes_1'
    ),
    agent_params = dict(
        agent_homes='h-x13-y11',
        agent_workplaces='w-x18-y8',
        seed_trajectory=list(range(N_reps*sparsity_samples)),
        seed_sparsity= list(range(N_reps*sparsity_samples)),
        beta_ping= np.repeat(np.linspace(1, 20, sparsity_samples), N_reps).tolist(),
        beta_durations=None,
        beta_start=None,
        ha=11.5/15
    )
)
with open('config_low_ha.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

# option 2 (reduced for quick demo run)
N_reps = 2
sparsity_samples = 3
config_2 = dict(
    dt = 0.5,
    N = N_reps*sparsity_samples,
    name_count=2,
    name_seed=2025,
    city_file='./../../../nomad/data/garden-city.gpkg',
    destination_diary_file='exp_1_destinations_unbalanced.csv',
    output_files = dict(
        sparse_path='./sparse_traj_2',
        diaries_path='./diaries_2',
        homes_path='./homes_2'
    ),
    agent_params = dict(
        agent_homes='h-x13-y11',
        agent_workplaces='w-x18-y8',
        seed_trajectory=list(range(N_reps*sparsity_samples)),
        seed_sparsity= list(range(N_reps*sparsity_samples)),
        beta_ping= np.repeat(np.linspace(1, 20, sparsity_samples), N_reps).tolist(),
        beta_durations=None,
        beta_start=None,
        ha=15/15
    )
)
with open('config_high_ha.json', 'w', encoding='utf-8') as f:
    json.dump(config_2, f, ensure_ascii=False, indent=4)

# %% [markdown]
# ## Generate trajectories

# %%
# Parameters according to the config file
with open('config_high_ha.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    
# Load city and destination diary from config
city = City.from_geopackage(config["city_file"])
# Build POI data from buildings_gdf door info
poi_data = pd.DataFrame({
    'building_id': city.buildings_gdf['id'].values,
    'x': city.buildings_gdf['door_point'].apply(lambda p: p[0]).values,
    'y': city.buildings_gdf['door_point'].apply(lambda p: p[1]).values
})

destinations = pd.read_csv(config["destination_diary_file"], parse_dates=["datetime"])

population = Population(city)
population.generate_agents(
    N=config["N"], 
    seed=config["name_seed"], 
    name_count=config["name_count"],
    agent_homes=config["agent_params"]["agent_homes"],
    agent_workplaces=config["agent_params"]["agent_workplaces"]
)

for i, agent in enumerate(tqdm(population.roster.values(), desc="Generating trajectories")):
    agent.generate_trajectory(
        destination_diary=destinations,
        dt=config["dt"],
        seed=config["agent_params"]["seed_trajectory"][i],
        step_seed=config["agent_params"]["seed_trajectory"][i])
    
    agent.sample_trajectory(
        beta_ping=config["agent_params"]["beta_ping"][i],
        seed=config["agent_params"]["seed_sparsity"][i],
        ha=config["agent_params"]["ha"],
        replace_sparse_traj=True)

# Reproject all trajectories to Web Mercator at population level
print("Reprojecting trajectories to Web Mercator...")
population.reproject_to_mercator(sparse_traj=True, full_traj=False, diaries=True, poi_data=poi_data)

# %%
# Save output files using save_pop method
print("Saving output files...")
population.save_pop(
    sparse_path=config["output_files"]["sparse_path"],
    diaries_path=config["output_files"]["diaries_path"],
    homes_path=config["output_files"]["homes_path"],
    beta_ping=config["agent_params"]["beta_ping"],
    ha=config["agent_params"]["ha"]
)
print("All output files saved successfully!")
