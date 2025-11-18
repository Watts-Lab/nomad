# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Install the `nomad` package from GitHub

# %%
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as npr
import random
from shapely.geometry import box
from pprint import pprint

import nomad.city_gen as cg
from nomad.city_gen import City, RandomCityGenerator
import nomad.traj_gen as tg
from nomad.traj_gen import Agent, Population
import nomad.stop_detection as sd

from nomad.constants import DEFAULT_SPEEDS, FAST_SPEEDS, SLOW_SPEEDS, DEFAULT_STILL_PROBS
from nomad.constants import FAST_STILL_PROBS, SLOW_STILL_PROBS, ALLOWED_BUILDINGS

import os
os.environ['TZ'] = 'UTC'

import pdb

# %% [markdown]
# Create City

# %%
city_generator = RandomCityGenerator(width=101, 
                                     height=101, 
                                     street_spacing=5, 
                                     park_ratio=0.05, 
                                     home_ratio=0.4, 
                                     work_ratio=0.3, 
                                     retail_ratio=0.25, 
                                     seed=100,
                                     verbose=False)
print("Generating city...")
clustered_city = city_generator.generate_city()
print(f"City generated: {len(clustered_city.buildings_gdf)} buildings, {len(clustered_city.streets_gdf) if hasattr(clustered_city, 'streets_gdf') else 'N/A'} streets")
print("Building hub network...")
clustered_city._build_hub_network(hub_size=16)
print("Computing gravity...")
clustered_city.compute_gravity(exponent=2.0)
print("Done!")

# %%
# %matplotlib inline

fig, ax = plt.subplots(figsize=(10, 10))
plt.box(on=False)

clustered_city.plot_city(ax, doors=True, address=False, legend=True)

# remove axis labels and ticks
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])

plt.show()
plt.savefig("random-city.png")

# %%
population = Population(clustered_city)
population.generate_agents(N=1, seed=100, datetimes="2025-01-01 07:00")

for i, agent_id in enumerate(population.roster):
    agent = population.roster[agent_id]
    agent.generate_trajectory(datetime=pd.Timestamp(2025, 1, 1, hour=7, minute=0),
                              end_time=pd.Timestamp(2025, 1, 8, hour=0, minute=0),
                              seed=100+i)
    agent.sample_trajectory(
        beta_start=300,
        beta_durations=60,
        beta_ping=10,
        seed=100+i)
    sampled_traj = agent.sparse_traj

# %%
Zach = population.roster['nifty_saha']

# %%
fig, ax = plt.subplots(figsize=(10, 10))
clustered_city.plot_city(ax, doors=True, address=False, zorder=1, legend=True)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])

ax.scatter(x=Zach.trajectory.x, 
           y=Zach.trajectory.y, 
           s=0.5, color='red', alpha=0.1)

plt.savefig("random-city-one-user.png")
