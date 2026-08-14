# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
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
# # Random City Generation

# %% execution={"iopub.execute_input": "2026-04-24T17:16:13.633934Z", "iopub.status.busy": "2026-04-24T17:16:13.633934Z", "iopub.status.idle": "2026-04-24T17:16:18.079970Z", "shell.execute_reply": "2026-04-24T17:16:18.078969Z"} tags=[]
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as npr
import random
from shapely.geometry import box
from pprint import pprint

import nomad.city_gen as cg
from nomad.city_gen import City, Street, RandomCityGenerator
import nomad.traj_gen as tg
from nomad.traj_gen import Agent, Population

from nomad.constants import DEFAULT_SPEEDS, FAST_SPEEDS, SLOW_SPEEDS, DEFAULT_STILL_PROBS
from nomad.constants import FAST_STILL_PROBS, SLOW_STILL_PROBS, ALLOWED_BUILDINGS

import os

# %% [markdown]
# Create City

# %% execution={"iopub.execute_input": "2026-04-24T17:16:18.082970Z", "iopub.status.busy": "2026-04-24T17:16:18.081971Z", "iopub.status.idle": "2026-04-24T17:17:15.300300Z", "shell.execute_reply": "2026-04-24T17:17:15.300300Z"} tags=[]
city_generator = RandomCityGenerator(width=101, 
                                     height=101, 
                                     street_spacing=5, 
                                     park_ratio=0.05, 
                                     home_ratio=0.4, 
                                     work_ratio=0.3, 
                                     retail_ratio=0.25, 
                                     seed=100)
clustered_city = city_generator.generate_city()
clustered_city.compute_gravity()

# %% execution={"iopub.execute_input": "2026-04-24T17:17:15.304300Z", "iopub.status.busy": "2026-04-24T17:17:15.304300Z", "iopub.status.idle": "2026-04-24T17:17:22.368601Z", "shell.execute_reply": "2026-04-24T17:17:22.367601Z"} tags=[]
# %matplotlib inline

fig, ax = plt.subplots(figsize=(10, 10))
plt.box(on=False)

clustered_city.plot_city(ax, doors=True, address=False)

# remove axis labels and ticks
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])

plt.show()
plt.savefig("random-city.png")

# %% execution={"iopub.execute_input": "2026-04-24T17:17:22.370602Z", "iopub.status.busy": "2026-04-24T17:17:22.370602Z", "iopub.status.idle": "2026-04-24T17:17:22.848579Z", "shell.execute_reply": "2026-04-24T17:17:22.848579Z"} tags=[]
population = Population(clustered_city)
population.generate_agents(N=1, seed=100, datetimes=pd.Timestamp("2025-01-01 00:00", tz="America/New_York"))

for i, agent_id in enumerate(population.roster):
    agent = population.roster[agent_id]
    agent.generate_trajectory(end_time=pd.Timestamp("2025-01-02 00:59", tz="America/New_York"),
                              seed=100+i)
    agent.set_beta_params(
        beta_start=300,
        beta_durations=60,
        beta_ping=10)
    agent.sample_trajectory(
        seed=100+i)
    sampled_traj = agent.sparse_traj

# %% execution={"iopub.execute_input": "2026-04-24T17:17:22.850582Z", "iopub.status.busy": "2026-04-24T17:17:22.850582Z", "iopub.status.idle": "2026-04-24T17:17:30.536559Z", "shell.execute_reply": "2026-04-24T17:17:30.536559Z"} tags=[]
# Visualization
sample_user = population.roster['nifty_saha']
fig, ax = plt.subplots(figsize=(10, 10))
clustered_city.plot_city(ax, doors=True, address=False, zorder=1)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])

ax.scatter(x=sample_user.trajectory.x, 
           y=sample_user.trajectory.y, 
           s=0.5, color='red', alpha=0.1)

plt.savefig("random-city-one-user.png")
