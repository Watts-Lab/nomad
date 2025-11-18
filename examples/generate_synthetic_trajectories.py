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
# # Synthetic Trajectory Generation with Nomad
#
# This notebook demonstrates how to generate realistic synthetic human mobility trajectories.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import time
import os
from joblib import Parallel, delayed

from nomad.city_gen import City
from nomad.traj_gen import Agent, Population
from nomad.stop_detection.viz import plot_pings, plot_time_barcode

# %%
city = City.from_geopackage('garden-city.gpkg', edges_path='garden-city-edges.parquet')
city._build_hub_network(hub_size=16)
city.compute_gravity(exponent=2.0)

print(f"City: {city.name}")
print(f"Dimensions: {city.dimensions}")
print(f"Buildings: {len(city.buildings_gdf)}")

# %% [markdown]
# ## Part 1: Effect of Sampling Parameters on Sparsity
#
# Generate 3 agents with 2-day trajectories, varying beta_duration and beta_start 
# to show their effect on sparsity (q = observed points / ground truth points).

# %%
np.random.seed(42)
population = Population(city)
population.generate_agents(N=3, seed=42, name_count=2)

# Vary beta_duration and beta_start to target different sparsity levels
sampling_params = [
    {'beta_ping': 5, 'beta_start': 100, 'beta_durations': 60},   
    {'beta_ping': 5, 'beta_start': 250, 'beta_durations': 150},  
    {'beta_ping': 5, 'beta_start': 400, 'beta_durations': 240}   
]

# Generate 2-day trajectories for quick visualization
for i, (agent_id, agent) in enumerate(population.roster.items()):
    agent.generate_trajectory(
        datetime=pd.Timestamp("2024-01-01T07:00-04:00"),
        end_time=pd.Timestamp("2024-01-03T07:00-04:00"),
        seed=i
    )

    agent.sample_trajectory(
        **sampling_params[i],
        replace_sparse_traj=True,
        seed=i
    )
    
    q = len(agent.sparse_traj) / len(agent.trajectory)
    print(f"Agent {i}: q={q:.3f}, beta_start={sampling_params[i]['beta_start']}, "
          f"beta_dur={sampling_params[i]['beta_durations']}")

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10), 
                         gridspec_kw={'height_ratios': [10, 1]})

for i, (agent_id, agent) in enumerate(population.roster.items()):
    ax_map = axes[0, i]
    ax_barcode = axes[1, i]
    
    city.plot_city(ax=ax_map, doors=False, address=False)
    
    traj = agent.sparse_traj
    plot_pings(traj, ax=ax_map, s=15, point_color='red', 
               x='x', y='y', timestamp='timestamp')
    
    plot_time_barcode(traj['timestamp'], ax=ax_barcode, set_xlim=True)
    
    q = len(traj) / len(agent.trajectory)
    ax_map.set_title(f"Agent {i}: {len(traj)} obs (q={q:.2f})\n"
                     f"beta_start={sampling_params[i]['beta_start']}, "
                     f"beta_dur={sampling_params[i]['beta_durations']}")
    ax_map.set_axis_off()

plt.tight_layout()
plt.savefig('data/trajectories_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Part 2: Parallel Generation at Scale
#
# Generate trajectories for 15 users using parallelization.

# %%
def generate_agent_trajectory(args):
    """Worker function for parallel generation."""
    identifier, home, work, seed = args
    
    city = City.from_geopackage('garden-city.gpkg', edges_path='garden-city-edges.parquet')
    city._build_hub_network(hub_size=16)
    city.compute_gravity(exponent=2.0)
    agent = Agent(identifier=identifier, city=city, home=home, workplace=work)
    
    agent.generate_trajectory(
        datetime=pd.Timestamp("2024-01-01T07:00-04:00"),
        end_time=pd.Timestamp("2024-01-08T07:00-04:00"),
        seed=seed
    )

    agent.sample_trajectory(
        beta_ping=5,
        replace_sparse_traj=True,
        seed=seed
    )
    
    sparse_df = agent.sparse_traj.copy()
    sparse_df['user_id'] = identifier
    return sparse_df

# %%
np.random.seed(100)
n_agents = 15
homes = city.buildings_gdf[city.buildings_gdf['building_type'] == 'home']['id'].tolist()
workplaces = city.buildings_gdf[city.buildings_gdf['building_type'] == 'workplace']['id'].tolist()

agent_params = [
    (f'agent_{i:04d}', 
     np.random.choice(homes),
     np.random.choice(workplaces),
     i)
    for i in range(n_agents)
]

# %%
print(f"Generating {n_agents} agents in parallel...")
start_time = time.time()

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(generate_agent_trajectory)(params) for params in agent_params
)

generation_time = time.time() - start_time
print(f"Generated {n_agents} agents in {generation_time:.2f}s ({generation_time/n_agents:.2f}s per agent)")

# %%
all_trajectories = pd.concat(results, ignore_index=True)
all_trajectories = city.to_mercator(all_trajectories)
all_trajectories['date'] = pd.to_datetime(all_trajectories['datetime']).dt.date

output_path = 'data/trajectories_15_users'
for date, group in all_trajectories.groupby('date'):
    os.makedirs(f'{output_path}/date={str(date)}', exist_ok=True)
    group.to_parquet(f'{output_path}/date={str(date)}/data.parquet', index=False)

print(f"Saved {len(all_trajectories):,} records to {output_path}/")
