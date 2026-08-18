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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lachesis Stop Detection

# %%
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# Imports
import nomad.io.base as loader
import geopandas as gpd
from shapely.geometry import box
from nomad.stop_detection.viz import plot_stops_barcode, plot_time_barcode, plot_stops, plot_pings, _plot_base_geometry, plot_circles
from nomad.stop_detection.sequential_algs import lachesis, lachesis_labels


import nomad.data as data_folder
from pathlib import Path

from nomad.city_gen import City
import pandas as pd
import datetime
import json
import matplotlib.dates as mdates
import pandas as pd
import nomad.generation.viz as viz
from nomad.traj_gen import Agent, condense_destinations

# %%
# Parameters according to the config file
data_dir = Path(data_folder.__file__).parent
city = City.from_geopackage(data_dir / "garden-city.gpkg")
buildings = gpd.read_parquet(data_dir / "garden-city-buildings-mercator.parquet")

# %%
fig, ax = plt.subplots(figsize=(3,3))

city.plot_city(ax=ax)

# %% [markdown]
# # Destination Diary for 3 stop trajectory

# %%
start = '2024-06-01 00:00-08:00'

start_time = pd.date_range(start=start, periods=6, freq='30min')
unix_timestamp = [int(t.timestamp()) for t in start_time]
duration = [30]*6  # in minutes

location = ['h-x14-y11'] * 1 + ['w-x17-y10'] * 2 + ['r-x19-y11'] * 2 + ['r-x20-y13'] * 1


destinations = pd.DataFrame(
    {
        "datetime":start_time,
         "timestamp":unix_timestamp,
         "duration":duration,
         "location":location
    }
)

destinations = condense_destinations(destinations)
destinations = destinations[:-1]
destinations

# %%
Charlie = Agent(identifier="Andres",
                city=city)

# Convert datetime column
# destinations['datetime'] = pd.to_datetime(destinations['datetime'])
Charlie.generate_trajectory(destination_diary=destinations,
                            seed=19104,
                            dt=0.2)

Charlie.set_beta_params(beta_start=None, beta_durations=None, beta_ping=7)
Charlie.sample_trajectory(seed=60,
                          ha=11/15)

traj = Charlie.sparse_traj

# %% [markdown]
# ### Plot normal stops

# %%
# outer_box = box(*city.city_boundary.bounds)
delta_roam = 2.5
dt_max=120
dur_min=5
stops = lachesis(traj, delta_roam=delta_roam, dt_max=dt_max, dur_min=dur_min, 
                         complete_output=True, keep_col_names=True, timestamp='timestamp')
labels = lachesis_labels(traj, delta_roam=delta_roam, dt_max=dt_max, dur_min=dur_min, timestamp='timestamp')
traj['cluster'] = labels

# %%
fig, (ax_map, ax_barcode) = plt.subplots(2, 1, figsize=(6,4.5),
                                         gridspec_kw={'height_ratios':[10,1]})


# Plot colored pings
plot_circles(traj, ax=ax_map, radius=0.1, color='cluster', cmap='inferno_r', base_geometry=city.buildings_gdf, base_geom_color='#8c8c8c', base_geom_background='#d3d3d3')
plot_pings(traj, ax=ax_map, s=5, color='black')
#ax_map.set_axis_off()
plot_time_barcode(traj['timestamp'], ax=ax_barcode, set_xlim=True)
plot_stops_barcode(stops, ax=ax_barcode, stop_alpha=0.5, cmap='inferno_r', set_xlim=False, timestamp='timestamp')
#plot_stops_barcode(stops, ax=ax_barcode, stop_color='#d3d3d3', cmap='inferno_r', set_xlim=False, timestamp='timestamp')
plot_time_barcode(traj, color='cluster', ax=ax_barcode, cmap='inferno_r', set_xlim=False, lw=1)
ax_barcode.set_title("timestamps")
plt.tight_layout(pad=2)
plt.savefig('normal_clusters.svg' , format='svg')
plt.savefig('normal_clusters.png' , format='png', dpi=300)
plt.show()

# %% [markdown]
# ### Ground truth, undersmoothing oversmoothing

# %%
start = '2024-06-01 00:00-08:00'

start_time = pd.date_range(start=start, periods=6, freq='30min')
unix_timestamp = [int(t.timestamp()) for t in start_time]
duration = [30]*6  # in minutes

location = ['p-x12-y11'] * 1 + ['h-x14-y11'] * 1 + ['w-x17-y10'] * 2 + ['r-x19-y11'] * 2

destinations = pd.DataFrame(
    {
        "datetime":start_time,
         "timestamp":unix_timestamp,
         "duration":duration,
         "location":location
    }
)
destinations = condense_destinations(destinations)

Charlie = Agent(identifier="Paco", city=city)

# Convert datetime column
# destinations['datetime'] = pd.to_datetime(destinations['datetime'])
Charlie.generate_trajectory(destination_diary=destinations,
                            seed=1005,
                            dt=0.15)

Charlie.set_beta_params(beta_start=None, beta_durations=None, beta_ping=5.3)
Charlie.sample_trajectory(seed=1005,
                          ha=9/15)



# %%
dt_max=220
dur_min=5

output = []
## Undersmoothing
delta_roam = 0.8
traj = Charlie.sparse_traj.copy()
stops = lachesis(traj, delta_roam=delta_roam, dt_max=dt_max, dur_min=dur_min, 
                         complete_output=True, keep_col_names=True, timestamp='timestamp')
labels = lachesis_labels(traj, delta_roam=delta_roam, dt_max=dt_max, dur_min=dur_min, timestamp='timestamp')
traj['cluster'] = labels
output += [(traj, stops)]
## Normal
delta_roam = 3.9
traj = Charlie.sparse_traj.copy()
stops = lachesis(traj, delta_roam=delta_roam, dt_max=dt_max, dur_min=dur_min, 
                         complete_output=True, keep_col_names=True, timestamp='timestamp')
labels = lachesis_labels(traj, delta_roam=delta_roam, dt_max=dt_max, dur_min=dur_min, timestamp='timestamp')
traj['cluster'] = labels
output += [(traj, stops)]
## Oversmoothing
delta_roam = 6
traj = Charlie.sparse_traj.copy()
stops = lachesis(traj, delta_roam=delta_roam, dt_max=dt_max, dur_min=dur_min, 
                         complete_output=True, keep_col_names=True, timestamp='timestamp')
labels = lachesis_labels(traj, delta_roam=delta_roam, dt_max=dt_max, dur_min=dur_min, timestamp='timestamp')
traj['cluster'] = labels
output += [(traj, stops)]


# %%
## Undersmoothing 
traj, stops = output[0]
fig, (ax_map, ax_barcode) = plt.subplots(2, 1, figsize=(6,3.5),
                                         gridspec_kw={'height_ratios':[10,1]})


# Plot colored pings
plot_circles(traj, ax=ax_map, radius=0.13, alpha=0.7, color='cluster', cmap='inferno_r', base_geometry=city.buildings_gdf, base_geom_color='#8c8c8c', base_geom_background='#d3d3d3')
plot_pings(traj, ax=ax_map, s=4, color='black')
#ax_map.set_axis_off()
plot_time_barcode(traj['timestamp'], ax=ax_barcode, set_xlim=True)
plot_stops_barcode(stops, ax=ax_barcode, stop_alpha=0.6, cmap='inferno_r', set_xlim=False, timestamp='timestamp')
#plot_stops_barcode(stops, ax=ax_barcode, stop_color='#d3d3d3', cmap='inferno_r', set_xlim=False, timestamp='timestamp')
plot_time_barcode(traj, color='cluster', ax=ax_barcode, cmap='inferno_r', set_xlim=False, lw=1)
ax_barcode.set_title("timestamps")
plt.tight_layout(pad=2)
plt.savefig('undersmooth_clusters.svg' , format='svg')
plt.savefig('undersmooth_clusters.png' , format='png', dpi=300)
plt.show()

# %%
## Normal 
traj, stops = output[1]
fig, (ax_map, ax_barcode) = plt.subplots(2, 1, figsize=(6,3.5),
                                         gridspec_kw={'height_ratios':[10,1]})


# Plot colored pings
plot_circles(traj, ax=ax_map, radius=0.13, alpha=0.7, color='cluster', cmap='inferno_r', base_geometry=city.buildings_gdf, base_geom_color='#8c8c8c', base_geom_background='#d3d3d3')
plot_pings(traj, ax=ax_map, s=4, color='black')
#ax_map.set_axis_off()
plot_time_barcode(traj['timestamp'], ax=ax_barcode, set_xlim=True)
plot_stops_barcode(stops, ax=ax_barcode, stop_alpha=0.6, cmap='inferno_r', set_xlim=False, timestamp='timestamp')
#plot_stops_barcode(stops, ax=ax_barcode, stop_color='#d3d3d3', cmap='inferno_r', set_xlim=False, timestamp='timestamp')
plot_time_barcode(traj, color='cluster', ax=ax_barcode, cmap='inferno_r', set_xlim=False, lw=1)
ax_barcode.set_title("timestamps")
plt.tight_layout(pad=2)
plt.savefig('good_clusters.svg' , format='svg')
plt.savefig('good_clusters.png' , format='png', dpi=300)
plt.show()

# %%
## Oversmoothing 
traj, stops = output[2]
fig, (ax_map, ax_barcode) = plt.subplots(2, 1, figsize=(6,3.5),
                                         gridspec_kw={'height_ratios':[10,1]})


# Plot colored pings
plot_circles(traj, ax=ax_map, radius=0.13, alpha=0.7, color='cluster', cmap='inferno_r', base_geometry=city.buildings_gdf, base_geom_color='#8c8c8c', base_geom_background='#d3d3d3')
plot_pings(traj, ax=ax_map, s=4, color='black')

plot_time_barcode(traj['timestamp'], ax=ax_barcode, set_xlim=True)
plot_stops_barcode(stops, ax=ax_barcode, stop_alpha=0.6, cmap='inferno_r', set_xlim=False, timestamp='timestamp')
#plot_stops_barcode(stops, ax=ax_barcode, stop_color='#d3d3d3', cmap='inferno_r', set_xlim=False, timestamp='timestamp')
plot_time_barcode(traj, color='cluster', ax=ax_barcode, cmap='inferno_r', set_xlim=False, lw=1)
ax_barcode.set_title("timestamps")
plt.tight_layout(pad=2)
plt.savefig('oversmooth_clusters.svg' , format='svg')
plt.savefig('oversmooth_clusters.png' , format='png', dpi=300)
plt.show()

# %%
## Ground truth only 
traj = Charlie.trajectory
stops = Charlie.diary.dropna()

fig, (ax_map, ax_barcode) = plt.subplots(2, 1, figsize=(6,3.5),
                                         gridspec_kw={'height_ratios':[10,1]})


# Plot colored pings
plot_pings(traj, ax=ax_map, radius=0.13, alpha=0, base_geometry=city.buildings_gdf, base_geom_color='#8c8c8c', base_geom_background='#d3d3d3')
ax_map.plot(traj.x, traj.y, alpha=0.35)
#ax_map.set_axis_off()
#plot_stops_barcode(stops, ax=ax_barcode, stop_alpha=0.5, cmap='inferno_r', set_xlim=False, timestamp='timestamp')
plot_stops_barcode(stops, ax=ax_barcode, stop_alpha=0.6, cmap='inferno_r', set_xlim=False, timestamp='timestamp')

ax_barcode.set_title("timestamps")
plt.tight_layout(pad=2)
plt.savefig('true_clusters.svg' , format='svg')
plt.savefig('true_clusters.png' , format='png', dpi=300)
plt.show()
