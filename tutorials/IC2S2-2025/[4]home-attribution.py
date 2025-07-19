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
#     display_name: Python 3.10 (daphme)
#     language: python
#     name: daphme
# ---

# %%
# # !pip install -q git+https://github.com/Watts-Lab/nomad.git@IC2S2-tutorial

# import gdown
# gdown.cached_download(
#     "https://drive.google.com/uc?id=1wk3nrNsmAiBoTtWznHjjjPkmWAZfxk0P",
#     path="IC2S2_2025.zip",
#     quiet=False,
#     postprocess=gdown.extractall,  # auto-unzip
# )

# %% [markdown]
# # **Tutorial 4: Attributing visits and detecting homes**
#
# In this notebook we will explore how to attribute stops to building geometries in two ways, and use the output to detect user homes and workplaces. 

# %%
import geopandas as gpd
from shapely.geometry import Polygon, box, Point
import matplotlib.pyplot as plt

import nomad.io.base as loader
from nomad.stop_detection.viz import plot_pings

# %%
poi_table = gpd.read_file('garden_city.gpkg').set_index('building_id')

city_bounds = box(*poi_table.total_bounds)
color_map = {"home":"#86ceeb", "park":"#90ee90", "retail":"#d3d3d3", "work":"#c9a0dc"}

tc = {"user_id": "gc_identifier","timestamp": "unix_ts","x":"dev_x", "y":"dev_y"}

traj = loader.sample_from_file('gc_data_long/',
                               format='parquet',
                               users=['confident_aryabhata'],
                               within=city_bounds, # spatial filters
                               filters=[("ha", "<", 30), ("date", "<", '2024-01-04')], # regular filters
                               data_crs="EPSG:3857",
                               traj_cols=tc)

fix, ax = plt.subplots(figsize=(5,5))

ax.set_axis_off()
poi_table.plot(ax=ax, color=poi_table["type"].map(color_map), edgecolor="black")

plot_pings(traj, ax=ax, point_color='black',
           radius="ha", circle_alpha=0.06, circle_color="red", # for horizontal accuracy
           s=5, alpha=0.4, # <<<<<< for pings
           traj_cols=tc)

[plt.plot([],[], marker="s", ls="", color=color_map[t], label=t) for t in color_map]
ax.legend(loc="upper center", ncol=4, fontsize=10, framealpha=1)

plt.tight_layout()
plt.title("Homes and workplaces are in the \n inner two rings, can we detect them?")
plt.show()

# %%
import numpy as np
import pandas as pd
pd.set_option("mode.copy_on_write", True)
from nomad import filters


stops = loader.sample_from_file("gc_data_stops/",
                                format='parquet',
                                users=['confident_aryabhata'], # <<<< single user
                                user_id="gc_identifier")

stops['datetime'] = pd.to_datetime(stops['start_timestamp'], unit='s', utc=True).dt.tz_convert('Etc/GMT-1')

# %%
start_datetime, end_datetime = pd.Timestamp('2024-01-05', tz='Etc/GMT-1'), pd.Timestamp('2024-01-07', tz='Etc/GMT-1')
stops_sample = stops.query('@start_datetime <= datetime <=@end_datetime')

# %% [markdown]
# ## Attribute visits from centroid of stops

# %%
import nomad.visit_attribution.visit_attribution as visits

stops["location_id"] = visits.point_in_polygon(
                         data=stops,
                         poi_table=poi_table,
                         max_distance=15,  # TRY max_distance = 0
                         x='x',
                         y='y',
                         method='centroid',
                         data_crs='EPSG:3857')

stops.location_id.value_counts().head(15)

# %% [markdown]
# In the presence of **noise**, it is possible that pings for a single stop are scattered, and maybe are **split between several buildings**. `nomad.visit_attribution` implements the method `majority`, with more robustness to such cases, in which the location assigned to a stop is the **"majority vote" of the pings making up the stop**. For this we need the raw data with cluster labels. 

# %%
import nomad.stop_detection.lachesis as LACHESIS
from nomad.stop_detection.utils import summarize_stop

tc = {"user_id": "gc_identifier", "timestamp": "unix_ts", "x": "dev_x", "y": "dev_y", "ha":"ha", "date":"date"}

traj = loader.sample_from_file("gc_data_long/", format='parquet', users=['confident_aryabhata'], traj_cols=tc)
traj["cluster"] = LACHESIS.lachesis_labels(traj, delta_roam=30, dt_max=240, complete_output=True, traj_cols=tc)

traj["location_id"] = visits.point_in_polygon(
                         data=traj,
                         poi_table=poi_table,
                         max_distance=15,  # try switching to max_distance = 0
                         x='dev_x',
                         y='dev_y',
                         method='majority',
                         data_crs='EPSG:3857')

stops_maj = traj[traj.cluster!=-1].groupby('cluster', as_index=False).apply(lambda df: summarize_stop(
                                                                                df,
                                                                                complete_output=True,
                                                                                keep_col_names=False,
                                                                                passthrough_cols=['location_id', 'gc_identifier'],
                                                                                traj_cols=tc), include_groups=False
                                                                            )

stops_maj.location_id.value_counts().head(15)

# %% [markdown]
# ## Detecting homes from a stop table with locations

# %%
stops = loader.from_file("gc_data_stops/", format='parquet', user_id="gc_identifier")
stops['datetime'] = pd.to_datetime(stops['start_timestamp'], unit='s', utc=True).dt.tz_convert('Etc/GMT-1')

stops["location_id"] = visits.point_in_polygon(
                         data=stops,
                         poi_table=poi_table,
                         max_distance=10,
                         x='x',
                         y='y',
                         method='centroid',
                         data_crs='EPSG:3857')

# %% [markdown]
# Let's plot the most visited locations, normalizing by time visited (keep in mind some stops are split!)

# %%
import matplotlib.pyplot as plt

poi_table['total_visit_time_hrs'] = stops.groupby("location_id").agg({"duration":"sum"})//60

fig, ax1 = plt.subplots(figsize=(5,5))
ax1.set_axis_off()

poi_table.plot(ax=ax1, column='total_visit_time_hrs', cmap='Reds', edgecolor='black', linewidth=0.75, legend=True, legend_kwds={'shrink': 0.75})
plt.title("Total visit time (h) by building")
plt.show()

# %% [markdown]
# Homes can be estimated from attributed stops by identifying **recurrent night time locations**. Naturally, we need a zoned datetime to reason about "nighttime"

# %%
stops['start_datetime'] = pd.to_datetime(stops['start_timestamp'], unit='s', utc=True).dt.tz_convert('Etc/GMT-1')
stops['end_datetime'] = pd.to_datetime(stops['end_timestamp'], unit='s', utc=True).dt.tz_convert('Etc/GMT-1')
stops.drop(['start_timestamp', 'end_timestamp'], axis=1, inplace=True)

# %%
# %%time
import nomad.visit_attribution.home_attribution as homes
from datetime import date

cand_homes = homes.compute_candidate_homes(stops,
                                           datetime="datetime",
                                           location_id="location_id",
                                           user_id="gc_identifier",
                                           dawn_hour=6,
                                           dusk_hour=19
                                           )
cand_homes

# %% [markdown]
# A simple query can now find candidate locations satisfying:
# - `num_nights >= min_days`
# - `num_weeks >= min_weeks`
# - break ties using the total dwell at night (`total_duration`)

# %%
last_date = date(year=2024, month=1, day=21) # needed for rolling home computations
home_table = homes.select_home(cand_homes, min_days=4, min_weeks=2, last_date=last_date, user_id='gc_identifier')
home_table

# %%
print(f"{100*len(home_table)/len(stops.gc_identifier.unique()):.2f}% of users have a detected home")
print(f"{100*(home_table.location_id.str[0] == 'h').sum()/len(stops.gc_identifier.unique()):.2f}% of users have a home of type 'home'")

# %% [markdown]
# ## Work locations and OD matrix

# %%
homes.workday_stops(stops, work_start_hour=11,
    work_end_hour=13).query("gc_identifier == 'admiring_cray'").groupby('location_id').duration.sum().sort_values()

# %%
stops_work = homes.workday_stops(stops)

# %%
stops_work.query("datetime < start_datetime")

# %%
cand_works =homes.compute_candidate_workplaces(stops,
                                               datetime="datetime",
                                               location_id="location_id",
                                               user_id="gc_identifier",
                                               work_start_hour=10,
                                               work_end_hour=12,
                                               include_weekends=False)

work_table = homes.select_workplace(cand_works, last_date=last_date, min_days=3,min_weeks=2, user_id='gc_identifier')

print(f"{100*len(work_table)/len(stops.gc_identifier.unique()):.2f}% of users have a detected workplace")
print(f"{100*(work_table.location_id.str[0] == 'w').sum()/len(stops.gc_identifier.unique()):.2f}% of users have a workplace of type 'work'.")

# %% [markdown]
# ## Visualization of network from home to work using pydeck

# %%
origin = home_table.set_index('gc_identifier').location_id
origin.name = "origin"

destination = work_table.set_index('gc_identifier').location_id
destination.name = "destination"

od = (pd.DataFrame([origin,destination]).T).dropna()
od = od.groupby(by=['origin', 'destination']).size().reset_index(name='count')

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from nomad.visit_attribution.viz import plot_od_map

# a little background
outer_box = box(*poi_table.total_bounds).buffer(15, join_style='mitre')
outer_box = outer_box.difference(poi_table.geometry.union_all())

background = poi_table.copy()
background['color'] = '#adadad'
background = pd.concat([background, gpd.GeoDataFrame({'geometry':outer_box, 'color':'#616161'}, index=['outline'], crs="EPSG:3857")])

plot_od_map(od_df=od,
   region_gdf=poi_table,
   origin_col="origin",
   dest_col="destination",
   weight_col="count",
   edge_alpha=0.85,
   edge_cmap="Reds", # try Reds, viridis, plasma
   w_min=1, # try 2
   background_gdf=background)

# %%
od.loc[od["count"]>2]

# %%
