# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TADBSCAN Stop Detection

# %% [markdown]
# The second stop detection algorithm implemented in ```nomad``` is an adaptation of DBSCAN. Unlike in plain DBSCAN, we also incorporate the time dimension to determine if two pings are "neighbors". This implementation relies on 3 parameters
#
# * `time_thresh` defines the maximum time difference (in minutes) between two consecutive pings for them to be considered neighbors within the same cluster.
# * `dist_thresh` specifies the maximum spatial distance (in meters) between two pings for them to be considered neighbors.
# * `min_pts` sets the minimum number of neighbors required for a ping to form a cluster.
#
# Notice that this method also works with **geographic coordinates** (lon, lat), using Haversine distance. 

# %%
# %matplotlib inline

# Imports
import nomad.io.base as loader
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nomad.stop_detection.viz import plot_stops_barcode, plot_time_barcode
import nomad.stop_detection.dbscan as DBSCAN
import nomad.filters as filters 
import nomad.stop_detection.postprocessing as post

# Load data
from nomad.city_gen import City
city_obj = City.from_geopackage("garden-city.gpkg")
# Create a simple bounds box for visualization
outer_box = box(0, 0, city_obj.dimensions[0], city_obj.dimensions[1])

filepath_root = 'gc_data_long/'
tc = {"user_id": "gc_identifier", "x": "dev_x", "y": "dev_y", "timestamp": "unix_ts"}

# Density based stop detection (Temporal DBSCAN)
users = ['confident_aryabhata']
traj = loader.sample_from_file(filepath_root, format='parquet', users=users, filters = ('date','<=', '2024-01-03'), traj_cols=tc)

user_data_tadb = traj.assign(cluster=DBSCAN.ta_dbscan_labels(traj, time_thresh=240, dist_thresh=15, min_pts=3, traj_cols=tc))
stops_tadb = DBSCAN.ta_dbscan(traj,
                    time_thresh=720,
                    dist_thresh=15,
                    min_pts=3,
                    complete_output=True,
                    traj_cols=tc)
stops_tadb["cluster"] = post.remove_overlaps(user_data_tadb, time_thresh=240, method='cluster', traj_cols=tc, min_pts=3, dur_min=5, min_cluster_size=3)    

# %%
fig, ax_barcode = plt.subplots(figsize=(10,1.5))

plot_time_barcode(traj['unix_ts'], ax=ax_barcode, set_xlim=True)
plot_stops_barcode(stops_tadb, ax=ax_barcode, stop_color='red', set_xlim=False, timestamp='unix_ts')
plt.title("TA-DBSCAN stops with post-processing")
plt.tight_layout()
plt.show()
