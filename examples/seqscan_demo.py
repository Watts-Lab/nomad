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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SeqScan Stop Detection
#
# **SeqScan** extends DBSCAN to spatio-temporal data while guaranteeing temporal separation between detected stops. Each stop is detected analogously to DBSCAN, but over a local time context. That is, a connected component of the mutual reachability graph, made up of pings with at least `min_pts` neighbors at distance `dist_thresh`, and with the additional requirement that the stop spans at least `dur_min` minutes. Unlike typical DBSCAN traversal, SeqScan discovers pings in chronological order, maintaining both an active cluster and an active time context for querying the neighbors of each incoming ping. If an incoming ping is a core point and reachable from the active cluster, it is assimilated. Otherwise, DBSCAN is applied to the tail of the trajectory, starting from the last core point of the active cluster up to the current ping, to try to detect a new active cluster. When we set a new active cluster, the active time context is also shifted, and the algorithm continues until reaching the end of the trajectory.
#  Damiani, Issa, and Cagnacci, [Extracting stay regions with uncertain boundaries from GPS trajectories: a case study in animal ecology](https://dl.acm.org/doi/10.1145/2666310.2666417).
#
# This implementation relies on 3 parameters:
#
# * `time_thresh` defines the maximum time gap (in minutes) used to build the temporal neighborhood graph.
# * `dist_thresh` specifies the maximum spatial distance (in meters) between two pings for them to be considered neighbors.
# * `min_pts` sets the minimum number of neighbors required for a ping to be treated as a core point.
#
# Notice that this method also works with **geographic coordinates** (`lon`, `lat`) using Haversine distance.

# %%
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# Imports
import nomad.io.base as loader
import geopandas as gpd
from shapely.geometry import box
from nomad.stop_detection.viz import plot_stops_barcode, plot_time_barcode, plot_stops, plot_pings
from nomad.stop_detection.density_algs import seqscan
import nomad.data as data_folder
from pathlib import Path
data_dir = Path(data_folder.__file__).parent
city = gpd.read_parquet(data_dir / 'garden-city-buildings-mercator.parquet')
outer_box = box(*city.total_bounds)

filepath_root = data_dir / "gc_data_long"
tc = {"user_id": "gc_identifier", "x": "dev_x", "y": "dev_y", "timestamp": "unix_ts"}

users = ['admiring_brattain']
traj = loader.sample_from_file(filepath_root, format='parquet', users=users, filters=('date','==', '2024-01-01'), traj_cols=tc)

stops_seqscan = seqscan(traj,
                    time_thresh=60,
                    dist_thresh=8,
                    min_pts=3,
                    complete_output=True,
                    traj_cols=tc)    

# %%
fig, (ax_map, ax_barcode) = plt.subplots(2, 1, figsize=(6,6.5),
                                         gridspec_kw={'height_ratios':[10,1]})

gpd.GeoDataFrame(geometry=[outer_box], crs='EPSG:3857').plot(ax=ax_map, color='#d3d3d3')
city.plot(ax=ax_map, edgecolor='white', linewidth=1, color='#8c8c8c')

plot_stops(stops_seqscan, ax=ax_map, cmap='Reds')
plot_pings(traj, ax=ax_map, s=6, color='black', alpha=0.5, traj_cols=tc)
ax_map.set_axis_off()

plot_time_barcode(traj['unix_ts'], ax=ax_barcode, set_xlim=True)
plot_stops_barcode(stops_seqscan, ax=ax_barcode, cmap='Reds', set_xlim=False, timestamp='unix_ts')

plt.tight_layout(pad=0.1)
plt.show()
