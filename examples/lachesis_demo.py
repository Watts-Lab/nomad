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
# # Lachesis Stop Detection

# %% [markdown]
# The first stop detection algorithm implemented in ```nomad``` is a sequential algorithm insipired by the one in _Project Lachesis: Parsing and Modeling Location Histories_ (Hariharan & Toyama). This algorithm for extracting stays is dependent on two parameters: the roaming distance and the stay duration. 
#
# * Roaming distance represents the maximum distance an object can move away from a point location and still be considered to be staying at that location.
# * Stop duration is the minimum amount of time an object must spend within the roaming distance of a location to qualify as a stop.
#
# The algorithm identifies stops as contiguous sequences of pings that stay within the roaming distance for at least the duration of the stop duration.
#
# This algorithm has the following parameters, which determine the size of the resulting stops:
# * ```dur_min```: Minimum duration for a stay in minutes.
# * ```dt_max```: Maximum time gap permitted between consecutive pings in a stay in minutes (dt_max should be greater than dur_min).
# * ```delta_roam```: Maximum roaming distance for a stay in meters.

# %%
# %matplotlib inline

# Imports
import nomad.io.base as loader
from shapely.geometry import box
import matplotlib.pyplot as plt
from nomad.stop_detection.viz import plot_stops_barcode, plot_time_barcode
import nomad.stop_detection.lachesis as LACHESIS

# Load data
filepath_root = 'gc_data_long/'
tc = {"user_id": "gc_identifier", "x": "dev_x", "y": "dev_y", "timestamp": "unix_ts"}

users = ['admiring_brattain']
traj = loader.sample_from_file(filepath_root, format='parquet', users=users, filters = ('date','==', '2024-01-01'), traj_cols=tc)

# Lachesis (sequential stop detection)
stops = LACHESIS.lachesis(traj, delta_roam=20, dt_max = 60, dur_min=5, complete_output=True, keep_col_names=True, traj_cols=tc)

# %%
fig, ax_barcode = plt.subplots(figsize=(10,1.5))

plot_time_barcode(traj['unix_ts'], ax=ax_barcode, set_xlim=True)
plot_stops_barcode(stops, ax=ax_barcode, stop_color='blue', set_xlim=False, timestamp='unix_ts')
fig.suptitle("Lachesis stops")
plt.tight_layout()
plt.show()
