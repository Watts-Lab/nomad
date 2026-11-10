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
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lachesis Stop Detection
#
# The optional Infomap comparison requires Python 3.11+ and installation from
# a NOMAD checkout:
# `python -m pip install -e ".[infomap]"`.

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

# %% execution={"iopub.execute_input": "2026-08-13T02:17:45.896420Z", "iopub.status.busy": "2026-08-13T02:17:45.896420Z", "iopub.status.idle": "2026-08-13T02:17:48.179005Z", "shell.execute_reply": "2026-08-13T02:17:48.179005Z"}
# # %matplotlib inline
from importlib.metadata import version

import infomap
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Imports
import nomad.io.base as loader
import geopandas as gpd
from shapely.geometry import box
from nomad.stop_detection.viz import plot_stops_barcode, plot_time_barcode, plot_stops, plot_pings
from nomad.stop_detection.sequential_algs import lachesis
from nomad.stop_detection.postprocessing import merge_stops
from nomad.visit_attribution.visit_attribution import detect_locations
import nomad.data as data_folder
from pathlib import Path

# Load data
data_dir = Path(data_folder.__file__).parent
city = gpd.read_parquet(data_dir / 'garden-city-buildings-mercator.parquet')
outer_box = box(*city.total_bounds)

filepath_root = data_dir / "gc_data_long"
tc = {"user_id": "gc_identifier", "x": "dev_x", "y": "dev_y", "timestamp": "unix_ts"}

users = ['admiring_brattain']
traj = loader.sample_from_file(filepath_root, format='parquet', users=users, filters=('date','==', '2024-01-01'), traj_cols=tc)

# Lachesis (sequential stop detection)
stops = lachesis(traj, delta_roam=20, dt_max = 60, dur_min=5, complete_output=True, keep_col_names=True, traj_cols=tc)

# %% execution={"iopub.execute_input": "2026-08-13T02:17:48.182006Z", "iopub.status.busy": "2026-08-13T02:17:48.181006Z", "iopub.status.idle": "2026-08-13T02:17:48.402817Z", "shell.execute_reply": "2026-08-13T02:17:48.402817Z"}
fig, (ax_map, ax_barcode) = plt.subplots(2, 1, figsize=(6,6.5),
                                         gridspec_kw={'height_ratios':[10,1]})

gpd.GeoDataFrame(geometry=[outer_box], crs='EPSG:3857').plot(ax=ax_map, color='#d3d3d3')
city.plot(ax=ax_map, edgecolor='white', linewidth=1, color='#8c8c8c')

plot_stops(stops, ax=ax_map, cmap='Blues')
plot_pings(traj, ax=ax_map, s=6, color='black', alpha=0.5, traj_cols=tc)
ax_map.set_axis_off()

plot_time_barcode(traj['unix_ts'], ax=ax_barcode, set_xlim=True)
plot_stops_barcode(stops, ax=ax_barcode, cmap='Blues', set_xlim=False, timestamp='unix_ts')

plt.tight_layout(pad=0.1)
plt.show()

# %% [markdown]
# ## Controlled location-attribution comparison
#
# Each row below is a known stop, with a declared physical-place ID. Five cases
# test repeated visits, neighboring places, a chain between places, unequal visit
# frequencies, and overlapping GPS scatter. Compare DBSCAN (minimum two stops),
# equal-weight spatial Infomap, and Gaussian-weighted spatial Infomap.
#
# These one-dimensional examples use projected meters. False merges/splits count
# erroneous stop pairs, not label-ID differences. Visit count and total duration
# follow the same 10-minute merge threshold for every method. Merging fills gaps,
# so incorrect merges can inflate duration. Parameters here illustrate sensitivity;
# the same settings are used across methods for each comparison.

# %%
print({name: version(name) for name in ['infomap', 'numpy', 'pandas', 'scikit-learn']})
cases = {
    'repeat': ([0, 1, 2, 30, 31, 32], [0, 0, 0, 1, 1, 1]),
    'neighboring': ([0, 1, 2, 8, 9, 10], [0, 0, 0, 1, 1, 1]),
    'chain': ([0, 1, 2, 5, 8, 11, 14, 15, 16], [0, 0, 0, 2, 3, 4, 1, 1, 1]),
    'imbalanced': ([0, 1, 2, 0, 1, 2, 0, 1, 2, 8, 9], [0] * 9 + [1, 1]),
    'overlap': ([0, 6, 2, 4, 10, 8], [0, 0, 0, 1, 1, 1]),
}
comparisons = []
for case, (positions, truth) in cases.items():
    sample = pd.DataFrame({
        'x': positions, 'y': 0., 'timestamp': np.arange(len(positions)) * 600,
        'duration': 5,
    })
    expected_visits = merge_stops(sample.assign(location_id=truth), time_thresh=10)
    for radius in [4., 10.]:
        for sigma in [1., 4.]:
            for seed in [123, 456]:
                graph = nx.Graph()
                graph.add_nodes_from(range(len(sample)))
                graph.add_edges_from(
                    (i, j) for i in range(len(sample)) for j in range(i + 1, len(sample))
                    if abs(positions[i] - positions[j]) <= radius
                )
                modules = infomap.run(
                    graph, directed=False, two_level=True, markov_time=1,
                    seed=seed, num_trials=10,
                ).modules()
                assignments = {
                    'dbscan': detect_locations(sample, epsilon=radius, min_pts=2),
                    'infomap_equal': pd.Series([modules[i] for i in range(len(sample))]),
                    'infomap_gaussian': detect_locations(
                        sample, method='infomap', epsilon=radius, sigma=sigma, seed=seed,
                    ),
                }
                for method, ids in assignments.items():
                    truth_pairs = np.equal.outer(truth, truth)
                    predicted_pairs = np.equal.outer(ids.to_numpy(), ids.to_numpy())
                    merged = merge_stops(sample.assign(location_id=ids), time_thresh=10)
                    comparisons.append({
                        'case': case, 'radius': radius, 'sigma': sigma, 'seed': seed,
                        'method': method,
                        'false_merge_pairs': np.triu(predicted_pairs & ~truth_pairs, k=1).sum(),
                        'false_split_pairs': np.triu(truth_pairs & ~predicted_pairs, k=1).sum(),
                        'visit_count_error': len(merged) - len(expected_visits),
                        'duration_error_minutes': merged.duration.sum() - expected_visits.duration.sum(),
                    })
comparison = pd.DataFrame(comparisons)
comparison.loc[
    (comparison.radius == 10) & (comparison.sigma == 1) & (comparison.seed == 123)
]

# %% [markdown]
# Inspect the full `comparison` table before selecting parameters. Identical
# coordinates from different real places cannot be resolved by spatial weighting
# alone. Equal weights are a baseline, not a separate NOMAD public API option.
#
# With Infomap 2.16.0, radius 10, sigma 1 and seed 123, Gaussian weights resolve
# the neighboring-place case (zero false merges versus nine for the other two
# methods). They still produce three false-merge pairs in the chain case and
# four false-merge plus four false-split pairs in the overlap case. In that overlap
# case the inferred visit count is two too high and total duration ten minutes
# too low. The examples illustrate how spatial layout and parameter choices affect
# the partition and subsequent stop merging.
