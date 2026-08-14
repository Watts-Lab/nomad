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
# # Compare Stop Detection Algorithms
#
# Dashboard-style preview: choose two algorithms and their own parameters, run, and inspect side-by-side map+barcode outputs.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import json
from pathlib import Path

import nomad.io.base as loader
import nomad.data as data_folder

from nomad.stop_detection.viz import (
    plot_circles,
    plot_stops_barcode,
    plot_time_barcode,
    plot_pings,
)

from nomad.stop_detection.density_algs import dbstop, dbstop_labels
from nomad.stop_detection.density_algs import ta_dbscan, ta_dbscan_labels
from nomad.stop_detection.density_algs import seqscan, seqscan_labels

base_dir = Path('research/robustness-of-algorithms')
with open(base_dir / 'config_2_stops.json', 'r', encoding='utf-8') as f:
    sim_config = json.load(f)

data_dir = Path(data_folder.__file__).parent
city = gpd.read_parquet(data_dir / 'garden-city-buildings-mercator.parquet')
sparse_path = base_dir / sim_config['output_files']['sparse_path']

tc = {'user_id': 'user_id', 'x': 'x', 'y': 'y', 'timestamp': 'timestamp'}
sparse_df = loader.from_file(str(sparse_path), format='parquet', traj_cols=tc)
seed = 1
rng = np.random.default_rng(seed)
random_user = rng.choice(sparse_df.user_id.unique())

algo_registry = {
    'dbstop': (dbstop, dbstop_labels),
    'tadbscan': (ta_dbscan, ta_dbscan_labels),
    'seqscan': (seqscan, seqscan_labels),
}

panel_cfg = [
    {'panel': 'left', 'algo': 'dbstop', 'params': {'time_thresh': 60, 'dist_thresh': 10, 'min_pts': 3, 'dur_min': 5}, 'cmap': 'inferno_r'},
    {'panel': 'right', 'algo': 'seqscan', 'params': {'time_thresh': 60, 'dist_thresh': 10, 'min_pts': 3, 'dur_min': 5}, 'cmap': 'inferno_r'},
]

# %%
fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 6.5),
    gridspec_kw={'height_ratios': [10, 1]},
)

panel_axes = {
    'left': (axes[0, 0], axes[1, 0]),
    'right': (axes[0, 1], axes[1, 1]),
}

for cfg in panel_cfg:
    ax_map, ax_barcode = panel_axes[cfg['panel']]
    run_fn, label_fn = algo_registry[cfg['algo']]
    traj = sparse_df.query('user_id == @random_user').copy()

    stops = run_fn(
        traj,
        complete_output=True,
        traj_cols=tc,
        timestamp=tc['timestamp'],
        **cfg['params'],
    )
    labels = label_fn(
        traj,
        traj_cols=tc,
        timestamp=tc['timestamp'],
        **cfg['params'],
    )
    traj['cluster'] = labels.to_numpy()

    plot_circles(
        traj,
        ax=ax_map,
        radius=1.5,
        color='cluster',
        cmap=cfg['cmap'],
        base_geometry=city,
        base_geom_color='#8c8c8c',
        base_geom_background='#d3d3d3',
        traj_cols=tc,
    )
    plot_pings(traj, ax=ax_map, s=6, color='black', traj_cols=tc)

    ax_map.set_title(
        f"{cfg['algo'].upper()} | t={cfg['params']['time_thresh']} min, d={cfg['params']['dist_thresh']} m, min_pts={cfg['params']['min_pts']}, dur_min={cfg['params']['dur_min']}"
    )

    plot_time_barcode(traj[tc['timestamp']], ax=ax_barcode, set_xlim=True)
    plot_stops_barcode(
        stops,
        ax=ax_barcode,
        stop_alpha=0.4,
        cmap=cfg['cmap'],
        set_xlim=False,
        timestamp=tc['timestamp'],
    )
    traj_barcode = traj[[tc['timestamp'], 'cluster']].rename(columns={tc['timestamp']: 'timestamp'})
    plot_time_barcode(traj_barcode, color='cluster', ax=ax_barcode, cmap=cfg['cmap'], set_xlim=False, lw=1)

plt.tight_layout(pad=1)
plt.show()
