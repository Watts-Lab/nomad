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
# # HDBSCAN Performance on Agent-Generated Trajectories
#
# Evaluate `st_hdbscan` across realistic synthetic trajectories produced by
# the `Agent` simulation (EPR mobility model on the Garden City map). Ground truth comes
# directly from `agent.diary`.
#
# **Sweep dimensions**
#
# | Dimension | Values | What it controls |
# |---|---|---|
# | `seed` | 0–9 | Agent home/workplace and movement |
# | `beta_start` | 100, 250, 400 | Burst inter-arrival time → trajectory sparsity |
# | `time_thresh` | 20, 60 | HDBSCAN temporal neighborhood (min) |
# | `min_pts` | 2, 3 | HDBSCAN core-point threshold |
#
# **Metrics** are time-based (seconds of overlap between detected and ground-truth stop intervals):
# - **Precision** — detected stop time that overlaps a true stop / total detected stop time  
# - **Recall** — true stop time covered by any detected stop / total true stop time  
# - **F1** — harmonic mean

# %%
# %matplotlib inline
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nomad.city_gen import City
from nomad.traj_gen import Agent
from nomad.stop_detection.density_algs import st_hdbscan
import nomad.data as data_folder

# %% [markdown]
# ## 1. Load
#
# Building hub network, gravity matrix, and shortest paths are computed once and reused for all agents.

# %%
data_dir = Path(data_folder.__file__).parent
city_path = data_dir / 'garden-city.gpkg'

t0 = time.perf_counter()
city = City.from_geopackage(str(city_path))
city._build_hub_network(hub_size=16)
city.compute_gravity(exponent=2.0)
city.compute_shortest_paths(callable_only=True)

# Pre-sample a pool of home/workplace pairs for the agents
rng_pool = np.random.default_rng(0)
homes      = city.buildings_gdf[city.buildings_gdf['building_type'] == 'home']['id'].to_numpy()
workplaces = city.buildings_gdf[city.buildings_gdf['building_type'] == 'workplace']['id'].to_numpy()
N_AGENTS = 10
home_pool = rng_pool.choice(homes,      size=N_AGENTS, replace=True)
work_pool = rng_pool.choice(workplaces, size=N_AGENTS, replace=True)

TC = {'timestamp': 'timestamp', 'x': 'x', 'y': 'y'}


# %% [markdown]
# ## 2. Metrics

# %%
# compare with contact_estimation, compute metrics and overlapping_visits
# see how results differ

def _overlap(a_s, a_e, b_s, b_e):
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def temporal_metrics(stops_df, gt, tc):
    """
    Time-based precision / recall / F1.

    Parameters
    ----------
    stops_df : output of st_hdbscan with complete_output=True
    gt       : DataFrame with columns start_ts (int, seconds) and end_ts (int, seconds)
    tc       : traj_cols dict (must include 'timestamp')
    """
    if stops_df is None or stops_df.empty:
        return dict(precision=0.0, recall=0.0, f1=0.0, n_detected=0)

    detected = [
        (int(r[tc['timestamp']]), int(r['end_timestamp']))
        for _, r in stops_df.iterrows()
    ]
    truth = [
        (int(r['start_ts']), int(r['end_ts']))
        for _, r in gt.iterrows()
    ]

    total_det = sum(e - s for s, e in detected)
    total_gt  = sum(e - s for s, e in truth)

    tp = sum(_overlap(ds, de, gs, ge) for ds, de in detected for gs, ge in truth)

    precision = tp / total_det if total_det > 0 else 0.0
    recall    = tp / total_gt  if total_gt  > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return dict(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        n_detected=len(stops_df),
    )


def diary_to_gt(diary):
    """
    Convert agent.diary to a ground-truth stop interval DataFrame.
    Skips transit entries (location is None) and consecutive duplicate
    locations are merged so back-to-back entries at the same building
    count as one stop.
    """
    df = diary[diary['location'].notna()].copy()
    df['end_ts'] = df['timestamp'] + (df['duration'] * 60).astype(int)
    df = df.rename(columns={'timestamp': 'start_ts'})

    # Merge consecutive entries at the same building
    rows = []
    for _, r in df.iterrows():
        if rows and rows[-1]['location'] == r['location'] and r['start_ts'] <= rows[-1]['end_ts']:
            rows[-1]['end_ts'] = r['end_ts']
        else:
            rows.append({'start_ts': r['start_ts'], 'end_ts': r['end_ts'],
                         'location': r['location']})

    gt = pd.DataFrame(rows)
    gt['duration_min'] = (gt['end_ts'] - gt['start_ts']) / 60
    return gt


# %% [markdown]
# ## 3. Battery Configuration

# %%
# Trajectory generation window
START_DT  = pd.Timestamp('2024-01-01T07:00-04:00')
END_DT    = pd.Timestamp('2024-01-02T07:00-04:00')  # 1-day trajectories

# Sampling sparsity: higher beta_start → more time between bursts → sparser
BETA_START_VALS = [100, 250, 400]   # burst inter-arrival time (min)
BETA_PING       = 5                 # ping inter-arrival within burst (min)

# HDBSCAN configurations
HDBSCAN_CONFIGS = [
    dict(time_thresh=20, min_pts=2, min_cluster_size=2, dur_min=5,  label='tt=20 mp=2 mcs=2'),
    dict(time_thresh=60, min_pts=2, min_cluster_size=2, dur_min=5,  label='tt=60 mp=2 mcs=2'),
    dict(time_thresh=60, min_pts=3, min_cluster_size=2, dur_min=5,  label='tt=60 mp=3 mcs=2'),
    dict(time_thresh=60, min_pts=2, min_cluster_size=1, dur_min=5,  label='tt=60 mp=2 mcs=1'),
]

total_runs = N_AGENTS * len(BETA_START_VALS) * len(HDBSCAN_CONFIGS)
print(f'Agents: {N_AGENTS}  |  Sparsity configs: {len(BETA_START_VALS)}  |  HDBSCAN configs: {len(HDBSCAN_CONFIGS)}')
print(f'Total HDBSCAN runs: {total_runs}')

# %% [markdown]
# ## 4. Generate Trajectories

# %%
# Generate dense ground-truth trajectories once per agent/seed
agents = {}
for seed in range(N_AGENTS):
    agent = Agent(
        identifier=f'agent_{seed:03d}',
        city=city,
        home=home_pool[seed],
        workplace=work_pool[seed],
        datetime=START_DT,
        seed=seed,
    )
    agent.generate_trajectory(end_time=END_DT, seed=seed)
    agents[seed] = agent

print(f'Generated {len(agents)} dense trajectories')
# Peek at one diary
ex = agents[0]
print(f'\nAgent 0 diary ({len(ex.diary)} entries, {len(ex.trajectory)} ticks):')
ex.diary.head(8)

# %% [markdown]
# ## 5. Run

# %%
records = []

for seed, agent in agents.items():
    gt = diary_to_gt(agent.diary)

    for beta_start in BETA_START_VALS:
        agent.set_beta_params(
            beta_ping=BETA_PING,
            beta_start=beta_start,
            beta_durations=beta_start,
        )
        agent.sample_trajectory(
            replace_sparse_traj=True,
            seed=seed,
        )
        # Drop datetime column to avoid timestamp/datetime ambiguity in st_hdbscan
        sparse = agent.sparse_traj[['x', 'y', 'timestamp']].copy()
        n_pings = len(sparse)
        q = n_pings / len(agent.trajectory)

        for cfg in HDBSCAN_CONFIGS:
            params = {k: v for k, v in cfg.items() if k != 'label'}

            t0 = time.perf_counter()
            try:
                stops = st_hdbscan(
                    sparse.copy(),
                    traj_cols=TC,
                    complete_output=True,
                    **params,
                )
            except Exception as exc:
                print(f'  seed={seed} beta_start={beta_start} {cfg["label"]}: {exc}')
                stops = None
            elapsed = time.perf_counter() - t0

            m = temporal_metrics(stops, gt, TC)
            records.append({
                'seed':         seed,
                'beta_start':   beta_start,
                'q':            round(q, 3),
                'n_pings':      n_pings,
                'n_true_stops': len(gt),
                'hdbscan_cfg':  cfg['label'],
                'runtime_s':    round(elapsed, 4),
                **m,
            })

raw_df = pd.DataFrame(records)
print(f'Done. {len(raw_df)} rows.')
raw_df.head(8)

# %% [markdown]
# ## 6. Summary Tables
#
# ### Mean ± std per HDBSCAN config (across agents and sparsities)

# %%
summary = (
    raw_df
    .groupby('hdbscan_cfg')[['precision', 'recall', 'f1', 'runtime_s']]
    .agg(['mean', 'std'])
    .round(3)
)
summary.columns = [f'{col}_{stat}' for col, stat in summary.columns]
summary.reset_index()

# %% [markdown]
# ## 7. Visualizations

# %%
configs = [cfg['label'] for cfg in HDBSCAN_CONFIGS]
palette = dict(zip(configs, sns.color_palette('tab10', n_colors=len(configs))))

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, metric in zip(axes, ['precision', 'recall', 'f1']):
    means = raw_df.groupby('hdbscan_cfg')[metric].mean().reindex(configs)
    stds  = raw_df.groupby('hdbscan_cfg')[metric].std().reindex(configs)
    ax.bar(range(len(configs)), means.values, yerr=stds.values,
           color=[palette[c] for c in configs], capsize=4, alpha=0.85)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=25, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title(metric.capitalize())
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Mean ± std across agents and sparsity levels')
plt.tight_layout()
plt.show()
