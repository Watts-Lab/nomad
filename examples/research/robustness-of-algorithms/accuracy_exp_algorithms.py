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
#     display_name: conda_python3
#     language: python
#     name: conda_python3
# ---

# %%
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import os
from functools import partial
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import nomad.io.base as loader
import nomad.stop_detection.utils as utils
from nomad.stop_detection.density_algs import seqscan_labels_per_user
from nomad.stop_detection.sequential_algs import lachesis_labels_per_user

import nomad.visit_attribution.visit_attribution as visits

from nomad.contact_estimation import compute_stop_detection_metrics
import warnings
warnings.filterwarnings('ignore', message='Input is timezone-aware; assuming UTC')

# %%
keys = ['ha13_beta5', 'ha13_beta6',
        'ha13_beta7', 'ha13_beta8',
        'ha15_beta5', 'ha15_beta6',
        'ha15_beta7', 'ha15_beta8',
        'ha17_beta5', 'ha17_beta6',
        'ha17_beta7', 'ha17_beta8']

dist_thresh_values = np.linspace(1, 70, 35)
delta_roam_values  = np.linspace(5, 80, 30)

algos = [*[("seqscan",{"func": seqscan_labels_per_user,
                      "params": {"time_thresh": 60, "dist_thresh": dr, "min_pts": 2,
                                 "back_merge": False, "n_jobs": 12},
                                 "visit_attr": "majority"}) for dr in dist_thresh_values],
        *[("seqscan", {"func": seqscan_labels_per_user,
                       "params": {"time_thresh": 60, "dist_thresh": dr, "min_pts": 3,
                                  "back_merge": False, "n_jobs": 12},
                                  "visit_attr": "majority"}) for dr in dist_thresh_values],
        *[("lachesis", {"func": lachesis_labels_per_user,
                        "params": {"dt_max": 60, "delta_roam": 1.7 * dr,
                                   "n_jobs": 12},
                                   "visit_attr": "majority"}) for dr in delta_roam_values]]

summarize_stops_with_loc = partial(
    utils.summarize_stop,
    x='x', y='y', timestamp='timestamp',
    keep_col_names=True,
    passthrough_cols=['location', 'user_id', 'cluster'],
    complete_output=False
)

# %%
all_results = []

for key in keys:
    config_file = f"config_{key}.json"

    if os.path.exists(f"results_{key}.parquet"):
        print(f"results_{key}.parquet already exists. Skipping save.")
        all_results.append(pd.read_parquet(f"results_{key}.parquet"))
        continue

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    par = config['agent_params']
    out = config['output_files']

    beta_ping = par['beta_ping']
    ha = par['ha'] * 15

    poi_table = gpd.read_parquet(config["buildings_file"]).rename(columns={"id": "location"})
    sparse_df = loader.from_file(out["sparse_path"], format="parquet")
    sparse_df.drop(columns=['datetime'], inplace=True)
    diaries_df = loader.from_file(out["diaries_path"], format="parquet").rename(
        columns={"identifier": "user_id"})

    poi_locations = sorted(diaries_df['location'].dropna().unique().tolist())
    poi_subset = poi_table.loc[poi_table.location.isin(poi_locations)]

    sparse_df['precomp_locations'] = visits.poi_map(
        sparse_df, poi_table=poi_subset, data_crs='EPSG:3857',
        max_distance=20, location_id='location', x='x', y='y'
    )

    results_list = []
    for algo_name, algo_config in tqdm(algos, desc=key):
        labels = algo_config["func"](sparse_df, **algo_config["params"],
                                     x="x", y="y", timestamp='timestamp',
                                     user_id='user_id')
        sparse_df['cluster'] = labels
        sparse_df['location'] = sparse_df['precomp_locations']

        sparse_df['location'] = visits.point_in_polygon(
            sparse_df, poi_table=poi_subset, data_crs='EPSG:3857',
            max_distance=20, location_id='location',
            method=algo_config["visit_attr"], x='x', y='y',
            recompute_location=False
        )

        stops_all = (
            sparse_df[sparse_df.cluster != -1]
            .groupby(['user_id', 'cluster'], as_index=False)
            .apply(summarize_stops_with_loc, include_groups=False).reset_index()
        )

        for user in diaries_df.user_id.unique():
            stops = stops_all[stops_all.user_id == user]
            truth = diaries_df.query("user_id==@user")

            metrics = compute_stop_detection_metrics(
                stops=stops, truth=truth, user_id=user, algorithm=algo_name,
                traj_cols={'location_id': 'location'}, timestamp='timestamp'
            )
            metrics['key'] = key
            metrics['beta_ping'] = beta_ping
            metrics['ha'] = ha
            metrics['dist_thresh'] = algo_config["params"].get("dist_thresh", np.nan)
            metrics['delta_roam'] = algo_config["params"].get("delta_roam", np.nan)
            metrics['min_pts'] = algo_config["params"].get("min_pts", np.nan)
            results_list.append(metrics)

        sparse_df.drop(columns=['cluster', 'location'], inplace=True)

    config_df = pd.DataFrame(results_list)
    config_df.to_parquet(f"results_{key}.parquet", index=False)
    all_results.append(config_df)
    print(f"  {len(config_df)} rows saved to results_{key}.parquet")

results_df = pd.concat(all_results, ignore_index=True)
results_df.to_parquet("results_all_configs.parquet", index=False)
print(f"\nTotal rows: {len(results_df)}")

# %%
'''
mean_acc_theta_star_i:
This takes each user's personal best score (idxmax() picks the row where each user peaked), then averages those best scores across users.
If every user could use their own perfect parameter setting, what's the average performance?

mean_acc_theta_star:
This picks one single parameter value (theta_star, the population-level median optimal) and evaluates performance for everyone using that same value.
If we had to pick one parameter setting for all users, and we chose the median-optimal one, what's the average performance?
'''

def stats_summary(data, metric="recall", param='delta_roam', param2=None):
    data = data.dropna(subset=[param])
    
    if param2:
        stats_by_user = (data.loc[data.groupby("user_id")[metric].idxmax(), ["user_id", metric, param, param2, "ha", "beta_ping"]].reset_index(drop=True))
        param2_star = stats_by_user[param2].value_counts().idxmax()
        theta_star = stats_by_user[stats_by_user[param2] == param2_star][param].quantile(0.5, interpolation='lower')
    else:
        stats_by_user = (data.loc[data.groupby("user_id")[metric].idxmax(), ["user_id", metric, param, "ha", "beta_ping"]].reset_index(drop=True))
        theta_star = stats_by_user[param].quantile(0.5, interpolation='lower')

    return pd.DataFrame({"algorithm": data['algorithm'].iloc[0],
                         "ha": stats_by_user["ha"].iloc[0],
                         "beta_ping": stats_by_user["beta_ping"].iloc[0],
                         "mean_acc_theta_star": data.loc[data[param]==theta_star][metric].mean(),
                         "mean_acc_theta_star_i": round(stats_by_user[metric].mean(), 4), # oracle performance
                         "std_acc_theta_star": data.loc[data[param]==theta_star][metric].std(),
                         "theta_star": theta_star,
                         'min_pts': param2_star if param2 else np.nan},
                         index=[param])


# %%
results1 = results_df.groupby("key").apply(stats_summary, metric="recall", param='delta_roam').reset_index().drop(columns=['key', 'level_1'])
results2 = results_df.groupby("key").apply(stats_summary, metric="recall", param='dist_thresh', param2='min_pts').reset_index().drop(columns=['key', 'level_1'])

# %%
results1

# %%
results2

# %%
final_results = pd.concat([results1, results2], ignore_index=True).sort_values(['beta_ping', 'ha'])

# %%
final_results

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
ax1, ax2, ax3, ax4 = axes.flatten()

# ha
sns.barplot(data=final_results, x='ha', y='mean_acc_theta_star',
            hue='algorithm', errorbar=None, ax=ax1)
ax1.set_xlabel('HA')
ax1.set_ylabel('Mean Accuracy')
ax1.set_title('Mean Accuracy (population-optimal θ) by HA')
ax1.legend(title='Algorithm')

sns.barplot(data=final_results, x='ha', y='mean_acc_theta_star_i',
            hue='algorithm', errorbar=None, ax=ax3)
ax3.set_xlabel('HA')
ax3.set_ylabel('Mean Accuracy')
ax3.set_title('Mean Accuracy (per-user optimal θ) by HA')
ax3.legend(title='Algorithm')

# beta_ping
sns.barplot(data=final_results, x='beta_ping', y='mean_acc_theta_star',
            hue='algorithm', errorbar=None, ax=ax2)
ax2.set_xlabel('Beta Ping')
ax2.set_ylabel('Mean Accuracy')
ax2.set_title('Mean Accuracy (population-optimal θ) by Beta Ping')
ax2.legend(title='Algorithm')

sns.barplot(data=final_results, x='beta_ping', y='mean_acc_theta_star_i',
            hue='algorithm', errorbar=None, ax=ax4)
ax4.set_xlabel('Beta Ping')
ax4.set_ylabel('Mean Accuracy')
ax4.set_title('Mean Accuracy (per-user optimal θ) by Beta Ping')
ax4.legend(title='Algorithm')

plt.suptitle('Mean Accuracy Comparison', fontsize=14)
plt.tight_layout()
plt.show()

# %%
final_results['ha_beta'] = final_results.apply(
    lambda row: f"ha={int(row['ha'])} \n β={int(row['beta_ping'])}", axis=1
)

# %%
final_results

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
ax1, ax2 = axes.flatten()

sns.barplot(data=final_results, x='ha_beta', y='mean_acc_theta_star',
            hue='algorithm', errorbar=None, ax=ax1)
ax1.set_xlabel('HA & Beta Ping')
ax1.set_ylabel('Mean Accuracy')
ax1.set_title('Mean Accuracy (population-optimal θ) by HA and β')
ax1.legend(title='Algorithm')

sns.barplot(data=final_results, x='ha_beta', y='mean_acc_theta_star_i',
            hue='algorithm', errorbar=None, ax=ax2)
ax2.set_xlabel('HA & Beta Ping')
ax2.set_ylabel('Mean Accuracy')
ax2.set_title('Mean Accuracy (per-user optimal θ) by HA and β')
ax2.legend(title='Algorithm')

plt.show()

# %%
final_results['gap'] = final_results['mean_acc_theta_star_i'] - final_results['mean_acc_theta_star']

# %%
fig, ax = plt.subplots(figsize=(12, 5))

sns.barplot(
    data=final_results,
    x='ha_beta',
    y='gap',
    hue='algorithm',
    errorbar=None,
    ax=ax
)

ax.set_xlabel('HA & Beta Ping')
ax.set_ylabel('Per-user − Population-optimal Accuracy')
ax.set_title('Per-user vs Population-optimal θ difference by Algorithm')
ax.tick_params(axis='x')
ax.legend(title='Algorithm')

plt.tight_layout()
plt.show()
