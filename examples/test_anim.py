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
#     display_name: nomad-clean
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stop Dashboard Animation

# %%
# %matplotlib inline

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from IPython.display import HTML

import nomad.data as data_folder
import nomad.filters as filters
import nomad.io.base as loader
from nomad.stop_detection.density_algs import ta_dbscan, ta_dbscan_labels
from nomad.stop_detection.density_algs import seqscan, seqscan_labels
from nomad.stop_detection.sequential_algs import grid_based, grid_based_labels
from nomad.stop_detection.density_algs import hdbscan_labels, st_hdbscan
from nomad.stop_detection.sequential_algs import lachesis, lachesis_labels
from nomad.stop_detection.viz import animate_stop_dashboard

# %%
tc = {
    "user_id": "gc_identifier",
    "x": "dev_x",
    "y": "dev_y",
    "timestamp": "unix_ts",
}

data_dir = Path(data_folder.__file__).parent

traj = loader.sample_from_file(
    data_dir / "gc_data_long",
    format="parquet",
    users=["admiring_brattain"],
    filters=("date", "==", "2024-01-01"),
    traj_cols=tc,
)
traj["h3_cell"] = filters.to_tessellation(
    traj,
    index="h3",
    res=11,
    traj_cols=tc,
    data_crs="EPSG:3857",
)
city = gpd.read_parquet(data_dir / "garden-city-buildings-mercator.parquet")

# %%
animation_cases = {
    "lachesis": {
        "run": lachesis,
        "labels": lachesis_labels,
        "params": {"delta_roam": 20, "dt_max": 60, "dur_min": 5},
    },
    "seqscan": {
        "run": seqscan,
        "labels": seqscan_labels,
        "params": {"time_thresh": 60, "dist_thresh": 8, "min_pts": 3},
    },
    "hdbscan": {
        "run": st_hdbscan,
        "labels": hdbscan_labels,
        "params": {"time_thresh": 720, "min_pts": 3},
    },
    "ta_dbscan": {
        "run": ta_dbscan,
        "labels": ta_dbscan_labels,
        "params": {"time_thresh": 60, "dist_thresh": 10, "min_pts": 3},
    },
    "grid_based": {
        "run": grid_based,
        "labels": grid_based_labels,
        "params": {"time_thresh": 240, "location_id": "h3_cell"},
    },
}

# %%
case = animation_cases["lachesis"]
stops = case["run"](
    traj,
    complete_output=True,
    traj_cols=tc,
    **case["params"],
)

fig, (ax_map, ax_barcode) = plt.subplots(
    2,
    1,
    figsize=(6, 6.5),
    gridspec_kw={"height_ratios": [10, 1]},
)

anim = animate_stop_dashboard(
    data=traj.assign(
        cluster=case["labels"](
            traj,
            traj_cols=tc,
            **case["params"],
        ).to_numpy()
    ),
    stops=stops,
    traj_cols=tc,
    show_path=False,
    show_stop_overlays=False,
    ping_color="cluster",
    ping_cmap="inferno_r",
    ping_size=30,
    base_geometry=city,
    base_geom_color="#8c8c8c",
    base_geom_background="#d3d3d3",
    ax_map=ax_map,
    ax_barcode=ax_barcode,
)

plt.tight_layout(pad=0.1)
fig.subplots_adjust(top=0.98, bottom=0.12, hspace=0.12)
html = anim.to_html5_video()
plt.close(fig)
HTML(html.replace("<video ", "<video autoplay muted playsinline ", 1))
