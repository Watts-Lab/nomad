# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lachesis Stop Detection
#
# Lachesis finds contiguous stays whose diameter is at most `delta_roam`. A stay must last at least `dur_min` minutes, with no ping gap above `dt_max` minutes.
#
# This example uses the same Garden City user and day as the other stop detection demos.

# %%
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

import nomad.data as data_folder
import nomad.io.base as loader
from nomad.stop_detection.sequential_algs import lachesis
from nomad.stop_detection.viz import (
    plot_pings, plot_stops, plot_stops_barcode, plot_time_barcode,
)

data_dir = Path(data_folder.__file__).parent
city = gpd.read_parquet(data_dir / "garden-city-buildings-mercator.parquet")
tc = {"user_id": "gc_identifier", "x": "dev_x", "y": "dev_y", "timestamp": "unix_ts"}
traj = loader.sample_from_file(
    data_dir / "gc_data_long", format="parquet", users=["admiring_brattain"],
    filters=("date", "==", "2024-01-01"), traj_cols=tc,
)

# %%
stops = lachesis(
    traj, delta_roam=20, dt_max=60, dur_min=5, complete_output=True, traj_cols=tc,
)

# %%
fig, (ax_map, ax_barcode) = plt.subplots(
    2, 1, figsize=(6, 6.5), gridspec_kw={"height_ratios": [10, 1]},
)
gpd.GeoDataFrame(geometry=[box(*city.total_bounds)], crs=city.crs).plot(
    ax=ax_map, color="#d3d3d3",
)
city.plot(ax=ax_map, edgecolor="white", linewidth=1, color="#8c8c8c")
plot_stops(stops, ax=ax_map, cmap="Blues", traj_cols=tc)
plot_pings(traj, ax=ax_map, s=6, color="black", alpha=0.5, traj_cols=tc)
ax_map.set_axis_off()
plot_time_barcode(traj[tc["timestamp"]], ax=ax_barcode, set_xlim=True)
plot_stops_barcode(stops, ax=ax_barcode, cmap="Blues", set_xlim=False, traj_cols=tc)
plt.tight_layout(pad=0.1)
plt.show()
