# ---
# jupyter:
#   jupytext:
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
# # Synthetic Philadelphia - Reworked Pipeline
#
# This version caches the fully rasterized city and parallelizes the per-agent workflow: destination diary generation, dense trajectory generation, sparse sampling, and dataframe extraction.
#
# `generate_dest_diary` uses optimization #2 (NumPy arrays for the EPR inner loop). Benchmark against baseline in `golden_dest_diary_benchmark.py`.
#
# City pickle caches baked gravity data (`restore_gravity` on load). Rebuild with `REBUILD_CITY_CACHE = True` after changing the city.

# %%
from pathlib import Path
import json
import time

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import pyarrow.dataset as ds
from joblib import Parallel, cpu_count, delayed
from shapely.geometry import box

import nomad.filters as filters
import nomad.map_utils as nm
import nomad.stop_detection.viz as viz
from nomad.city_gen import RasterCity, load as load_city_pickle, save as save_city_pickle
from nomad.io.base import from_df, from_file
from nomad.traj_gen import Agent, Population

# %% [markdown]
# ## Configuration

# %%
LARGE_BOX = box(-75.212193, 39.940800, -75.136933, 39.962847)

USE_FULL_CITY = False
OUTPUT_DIR = Path("output_rework")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ox.settings.cache_folder = "cache_rework"

if USE_FULL_CITY:
    BOX_NAME = "full"
    POLY = "Philadelphia, Pennsylvania, USA"
else:
    BOX_NAME = "large"
    POLY = LARGE_BOX

SPATIAL_GPKG = OUTPUT_DIR / f"spatial_data_{BOX_NAME}.gpkg"
ROTATION_METADATA_PATH = OUTPUT_DIR / f"rotation_metadata_{BOX_NAME}.json"
CITY_CACHE_PICKLE = OUTPUT_DIR / f"raster_city_{BOX_NAME}.pkl"

REGENERATE_SPATIAL_DATA = False
REBUILD_CITY_CACHE = False
N_JOBS = -1
RUN_ANALYSIS = True

config = {
    "box_name": BOX_NAME,
    "block_side_length": 15.0,
    "hub_size": 100,
    "N": 1000,
    "name_seed": 42,
    "name_count": 2,
    "epr_params": {
        "datetime": "2025-05-23 00:00-05:00",
        "end_time": "2025-07-01 00:00-05:00",
        "epr_time_res": 15,
        "rho": 0.4,
        "gamma": 0.3,
        "seed_base": 100,
    },
    "traj_params": {
        "dt": 0.5,
        "seed_base": 200,
    },
    "sampling_params": {
        "beta_ping": 7,
        "beta_start": 300,
        "beta_durations": 55,
        "ha": 11.5 / 15,
        "seed_base": 1,
    },
}


# %% [markdown]
# ## Helpers

# %%
def section(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def write_parquet_file(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def prepare_traj_parquet_dataset(df):
    out = from_df(df, mixed_timezone_behavior="naive")
    # Store datetimes as text so every batch parquet file has the same schema.
    out["datetime"] = out["datetime"].astype(str)
    return out


def write_dataset_from_files(paths, path, partition_cols):
    dataset = ds.dataset([str(file_path) for file_path in paths], format="parquet")
    ds.write_dataset(
        dataset.scanner(),
        base_dir=str(path),
        format="parquet",
        partitioning=partition_cols,
        existing_data_behavior="delete_matching",
    )


def write_traj_dataset_from_files(paths, path):
    dataset = ds.dataset([str(file_path) for file_path in paths], format="parquet")
    ds.write_dataset(
        dataset.scanner(),
        base_dir=str(path),
        format="parquet",
        partitioning=["date"],
        partitioning_flavor="hive",
        existing_data_behavior="delete_matching",
    )


# %% [markdown]
# ## Spatial Data Cache

# %%
section("SPATIAL DATA")

if REGENERATE_SPATIAL_DATA or not SPATIAL_GPKG.exists():
    t0 = time.time()
    buildings = nm.download_osm_buildings(
        POLY,
        crs="EPSG:3857",
        schema="garden_city",
        clip=True,
        infer_building_types=True,
        explode=True,
    )
    print(f"Buildings download: {time.time() - t0:>6.2f}s ({len(buildings):,} buildings)")

    if USE_FULL_CITY:
        boundary_polygon = nm.get_city_boundary_osm(POLY, simplify=True)[0]
        boundary_polygon = gpd.GeoSeries([boundary_polygon], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    else:
        boundary_polygon = gpd.GeoDataFrame(geometry=[POLY], crs="EPSG:4326").to_crs("EPSG:3857").geometry.iloc[0]

    buildings = gpd.clip(buildings, gpd.GeoDataFrame(geometry=[boundary_polygon], crs="EPSG:3857"))
    buildings = nm.remove_overlaps(buildings).reset_index(drop=True)

    t1 = time.time()
    streets = nm.download_osm_streets(
        POLY,
        crs="EPSG:3857",
        clip=True,
        explode=True,
        graphml_path=OUTPUT_DIR / "streets_consolidated.graphml",
    ).reset_index(drop=True)
    print(f"Streets download:   {time.time() - t1:>6.2f}s ({len(streets):,} streets)")

    t2 = time.time()
    rotated_streets, rotation_deg = nm.rotate_streets_to_align(streets, k=200)
    all_streets = streets.geometry.union_all()
    rotation_origin = (all_streets.centroid.x, all_streets.centroid.y)
    rotated_buildings = nm.rotate(buildings, rotation_deg=rotation_deg, origin=rotation_origin)
    rotated_boundary = nm.rotate(
        gpd.GeoDataFrame(geometry=[boundary_polygon], crs="EPSG:3857"),
        rotation_deg=rotation_deg,
        origin=rotation_origin,
    )
    print(f"Grid rotation:      {time.time() - t2:>6.2f}s ({rotation_deg:.2f} deg)")

    if SPATIAL_GPKG.exists():
        SPATIAL_GPKG.unlink()
    rotated_buildings.to_file(SPATIAL_GPKG, layer="buildings", driver="GPKG")
    rotated_streets.to_file(SPATIAL_GPKG, layer="streets", driver="GPKG", mode="a")
    rotated_boundary.to_file(SPATIAL_GPKG, layer="boundary", driver="GPKG", mode="a")

    with open(ROTATION_METADATA_PATH, "w") as f:
        json.dump({"rotation_deg": rotation_deg, "rotation_origin": rotation_origin}, f)
else:
    print(f"Loading spatial data from {SPATIAL_GPKG}")

with open(ROTATION_METADATA_PATH) as f:
    rotation_metadata = json.load(f)

rotation_deg = rotation_metadata["rotation_deg"]
rotation_origin = tuple(rotation_metadata["rotation_origin"])

# %% [markdown]
# ## Raster City Cache

# %%
section("RASTER CITY CACHE")

if REBUILD_CITY_CACHE or not CITY_CACHE_PICKLE.exists():
    buildings = gpd.read_file(SPATIAL_GPKG, layer="buildings")
    streets = gpd.read_file(SPATIAL_GPKG, layer="streets")
    boundary = gpd.read_file(SPATIAL_GPKG, layer="boundary")

    t0 = time.time()
    city_for_cache = RasterCity(
        boundary.geometry.iloc[0],
        streets,
        buildings,
        block_side_length=config["block_side_length"],
        resolve_overlaps=True,
        other_building_behavior="filter",
        rotation_deg=rotation_deg,
        rotation_origin=rotation_origin,
    )
    print(f"City generation:    {time.time() - t0:>6.2f}s")

    t1 = time.time()
    city_for_cache.get_street_graph()
    print(f"Street graph:       {time.time() - t1:>6.2f}s")

    t2 = time.time()
    city_for_cache._build_hub_network(hub_size=config["hub_size"])
    print(f"Hub network:        {time.time() - t2:>6.2f}s")

    t3 = time.time()
    city_for_cache.compute_gravity(exponent=2.0, callable_only=True)
    print(f"Gravity:            {time.time() - t3:>6.2f}s")

    t4 = time.time()
    city_for_cache.compute_shortest_paths(callable_only=True)
    print(f"Shortest paths:     {time.time() - t4:>6.2f}s")

    city_for_cache.grav = None
    city_for_cache.grav_for_candidates = None
    city_for_cache.shortest_paths = None

    save_city_pickle(city_for_cache, CITY_CACHE_PICKLE)
    print(f"Saved raster city cache to {CITY_CACHE_PICKLE}")
else:
    print(f"Loading raster city cache from {CITY_CACHE_PICKLE}")


# %%
def load_city_from_cache():
    city = load_city_pickle(CITY_CACHE_PICKLE)
    city.rotation_deg = rotation_metadata["rotation_deg"]
    city.rotation_origin = tuple(rotation_metadata["rotation_origin"])
    if city.grav is None or city.grav_for_candidates is None:
        city.restore_gravity(exponent=2.0, callable_only=True)
    if city.shortest_paths is None:
        city.compute_shortest_paths(callable_only=True)
    return city


city = load_city_from_cache()

summary_df = pd.DataFrame({
    "component": ["blocks", "streets", "buildings", "hubs", "nearby_door_pairs"],
    "count": [
        len(city.blocks_gdf),
        len(city.streets_gdf),
        len(city.buildings_gdf),
        len(city.hubs),
        len(city.mh_dist_nearby_doors),
    ],
})
print(summary_df.to_string(index=False))
print(city.buildings_gdf.building_type.value_counts())

# %% [markdown]
# ## Agent Parameters

# %%
section("AGENT PARAMETERS")

population_template = Population(city)
population_template.generate_agents(
    N=config["N"],
    seed=config["name_seed"],
    name_count=config["name_count"],
    datetimes=config["epr_params"]["datetime"],
)

agents = list(population_template.roster.values())

agent_params = pd.DataFrame({
    "identifier": [agent.identifier for agent in agents],
    "home": [agent.home for agent in agents],
    "workplace": [agent.workplace for agent in agents],
    "datetime": [config["epr_params"]["datetime"]] * len(agents),
    "dest_seed": config["epr_params"]["seed_base"] + np.arange(len(agents)),
    "traj_seed": config["traj_params"]["seed_base"] + np.arange(len(agents)),
    "sampling_seed": config["sampling_params"]["seed_base"] + np.arange(len(agents)),
})

agent_params["beta_start"] = config["sampling_params"]["beta_start"]
agent_params["beta_ping"] = config["sampling_params"]["beta_ping"]
agent_params["beta_durations"] = config["sampling_params"]["beta_durations"]
agent_params["ha"] = config["sampling_params"]["ha"]

agent_params_path = OUTPUT_DIR / f"agent_params_{BOX_NAME}.parquet"
agent_params.to_parquet(agent_params_path, index=False)
agent_params.head()

# %% [markdown]
# ## Per-Agent Generation

# %%
end_time = pd.Timestamp(config["epr_params"]["end_time"])


def effective_n_jobs(n_jobs, task_count):
    """Return the number of worker batches for the configured joblib setting."""
    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)
    return min(n_jobs, task_count)


def simulate_agent(row, city_worker):
    t_agent = time.perf_counter()
    t0 = time.perf_counter()
    agent = Agent(
        identifier=row.identifier,
        city=city_worker,
        home=row.home,
        workplace=row.workplace,
        datetime=row.datetime,
    )
    init_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    agent.generate_dest_diary(
        end_time=end_time,
        epr_time_res=config["epr_params"]["epr_time_res"],
        rho=config["epr_params"]["rho"],
        gamma=config["epr_params"]["gamma"],
        seed=int(row.dest_seed),
    )
    dest_time = time.perf_counter() - t0
    dest_diary = agent.destination_diary.copy()
    dest_diary["identifier"] = row.identifier

    t0 = time.perf_counter()
    agent.generate_trajectory(
        dt=config["traj_params"]["dt"],
        seed=int(row.traj_seed),
    )
    traj_time = time.perf_counter() - t0
    full_points = len(agent.trajectory)

    t0 = time.perf_counter()
    agent.sample_trajectory(
        beta_start=float(row.beta_start),
        beta_durations=float(row.beta_durations),
        beta_ping=float(row.beta_ping),
        ha=float(row.ha),
        seed=int(row.sampling_seed),
        replace_sparse_traj=True,
        cache_traj=True,
    )
    sample_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sparse = agent.sparse_traj.reset_index(drop=True).copy()
    diary = agent.diary.copy()
    extract_time = time.perf_counter() - t0
    summary = {
        "user_id": row.identifier,
        "dest_entries": len(dest_diary),
        "diary_entries": len(diary),
        "full_points": full_points,
        "sparse_points": len(sparse),
        "beta_start": row.beta_start,
        "beta_ping": row.beta_ping,
        "beta_durations": row.beta_durations,
        "ha": row.ha,
        "init_time": init_time,
        "dest_time": dest_time,
        "traj_time": traj_time,
        "sample_time": sample_time,
        "extract_time": extract_time,
        "agent_time": time.perf_counter() - t_agent,
    }
    return sparse, diary, dest_diary, summary


def simulate_agent_batch(batch_id, agent_batch, batch_output_dir):
    """Simulate one dataframe chunk and write its outputs to batch-local files."""
    t_batch = time.perf_counter()
    t0 = time.perf_counter()
    city_worker = load_city_from_cache()
    city_load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = [simulate_agent(row, city_worker) for row in agent_batch.itertuples(index=False)]
    agent_loop_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sparse_parts, diary_parts, dest_diary_parts, summary_rows = zip(*results)

    sparse_df = pd.concat(sparse_parts, ignore_index=True)
    diaries_df = pd.concat(diary_parts, ignore_index=True)
    dest_diaries_df = pd.concat(dest_diary_parts, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    concat_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    poi_data = pd.DataFrame({
        "building_id": city_worker.buildings_gdf["id"].values,
        "x": (city_worker.buildings_gdf["door_cell_x"].astype(float) + 0.5).values,
        "y": (city_worker.buildings_gdf["door_cell_y"].astype(float) + 0.5).values,
    })

    sparse_df = city_worker.to_mercator(sparse_df)
    diaries_df = diaries_df.merge(poi_data, left_on="location", right_on="building_id", how="left")
    diaries_df = diaries_df.drop(columns=["building_id"])
    diaries_df = city_worker.to_mercator(diaries_df)

    sparse_df["date"] = pd.to_datetime(sparse_df["timestamp"], unit="s").dt.date.astype(str)
    diaries_df["date"] = pd.to_datetime(diaries_df["timestamp"], unit="s").dt.date.astype(str)
    dest_diaries_df["date"] = pd.to_datetime(dest_diaries_df["datetime"]).dt.date.astype(str)
    summary_df["date"] = pd.to_datetime(config["epr_params"]["datetime"]).date().isoformat()
    reproject_time = time.perf_counter() - t0

    batch_dir = batch_output_dir / f"batch_{batch_id:04d}"
    paths = {
        "sparse": batch_dir / "sparse.parquet",
        "diaries": batch_dir / "diaries.parquet",
        "sparse_traj": batch_dir / "sparse_traj.parquet",
        "diaries_traj": batch_dir / "diaries_traj.parquet",
        "dest_diaries": batch_dir / "dest_diaries.parquet",
        "summary": batch_dir / "summary.parquet",
    }
    t0 = time.perf_counter()
    write_parquet_file(sparse_df, paths["sparse"])
    write_parquet_file(diaries_df, paths["diaries"])
    write_parquet_file(prepare_traj_parquet_dataset(sparse_df), paths["sparse_traj"])
    write_parquet_file(prepare_traj_parquet_dataset(diaries_df), paths["diaries_traj"])
    write_parquet_file(dest_diaries_df, paths["dest_diaries"])
    write_parquet_file(summary_df, paths["summary"])
    write_time = time.perf_counter() - t0

    counts = summary_df[["full_points", "sparse_points", "dest_entries", "diary_entries"]].sum()
    return {
        "batch_id": batch_id,
        "agents": len(agent_batch),
        "full_points": int(counts["full_points"]),
        "sparse_points": int(counts["sparse_points"]),
        "dest_entries": int(counts["dest_entries"]),
        "diary_entries": int(counts["diary_entries"]),
        "city_load_time": city_load_time,
        "agent_loop_time": agent_loop_time,
        "concat_time": concat_time,
        "reproject_time": reproject_time,
        "write_time": write_time,
        "batch_time": time.perf_counter() - t_batch,
        "paths": {key: str(value) for key, value in paths.items()},
    }


# %%
section("PARALLEL AGENT GENERATION")

n_workers = effective_n_jobs(N_JOBS, len(agent_params))
agent_batches = [agent_params.iloc[idx] for idx in np.array_split(np.arange(len(agent_params)), n_workers)]
run_id = pd.Timestamp.now('UTC').strftime("%Y%m%d_%H%M%S")
batch_output_dir = OUTPUT_DIR / f"batch_outputs_{BOX_NAME}" / run_id

t0 = time.time()
batch_results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(simulate_agent_batch)(batch_id, batch, batch_output_dir)
    for batch_id, batch in enumerate(agent_batches)
)
generation_time = time.time() - t0

batch_manifest = pd.DataFrame([
    {"batch_id": result["batch_id"], "output": key, "path": value}
    for result in batch_results
    for key, value in result["paths"].items()
])
batch_timing = pd.DataFrame([
    {key: result[key] for key in [
        "batch_id", "agents", "city_load_time", "agent_loop_time",
        "concat_time", "reproject_time", "write_time", "batch_time",
    ]}
    for result in batch_results
])
generation_summary = pd.concat(
    [pd.read_parquet(result["paths"]["summary"]) for result in batch_results],
    ignore_index=True,
)

print(f"Generated {len(agent_params)} agents in {generation_time:.2f}s")
print(f"Batch outputs: {batch_output_dir}")
print(generation_summary[["full_points", "sparse_points", "dest_entries", "diary_entries"]].sum())
print(batch_timing.drop(columns=["batch_id", "agents"]).sum().sort_values(ascending=False))
generation_summary.head()

# %% [markdown]
# ## Save Final Datasets

# %%
section("SAVE FINAL DATASETS")

config_path = OUTPUT_DIR / f"config_{BOX_NAME}.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

manifest_path = OUTPUT_DIR / f"batch_manifest_{BOX_NAME}.parquet"
batch_manifest.to_parquet(manifest_path, index=False)
timing_path = OUTPUT_DIR / f"batch_timing_{BOX_NAME}.parquet"
batch_timing.to_parquet(timing_path, index=False)

paths_by_output = batch_manifest.groupby("output")["path"].apply(list)
write_traj_dataset_from_files(paths_by_output["sparse_traj"], OUTPUT_DIR / f"sparse_traj_{BOX_NAME}")
write_traj_dataset_from_files(paths_by_output["diaries_traj"], OUTPUT_DIR / f"diaries_{BOX_NAME}")
write_dataset_from_files(paths_by_output["dest_diaries"], OUTPUT_DIR / f"dest_diaries_{BOX_NAME}", ["date"])
write_dataset_from_files(paths_by_output["summary"], OUTPUT_DIR / f"homes_{BOX_NAME}", ["date"])

write_dataset_from_files(paths_by_output["sparse"], OUTPUT_DIR / "device_level", ["date"])
write_dataset_from_files(paths_by_output["diaries"], OUTPUT_DIR / "travel_diaries", ["date"])

print(f"Config saved to {config_path}")
print(f"Agent params saved to {agent_params_path}")
print(f"Batch manifest saved to {manifest_path}")
print(f"Batch timing saved to {timing_path}")
print(f"Sparse trajectories: {OUTPUT_DIR / f'sparse_traj_{BOX_NAME}'}")
print(f"Realized diaries:    {OUTPUT_DIR / f'diaries_{BOX_NAME}'}")
print(f"Destination diaries: {OUTPUT_DIR / f'dest_diaries_{BOX_NAME}'}")

# %% [markdown]
# ## Trajectory Visualization

# %%
if RUN_ANALYSIS:
    section("TRAJECTORY VISUALIZATION")

    device_df = from_file(OUTPUT_DIR / f"sparse_traj_{BOX_NAME}", format="parquet")
    device_df["plot_date"] = pd.to_datetime(device_df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
    sample_dates = device_df["plot_date"].value_counts().head(3).sort_index().index.tolist()

    stop_paths = [OUTPUT_DIR / f"diaries_{BOX_NAME}" / f"date={date}" for date in sample_dates]
    stop_df = from_file(stop_paths, format="parquet")
    stop_df["plot_date"] = pd.to_datetime(stop_df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")

    device_df_day = device_df[device_df["plot_date"].isin(sample_dates)].copy()
    valid_stop_rows = stop_df[
        stop_df["location"].notna()
        & stop_df["x"].notna()
        & stop_df["y"].notna()
    ].copy()
    selected_users = (
        valid_stop_rows.loc[valid_stop_rows["user_id"].isin(device_df_day["user_id"]), "user_id"]
        .drop_duplicates()
        .head(3)
    )

    print(f"Selected dates: {', '.join(sample_dates)}")
    print(f"Pings in window: {len(device_df_day):,}")
    print(f"Stops with coordinates in window: {len(valid_stop_rows):,}")

    fig, axes = plt.subplots(1, len(selected_users), figsize=(6 * len(selected_users), 7), squeeze=False)
    axes = axes.ravel()

    for ax, user_id in zip(axes, selected_users):
        user_pings = device_df_day[device_df_day["user_id"] == user_id]
        user_stops = valid_stop_rows[valid_stop_rows["user_id"] == user_id].copy()
        user_stops["cluster"] = np.arange(1, len(user_stops) + 1)

        viz.plot_pings(user_pings, ax, color="black", s=1, alpha=0.6, data_crs="EPSG:3857")
        viz.plot_stops(
            user_stops,
            ax,
            radius=40,
            cmap="Reds",
            data_crs="EPSG:3857",
            traj_cols={"x": "x", "y": "y"},
        )

        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        ax.set_title(f"User: {user_id} ({', '.join(sample_dates)})")
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Data Completeness Analysis

# %%
if RUN_ANALYSIS:
    section("DATA COMPLETENESS ANALYSIS")

    comp_hourly = filters.completeness(device_df, periods=1, freq="h", user_id="user_id", timestamp="timestamp")
    comp_daily = filters.completeness(device_df, periods=1, freq="d", user_id="user_id", timestamp="timestamp")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(comp_hourly, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Completeness (proportion of hours with data)")
    axes[0].set_ylabel("Number of Users")
    axes[0].set_title("Hourly Completeness")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(comp_daily, bins=5, edgecolor="black", alpha=0.7, color="coral")
    axes[1].set_xlabel("Completeness (proportion of days with data)")
    axes[1].set_ylabel("Number of Users")
    axes[1].set_title("Daily Completeness")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Completeness Statistics:")
    print(f"  Hourly  - Mean: {comp_hourly.mean():.3f}, Median: {comp_hourly.median():.3f}")
    print(f"  Daily   - Mean: {comp_daily.mean():.3f}, Median: {comp_daily.median():.3f}")

# %% [markdown]
# ## Dataset Statistics

# %%
if RUN_ANALYSIS:
    section("DATASET STATISTICS")

    pings_per_user = device_df.groupby("user_id").size()

    print(f"Number of users: {device_df['user_id'].nunique()}")
    print(f"Total pings: {len(device_df):,}")
    print(f"Pings per user - Mean: {pings_per_user.mean():.0f}, Median: {pings_per_user.median():.0f}")
    print(f"Pings per user - Min: {pings_per_user.min()}, Max: {pings_per_user.max()}")

    device_df["loc_grid"] = (device_df["x"] // 100).astype(str) + "_" + (device_df["y"] // 100).astype(str)
    device_df["day_num"] = (
        pd.to_datetime(device_df["plot_date"])
        - pd.to_datetime(device_df["plot_date"]).min()
    ).dt.days

    first_seen = device_df.groupby(["user_id", "loc_grid"], as_index=False)["day_num"].min()
    new_locations = first_seen.groupby(["user_id", "day_num"]).size().rename("new_locations")
    full_index = pd.MultiIndex.from_product(
        [device_df["user_id"].drop_duplicates(), range(device_df["day_num"].max() + 1)],
        names=["user_id", "day"],
    )
    cum_loc_df = (
        new_locations.rename_axis(index={"day_num": "day"})
        .reindex(full_index, fill_value=0)
        .groupby(level="user_id")
        .cumsum()
        .reset_index()
        .rename(columns={"new_locations": "cumulative_locations"})
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(cum_loc_df["day"], cum_loc_df["cumulative_locations"], alpha=0.2, s=15, color="darkgreen")
    ax.set_xlabel("Day (relative to start)")
    ax.set_ylabel("Cumulative Unique Locations Visited (100m grid)")
    ax.set_title("Growth of Unique Locations Over Time")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Quick Checks

# %%
print("Sparse rows:", int(generation_summary["sparse_points"].sum()))
print("Diary rows:", int(generation_summary["diary_entries"].sum()))
print("Destination diary rows:", int(generation_summary["dest_entries"].sum()))
print("Sparse ratio:", f"{generation_summary['sparse_points'].sum() / generation_summary['full_points'].sum():.2%}")

agent_timing_cols = ["init_time", "dest_time", "traj_time", "sample_time", "extract_time", "agent_time"]
agent_timing = generation_summary[agent_timing_cols].sum().sort_values(ascending=False)
print("\nAgent CPU seconds (summed across agents)")
print(agent_timing.to_string())
print("\nAgent CPU share")
print((agent_timing.drop("agent_time") / agent_timing["agent_time"]).sort_values(ascending=False).map("{:.1%}".format).to_string())

batch_timing_cols = ["city_load_time", "agent_loop_time", "concat_time", "reproject_time", "write_time", "batch_time"]
print("\nBatch wall (max across workers)")
print(batch_timing[batch_timing_cols].max().sort_values(ascending=False).to_string())
print("\nBatch CPU seconds (summed across workers)")
print(batch_timing[batch_timing_cols].sum().sort_values(ascending=False).to_string())

generation_summary[agent_timing_cols + ["full_points", "sparse_points", "dest_entries", "diary_entries"]].describe()
