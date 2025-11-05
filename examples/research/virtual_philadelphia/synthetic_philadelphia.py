# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Synthetic Philadelphia
#
# This notebook will use the functions in `map_utils.py` to create a synthetic rasterized version of Philadelphia. It starts by downloading and classifying buildings from OSM in web mercator coordinates, and reporting on the building counts for each subtype and each of the _garden city building types_ which are:
#   - park
#   - home
#   - work
#   - retail
#   - other (will be randomly assigned to one of the above categories)
# It also identifies which rotation best aligns a random sample of streets with a N-S, E-W grid. 

# %%
from pathlib import Path
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry.geo import box

import nomad.map_utils as nm
from nomad.city_gen import RasterCity


# %% [markdown]
# ## 1. Download and Persist Philadelphia OSM Data (Web Mercator)
#
# This section downloads buildings and streets for Philadelphia in Web Mercator
# using chunked OSM queries, classifies buildings with the Garden City schema,
# removes overlaps, and stores the results in a GeoPackage for downstream use.


# %%
# if 
FULL_CITY = False
if FULL_CITY:
    CITY_NAME = "Philadelphia, Pennsylvania, USA"
    OUTPUT_DIR = Path(".")  # Relative path for Jupyter compatibility
    RAW_GPKG_PATH = OUTPUT_DIR / "philadelphia_osm_raw.gpkg"
    LOAD_FROM_CACHE = RAW_GPKG_PATH.exists()
    POLY = CITY_NAME
else:
    LARGE_BOX  = box(-75.1905, 39.9235, -75.1425, 39.9535)
    POLY = LARGE_BOX
    
if LOAD_FROM_CACHE:
    print("Loading persisted data from GeoPackage...")
    buildings = gpd.read_file(RAW_GPKG_PATH, layer="buildings")
    streets = gpd.read_file(RAW_GPKG_PATH, layer="streets")
    boundary_gdf = gpd.read_file(RAW_GPKG_PATH, layer="city_boundary")
    print(f"Loaded {len(buildings):,} buildings and {len(streets):,} streets")
else:
    print(f"Fetching city boundary for {CITY_NAME}...")
    boundary, city_center, population = nm.get_city_boundary_osm(CITY_NAME, simplify=True)
    
    boundary_gdf = gpd.GeoDataFrame(
        {
            "name": [CITY_NAME],
            "population": [population],
        },
        geometry=[boundary],
        crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    
    print("Downloading buildings (single query, Web Mercator)...")
    start_time = time.time()
    buildings = nm.download_osm_buildings(
        boundary,
        crs="EPSG:3857",
        schema="garden_city",
        clip=True,
        infer_building_types=True,
        explode=True,
    )
    elapsed = time.time() - start_time
    print(f"Downloaded {len(buildings):,} buildings in {elapsed:.1f}s")
    
    buildings = nm.remove_overlaps(buildings).reset_index(drop=True)
    
    print("Downloading streets (single query, Web Mercator)...")
    start_time = time.time()
    streets = nm.download_osm_streets(
        boundary,
        crs="EPSG:3857",
        clip=True,
        explode=True,
    )
    elapsed = time.time() - start_time
    print(f"Downloaded {len(streets):,} streets in {elapsed:.1f}s")
    
    streets = streets.reset_index(drop=True)
    
    RAW_GPKG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_GPKG_PATH.exists():
        RAW_GPKG_PATH.unlink()
    
    print(f"Persisting raw data layers to {RAW_GPKG_PATH}...")
    buildings.to_file(RAW_GPKG_PATH, layer="buildings", driver="GPKG")
    streets.to_file(RAW_GPKG_PATH, layer="streets", driver="GPKG", mode="a")
    boundary_gdf.to_file(RAW_GPKG_PATH, layer="city_boundary", driver="GPKG", mode="a")

# %%
category_counts = buildings["building_type"].value_counts().to_dict()
print("\nBuilding category counts:")
for key, value in sorted(category_counts.items()):
    print(f"  {key}: {value:,}")

subtype_counts = buildings["subtype"].value_counts().head(15).to_dict()
print("\nTop building subtypes (first 15):")
for key, value in sorted(subtype_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {key}: {value:,}")


# %% [markdown]
# ## 2. Find Optimal Grid Alignment and Rotate
#
# Estimate the best rotation to align the street network with a North-South,
# East-West grid, apply it to the streets and buildings, persist the rotated
# geometries, and produce diagnostic plots.


# %%
print("Estimating optimal rotation from street bearings...")
rotation_start = time.time()
rotated_streets, rotation_deg = nm.rotate_streets_to_align(streets, k=600)
rotation_elapsed = time.time() - rotation_start
print(f"Rotation estimated at {rotation_deg:.2f} degrees (computed in {rotation_elapsed:.1f}s).")

rotated_buildings = nm.rotate(buildings, rotation_deg=rotation_deg)
rotated_boundary = nm.rotate(boundary_gdf, rotation_deg=rotation_deg)

print("Persisting rotated layers to GeoPackage...")
rotated_buildings.to_file(RAW_GPKG_PATH, layer="buildings_rotated", driver="GPKG", mode="a")
rotated_streets.to_file(RAW_GPKG_PATH, layer="streets_rotated", driver="GPKG", mode="a")
rotated_boundary.to_file(RAW_GPKG_PATH, layer="city_boundary_rotated", driver="GPKG", mode="a")

# %% [markdown]
# ### A/B check: category counts with and without speculative inference

# %%
try:
    def _category_counts(df):
        return df['building_type'].value_counts().to_dict()

    # Re-categorize using internal categorizer (garden_city schema)
    ab_no_infer = nm._categorize_features(rotated_buildings, schema='garden_city', infer_building_types=False)
    ab_yes_infer = nm._categorize_features(rotated_buildings, schema='garden_city', infer_building_types=True)
    print("Category counts (infer_building_types=False):", nm.get_category_summary(ab_no_infer))
    print("Category counts (infer_building_types=True): ", nm.get_category_summary(ab_yes_infer))
except Exception as e:
    print(f"A/B category check skipped: {e}")


# %%
print("Generating diagnostic plots...")

def _plot_streets(ax, s_gdf, title):
    s_gdf.plot(ax=ax, color="#000000", linewidth=0.3, alpha=0.7)
    ax.set_title(title)
    ax.set_axis_off()
    ctx.add_basemap(ax, crs=s_gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik, attribution="")


fig, axes = plt.subplots(1, 2, figsize=(18, 9))

_plot_streets(axes[0], streets, "Original Streets (Web Mercator)")
_plot_streets(axes[1], rotated_streets, f"Rotated Streets ({rotation_deg:.2f}°)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "philadelphia_rotation_diagnostics.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Streets-only rasterization and graph timing
#
# Baby step: rasterize only the rotated streets and measure graph build time.

# %%
print("\nStreets-only rasterization and graph timing...")
rot_streets = gpd.read_file(RAW_GPKG_PATH, layer="streets_rotated")
rot_buildings = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=rot_streets.crs)
rot_boundary = gpd.read_file(RAW_GPKG_PATH, layer="city_boundary_rotated")

boundary_geom = rot_boundary.geometry.iloc[0]
block_side_length = 15.0

import time as _t
_t0 = _t.time()
city_streets = RasterCity(
    boundary_polygon=boundary_geom,
    streets_gdf=rot_streets,
    buildings_gdf=rot_buildings,
    block_side_length=block_side_length,
)
streets_only_elapsed = _t.time() - _t0
print(f"Streets-only rasterization: {streets_only_elapsed:.2f}s; streets={len(city_streets.streets_gdf):,}, blocks={len(city_streets.blocks_gdf):,}")

# Graph build timing (includes shortcut network)
_t1 = _t.time()
G = city_streets.get_street_graph(lazy=True)
graph_elapsed = _t.time() - _t1
print(f"Street graph build (lazy + shortcuts): {graph_elapsed:.2f}s; nodes={len(G.nodes):,}, edges={len(G.edges):,}")

# %% [markdown]
# ## 4. Rasterize City with Gravity Computation
#
# Rasterize the city, build hub network, and compute gravity using callable_only=True
# for memory efficiency.

# %%
print("\nRasterizing city...")
rot_buildings = gpd.read_file(RAW_GPKG_PATH, layer="buildings_rotated")
rot_streets = gpd.read_file(RAW_GPKG_PATH, layer="streets_rotated")
rot_boundary = gpd.read_file(RAW_GPKG_PATH, layer="city_boundary_rotated")

boundary_geom = rot_boundary.geometry.iloc[0]
block_side_length = 10.0
hub_size = 100

_t0 = time.time()
city = RasterCity(
    boundary_polygon=boundary_geom,
    streets_gdf=rot_streets,
    buildings_gdf=rot_buildings,
    block_side_length=block_side_length,
    resolve_overlaps=True
)
gen_time = time.time() - _t0
print(f"City generation: {gen_time:.2f}s")
print(f"  Blocks: {len(city.blocks_gdf):,}")
print(f"  Streets: {len(city.streets_gdf):,}")
print(f"  Buildings: {len(city.buildings_gdf):,}")

_t1 = time.time()
G = city.get_street_graph()
graph_time = time.time() - _t1
print(f"Street graph: {graph_time:.2f}s ({len(G.nodes):,} nodes, {len(G.edges):,} edges)")

_t2 = time.time()
city._build_hub_network(hub_size=hub_size)
hub_time = time.time() - _t2
print(f"Hub network: {hub_time:.2f}s ({hub_size}×{hub_size})")

_t3 = time.time()
city.compute_gravity(exponent=2.0, callable_only=True)
grav_time = time.time() - _t3
print(f"Gravity computation: {grav_time:.2f}s (callable)")

total_time = gen_time + graph_time + hub_time + grav_time
print(f"Total rasterization: {total_time:.2f}s")

RASTER_GPKG = OUTPUT_DIR / "philadelphia_rasterized.gpkg"
if RASTER_GPKG.exists():
    RASTER_GPKG.unlink()
city.save_geopackage(str(RASTER_GPKG), persist_blocks=True, persist_city_properties=True)
print(f"Saved rasterized city to {RASTER_GPKG}")

# %% [markdown]
# ## 5. Generate Population and Destination Diaries
#
# Create 10 agents and generate destination diaries using EPR for 1 week.

# %%
import json
import pandas as pd
from nomad.traj_gen import Population

print("\nGenerating population and destination diaries...")

config = {
    "city_file": str(RASTER_GPKG),
    "block_side_length": block_side_length,
    "hub_size": hub_size,
    "N": 10,
    "name_seed": 42,
    "name_count": 2,
    "epr_params": {
        "end_time": "2024-01-08 00:00-05:00",
        "epr_time_res": 15,
        "rho": 0.4,
        "gamma": 0.3,
        "seed_base": 100
    },
    "output_dir": str(OUTPUT_DIR / "diaries")
}

config_path = OUTPUT_DIR / "simulation_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Saved config to {config_path}")

population = Population(city)
population.generate_agents(
    N=config["N"],
    seed=config["name_seed"],
    name_count=config["name_count"]
)
print(f"Generated {len(population.roster)} agents")

end_time = pd.Timestamp(config["epr_params"]["end_time"])
output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

for i, agent in enumerate(population.roster.values()):
    agent.generate_dest_diary(
        end_time=end_time,
        epr_time_res=config["epr_params"]["epr_time_res"],
        rho=config["epr_params"]["rho"],
        gamma=config["epr_params"]["gamma"],
        seed=config["epr_params"]["seed_base"] + i
    )
    
    diary_path = output_dir / f"{agent.identifier}_diary.csv"
    agent.destination_diary.to_csv(diary_path, index=False)
    print(f"  Saved {agent.identifier} diary ({len(agent.destination_diary)} entries) to {diary_path}")

print(f"\nAll destination diaries saved to {output_dir}")
