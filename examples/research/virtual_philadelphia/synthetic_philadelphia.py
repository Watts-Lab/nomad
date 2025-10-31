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
#
# It also identifies which rotation best aligns a random sample of streets with a N-S, E-W grid. 

# %%
from pathlib import Path
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

import nomad.map_utils as nm
from nomad.city_gen import RasterCityGenerator


# %% [markdown]
# ## 1. Download and Persist Philadelphia OSM Data (Web Mercator)
#
# This section downloads buildings and streets for Philadelphia in Web Mercator
# using chunked OSM queries, classifies buildings with the Garden City schema,
# removes overlaps, and stores the results in a GeoPackage for downstream use.


# %%
CITY_NAME = "Philadelphia, Pennsylvania, USA"
OUTPUT_DIR = Path(".")  # Relative path for Jupyter compatibility
RAW_GPKG_PATH = OUTPUT_DIR / "philadelphia_osm_raw.gpkg"

LOAD_FROM_CACHE = RAW_GPKG_PATH.exists()

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
category_counts = buildings["garden_city_category"].value_counts().to_dict()
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
        return df['category'].value_counts().to_dict() if 'category' in df.columns else {}

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
block_size = 15.0

gen_streets_only = RasterCityGenerator(
    boundary_polygon=boundary_geom,
    streets_gdf=rot_streets,
    buildings_gdf=rot_buildings,
    block_size=block_size,
)

import time as _t
_t0 = _t.time()
city_streets = gen_streets_only.generate_city()
streets_only_elapsed = _t.time() - _t0
print(f"Streets-only rasterization: {streets_only_elapsed:.2f}s; streets={len(city_streets.streets_gdf):,}, blocks={len(city_streets.blocks_gdf):,}")

# Graph build timing (includes shortcut network)
_t1 = _t.time()
G = city_streets.get_street_graph(lazy=True)
graph_elapsed = _t.time() - _t1
print(f"Street graph build (lazy + shortcuts): {graph_elapsed:.2f}s; nodes={len(G.nodes):,}, edges={len(G.edges):,}")

# %% [markdown]
# ## 4. Rasterize Rotated City (Scaled)
#
# Using the rotated layers, rasterize the full city at 15m block size and persist
# the resulting `City` layers. This is the next step toward a complete pipeline.

# %%
print("\nRasterizing rotated city at 15m (buildings + streets)...")
rot_buildings = gpd.read_file(RAW_GPKG_PATH, layer="buildings_rotated")

boundary_geom = rot_boundary.geometry.iloc[0]
block_size = 15.0

gen = RasterCityGenerator(
    boundary_polygon=boundary_geom,
    streets_gdf=rot_streets,
    buildings_gdf=rot_buildings,
    block_size=block_size,
)

import time as _t
_t0 = _t.time()
city = gen.generate_city()
elapsed = _t.time() - _t0
print(f"Rasterization completed in {elapsed:.1f}s: blocks={len(city.blocks_gdf):,}, streets={len(city.streets_gdf):,}, buildings={len(city.buildings_gdf):,}")

RASTER_GPKG = OUTPUT_DIR / "philadelphia_rasterized.gpkg"
if RASTER_GPKG.exists():
    RASTER_GPKG.unlink()
city.save_geopackage(str(RASTER_GPKG), persist_blocks=True, persist_city_properties=True)
print(f"Saved rasterized layers to {RASTER_GPKG}")
