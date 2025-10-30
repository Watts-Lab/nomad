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


# %% [markdown]
# ## 1. Download and Persist Philadelphia OSM Data (Web Mercator)
#
# This section downloads buildings and streets for Philadelphia in Web Mercator
# using chunked OSM queries, classifies buildings with the Garden City schema,
# removes overlaps, and stores the results in a GeoPackage for downstream use.


# %%
CITY_NAME = "Philadelphia, Pennsylvania, USA"
OUTPUT_DIR = Path(__file__).resolve().parent
RAW_GPKG_PATH = OUTPUT_DIR / "philadelphia_osm_raw.gpkg"

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

cache_mode = "persistent"

print("Downloading buildings (single query, Web Mercator)...")
start_time = time.time()
buildings = nm.download_osm_buildings(
    boundary,
    crs="EPSG:3857",
    schema="garden_city",
    clip=True,
    infer_building_types=True,
    explode=True,
    by_chunks=False,
    cache_mode=cache_mode,
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
    by_chunks=False,
    cache_mode=cache_mode,
)
elapsed = time.time() - start_time
print(f"Downloaded {len(streets):,} streets in {elapsed:.1f}s")

streets = streets.reset_index(drop=True)

print(f"Downloaded {len(buildings):,} building polygons and {len(streets):,} street segments.")

RAW_GPKG_PATH.parent.mkdir(parents=True, exist_ok=True)
if RAW_GPKG_PATH.exists():
    RAW_GPKG_PATH.unlink()

print(f"Persisting raw data layers to {RAW_GPKG_PATH}...")
buildings.to_file(RAW_GPKG_PATH, layer="buildings", driver="GPKG")
streets.to_file(RAW_GPKG_PATH, layer="streets", driver="GPKG", mode="a")
boundary_gdf.to_file(RAW_GPKG_PATH, layer="city_boundary", driver="GPKG", mode="a")


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


# %%
print("Generating diagnostic plots...")

def _plot_city(ax, b_gdf, s_gdf, title):
    if len(b_gdf) > 0:
        color_map = {
            "park": "#74c476",
            "residential": "#6baed6",
            "retail": "#fd8d3c",
            "workplace": "#9e9ac8",
            "other": "#969696",
        }
        column_name = "garden_city_category" if "garden_city_category" in b_gdf.columns else None
        if column_name:
            for cat, color in color_map.items():
                subset = b_gdf[b_gdf[column_name] == cat]
                if len(subset) > 0:
                    subset.plot(ax=ax, color=color, linewidth=0, alpha=0.6)
            others = b_gdf[~b_gdf[column_name].isin(color_map.keys())]
            if len(others) > 0:
                others.plot(ax=ax, color="#bdbdbd", linewidth=0, alpha=0.5)
        else:
            b_gdf.plot(ax=ax, color="#9ecae1", linewidth=0, alpha=0.6)

    if len(s_gdf) > 0:
        s_gdf.plot(ax=ax, color="#000000", linewidth=0.3, alpha=0.6)

    ax.set_title(title)
    ax.set_axis_off()
    try:
        ctx.add_basemap(ax, crs=b_gdf.crs if len(b_gdf) > 0 else s_gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik, attribution="")
    except Exception:
        pass


fig, axes = plt.subplots(1, 2, figsize=(18, 9))

_plot_city(axes[0], buildings.sample(n=min(20000, len(buildings))) if len(buildings) > 20000 else buildings, streets, "Original (Web Mercator)")
_plot_city(axes[1], rotated_buildings.sample(n=min(20000, len(rotated_buildings))) if len(rotated_buildings) > 20000 else rotated_buildings, rotated_streets, f"Rotated ({rotation_deg:.2f}°)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "philadelphia_rotation_diagnostics.png", dpi=200, bbox_inches="tight")
plt.show()
