import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import nomad.filters as filters
import nomad.io.base as loader
from nomad.visit_attribution.visit_attribution import point_in_polygon

REPO_ROOT = Path(__file__).resolve().parents[2]
PHILLY_OUTPUT = REPO_ROOT / "examples" / "research" / "virtual_philadelphia" / "output"
SICSS_DATA = REPO_ROOT / "tutorials" / "SICSS_2026" / "data"
OUT_DIR = SICSS_DATA / "philly"
CBG_PATH = REPO_ROOT / "tutorials" / "IC2S2-2025" / "Census_Block_Groups_2010.geojson"


# Convert NOMAD location ids into readable activity labels.
def activity_type_from_location(location):
    """Map NOMAD synthetic building ids to the activity labels used in SICSS."""
    if pd.isna(location):
        return "Travel"

    # NOMAD location ids are prefixed by land-use type, e.g. h-x343-y121.
    prefix = str(location).split("-", 1)[0]
    return {
        "h": "Home",
        "w": "Work",
        "r": "Retail",
        "p": "Park",
    }.get(prefix, "Other")


# Load the generated synthetic Philly travel diaries.
def load_travel_diaries():
    """Load the already-generated synthetic Philly activity diary parquet files."""
    return loader.from_file(
        PHILLY_OUTPUT / "travel_diaries",
        format="parquet",
        traj_cols={
            "user_id": "user_id",
            "datetime": "datetime",
            "timestamp": "timestamp",
            "x": "x",
            "y": "y",
            "duration": "duration",
        },
    )


def load_agent_homes():
    """Load each synthetic agent's assigned home building.

    Returns
    -------
    pandas.DataFrame
        One row per agent with ``user_id`` and ``home`` columns.
    """
    return pd.read_parquet(
        PHILLY_OUTPUT / "homes_large",
        columns=["user_id", "home"],
    )


def load_philly_areas():
    """Load Philadelphia Census Block Groups with tutorial-ready ids."""
    areas = gpd.read_file(CBG_PATH)
    areas = areas.rename(columns={"GEOID10": "area_id", "NAMELSAD10": "area_name"})

    # Prefix the numeric GEOID so CSV readers do not coerce it to an integer.
    areas["area_id"] = "BG" + areas["area_id"].astype(str)
    return areas


# MAIN FUNCTION: Create the tutorial-ready Philly activity and geography files.
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Start from the generated synthetic travel diary.
    raw = load_travel_diaries()
    agent_homes = load_agent_homes().rename(columns={"user_id": "userid"})

    # Rows with null x/y are travel intervals between activities, and rows with
    # zero duration are instantaneous diary transitions rather than true stops.
    # This tutorial operates on observed stops/activities, so keep only located
    # rows with positive duration.
    raw = raw.dropna(subset=["x", "y"]).copy()
    raw = raw[raw["duration"] > 0].copy()

    # Convert NOMAD output into SICSS activity columns (with start and stop)
    raw["activity_type"] = raw["location"].apply(activity_type_from_location)
    raw["start_time"] = pd.to_datetime(raw["datetime"])
    raw["end_time"] = raw["start_time"] + pd.to_timedelta(raw["duration"], unit="m")

    # Synthetic Philly coordinates are Web Mercator meters. Keeping EPSG:3857
    # preserves meter-based distance calculations used throughout the tutorial.
    gdf = gpd.GeoDataFrame(
        raw,
        geometry=gpd.points_from_xy(raw["x"], raw["y"]),
        crs="EPSG:3857",
    )
    gdf = gdf.rename(columns={"user_id": "userid", "x": "o_lon", "y": "o_lat"})
    gdf = gdf.sort_values(["userid", "start_time"]).reset_index(drop=True)

    xy_cols = {"x": "o_lon", "y": "o_lat"}
    gdf["o_h7"] = filters.to_tessellation(
        gdf, "h3", 7, data_crs=gdf.crs, traj_cols=xy_cols
    )
    gdf["o_h8"] = filters.to_tessellation(
        gdf, "h3", 8, data_crs=gdf.crs, traj_cols=xy_cols
    )
    gdf["o_h9"] = filters.to_tessellation(
        gdf, "h3", 9, data_crs=gdf.crs, traj_cols=xy_cols
    )

    # Add activity-area ids and assign home areas from the simulator's home building.
    areas = load_philly_areas()
    area_polygons = areas[["area_id", "geometry"]]
    gdf["loc_area"] = point_in_polygon(
        gdf,
        poi_table=area_polygons,
        data_crs=gdf.crs,
        location_id="area_id",
        traj_cols=xy_cols,
    )
    missing_area = gdf["loc_area"].isna()
    if missing_area.any():
        gdf.loc[missing_area, "loc_area"] = point_in_polygon(
            gdf.loc[missing_area],
            poi_table=area_polygons,
            data_crs=gdf.crs,
            location_id="area_id",
            max_distance=500,
            traj_cols=xy_cols,
        )
    gdf["home"] = gdf["userid"].map(agent_homes.set_index("userid")["home"])
    home_area = (
        gdf[gdf["location"] == gdf["home"]]
        .drop_duplicates("userid")
        .set_index("userid")["loc_area"]
    )
    gdf["home_area"] = gdf["userid"].map(home_area)
    gdf["home_activity"] = np.where(gdf["location"] == gdf["home"], "Y", "N")

    keep_cols = [
        "userid",
        "start_time",
        "end_time",
        "duration",
        "loc_area",
        "home_area",
        "activity_type",
        "geometry",
        "o_lat",
        "o_lon",
        "o_h7",
        "o_h8",
        "o_h9",
        "home_activity",
        "location",
        "timestamp",
    ]
    gdf = gdf[keep_cols]

    # Persist the tutorial-ready snapshot. These are the files the SICSS notebook should read.
    #
    # MAKE SURE TO UPLOAD FILES TO GOOGLE DRIVE OR TUTORIAL WILL NOT WORK.
    #
    gdf.to_file(OUT_DIR / "df_samp.geojson", driver="GeoJSON")
    stop_table = gdf.drop(columns="geometry")
    stop_table.to_parquet(OUT_DIR / "df_samp.parquet", index=False)
    stop_table.to_csv(OUT_DIR / "df_samp.csv", index=False)
    areas.to_file(OUT_DIR / "philly_areas.geojson", driver="GeoJSON")

    # Keep the synthetic-run config next to the prepared data for provenance.
    for name in ["config_large.json", "rotation_metadata_large.json"]:
        src = PHILLY_OUTPUT / name
        if src.exists():
            shutil.copy2(src, OUT_DIR / name)

    # Machine-readable provenance and counts for quick sanity checks.
    manifest = {
        "source": str(PHILLY_OUTPUT.relative_to(REPO_ROOT)),
        "outputs": {
            "df_samp_rows": int(len(gdf)),
            "users": int(gdf["userid"].nunique()),
            "areas": int(areas["area_id"].nunique()),
            "home_areas": int(gdf["home_area"].nunique()),
            "hexes_resolution_9": int(gdf["o_h9"].nunique()),
            "start_time_min": str(gdf["start_time"].min()),
            "start_time_max": str(gdf["start_time"].max()),
        },
        "notes": [
            "df_samp contains non-travel diary rows with non-null coordinates.",
            "Coordinates and geometry are EPSG:3857 for meter-based distance calculations.",
            "H3 columns were computed after transforming points to EPSG:4326.",
            "loc_area/home_area use Philadelphia 2010 Census Block Group GEOID10 values with a BG prefix.",
            "home_area is assigned from each synthetic agent's designated home building.",
            "home_activity is Y only at the agent's designated home building.",
            "Mobility metrics are intentionally calculated in the tutorial notebook, not precomputed here.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Human-readable notes for future users of the prepared data folder.
    readme = f"""# Prepared Synthetic Philadelphia Data for SICSS 2026

Generated by `tutorials/SICSS_2026/prep_philly_tutorial_data.py` from
`examples/research/virtual_philadelphia/output`.

## Files

- `df_samp.parquet`: activity-level stop table used by the SICSS 2026 tutorial.
- `df_samp.csv` / `df_samp.geojson`: CSV and GeoJSON versions of the same stop table.
- `philly_areas.geojson`: Philadelphia Census Block Groups with `area_id` and `area_name` columns.
- `config_large.json` and `rotation_metadata_large.json`: copied from the synthetic Philly run.
- `manifest.json`: row counts, date range, and notes.

## Key Columns in df_samp

- `userid`: synthetic agent id.
- `start_time`, `end_time`, `duration`: activity timing, with duration in minutes.
- `activity_type`: building type inferred from the NOMAD location id prefix (`Home`, `Work`, `Retail`, `Park`); `Home` denotes a residential building, not necessarily the agent's assigned home.
- `home_activity`: `Y` only when the activity occurs at the agent's assigned home building.
- `o_lon`, `o_lat`, `geometry`: EPSG:3857 coordinates for meter-based distance calculations.
- `o_h7`, `o_h8`, `o_h9`: H3 cells computed from WGS84 coordinates.
- `loc_area`: Philadelphia Census Block Group containing the activity.
- `home_area`: Philadelphia Census Block Group containing the agent's assigned home building.

## Current Counts

- Rows in `df_samp`: {len(gdf):,}
- Users: {gdf["userid"].nunique():,}
- Date range: {gdf["start_time"].min()} to {gdf["start_time"].max()}
- Home areas: {gdf["home_area"].nunique():,}

Mobility metrics such as radius of gyration, self-containment, and social interaction
potential are calculated in the tutorial notebook from these prepared activity files.
"""
    (OUT_DIR / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
