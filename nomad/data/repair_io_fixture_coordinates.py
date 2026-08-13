"""Regenerate packaged IO fixtures from aligned IC2S2 mobility data.

Usage:
  python -m nomad.data.repair_io_fixture_coordinates
  python -m nomad.data.repair_io_fixture_coordinates --commit
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import nomad.data as data_folder
from nomad.io import base as loader


DATA_DIR = Path(data_folder.__file__).parent
REPO_DIR = DATA_DIR.parent.parent
SOURCE_CSV = REPO_DIR / "tutorials" / "IC2S2-2025" / "IC2S2-DATA" / "gc_data.csv"
STAGING_DIR = REPO_DIR / "cache" / "corrected_io_fixtures"
BACKUP_DIR = REPO_DIR / "cache" / "io_fixture_backups"
PARTITIONED_CSV_COLS = {
    "user_id": "user_id",
    "latitude": "dev_lat",
    "longitude": "dev_lon",
    "datetime": "local_datetime",
}
CANONICAL_COLS = {
    "user_id": "uid",
    "timestamp": "timestamp",
    "latitude": "latitude",
    "longitude": "longitude",
    "date": "date",
}
CANONICAL_FILE_COLS = {
    "user_id": "uid",
    "timestamp": "timestamp",
    "latitude": "latitude",
    "longitude": "longitude",
}

def load_source():
    """Load the aligned IC2S2 source table."""
    return pd.read_csv(SOURCE_CSV)


def partitioned_csv_table(df):
    """Return the staged partitioned CSV table with target column names."""
    return pd.DataFrame({
        "user_id": df["identifier"],
        "dev_lat": df["device_lat"],
        "dev_lon": df["device_lon"],
        "local_datetime": df["local_datetime"],
        "date": df["date"],
    })


def canonical_table(df):
    """Return the canonical fixture table used by parquet and single-file outputs."""
    return pd.DataFrame({
        "uid": df["identifier"],
        "timestamp": df["unix_timestamp"].astype("int64"),
        "latitude": df["device_lat"],
        "longitude": df["device_lon"],
        "date": df["date"],
    })


def gc_sample_table(df):
    """Return gc_sample.csv with target column names."""
    local_datetime = pd.to_datetime(df["local_datetime"])
    tz_offset = local_datetime.map(lambda x: int(x.utcoffset().total_seconds()))
    return pd.DataFrame({
        "uid": df["identifier"],
        "timestamp": df["unix_timestamp"].astype("int64"),
        "tz_offset": tz_offset,
        "longitude": df["device_lon"],
        "latitude": df["device_lat"],
    })


def reset_dir(path):
    """Remove and recreate a staging directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def strip_hive_prefix(path):
    """Rename date=YYYY-MM-DD partition folders to YYYY-MM-DD for the CSV fixture."""
    for folder in path.glob("date=*"):
        folder.rename(path / folder.name.split("=", 1)[1])


def write_outputs(partitioned_csv, canonical, staging_dir):
    """Write all fixture shapes through nomad.io."""
    reset_dir(staging_dir)

    normalized_csv = loader.from_df(partitioned_csv, parse_dates=False, traj_cols=PARTITIONED_CSV_COLS)
    csv_dir = staging_dir / "partitioned_csv"
    csv_dir.mkdir()
    loader.to_file(
        normalized_csv,
        csv_dir,
        format="csv",
        traj_cols=PARTITIONED_CSV_COLS,
        partition_by=["date"],
        existing_data_behavior="delete_matching",
    )
    strip_hive_prefix(csv_dir)

    normalized = loader.from_df(canonical, traj_cols=CANONICAL_COLS)
    loader.to_file(
        normalized,
        staging_dir / "partitioned_parquet",
        format="parquet",
        traj_cols=CANONICAL_COLS,
        partition_by=["date"],
        existing_data_behavior="delete_matching",
    )
    loader.to_file(
        normalized.drop(columns=["date"]),
        staging_dir / "single_parquet",
        format="parquet",
        traj_cols=CANONICAL_FILE_COLS,
        existing_data_behavior="delete_matching",
    )

    single_csv_dir = staging_dir / "single_csv"
    single_csv_dir.mkdir()
    loader.to_file(
        normalized.drop(columns=["date"]),
        single_csv_dir / "sample2.csv",
        format="csv",
        traj_cols=CANONICAL_FILE_COLS,
    )

    gc_sample = gc_sample_table(load_source())
    loader.to_file(
        gc_sample,
        staging_dir / "gc_sample.csv",
        format="csv",
        traj_cols=CANONICAL_FILE_COLS,
    )


def normalize_for_compare(df, source):
    """Return common columns for equality checks across fixture shapes."""
    if source == "partitioned_csv":
        out = df.rename(columns={
            "user_id": "uid",
            "dev_lat": "latitude",
            "dev_lon": "longitude",
            "local_datetime": "datetime",
        })
        out["timestamp"] = pd.to_datetime(out["datetime"]).astype("int64") // 10**9
        if "tz_offset" in out.columns:
            out["timestamp"] = out["timestamp"] - out["tz_offset"].astype("int64")
    else:
        out = df.copy()
    cols = ["uid", "timestamp", "latitude", "longitude"]
    return out[cols].sort_values(cols).reset_index(drop=True)


def city_smoke_report(df):
    """Return containment diagnostics against the Garden City GeoJSON extent."""
    city = gpd.read_file(DATA_DIR / "garden-city-buildings.geojson").to_crs("EPSG:3857")
    city_props = gpd.read_file(DATA_DIR / "garden-city.gpkg", layer="city_properties").iloc[0]
    points = gpd.GeoSeries(
        gpd.points_from_xy(df["dev_lon"], df["dev_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    city_bounds = city.total_bounds
    point_bounds = points.total_bounds
    inside_building_bounds = (
        points.x.between(city_bounds[0], city_bounds[2])
        & points.y.between(city_bounds[1], city_bounds[3])
    )
    minx, miny, maxx, maxy = city_props.geometry.bounds
    block_size = city_props["block_side_length"]
    full_bounds = [
        city_props["web_mercator_origin_x"] + minx * block_size,
        city_props["web_mercator_origin_y"] + miny * block_size,
        city_props["web_mercator_origin_x"] + maxx * block_size,
        city_props["web_mercator_origin_y"] + maxy * block_size,
    ]
    inside_full_bounds = (
        points.x.between(full_bounds[0], full_bounds[2])
        & points.y.between(full_bounds[1], full_bounds[3])
    )
    return {
        "point_bounds": point_bounds,
        "city_bounds": city_bounds,
        "full_bounds": full_bounds,
        "inside_building_bounds": inside_building_bounds.mean(),
        "inside_full_bounds": inside_full_bounds.mean(),
    }


def validate_outputs(partitioned_csv, canonical, staging_dir):
    """Validate staged outputs before migration."""
    smoke = city_smoke_report(partitioned_csv)

    expected = normalize_for_compare(canonical.drop(columns=["date"]), "canonical")
    staged_csv = loader.from_file(
        staging_dir / "partitioned_csv",
        format="csv",
        traj_cols=PARTITIONED_CSV_COLS,
    )
    staged_parquet = loader.from_file(
        staging_dir / "partitioned_parquet",
        format="parquet",
        traj_cols=CANONICAL_COLS,
    )
    staged_single_csv = loader.from_file(
        staging_dir / "single_csv" / "sample2.csv",
        format="csv",
        traj_cols=CANONICAL_FILE_COLS,
    )
    staged_single_parquet = loader.from_file(
        staging_dir / "single_parquet",
        format="parquet",
        traj_cols=CANONICAL_FILE_COLS,
    )
    staged_gc_sample = loader.from_file(
        staging_dir / "gc_sample.csv",
        format="csv",
        traj_cols=CANONICAL_FILE_COLS,
    )

    checks = [
        normalize_for_compare(staged_csv, "partitioned_csv"),
        normalize_for_compare(staged_parquet, "canonical"),
        normalize_for_compare(staged_single_csv, "canonical"),
        normalize_for_compare(staged_single_parquet, "canonical"),
    ]
    for check in checks:
        pd.testing.assert_frame_equal(check, expected, check_dtype=False, rtol=0, atol=1e-12)

    gc_sample_expected = normalize_for_compare(gc_sample_table(load_source()), "canonical")
    pd.testing.assert_frame_equal(
        normalize_for_compare(staged_gc_sample, "canonical"),
        gc_sample_expected,
        check_dtype=False,
        rtol=0,
        atol=1e-12,
    )
    validate_schemas(staging_dir)
    if smoke["inside_building_bounds"] < 0.95 or smoke["inside_full_bounds"] < 0.99:
        raise AssertionError("Staged points do not align with the Garden City extents.")

    print(f"Rows: {len(expected):,}")
    print(f"Users: {expected['uid'].nunique():,}")
    print(f"Dates: {canonical['date'].min()} to {canonical['date'].max()}")
    print(f"Point bounds: {smoke['point_bounds']}")
    print(f"Garden City bounds: {smoke['city_bounds']}")
    print(f"Full city bounds: {smoke['full_bounds']}")
    print(f"Share inside building bounds: {smoke['inside_building_bounds']:.3f}")
    print(f"Share inside full city bounds: {smoke['inside_full_bounds']:.3f}")


def validate_schemas(staging_dir):
    """Validate staged fixture schemas and absence of accidental index columns."""
    schemas = {
        "partitioned_csv": loader.table_columns(staging_dir / "partitioned_csv", format="csv").tolist(),
        "partitioned_parquet": loader.table_columns(staging_dir / "partitioned_parquet", format="parquet").tolist(),
        "single_csv": loader.table_columns(staging_dir / "single_csv" / "sample2.csv", format="csv").tolist(),
        "single_parquet": loader.table_columns(staging_dir / "single_parquet", format="parquet").tolist(),
        "gc_sample": loader.table_columns(staging_dir / "gc_sample.csv", format="csv").tolist(),
    }
    expected = {
        "partitioned_csv": ["user_id", "dev_lat", "dev_lon", "local_datetime"],
        "partitioned_parquet": ["uid", "timestamp", "latitude", "longitude", "date"],
        "single_csv": ["uid", "timestamp", "latitude", "longitude"],
        "single_parquet": ["uid", "timestamp", "latitude", "longitude"],
        "gc_sample": ["uid", "timestamp", "tz_offset", "longitude", "latitude"],
    }
    if schemas != expected:
        raise AssertionError(f"Unexpected staged schemas: {schemas}")
    if any("__index_level_0__" in schema for schema in schemas.values()):
        raise AssertionError("Unexpected __index_level_0__ column in staged fixtures.")


def backup_and_commit(staging_dir):
    """Backup current fixtures and replace them with staged outputs."""
    backup_dir = BACKUP_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir.mkdir(parents=True)
    targets = [
        "partitioned_csv",
        "partitioned_parquet",
        "single_csv",
        "single_parquet",
        "gc_sample.csv",
    ]
    for name in targets:
        src = DATA_DIR / name
        backup = backup_dir / name
        if src.is_dir():
            shutil.copytree(src, backup)
            shutil.rmtree(src)
            shutil.copytree(staging_dir / name, src)
        else:
            shutil.copy2(src, backup)
            shutil.copy2(staging_dir / name, src)
    print(f"Backup written to {backup_dir}")
    print(f"Corrected fixtures copied to {DATA_DIR}")


def main():
    """Build staged fixture outputs and optionally migrate them into nomad/data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()

    source = load_source()
    partitioned_csv = partitioned_csv_table(source)
    canonical = canonical_table(source)
    write_outputs(partitioned_csv, canonical, STAGING_DIR)
    validate_outputs(partitioned_csv, canonical, STAGING_DIR)
    print(f"Staged corrected fixtures in {STAGING_DIR}")

    if args.commit:
        backup_and_commit(STAGING_DIR)
    else:
        print("Run again with --commit to backup and replace nomad/data fixtures.")


if __name__ == "__main__":
    main()
