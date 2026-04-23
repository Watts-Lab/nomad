# AI Copilot Instructions for NOMAD Codebase

## Project Overview
NOMAD is a production-ready Python library for large-scale GPS trajectory analysis. It provides end-to-end processing: data ingestion, filtering, stop detection, mobility metrics, and synthetic trajectory generation. The library supports both local (pandas/GeoPandas) and distributed (PySpark) execution with identical APIs.

## Architecture & Data Flow

### Core Module Structure
- **`nomad.io.base`** (base.py, ~1500 lines): Primary data loader. Handles partitioned datasets (CSV, Parquet), automatic type casting, and column mapping via `traj_cols` dict. See `from_file()`, `sample_from_file()`.
- **`nomad.stop_detection`**: 5+ algorithms (Lachesis, SeqScan, HDBSCAN, TA-DBSCAN, Grid-Based) with consistent interface. Each has `*_labels()` (per-ping classification) and `*()` (full stop table output).
- **`nomad.filters` / `nomad.filters_spark`**: Parallel implementations (pandas/PySpark) for tessellation (H3, S2), projections, completeness metrics, and spatial filtering.
- **`nomad.constants`**: `DEFAULT_SCHEMA` dict maps canonical column names ("timestamp", "latitude", etc.) to actual DataFrame columns.

### Critical Data Pattern: `traj_cols` Dictionary
Almost all functions accept `traj_cols` parameter to map data columns to canonical names:
```python
traj_cols = {
    "user_id": "gc_identifier",       # actual column name
    "x": "dev_x",                     # or lat/lon
    "y": "dev_y",
    "timestamp": "unix_ts"            # or "datetime"
}
```
Use `loader._parse_traj_cols(df.columns, traj_cols, kwargs)` to normalize, and `loader._has_spatial_cols()` / `loader._has_time_cols()` to validate.

### Common Detection Pattern
All stop detection algorithms follow:
1. `algo.lachesis_labels()` → returns Series of cluster IDs per ping (-1 = noise)
2. `algo.lachesis()` with `complete_output=True` → returns DataFrame with columns: `timestamp`, `end_timestamp`, `cluster`, `duration`, `n_pings`, `diameter`, etc.
3. Preprocessing/postprocessing functions in `preprocessing.py` for smoothing/filtering.

## Key Files to Understand First
1. **`nomad/io/base.py`** (lines 1-200): Understand `from_file()`, `_parse_traj_cols()`, `_fallback_spatial_cols()`
2. **`nomad/constants.py`**: `DEFAULT_SCHEMA`, `FILTER_OPERATORS`, trajectory parameter sets
3. **`nomad/stop_detection/lachesis.py`** (lines 1-100): Cleanest stop detection implementation; shows `traj_cols` pattern, metric selection (haversine vs euclidean)
4. **`nomad/stop_detection/utils.py`**: `_diameter()`, `_fallback_st_cols()` — used by all algorithms
5. **`nomad/tests/test_stop_detection.py`** (lines 1-60): Pytest fixtures and algorithm registry pattern

## Development Workflows

### Running Tests
```bash
cd /Users/anoushkamenon/Desktop/nomad
pytest nomad/tests/test_stop_detection.py -v
pytest nomad/tests/test_filter.py -v
```
Tests use fixtures for shared test trajectories and parameter sets. Check `@pytest.fixture` decorators in test files.

### Debugging Trajectories
Notebook: `nomad/tests/test_anim.ipynb` (animation visualization for all 5 stop detection algorithms)
- Load data via `loader.sample_from_file()` with `traj_cols` mapping
- Run algorithm with `complete_output=True`
- Use `animate_stop_dashboard()` from `nomad.stop_detection.viz` for visual validation

## Project-Specific Patterns & Conventions

### 1. Spatial Coordinates: Auto-Detection with Fallback
Functions attempt to auto-detect `x/y` vs `latitude/longitude` but require explicit `traj_cols`:
```python
# In nomad.stop_detection.utils._fallback_st_cols():
# Returns: t_key, coord_key1, coord_key2, use_datetime, use_lon_lat
# Decides metric: 'haversine' if lon/lat, else 'euclidean'
```
Always validate with `loader._has_spatial_cols(df.columns, traj_cols)`.

### 2. Timestamp Handling: datetime vs Unix
Functions support both `datetime` columns (pd.Timestamp) and `timestamp` (Unix seconds).
- `loader._fallback_time_cols_dt()` detects type and returns `use_datetime` flag
- Use `nomad.filters.to_timestamp()` for parsing if needed
- Lachesis example: line 50 shows conditional time series parsing

### 3. Stop Algorithm Interface (Consistent Across All)
Every stop detection has same signature:
```python
stops = ALGORITHM.lachesis(
    traj,
    delta_roam=30,           # spatial radius
    dt_max=60,               # temporal gap (minutes)
    dur_min=5,               # minimum duration
    complete_output=True,    # True = DataFrame with all cols, False = minimal
    keep_col_names=True,     # preserve user column names
    traj_cols=traj_cols
)
```
Output structure (stops table):
- `timestamp`, `end_timestamp` (or `datetime`/`end_datetime`)
- `cluster` (integer ID, -1 not used for stops)
- `duration` (minutes), `n_pings`, `diameter`, `max_gap`

### 4. Parallel Python/Spark Implementations
Some modules have dual implementations (`filters.py` + `filters_spark.py`). Key difference:
- **Pandas**: Direct DataFrame operations, returns pd.DataFrame
- **Spark**: Takes `spark_session` kwarg, returns Spark DataFrame
- **Same API**: Client code unchanged, just pass `spark_session=spark` to enable Spark

Example: `to_projection()` appears in both modules with identical signatures.

### 5. Preprocessing & Postprocessing
Stop detection results often need cleanup:
- `nomad.stop_detection.preprocessing`: Handle duplicates, sort, validate temporal order
- `nomad.stop_detection.postprocessing`: Merge adjacent stops, clip to time windows (see `clip_stops_datetime()` in utils.py)
- Example: test_anim.ipynb lines 100-130 show renaming `unix_ts` → `timestamp` and assigning pings to stops

### 6. Column Naming Conventions
- Always use `traj_cols` dict to map; never hardcode column names
- Standard keys: `"user_id"`, `"x"`, `"y"`, `"latitude"`, `"longitude"`, `"timestamp"`, `"datetime"`, `"duration"`, `"h3_cell"`, `"location_id"`
- See `nomad.constants.DEFAULT_SCHEMA` for full list

## Testing Patterns
- **Unit tests**: `nomad/tests/test_*.py` use pytest with parametrized fixtures
- **Shared algorithm registry** (test_stop_detection.py): Registers all algorithms with `label_fn`, `stop_fn`, `extra_kwargs`
- **Test data**: `examples/gc_data_long/` (partitioned parquet); `nomad/data/` (smaller static data for fast tests)
- **Assertions**: Use `pandas.testing.assert_frame_equal()` for DataFrame comparisons

## Integration Points
- **Data sources**: CSV, Parquet, GeoJSON, partitioned directories (handled by `nomad.io`)
- **External deps**: geopandas, shapely, h3, osmnx, networkx, scipy, pyspark
- **Visualization**: `nomad.stop_detection.viz.animate_stop_dashboard()` generates GIFs; `matplotlib` for static plots

## Common Gotchas
1. **Missing `traj_cols`**: Function will attempt auto-detection; always pass explicitly to avoid surprises
2. **Timezone-aware timestamps**: `clip_stops_datetime()` requires consistent tz handling; see lines 50-60 for pattern
3. **Column renaming before analysis**: test_anim.ipynb shows renaming `unix_ts` → `timestamp` because animation function expects canonical names
4. **Metric selection**: haversine for lon/lat (kilometers), euclidean for projected x/y
5. **Empty DataFrames**: Algorithms return empty Series/DataFrame gracefully; check with `if data.empty:`
