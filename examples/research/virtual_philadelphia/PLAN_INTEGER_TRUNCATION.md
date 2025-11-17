# Plan: Integer Truncation Optimization for Point-in-Block Checks

## Architecture Understanding
- **RasterCity**: All building and street geometries are rasterized to blocks with integer coordinates
- **blocks_gdf**: MultiIndex (coord_x, coord_y), has `building_id` column mapping blocks → buildings  
- **buildings_gdf**: Has geometry, but missing `blocks` column (list of block tuples)
- **All geometry checks can use integer truncation instead of shapely operations**

## What Gets Optimized
**Building checks (201,600 + 27,000 calls):**
- Lines 363-364: Check if agent in current/previous destination building
- Line 382: Check if sampled point inside current destination building

**Bound polygon checks (43,600 calls):**
- Line 410: Validation check if agent in bound_poly  
- Line 451: Check if sampled path point inside bound_poly
- bound_poly = unary_union([start_geom, street_geom]) - ALL blocks!

**Total: ~272,000 shapely operations → integer checks**

## Implementation Steps

### 1. Add 'blocks' column during rasterization
**File:** `nomad/city_gen.py`
**Location:** Line 2272 in `_rasterize()` method

```python
new_building_rows.append({
    'id': building_id,
    'building_type': building_type,
    'door_cell_x': door[0],
    'door_cell_y': door[1],
    'door_point': door_centroid,
    'size': len(building_blocks),
    'blocks': building_blocks,  # ADD: list of (x,y) tuples
    'geometry': building_geom,
})
```

### 2. Populate 'blocks' column on load (if missing)
**File:** `nomad/city_gen.py`
**Location:** In `from_geodataframes()` after line 1164 (after buildings_gdf is set)

```python
# Populate blocks column from blocks_gdf if missing
if 'blocks' not in self.buildings_gdf.columns and hasattr(self, 'blocks_gdf') and not self.blocks_gdf.empty:
    building_blocks_map = {}
    for building_id in self.blocks_gdf['building_id'].dropna().unique():
        mask = self.blocks_gdf['building_id'] == building_id
        building_blocks_map[building_id] = self.blocks_gdf[mask].index.tolist()
    self.buildings_gdf['blocks'] = self.buildings_gdf['id'].map(building_blocks_map)
```

**Why here:** from_geodataframes is called by from_geopackage (line 1300), ensures blocks populated for all load paths

### 3. Add helper function for integer truncation check
**File:** `nomad/traj_gen.py`
**Location:** After line 1092, before `_cartesian_coords`

```python
def _point_in_blocks(point_arr, blocks_set):
    """Check if point is in any block using integer truncation."""
    if blocks_set is None:
        return False
    block_idx = (int(np.floor(point_arr[0])), int(np.floor(point_arr[1])))
    return block_idx in blocks_set
```

### 4. Convert blocks to set after .to_dict()
**File:** `nomad/traj_gen.py`
**Location:** Lines 481, 487, 496 in `_traj_from_dest_diary()`

Replace all 3 occurrences:
```python
# OLD:
self._current_dest_building_row = city.buildings_gdf.loc[building_id].to_dict()

# NEW:
building_dict = city.buildings_gdf.loc[building_id].to_dict()
building_dict['blocks_set'] = set(building_dict.get('blocks', []))
self._current_dest_building_row = building_dict
```

**One line after .to_dict()** - no auxiliary functions

### 5. Cache bound_poly blocks alongside bound_poly
**File:** `nomad/traj_gen.py`
**Location:** In `_sample_step()` method

**Initialize cache (line 286 in reset_trajectory):**
```python
self._cached_bound_poly_blocks_set = None  # ADD
```

**Use cache (line 406-408):**
```python
if use_cache:
    path_ml = self._cached_path_ml
    bound_poly = self._cached_bound_poly
    bound_poly_blocks_set = self._cached_bound_poly_blocks_set  # ADD
```

**Build and cache (after line 427):**
```python
bound_poly = unary_union([start_geom, street_geom])

# Build blocks set for bound_poly
if in_previous_dest and self._previous_dest_building_row is not None:
    start_blocks = self._previous_dest_building_row.get('blocks_set', set())
else:
    start_blocks = {start_block}
bound_poly_blocks_set = start_blocks | set(street_path)  # ADD

# Cache the results
self._cached_path_ml = path_ml
self._cached_bound_poly = bound_poly
self._cached_dest_id = brow['id']
self._cached_bound_poly_blocks_set = bound_poly_blocks_set  # ADD
```

### 6. Replace geometry checks with integer truncation
**File:** `nomad/traj_gen.py`, `_sample_step()` method

**Line 360 - REMOVE:**
```python
start_pt = Point(start_point_arr)  # DELETE - no longer needed
```

**Lines 363-364 - REPLACE:**
```python
in_current_dest = _point_in_blocks(start_point_arr, self._current_dest_building_row.get('blocks_set')) if self._current_dest_building_row is not None else False
in_previous_dest = _point_in_blocks(start_point_arr, self._previous_dest_building_row.get('blocks_set')) if self._previous_dest_building_row is not None else False
```

**Line 382 - REPLACE:**
```python
if _point_in_blocks(coord, brow.get('blocks_set')):
    break
```

**Line 410 - REPLACE:**
```python
if not in_current_dest and not in_previous_dest and not _point_in_blocks(start_point_arr, bound_poly_blocks_set):
    raise ValueError(f"Agent at {start_point_arr} is outside cached bound_poly for destination {brow['id']}")
```

**Line 451 - REPLACE:**
```python
if _point_in_blocks(coord, bound_poly_blocks_set):
    break
```

## Coding Principles Compliance

✅ **No Bloat**
- One helper function `_point_in_blocks()` (3 lines)
- One-line conversion: `building_dict['blocks_set'] = set(building_dict.get('blocks', []))`
- Minimal caching: one new cache variable

✅ **No Blind Coding**
- Verified blocks_gdf structure has building_id column
- Confirmed all geometries are block-based in RasterCity
- Identified all uses of Point/intersects/contains

✅ **No Ad-Hoc Patches**
- Systematic solution using existing blocks_gdf architecture
- Proper caching strategy for bound_poly_blocks_set
- Backward compatible via .get() with default empty list

✅ **No Type Hints** - None added

✅ **Imports at Beginning** - No new imports needed

✅ **Avoid Double Loops** - None introduced

✅ **Plan Before Editing** - This document

## Expected Performance Impact
**Baseline:** 49.3s trajectory generation (5 agents, 14 days)

**Removals:**
- ~201,600 Point creations for building checks
- ~27,000 Point creations in stay-within-building loop
- ~43,600 Point creations for bound_poly checks
- **Total: ~272,000 Point + shapely operations → integer operations**

**Estimated speedup:** 1.76x faster per test (0.060s → 0.034s)
- Building checks: ~11s savings
- Bound poly checks: ~4s savings  
- **Total estimated savings: ~15s (49.3s → ~34s trajectory generation)**

## Testing Strategy
1. Run full test suite: `pytest nomad/tests/test_traj_gen.py -v`
2. Profile: `python profile_trajectory.py`
3. Verify backward compatibility with old geopackage (no 'blocks' column)
4. Check that RandomCityGenerator still works

