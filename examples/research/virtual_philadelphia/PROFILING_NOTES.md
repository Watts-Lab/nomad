# Profiling Notes: Trajectory Generation Optimization

**Setup:** 5 agents, 14 days (May 23 - Jun 6, 2025), Philadelphia large box (~20k buildings)
**Method:** `cProfile` for function-level, `line_profiler` (`kernprof`) for line-by-line

## Baseline (With start_pt caching optimization)
- **Dest Diary:** 0.560s per agent per day
- **Trajectory:** 0.969s per agent per day
- **Total:** 1.529s per agent per day

### Current optimizations included:
1. Cache `Point(start_point_arr)` as `start_pt` in `_sample_step` (lines 359-360)
   - Reuse cached Point instead of creating 3x per step

### Detailed Line-by-Line Analysis (_sample_step: 151.2s total):

**Top Bottlenecks:**
1. Line 414: `get_shortest_path()` - **41.4s (27.4%)** - 693 calls @ 59.7ms each
2. Line 434: `_path_coords()` - **15.5s (10.2%)** - 44,344 calls
3. Line 363: `.intersects()` (current dest) - **10.9s (7.2%)** - 201,600 calls
4. Line 360: `Point(start_point_arr)` - **10.9s (7.2%)** - 201,600 calls
5. Line 364: `.intersects()` (previous dest) - **9.8s (6.5%)** - 201,600 calls
6. Line 448: `_cartesian_coords()` - **9.0s (5.9%)** - 43,927 calls
7. Line 403: `Point()` for cache check - **7.8s (5.1%)** - 43,922 calls
8. Line 415: `.loc[street_path]` - **5.5s (3.6%)** - 693 calls
9. Line 355: `brow['building_type']` - **5.1s (3.4%)** - 201,600 calls (Series lookup!)
10. Line 419: `[(pt.x, pt.y) for pt in centroids]` - **4.5s (3.0%)** - 693 calls
11. Line 421: `unary_union(street_blocks)` - **3.9s (2.6%)** - 693 calls
12. Line 450: `.contains(Point(coord))` - **3.9s (2.6%)** - 43,927 calls

**Total accounted:** 128s / 151s

### Low-Hanging Fruit Candidates:
1. **Cache building attributes as dict** - Line 355 shows pandas Series lookups are expensive (5.1s for just building_type!)
   - Currently: `brow['building_type']`, `brow['door_cell_x']`, etc. hit pandas overhead 201k+ times
   - Solution: Extract to simple dict/named tuple when destination changes
   - Estimated savings: 7-10s

2. **Use shapely prepared geometries** - `.contains()` and `.intersects()` called repeatedly on same geometries
   - Lines 363, 364, 381, 410, 450 all use same geometries repeatedly
   - `shapely.prepared.prep()` creates spatial index for faster queries
   - Estimated savings: 10-15s

3. **Avoid unnecessary Point() creations** - Line 403 creates Point just for intersects check
   - Could compare coordinates directly without Point wrapper
   - Estimated savings: 7-8s

4. **Cache dest_cell calculation** - Line 398 recalculates every call but only changes with destination
   - Should be part of cached building attributes
   - Estimated savings: 2s

### Dest Diary Notes:
- `compute_gravity_row` dominates (30.2s of 39.2s total)

## Optimization Attempts

### ✅ **Opt 1: Cache destination ID instead of Point creation** (IMPLEMENTED)
- **Problem:** Line 403 created `Point()` + `.intersects()` for cache validation (7.8s, 43k calls)
- **Solution:** Cache destination building ID (`_cached_dest_id`), compare IDs directly
- **Status:** Implemented, tests pass
- **Estimated savings:** ~7-8s

### ❌ **Opt 2: Prepared geometries** (REJECTED)
- **Hypothesis:** `shapely.prepared.prep()` would speed up repeated `.contains()`/`.intersects()` calls
- **Test Results:**
  - `contains()`: 1.00x speedup (no improvement)
  - `intersects()`: 1.07x speedup (marginal)
- **Conclusion:** Spatial indexing overhead not worth it for our polygon complexity
- **Status:** Rejected based on empirical testing

### ✅ **Opt 3: Cache building attributes as dict** (IMPLEMENTED)
- **Problem:** Pandas Series lookups expensive (line 355: 5.1s for 201k calls to `brow['building_type']`)
- **Solution:** Use `.to_dict()` when setting `_current_dest_building_row` and `_previous_dest_building_row`
- **Changes:** Lines 481, 487, 496 in `_traj_from_dest_diary()`
- **Status:** Implemented, tests pass
- **Actual savings:** 18.5s trajectory generation (67.8s → 49.3s)

## Results After Opt 1 + Opt 3

**Baseline (with start_pt caching):**
- Dest Diary: 0.560s per agent per day
- Trajectory: 0.969s per agent per day
- Total: 1.529s per agent per day

**After Opt 1 (cache dest ID) + Opt 3 (dict caching):**
- Dest Diary: 0.502s per agent per day (**10.4% faster**)
- Trajectory: 0.704s per agent per day (**27.3% faster**)
- Total: 1.206s per agent per day (**21.1% faster**)

**Absolute improvements:**
- Dest Diary: 4.0s faster (39.2s → 35.2s)
- Trajectory: 18.5s faster (67.8s → 49.3s)
- Total: 22.5s faster (107.0s → 84.5s)

## Remaining Bottlenecks (After Opt 1 + Opt 3)

From profiling at 49.3s trajectory generation:

**Major bottlenecks:**
1. `get_shortest_path()` - **10.2s** (693 calls @ 14.7ms/call)
   - NetworkX bidirectional search on street graph
   - Only called on cache miss (693 / 201,600 = 0.3%)
   
2. Shapely decorator overhead - **28.9s total**
   - `decorators.py:73(wrapped)` - 18.3s (1.75M calls)
   - `decorators.py:171(wrapper)` - 10.6s (762k calls)
   - Wraps C calls for shapely operations

3. `_sample_step()` internal - **43.6s** (201,600 calls)
   - Remaining time after subtracting known functions
   - Includes geometry checks, path computation

**Breakdown of _sample_step time (from previous line profiling):**
- `get_shortest_path`: 41.4s → now 10.2s (huge improvement!)
- `_path_coords`: 15.5s (44k calls) - path coordinate transformation
- Point creation + intersects (lines 363-364): ~20s
- `_cartesian_coords`: 9.0s (44k calls)
- Other shapely operations: ~10s

## Geometry Check Method Comparison

**Test:** 10,000 point-in-geometry checks on 13-block L-shaped region

| Method | Time | Speedup | Notes |
|--------|------|---------|-------|
| 1. unary_union + intersects | 0.060s | 1.00x | Current method |
| 2. MultiPolygon + intersects | 0.061s | 0.98x | No improvement |
| 3. GeoDataFrame + intersects | 0.526s | 0.11x | Much slower |
| 4. Integer truncation + set | 0.034s | **1.76x** | Fastest |

**Method 4 (int truncation):** Truncate coordinates to integers, check `(floor(x), floor(y)) in block_set`

### Applicability to Current Code

**NOT APPLICABLE** for current OSM-based simulations because:
- Building geometries are **arbitrary polygons** (downloaded from OSM)
- `bound_poly` is **mixed**: unary_union of blocks + building geometry
- Integer truncation only works for **axis-aligned block-based** geometries

**APPLICABLE** for:
- Synthetic cities with block-based buildings (RandomCityGenerator)
- Block-only geometry checks (rare in current code)

**Current approach (unary_union + intersects) is appropriate** given arbitrary building geometries.

