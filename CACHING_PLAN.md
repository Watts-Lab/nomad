# Geometry Caching Plan for `_sample_step`

## Goal
Cache geometry computations to avoid redundant `unary_union` and `MultiLineString` calculations.

## Coding Principles
See `CODING_PRINCIPLES.md` for full list. Key points: no bloating, no guessing, no ad-hoc patches, use vectorized operations.

## Implementation Steps

### Step 1: Add Cache Attributes
Add 4 attributes to `Agent.__init__`:
- `self._cached_bound_poly = None`
- `self._cached_path_ml = None`
- `self._previous_dest_building_row = None`
- `self._current_dest_building_row = None`

Group with inline comment: "Trajectory simulation parameters (caching for performance)"

**Why**: Minimal change, no behavior modification.

### Step 2: Update `_traj_from_dest_diary` to Set Building Rows
At start of each destination diary entry loop (line 479):
- `self._previous_dest_building_row = self._current_dest_building_row`
- `self._current_dest_building_row = buildings_gdf row for current destination`

**Why**: External to `_sample_step`, minimal impact. Makes building data available without querying.

### Step 3: Add Inconsequential Geometry Checks
In `_sample_step`, after `get_block` call, add checks (no behavior change):
- Check if `start_point_arr` is inside `_current_dest_building_row['geometry']` (if cached)
- Check if `start_point_arr` is inside `_previous_dest_building_row['geometry']` (if cached)
- Check if `start_point_arr` is inside `_cached_bound_poly` (if cached)

Let code flow normally regardless of check results.

**Why**: Validates cache logic without changing behavior. Uses geometry directly, no `get_block` needed.

### Step 4: Conditional Path Recomputation
Modify path computation section (lines 423-433):
- Compute path between start building and destination building (not just start_node to dest_cell)
- Build bounding polygon from path blocks
- If point is NOT inside bounding polygon AND NOT inside start building AND NOT inside destination building, THEN recompute path_ml and bound_poly
- Otherwise, use cached values if available

**Why**: Only recomputes when necessary. Tests should still pass.

## What We're NOT Doing Yet
- Not modifying `get_block` logic
- Not changing rejection sampling loops
- Not optimizing cache hits (that comes later)
- Not removing original queries (keep them as fallback)
