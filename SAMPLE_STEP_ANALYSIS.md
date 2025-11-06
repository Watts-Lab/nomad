# `_sample_step` Coordinate Transformation Analysis

## Overview

`_sample_step` is called **repeatedly** in a tight loop (`_traj_from_dest_diary`, line 487) to generate individual position pings as an agent moves toward a destination. Each call is **independent** and reconstructs all geometric objects from scratch.

---

## The Coordinate Transformation System

### Purpose
Convert between **Cartesian (x, y)** space and **path-based (distance_along, perpendicular_offset)** space to enable realistic random walks that:
1. Drift forward along the path (toward destination)
2. Wiggle perpendicular to the path (crossing the street, lane changes)
3. Stay within the "street corridor" (not wandering into buildings)

---

## Step-by-Step Breakdown

### 1. Path Construction (Lines 424-431) - **RECONSTRUCTED EVERY CALL**

```python
# Get path as list of block tuples
street_path = city.get_shortest_path(start_node, dest_cell)  # e.g., [(8,8), (8,9), (8,10)]

# Convert to block centers
path = [(x + 0.5, y + 0.5) for (x, y) in street_path]  # e.g., [(8.5, 8.5), (8.5, 9.5), ...]

# Prepend start segment (if inside building) and append destination door
path = start_segment + path + [(dest_door_centroid.x, dest_door_centroid.y)]

# Create MultiLineString - a continuous line through these points
path_ml = MultiLineString([path])
```

**Result**: `path_ml` is a **continuous polyline** connecting:
- Start position (or start building door)
- Block centers along shortest path
- Destination door

**Cost**: **EXPENSIVE** - Calls `get_shortest_path()` and constructs `MultiLineString` every iteration

---

### 2. Boundary Polygon Construction (Line 432-433) - **RECONSTRUCTED EVERY CALL**

```python
# Get geometry of each street block along path
street_geom = unary_union([city.get_block(b)['geometry'] for b in street_path])

# Union with start building (if applicable)
bound_poly = unary_union([start_info['geometry'], street_geom]) if start_info['geometry'] is not None else street_geom
```

**Result**: `bound_poly` is a **merged polygon** representing the allowable movement area:
- All street blocks along the path
- Plus start building (if agent is exiting a building)

**Purpose**: Agent can only move within this corridor (checked at line 452)

**Cost**: **VERY EXPENSIVE** - `unary_union()` merges potentially hundreds of block polygons

---

### 3. Coordinate Transformation (Lines 436, 450)

#### Forward Transform: Cartesian → Path-based (`_path_coords`)

```python
# Convert current position to path coordinates
path_coord = _path_coords(path_ml, start_point_arr)
# Returns: (distance_along_path, perpendicular_offset)
```

**How it works** (lines 1117-1154):
1. `distance = multilines.project(point)` - Find closest point on path, return distance along path
2. `point_on_path = multilines.interpolate(distance)` - Get that point's coordinates
3. Compute tangent direction using `distance - eps` 
4. Rotate tangent 90° to get normal vector
5. Project `(point - point_on_path)` onto normal to get signed perpendicular offset

**Output**: 
- `distance`: How far along the path (scalar, in block units)
- `offset`: How far perpendicular to path (scalar, positive = left, negative = right)

#### Backward Transform: Path-based → Cartesian (`_cartesian_coords`)

```python
# Convert path coordinates back to cartesian
coord = _cartesian_coords(path_ml, *path_coord)
# Input: (distance_along, perpendicular_offset)
# Returns: (x, y)
```

**How it works** (lines 1083-1115):
1. `point_on_path = multilines.interpolate(distance)` - Get point at distance along path
2. Compute tangent direction using `distance - eps`
3. Rotate tangent 90° to get normal vector
4. `cartesian = point_on_path + offset * normal`

---

### 4. Random Walk in Path Space (Lines 438-454)

```python
heading_drift = 3.33 * dt  # Forward bias (moves toward destination)
sigma = 0.5 * dt / 1.96     # Random walk variance

while True:
    # Random step in path space
    step = rng.normal(loc=[heading_drift, 0], scale=sigma * np.sqrt(dt), size=2)
    
    # Update path coordinates
    path_coord = (path_coord[0] + step[0],      # distance_along += drift + noise
                  0.7 * path_coord[1] + step[1]) # perpendicular *= 0.7 + noise
    
    # Check if reached destination
    if path_coord[0] > path_ml.length:
        coord = np.array([dest_geom.centroid.x, dest_geom.centroid.y])
        break
    
    # Convert back to cartesian
    coord = _cartesian_coords(path_ml, *path_coord)
    
    # Check if inside allowed corridor
    if bound_poly.contains(Point(coord)):
        break  # Valid position found
```

**Key behaviors**:
- **Forward drift**: `heading_drift = 3.33 * dt` biases movement toward destination
- **Perpendicular damping**: `0.7 * path_coord[1]` pulls agent back toward centerline
- **Rejection sampling**: If `coord` falls outside `bound_poly`, try again

---

## Critical Performance Issues

### Issue 1: Polygons Reconstructed Every Iteration ❌

**Current**: Lines 424-433 execute **every time `_sample_step()` is called**

For a 4-hour trajectory with `dt=0.5` minutes:
- **480 calls** to `_sample_step()`
- **480 calls** to `get_shortest_path()`
- **480 calls** to `unary_union()` (merging street geometries)
- **480 `MultiLineString` constructions**

**Why this is terrible**:
1. Path between same (start_building, dest_building) pair is recomputed hundreds of times
2. `unary_union()` is O(n log n) for n polygons
3. No caching or reuse

### Issue 2: Path Recomputation Between Same Doors

Agent moving from `h-x8-y8` to `w-x15-y10` over 60 minutes (120 steps at `dt=0.5`):
- **All 120 steps** call `get_shortest_path((8,8), (15,10))`
- **All 120 steps** build the same `MultiLineString`
- **All 120 steps** union the same street polygons

**Only variable**: Agent's position along the path (start_point)

---

## Your MultiLineString Optimization Proposal

### Key Insight
**The path between two doors doesn't change during a trip.** 

Only the agent's position along that path changes.

### Proposed Architecture

#### Phase 1: Cache Door-to-Door MultiLineStrings

```python
# In City class or as separate cache
self.door_to_door_paths = {}  # Dict or callable

def get_shortest_path_ml(start_door, end_door):
    """
    Returns MultiLineString from start_door to end_door.
    Cached for repeated queries.
    """
    if (start_door, end_door) in self.door_to_door_paths:
        return self.door_to_door_paths[(start_door, end_door)]
    
    # Compute once
    street_path = self.get_shortest_path(start_door, end_door)  # List of blocks
    path_coords = [(x + 0.5, y + 0.5) for (x, y) in street_path]
    path_ml = MultiLineString([path_coords])
    
    self.door_to_door_paths[(start_door, end_door)] = path_ml
    return path_ml
```

**Benefit**: For agent going from home → work → lunch → work → home, only **4 unique paths** are computed (not 480).

---

#### Phase 2: Hub-Based MultiLineString Network

Your proposal for sparse hub network:

```python
# Precomputed at city creation
self.hub_to_hub_paths = {}  # MultiLineStrings between hub nodes

def get_shortest_path_ml(start_door, end_door):
    """
    Returns MultiLineString using hub network for long distances.
    """
    # If doors are "close" (within hub radius)
    if manhattan_distance(start_door, end_door) < threshold:
        # Direct path
        return compute_direct_multilinestring(start_door, end_door)
    
    # Otherwise: door → nearest_hub → hub_path → nearest_hub → door
    start_hub = self.nearest_hub[start_door]
    end_hub = self.nearest_hub[end_door]
    
    # Concatenate segments
    seg1 = door_to_hub_ml(start_door, start_hub)     # Precomputed or on-demand
    seg2 = self.hub_to_hub_paths[(start_hub, end_hub)]  # Precomputed
    seg3 = hub_to_door_ml(end_hub, end_door)         # Precomputed or on-demand
    
    # Merge into single MultiLineString
    return merge_multilinestrings([seg1, seg2, seg3])
```

**Benefit**: 
- Hub-to-hub paths computed once (e.g., 200×200 = 40k pairs)
- Door-to-hub paths computed once per door (e.g., 20k doors)
- Total: ~60k MultiLineStrings for 2M block city
- vs. recomputing billions of paths on-demand

---

## Existing Projection Methods in traj_gen

### Yes, Already Implemented! ✅

The coordinate transformation functions **already do** what you need:

**`_path_coords(multilines, point)`** (line 1117):
- Projects cartesian point onto MultiLineString
- Returns (distance_along, perpendicular_offset)
- Uses `multilines.project()` internally (Shapely's efficient projection)

**`_cartesian_coords(multilines, distance, offset)`** (line 1083):
- Inverse operation
- Given distance and offset, returns cartesian (x, y)
- Uses `multilines.interpolate()` internally

**These are exactly what you need** - they work with **any** MultiLineString, not just the specific path construction.

---

## Implementation Strategy

### What Changes in `_sample_step`

**Before** (lines 424-433):
```python
street_path = city.get_shortest_path(start_node, dest_cell)  # Returns list
path = [(x + 0.5, y + 0.5) for (x, y) in street_path]
path = start_segment + path + [dest_door_centroid]
path_ml = MultiLineString([path])
street_geom = unary_union([city.get_block(b)['geometry'] for b in street_path])
bound_poly = unary_union([start_info['geometry'], street_geom])
```

**After**:
```python
# get_shortest_path now returns MultiLineString (with optional start/end segments)
path_ml = city.get_shortest_path_ml(start_node, dest_cell, 
                                     start_point=start_point if in_building else None,
                                     end_point=dest_door_centroid)

# Optionally also return bound_poly (if needed), or skip it entirely
bound_poly = city.get_path_corridor(start_node, dest_cell)  # Cached
```

**Benefit**: One method call, returns cached MultiLineString

---

### Alternative: Remove `bound_poly` Check Entirely

**Question**: Is the `bound_poly.contains()` check (line 452) necessary?

**Current purpose**: Prevents agent from wandering outside street corridor

**Alternative**: Trust the path-based coordinate system
- Perpendicular offset is damped (`0.7 * path_coord[1]`)
- Drift is forward-biased
- Agent naturally stays near path

**If removed**:
- No need for `unary_union()` of street geometries
- Significant speedup
- May lose some realism (agents could "cut corners")

---

## Summary: Answering Your Questions

### 1. Does traj_gen have MultiLineString projection methods?

**YES**: 
- `_path_coords(multilines, point)` - cartesian → path-based
- `_cartesian_coords(multilines, distance, offset)` - path-based → cartesian

These work with **any MultiLineString**, so your optimization is feasible.

---

### 2. How do polygons persist?

**THEY DON'T**: 
- `path_ml` and `bound_poly` are **reconstructed every call**
- No caching or reuse between calls
- **Major performance bottleneck**

---

### 3. Optimization Path

**Step 1** (Easy): Cache door-to-door MultiLineStrings
- Modify `get_shortest_path()` to return MultiLineString (or add new method)
- Add simple dict cache: `{(start_door, end_door): path_ml}`
- Modify `_sample_step` to use cached MultiLineString
- **Estimated speedup**: 100-1000x for repeated trips

**Step 2** (Medium): Hub-based MultiLineString network  
- Precompute hub-to-hub MultiLineStrings at city initialization
- Compute door-to-hub on first use (or precompute for all doors)
- Concatenate segments on demand
- **Estimated speedup**: Scalable to millions of doors

**Step 3** (Optional): Remove `bound_poly` check
- Simplifies code
- Further speedup
- Test if realistic trajectories still generated

---

## Recommendation

Your intuition is **spot on**. The current implementation is doing massive redundant work. The MultiLineString approach with hub network caching is the right solution.

**Next steps**:
1. Implement `compute_shortest_paths_ml(callable_only=True)` - mirrors our current work
2. Cache hub-to-hub MultiLineStrings
3. Modify `_sample_step` to accept/use MultiLineString directly
4. Profile and compare

The coordinate transformation functions (`_path_coords`, `_cartesian_coords`) are already perfect for this - no changes needed there!

