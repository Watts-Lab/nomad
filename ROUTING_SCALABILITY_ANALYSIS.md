# Routing Scalability Crisis: Analysis and Path Forward

## Coding Principles

This codebase adheres to the following principles:

1. **No bloat**: Avoid unnecessary variables, checks, defensive code
2. **No imports inside functions**: All imports at the beginning of the file
3. **Efficient operations**: Use vectorized pandas/numpy, avoid loops where possible
4. **Clear intent**: Self-documenting code
5. **Consistency**: Follow existing patterns
6. **No defensive guards that hide bugs**: Let failures surface clearly
7. **No fallbacks or legacy support**: One clear way to do things

---

## The Problem

### Scale of Philadelphia Rasterization

- **Full Philadelphia**: ~2 million street blocks
- **Large box** (current test): ~19,000 buildings, hundreds of thousands of street blocks
- **Current implementation**: **Infeasible at scale**

### Two Critical Bottlenecks

#### 1. All-Pairs Shortest Path Computation (Memory)

**Location**: `nomad/city_gen.py:1190`

```python
city.shortest_paths = dict(nx.all_pairs_shortest_path(G))
```

**The Crisis**:
- For **2 million nodes**: ~4 trillion pairs
- Even storing just paths (not full all-pairs matrix) is **impossible**
- This line is called in `City.from_geodataframes()` when loading from edges
- **This was removed** from the main code path but **still exists as a vestige**

**Current Status**: The new `get_shortest_path()` method (lines 1483-1620) **does NOT use** `shortest_paths` dictionary. It uses:
- Hub-based shortcut network (`_build_hub_network`)
- On-demand NetworkX shortest path when needed
- Simple LRU cache (50,000 entries) in `get_distance_fast()`

**The Problem**: Line 1190 still computes this infeasible structure during city loading, even though it's never used!

---

#### 2. Geometric Path Construction in `_sample_step` (Performance)

**Location**: `nomad/traj_gen.py:343-455`

**The Expensive Operations** (lines 424-433):

```python
# 1. Get shortest path (this part is OK with hub network)
street_path = city.get_shortest_path(start_node, dest_cell)

# 2. Build MultiLineString from path (EXPENSIVE)
path = [(x + 0.5, y + 0.5) for (x, y) in street_path]
path = start_segment + path + [(dest_door_centroid.x, dest_door_centroid.y)]
path_ml = MultiLineString([path])

# 3. Union all geometries along path (VERY EXPENSIVE)
street_geom = unary_union([city.get_block(b)['geometry'] for b in street_path])
bound_poly = unary_union([start_info['geometry'], street_geom]) if start_info['geometry'] is not None else street_geom
```

**Why This Is Expensive**:
1. **`MultiLineString` construction**: Creates complex geometric object for every step
2. **`unary_union` of path geometries**: Merges potentially hundreds of block polygons
3. **Transformed coordinate system**: Uses `_path_coords()` and `_cartesian_coords()` to map between:
   - Cartesian (x, y) space
   - Path-based (distance_along_path, perpendicular_offset) space
4. **`bound_poly.contains()` check**: Tests if point is inside complex polygon (line 452)
5. **While loop**: Repeats geometric operations until valid position found

**The Logic** (lines 435-454):
```python
# Transform to path coordinates
path_coord = _path_coords(path_ml, start_point_arr)

heading_drift = 3.33 * dt
sigma = 0.5 * dt / 1.96

while True:
    # Step in transformed space
    step = rng.normal(loc=[heading_drift, 0], scale=sigma * np.sqrt(dt), size=2)
    path_coord = (path_coord[0] + step[0], 0.7 * path_coord[1] + step[1])
    
    if path_coord[0] > path_ml.length:
        coord = np.array([dest_geom.centroid.x, dest_geom.centroid.y])
        break
    
    # Transform back to cartesian
    coord = _cartesian_coords(path_ml, *path_coord)
    
    # Check if inside bounded corridor
    if bound_poly.contains(Point(coord)):
        break
```

**Purpose**: Agent moves along street path with:
- Forward drift (`heading_drift`)
- Random walk in path-perpendicular direction
- Constrained to stay within street corridor (`bound_poly`)

---

## Code References

### City Generation & Routing

#### `city_gen.py:512-552` - `get_street_graph()`
- Builds NetworkX graph from streets_gdf
- Creates edges for cardinally adjacent street blocks
- **Does NOT compute all-pairs shortest paths** (good!)
- Returns `nx.Graph` object

#### `city_gen.py:1190` - **VESTIGIAL ALL-PAIRS COMPUTATION**
```python
city.shortest_paths = dict(nx.all_pairs_shortest_path(G))
```
- Called in `from_geodataframes()` when loading from edges
- **Never used** by any downstream code
- **Should be removed immediately**

#### `city_gen.py:1483-1620` - `get_shortest_path()`
- Hub-based routing using shortcut network
- Falls back to on-demand `nx.shortest_path()` when needed
- **Scalable approach** (only computes paths as needed)
- Uses `_build_hub_network()` for efficient routing

#### `city_gen.py:1625-1665` - `get_distance_fast()`
- Wraps `get_shortest_path()` with simple cache
- 50,000 entry limit with eviction
- Returns path length as distance
- **Already scalable**

---

### Trajectory Generation

#### `traj_gen.py:343-455` - `_sample_step()`
**The Core Simulation Method**

**Input**:
- `start_point`: Current (x, y) position
- `dest_building_id`: Target building
- `dt`: Time step (minutes)
- `rng`: Random number generator

**Output**:
- `coord`: New (x, y) position
- `location`: Building ID if arrived, else None

**Logic Flow**:
1. **Lines 367-378**: Resolve destination building geometry and door
2. **Lines 379-404**: If already at destination → stay-within-building dynamics
3. **Lines 406-421**: If starting in building → plan route from building door
4. **Lines 423-426**: Get shortest path between doors (uses `get_shortest_path`)
5. **Lines 428-433**: **EXPENSIVE** - Build `MultiLineString` and `bound_poly`
6. **Lines 435-454**: **EXPENSIVE** - Transformed coordinate random walk

**The Transformation**:
- `_path_coords(path_ml, point)` → (distance_along, perpendicular_offset)
- Random walk in transformed space
- `_cartesian_coords(path_ml, distance, offset)` → (x, y)
- Check if inside `bound_poly` corridor

---

#### `traj_gen.py:1083-1154` - Coordinate Transformation Helpers

**`_cartesian_coords(multilines, distance, offset, eps=0.001)`**:
- Converts path-based coords → cartesian (x, y)
- Finds point on path at `distance`
- Applies perpendicular `offset`
- Uses tangent vector estimation with `eps` delta

**`_path_coords(multilines, point, eps=0.001)`**:
- Converts cartesian (x, y) → path-based coords
- Projects point onto path using `multilines.project()`
- Computes perpendicular offset using normal vector
- Returns (distance_along_path, offset)

---

### Tests

#### `test_traj_gen.py` - NO DIRECT TESTS FOR `_sample_step`
- Tests exist for `generate_trajectory()` and `sample_trajectory()`
- Tests workflow: destination diary → trajectory → sampling
- **No unit tests specifically for `_sample_step` geometry**
- Tests pass for small `RandomCity` (101x101 blocks)

---

### Notebooks Using This Code

#### `examples/random_city.ipynb`
**Currently Works** (small scale):
```python
city_generator = RandomCityGenerator(width=101, height=101, 
                                     street_spacing=5, seed=100)
clustered_city = city_generator.generate_city()

population = Population(clustered_city)
population.generate_agents(N=1, seed=100)

agent.generate_trajectory(end_time=pd.Timestamp(2025, 1, 8))
agent.sample_trajectory(beta_start=300, beta_durations=60, beta_ping=10)
```

**Scale**: 
- 101x101 = 10,201 total blocks
- Street spacing 5 → ~400 street blocks
- ~20 streets per dimension
- **Feasible at this scale**

**Challenge**: Doesn't scale to 1,000x1,000 or 2,000x2,000 (Philadelphia scale)

---

## The OSM Street Graph Alternative

### What We Have (RasterCity)

**Location**: `city_gen.py:1924-2149` - `RasterCity` class

**Input Data**:
- `streets_gdf`: Simplified OSM street network (already processed by OSMnx)
  - Nodes at intersections
  - Edges are street segments
  - Already cleaned and consolidated
- `buildings_gdf`: OSM buildings with types

**Current Transformation**:
1. Rasterize streets → uniform grid blocks
2. Buildings → door cells on grid
3. Routing on **discrete grid** (not original OSM graph)

**The Opportunity**:
- OSM graph is **already optimal** for routing
- Nodes ~= intersections (sparse, ~tens of thousands)
- Edges have real lengths, geometries
- NetworkX can route efficiently on this graph

---

### The Integration Challenge

**Problem**: Two different coordinate systems:

1. **Garden City Grid** (current simulation):
   - Integer block coordinates (x, y)
   - Buildings occupy 1x1 blocks
   - Agents move in continuous (x, y) with noise
   - All geometry in "garden city units"

2. **OSM Street Network** (reality):
   - Node IDs (arbitrary integers)
   - Edge geometries (LineStrings in meters)
   - Real-world coordinates (Web Mercator)
   - Irregular graph structure

**Mapping Challenge**:
- Building doors → nearest OSM nodes?
- Agent position (x, y) → nearest OSM edge?
- Path following → interpolation along OSM LineString?
- How to preserve stochastic movement model?

---

## Potential Solutions

### Option 1: Eliminate All-Pairs Shortest Path (Quick Win)

**Action**: Remove line 1190 from `city_gen.py:from_geodataframes()`

**Impact**:
- Eliminates infeasible memory allocation
- No functional change (never used)
- **Immediate fix**

---

### Option 2: Simplify `_sample_step` Geometry (Performance)

**Current Overhead**:
- `unary_union` of path geometries
- `MultiLineString` construction
- Transformed coordinate system
- Polygon containment checks

**Possible Simplification**:
1. **Skip `bound_poly` construction**: Just trust the path, add noise perpendicular
2. **Simplified corridor**: Use buffer around path centerline (cheaper than union)
3. **Direct cartesian movement**: Move along path without coordinate transformation
4. **Vectorize**: Compute multiple steps at once instead of while loop

**Trade-off**: May lose some realism in movement dynamics

---

### Option 3: Hybrid OSM + Grid Routing (Ambitious)

**Concept**:
- **Long-distance routing**: Use OSM graph (intersection to intersection)
- **Local movement**: Use grid blocks (within ~5 blocks of current location)
- **Mapping layer**: Buildings have both door_cell (grid) and nearest_osm_node

**Implementation**:
1. Each building stores: `door_cell` (grid) + `nearest_osm_node` (OSM graph ID)
2. `get_shortest_path(start_building, dest_building)`:
   - Map buildings → OSM nodes
   - Route on OSM graph (NetworkX, efficient)
   - Map OSM nodes back to grid cells for final approach
3. `_sample_step()`:
   - If far from destination: Use OSM graph segment
   - If near destination: Use grid-based movement

**Challenges**:
- Complex coordinate mapping
- Two different graph representations
- Need to preserve stochastic model
- Testing and validation

---

### Option 4: Pure OSM Graph Simulation (Radical)

**Concept**: Abandon grid entirely, simulate on OSM graph

**Changes**:
- Agent position: (edge_id, distance_along_edge)
- Buildings: Attached to edges
- Movement: Walk along edges, switch at nodes
- Coordinate transform: Only for output (export to lat/lon)

**Advantages**:
- Direct use of OSM graph (no rasterization needed)
- Inherently scalable
- Real street network topology

**Challenges**:
- Complete rewrite of simulation logic
- How to handle building interiors?
- Perpendicular noise model doesn't translate
- Departure from current "garden city" paradigm

---

## Recommendations

### Immediate (This Week)

1. **Remove line 1190** (`all_pairs_shortest_path`) - Easy win, no downside
2. **Profile `_sample_step`** on large city to quantify geometry overhead
3. **Benchmark**: 
   - Time per `_sample_step` call
   - Path length distribution
   - Geometry operation breakdown

### Short-term (Next Sprint)

4. **Simplify `_sample_step` geometry**:
   - Test removing `bound_poly` construction
   - Use simpler corridor constraint
   - Measure performance gain vs. realism loss

5. **Optimize coordinate transforms**:
   - Cache `MultiLineString` objects?
   - Vectorize transformation functions?
   - Pre-compute path properties?

### Long-term (Research)

6. **Prototype hybrid OSM + Grid routing**:
   - Proof-of-concept with small example
   - Validate trajectory realism
   - Assess complexity vs. benefit

7. **Investigate pure OSM graph simulation**:
   - Literature review: How do other models handle this?
   - Prototype minimal viable implementation
   - Compare results with current model

---

## Open Questions

1. **How much geometric fidelity do we need?**
   - Is the `bound_poly` corridor constraint essential?
   - Can we simplify to straight-line paths with noise?
   
2. **What is the actual bottleneck?**
   - Profile needed: Is it geometry or path computation?
   - How many `_sample_step` calls per agent per day?

3. **Can we vectorize `_sample_step`?**
   - Generate multiple steps at once?
   - Pre-compute paths for common routes?

4. **OSM graph integration strategy?**
   - Should we commit to hybrid or pure OSM?
   - What is the migration path?
   - How to maintain backward compatibility?

5. **Testing and validation?**
   - How to verify trajectory realism after changes?
   - What metrics matter: path length, dwell time, visit patterns?
   - Do we need ground truth comparison?

---

## Next Steps

**Priority**: **Profile first, optimize second**

1. Run `synthetic_philadelphia.py` with profiling on LARGE_BOX
2. Measure time spent in `_sample_step` components
3. Count calls to `get_shortest_path` vs. geometry operations
4. Make data-driven decisions about where to optimize

**Then**: Choose between simplification (Option 2) vs. hybrid routing (Option 3)

