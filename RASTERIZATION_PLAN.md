# Philadelphia Rasterization Implementation Plan

## Overview
Implement scalable, efficient rasterization of Philadelphia OSM data into a block-based City compatible with `traj_gen.py`. Focus on spatial indexing, vectorized operations, and lazy evaluation to handle 500k+ buildings efficiently.

## Core Requirements

### 1. Block Size
- **Parameter**: `block_side_length` (default: 15m for Philadelphia)
- **Stored in**: City attribute (already exists)
- **Rationale**: 15m balances detail vs scalability; too fine causes performance issues

### 2. Building Type Handling
- **"Other" buildings**: Include in city, assign to blocks with lowest priority
- **Future**: May sample type randomly later
- **Priority order**: street > park > workplace > home > retail > other

### 3. Block Type Assignment Priority
Blocks are assigned types based on intersection priority:
1. Street (if intersects street geometry)
2. Park (if no street, but intersects park)
3. Workplace (if no street/park, but intersects workplace)
4. Home (if no street/park/workplace, but intersects home)
5. Retail (if only retail intersects)
6. Other (if only "other" buildings intersect)

**Special case**: Buildings spanning multiple disconnected groups → split into multiple Building objects of same type.

### 4. Building-Street Connectivity
- Buildings must have adjacent street block for door assignment
- If no adjacent street: **skip building** (don't add to city)
- Door assignment: Use first adjacent street found during building addition

### 5. Development Scope
- **Phase 1**: Small bounding box (Old City Philadelphia bbox from `virtual_philly.py`)
- **Phase 2**: Scale to full Philadelphia (later)

### 6. Backwards Compatibility
- City methods (`get_street_graph`, `get_shortest_path`, etc.) must work unchanged
- Optional parameters for real vs synthetic data:
  - `manual_streets=True`: Streets must be explicitly added
  - `residual_streets=False`: Don't auto-assign all non-building blocks as streets
  - `randomize_types=False`: For synthetic data fallback

### 7. Canvas/Grid Generation
- **Strategy**: Only generate blocks that intersect rotated city boundary polygon
- **Lazy evaluation**: Don't instantiate full grid upfront
- **Method**: Use spatial index to generate blocks on-demand or batch-by-region

### 8. Street Connectivity Verification
- **Method**: `verify_street_connectivity()` 
- **Action**: Keep only largest connected component of street blocks
- **Output**: Summary of discarded blocks (count, locations)
- **Implementation**: Use NetworkX or vectorized BFS

### 9. Connected Component Detection
- **Need**: Efficient algorithm to find connected components of non-street blocks
- **Use cases**: 
  - Splitting buildings that span disconnected groups
  - Verifying building connectivity
- **Optimization**: Leverage grid structure (integer coordinates) for fast neighbor lookup

## Architecture Design

### New Module: `nomad/rasterization.py`
Contains:
- `RasterCityGenerator` class (replaces `RealCityGenerator`)
- Helper functions for spatial operations
- Connected component utilities

### Integration Points
- `City` class: Extend to support `manual_streets=True` with `residual_streets=False`
- `City.add_building()`: Must handle door assignment from adjacent streets
- `City.get_street_graph()`: Already compatible, uses `streets_gdf`
- `City.get_shortest_path()`: Already compatible

## Implementation Phases

### Phase 1: Infrastructure & Spatial Operations
**Goal**: Build scalable spatial primitives

1. **Canvas Generation**
   - Function: `generate_canvas_blocks(boundary_polygon, block_size)`
   - Returns: GeoDataFrame of blocks with `(x, y)` coordinates
   - Optimization: Use spatial index, lazy generation
   - Test: Small bbox, verify all blocks intersect boundary

2. **Spatial Intersection Helpers**
   - Function: `find_intersecting_blocks(geometry, blocks_gdf)`
   - Uses: `gpd.sjoin()` with spatial index
   - Returns: DataFrame of intersecting blocks
   - Test: Benchmark vs double-loop on small dataset

3. **Block Type Assignment**
   - Function: `assign_block_types(blocks_gdf, streets_gdf, buildings_by_type)`
   - Priority: street > park > workplace > home > retail > other
   - Returns: Blocks with `type` column
   - Test: Verify priority ordering, verify no double-assignment

### Phase 2: Connected Component Utilities
**Goal**: Fast connectivity analysis

4. **Connected Component Detection**
   - Function: `find_connected_components(blocks_gdf, connectivity='4-connected')`
   - Uses: Grid-based BFS (fast for integer coordinates)
   - Returns: List of component sets (each set contains block coordinates)
   - Optimization: Vectorized where possible
   - Test: Known disconnected groups, verify correct split

5. **Street Connectivity Verification**
   - Method: `City.verify_street_connectivity()`
   - Action: Find largest component, remove others
   - Returns: Summary dict with discarded count
   - Test: Artificially disconnected streets, verify cleanup

### Phase 3: Building Assignment
**Goal**: Assign buildings to blocks efficiently

6. **Building-to-Blocks Assignment**
   - Function: `assign_buildings_to_blocks(buildings_gdf, blocks_gdf, block_types)`
   - Strategy: 
     - Spatial join to find intersecting blocks
     - Group by building, find connected components
     - Split disconnected groups into separate Building objects
   - Returns: List of Building objects with door assignments
   - Test: Buildings spanning multiple blocks, buildings straddling streets

7. **Door Assignment**
   - Function: `assign_door_to_building(building_blocks, street_blocks_gdf)`
   - Logic: Find first adjacent street block using grid coordinates
   - Returns: Door coordinates or None (if disconnected)
   - Test: Buildings with/without adjacent streets

### Phase 4: RasterCityGenerator Class
**Goal**: Main orchestration class

8. **Class Structure**
   ```python
   class RasterCityGenerator:
       def __init__(self, boundary_polygon, streets_gdf, buildings_by_type_dict, block_size=15.0)
       def generate_city(self) -> City
       def _create_canvas(self)
       def _assign_streets(self)
       def _assign_buildings(self)
       def _verify_connectivity(self)
   ```

9. **Integration with City Class**
   - Ensure `City` accepts `manual_streets=True, residual_streets=False`
   - Verify all existing methods work with sparse street grid
   - Test: Generate city, verify `get_street_graph()` works

### Phase 5: Testing & Validation
**Goal**: Ensure correctness and compatibility

10. **Unit Tests**
    - Canvas generation correctness
    - Intersection performance (benchmark vs naive)
    - Connected component detection
    - Building assignment edge cases

11. **Integration Tests**
    - Generate city from Old City bbox
    - Verify compatibility with `Agent.generate_trajectory()`
    - Verify `shortest_paths` works correctly
    - Performance benchmarks

12. **Sandbox Dataset**
    - Create `examples/research/virtual_philadelphia/sandbox/` directory
    - Save small bbox data (Old City Philadelphia)
    - Document expected outputs

## Key Optimizations

### Spatial Indexing
- Use `geopandas.sjoin()` with spatial index (`sindex`) for all intersections
- Pre-index `streets_gdf` and `buildings_gdf` before batch operations
- Never use double loops for spatial operations

### Grid-Based Neighbor Lookup
- For connected components: Use `(x±1, y)` and `(x, y±1)` lookups
- Build lookup dict: `{(x, y): block_id}` for O(1) neighbor checks
- Avoid geometric distance calculations for grid neighbors

### Lazy Block Generation
- Generate blocks only for regions that intersect buildings/streets
- Use bounding box expansion: `bbox.buffer(block_size)` to find candidate regions
- Don't instantiate full grid upfront

### Vectorized Operations
- Use pandas `merge()` for neighbor finding (already in `get_street_graph()`)
- Use `groupby()` for building-to-blocks grouping
- Minimize row-by-row operations

## Compatibility Requirements

### City Class Interface
Must maintain compatibility with:
- `city.get_street_graph()` → Returns NetworkX graph
- `city.get_shortest_path(start, end)` → Returns list of coordinates
- `city.get_block(coords)` → Returns block info dict
- `city.buildings_gdf` → GeoDataFrame with columns: `['id', 'type', 'door_cell_x', 'door_cell_y', 'door_point', 'size', 'geometry']`
- `city.streets_gdf` → GeoDataFrame with columns: `['coord_x', 'coord_y', 'geometry']`
- `city.blocks_gdf` → GeoDataFrame with `['kind', 'building_id', 'building_type', 'geometry']`

### Agent Class Requirements
Must work with:
- `Agent.__init__(city=city, home=home_id, workplace=work_id)`
- `agent.generate_trajectory()` uses `city.get_shortest_path()` and `city.get_block()`
- Building IDs must be valid strings
- Door cells must be valid street coordinates

## Testing Strategy

### Small Test Dataset
- **Location**: `examples/research/virtual_philadelphia/sandbox/`
- **Extent**: Old City Philadelphia bbox (-75.1662060, 39.9411582, -75.1456557, 39.9557201)
- **Files**: 
  - `sandbox_streets.gpkg` (from rotated streets, clipped to bbox)
  - `sandbox_buildings.gpkg` (from rotated buildings, clipped to bbox)
  - `sandbox_boundary.gpkg` (bbox polygon)

### Test Cases
1. **Canvas generation**: Verify all blocks intersect boundary, none outside
2. **Street assignment**: Verify street blocks match intersections
3. **Building assignment**: Verify priority ordering, door assignment
4. **Disconnected buildings**: Verify splitting works correctly
5. **Street connectivity**: Verify largest component kept, others discarded
6. **Agent compatibility**: Generate trajectory, verify no errors
7. **Performance**: Benchmark vs naive double-loop approach

## File Structure

```
nomad/
  rasterization.py         # New module with RasterCityGenerator
  city_gen.py             # Extend City.__init__ for residual_streets parameter

examples/research/virtual_philadelphia/
  sandbox/
    sandbox_streets.gpkg
    sandbox_buildings.gpkg  
    sandbox_boundary.gpkg
    test_rasterization.py  # Development test script
  synthetic_philadelphia.py  # Updated to use RasterCityGenerator

nomad/tests/
  test_rasterization.py    # Unit tests for new module
```

## Success Criteria

1. ✅ Generate City from Old City bbox in < 30 seconds
2. ✅ All street blocks are connected (single component)
3. ✅ All buildings have valid door assignments
4. ✅ Agent can generate trajectories successfully
5. ✅ `get_shortest_path()` works correctly
6. ✅ No double-loop spatial operations
7. ✅ Code is readable, well-documented, testable

## Next Steps

1. Research spatial indexing best practices (geopandas sjoin)
2. Implement canvas generation with lazy evaluation
3. Implement efficient intersection helpers
4. Build connected component utilities
5. Integrate with City class
6. Test with sandbox dataset
7. Validate Agent compatibility

