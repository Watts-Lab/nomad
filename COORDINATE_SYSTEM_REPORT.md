# RasterCity Coordinate Systems Report

## Executive Summary

RasterCity maintains **TWO SEPARATE COORDINATE SYSTEMS** simultaneously:

1. **Garden City Block Coordinates** (`coord_x`, `coord_y`): Integer grid coordinates for simulation
2. **Web Mercator Geometries** (`geometry` column): Real-world coordinates in meters (EPSG:3857)

These systems are **offset** from each other but use the **same scale** (block_side_length in meters).

---

## The Problem: Why Two Systems?

### Historical Context
The original `City` class worked in abstract "garden city blocks":
- Coordinates like `(0, 0)`, `(1, 2)`, etc.
- Buildings have `door_cell_x`, `door_cell_y` as integers
- `traj_gen.py` expects integer block coordinates
- Method `to_mercator()` converts trajectories to real-world coordinates **after** simulation

### RasterCity's Challenge
`RasterCity` starts with **real OSM data** in Web Mercator (EPSG:3857):
- Buildings at coordinates like `(-8365127.3, 4864231.8)` meters
- Streets at similar real-world coordinates
- But still needs to feed integer block coordinates to `traj_gen.py`

---

## The Solution: Dual Coordinate System

### Step 1: Generate Canvas Blocks (line 1903)
```python
self.blocks_gdf = generate_canvas_blocks(self.boundary_polygon, self.block_side_length, self.crs)
```

**What `generate_canvas_blocks` does:**
- Takes boundary polygon with bounds like `minx=-8365200, miny=4864100, maxx=-8364800, maxy=4864500`
- Calculates integer grid range: `x_min = int(minx // block_size)` → `-836520` (line 1720)
- Creates blocks with:
  - **coord_x, coord_y**: integers like `(-836520, 486410)`
  - **geometry**: box from `(-836520 * 10, 486410 * 10)` to `(-836520 * 10 + 10, 486410 * 10 + 10)`
  
**Result**: Blocks have enormous integer coordinates, but geometries align perfectly with input data.

### Step 2: Calculate Offset (lines 1906-1909)
```python
minx, miny, maxx, maxy = self.boundary_polygon.bounds
self.offset_x = int(minx // self.block_side_length)  # e.g., -836520
self.offset_y = int(miny // self.block_side_length)  # e.g., 486410
offset_meters = (self.offset_x * self.block_side_length, self.offset_y * self.block_side_length)
```

**Purpose**: Store the "origin shift" needed to convert between systems.

### Step 3: Shift Block Coordinates to Garden City Space (lines 1911-1912)
```python
self.blocks_gdf['coord_x'] = self.blocks_gdf['coord_x'] - self.offset_x
self.blocks_gdf['coord_y'] = self.blocks_gdf['coord_y'] - self.offset_y
```

**Example:**
- Before: `coord_x = -836520`
- After: `coord_x = -836520 - (-836520) = 0`

**Result**: Block coordinates now start at `(0, 0)` like garden city expects.

### Step 4: Shift All Geometries to Match (lines 1914, 1918-1919)
```python
self.blocks_gdf['geometry'] = self.blocks_gdf['geometry'].translate(
    xoff=-offset_meters[0], yoff=-offset_meters[1]
)
self.streets_gdf_input['geometry'] = self.streets_gdf_input['geometry'].translate(...)
self.buildings_gdf_input['geometry'] = self.buildings_gdf_input['geometry'].translate(...)
```

**Example with block_side_length=10:**
- offset_meters = `(-836520 * 10, 486410 * 10)` = `(-8365200, 4864100)`
- Original geometry at `(-8365127.3, 4864231.8)`
- Shifted geometry: `(-8365127.3 - (-8365200), 4864231.8 - 4864100)` = `(72.7, 131.8)`

**Result**: Geometries are now in a local coordinate system starting near `(0, 0)`, aligned with the block grid.

### Step 5: Rasterization Creates Buildings (lines 1986-2003)
```python
block_polys = [box(x, y, x+1, y+1) for x,y in building_blocks]  # x, y are shifted coords
building_geom = unary_union(block_polys)
```

**Example:**
- Building covers blocks `[(0, 5), (1, 5), (0, 6)]` (garden city coords)
- Geometry: `box(0, 5, 2, 7)` in shifted Web Mercator
- `door_cell_x=0, door_cell_y=5` (garden city coords)

---

## The Relationship Between Systems

### Coordinate Transformation
```
Garden City Coords → Web Mercator (shifted):
    x_shifted_meters = coord_x * block_side_length
    y_shifted_meters = coord_y * block_side_length

Web Mercator (shifted) → Original Web Mercator:
    x_original = x_shifted + offset_x * block_side_length
    y_original = y_shifted + offset_y * block_side_length

Combined:
    x_original = (coord_x + offset_x) * block_side_length
    y_original = (coord_y + offset_y) * block_side_length
```

This is **exactly** what `blocks_to_mercator` does (map_utils.py line 854):
```python
result['x'] = block_size * result['x'] + false_easting
```
Where `false_easting` = `web_mercator_origin_x` = starting point in original Web Mercator.

But in `RasterCity`:
- `web_mercator_origin_x` is **not set** in constructor (uses default from base City)
- The "origin" is implicitly `offset_x * block_side_length`

### Why Geometries are Preserved

The `geometry` columns in `buildings_gdf`, `streets_gdf`, `blocks_gdf` serve several purposes:

1. **Visualization**: Plot the city using real shapes, not just grid cells
2. **Spatial Queries**: Check if building overlaps street, find nearest door, etc.
3. **Export**: Save city back to GeoJSON/GPKG with meaningful shapes
4. **Trajectory Conversion**: After simulation, convert back to real coordinates

---

## Why Integer Division?

```python
self.offset_x = int(minx // self.block_side_length)
offset_meters = self.offset_x * self.block_side_length
```

**Purpose**: Ensure the offset is a **multiple of block_side_length**.

**Example with block_side_length=10:**
- `minx = -8365127.3`
- `offset_x = int(-8365127.3 // 10)` = `int(-836512.73)` = `-836513`
- `offset_meters[0] = -836513 * 10` = `-8365130`

**Result**: The shifted coordinate system has block boundaries at integer multiples of block_side_length, which aligns perfectly with the grid where each block is exactly `block_side_length` wide.

If we didn't do integer division:
- Shifted geometry might start at fractional block position like `(0.27, 0.18)`
- Block `(0, 0)` would be `box(0, 0, 1, 1)` but geometry would be misaligned
- Spatial queries would fail

---

## Current State of Affairs

### What Works ✓
- `coord_x`, `coord_y` are garden city integers: `(0, 1, 2, ...)`
- `geometry` columns are in shifted Web Mercator, aligned with grid
- `traj_gen.py` uses integer coordinates for simulation
- Buildings have both integer door cells AND real geometry

### What's Confusing ✗
1. **Geometries are in shifted Web Mercator, not garden city units**
   - A building at `coord=(5, 10)` has `geometry=box(5*10, 10*10, 6*10, 11*10)` = `box(50, 100, 60, 110)` meters
   - Not `box(5, 10, 6, 11)` in abstract units

2. **web_mercator_origin_x/y are misleading**
   - Set to default values from base `City` class
   - Don't represent the actual origin of this rasterized city
   - Should be `offset_x * block_side_length` and `offset_y * block_side_length`

3. **CRS is EPSG:3857 but coordinates are shifted**
   - `gdf.to_file()` will write CRS as EPSG:3857
   - But coordinates are NOT in true Web Mercator
   - They're in "shifted Web Mercator" that only makes sense relative to `offset_x`, `offset_y`

---

## Implications for `to_file` Refactor

### Use Case (i): Save for downstream pipeline
```python
city.to_file(buildings_path='buildings.gpkg', driver='GPKG')
```
**Expectation**: Save as-is
- Geometries in shifted Web Mercator
- `coord_x`, `coord_y` as garden city integers
- Can reload with `RasterCity.from_geopackage()` and continue working

**Issue**: CRS claims EPSG:3857 but it's actually shifted. Should we:
- Set CRS to None?
- Add metadata about offset?
- Document that these are "working coordinates"?

### Use Case (ii): Export to real-world coordinates
```python
city.to_file(
    buildings_path='buildings.geojson',
    driver='GeoJSON',
    reverse_affine_transformation=True,
    to_crs='EPSG:4326'
)
```
**Expectation**: Undo shift, convert to WGS84, export
- Geometries back at original OSM coordinates
- But what about `coord_x`, `coord_y`? They're still garden city integers!
- Do we drop those columns? Keep them? Recalculate them?

**Critical Question**: If we reverse the affine transformation, the geometry columns become "real world" but the coordinate columns are still "garden city". This is inconsistent. Should we:
1. Drop `coord_x`, `coord_y` when exporting with `reverse_affine_transformation=True`?
2. Recalculate them to original grid coords (like `-836520`)?
3. Keep them as-is and document the mismatch?

---

## Recommendations

1. **Add `rotation_deg` to base City** ✓ (as planned)

2. **Recalculate `web_mercator_origin_x/y` in RasterCity**:
   ```python
   self.web_mercator_origin_x = self.offset_x * self.block_side_length
   self.web_mercator_origin_y = self.offset_y * self.block_side_length
   ```
   This makes them accurate for use with `to_mercator()`.

3. **For `to_file` with `reverse_affine_transformation=True`**:
   - Undo rotation (if non-zero)
   - Undo translation: add back `(offset_x * block_side_length, offset_y * block_side_length)`
   - **Drop `coord_x`, `coord_y`, `door_cell_x`, `door_cell_y` columns** (they only make sense in garden city space)
   - Keep `id`, `building_type`, `size`, `geometry`

4. **For `to_file` without `reverse_affine_transformation`**:
   - Keep everything as-is
   - Maybe set CRS to None? Or document that it's shifted?
   - Or add offset metadata to the file attributes?

5. **Warning/Error logic**:
   - If `driver=='GeoJSON'` and `reverse_affine_transformation=False`, warn that coordinates are shifted
   - If `driver=='GeoJSON'` and final CRS is not WGS84, raise error

---

## Conclusion

The current implementation maintains two parallel coordinate systems that are related by a simple affine transformation. This is **correct and necessary** for compatibility with `traj_gen.py`. The confusion arises from:

1. Geometries being in meters (shifted Web Mercator) not abstract units
2. The offset calculation using integer division to align grids
3. The `web_mercator_origin_x/y` attributes being stale

The refactor should respect this dual system and handle export cases appropriately.

