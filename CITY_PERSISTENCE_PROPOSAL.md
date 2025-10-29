# City Persistence Enhancement Proposal

## Current State Analysis

### Transformation Functions Found

1. **examples/synthetic-data-generation.ipynb** - `garden_city_to_lat_long()`
   - Block size: 15m
   - Web Mercator offsets: x=-4265699, y=+4392976
   - Converts to EPSG:4326 (lat/lon)

2. **examples/synthetic-data-generation.ipynb** - `philly_to_lat_long()`
   - Block size: 10m  
   - Uses grid_map for custom affine transformation

3. **examples/[3]stop-detection.ipynb** & **examples/[4]home-attribution.ipynb**
   - Manual transformations with same constants (4265699, 4392976)
   - Some use 2x multipliers: `(x - 2*4265699)`, `(y + 2*4392976)`

4. **nomad/traj_gen.py** - `garden_city_to_mercator()`
   - Already implemented in the module
   - Parameters: block_size=15, false_easting=-4265699, false_northing=4392976

### Current City Attributes (Not Persisted)

- `city_boundary`: Polygon geometry
- `buildings_outline`: Polygon geometry (union of all buildings)
- No name attribute
- No block size metadata
- No web mercator origin metadata

## Proposed Changes

### 1. Add New City Attributes (with defaults for backwards compatibility)

```python
class City:
    def __init__(self, 
                 dimensions: tuple = (0,0),
                 manual_streets: bool = False,
                 name: str = "Garden City",
                 block_side_length: float = 15.0,
                 web_mercator_origin_x: float = -4265699.0,
                 web_mercator_origin_y: float = 4392976.0):
        
        self.name = name
        self.block_side_length = block_side_length
        self.web_mercator_origin_x = web_mercator_origin_x
        self.web_mercator_origin_y = web_mercator_origin_y
        # ... existing code ...
```

### 2. Create City Properties Layer

Add a `city_properties` layer to the geopackage with:
- Single row GeoDataFrame
- Columns:
  - `name`: str
  - `block_side_length`: float
  - `web_mercator_origin_x`: float  
  - `web_mercator_origin_y`: float
  - `city_boundary`: geometry (Polygon)
  - `buildings_outline`: geometry (Polygon)

### 3. Update save_geopackage Method

```python
def save_geopackage(self, gpkg_path, persist_blocks: bool = False, 
                    persist_city_properties: bool = True, edges_path: str = None):
    """Save buildings/streets/properties to a GeoPackage.
    
    Parameters
    ----------
    persist_city_properties : bool, default True
        If True, persist city properties (name, block_side_length, 
        web_mercator_origin, city_boundary, buildings_outline)
    """
    # ... existing code ...
    
    if persist_city_properties:
        city_props_gdf = gpd.GeoDataFrame({
            'name': [self.name],
            'block_side_length': [self.block_side_length],
            'web_mercator_origin_x': [self.web_mercator_origin_x],
            'web_mercator_origin_y': [self.web_mercator_origin_y],
            'city_boundary': [self.city_boundary],
            'buildings_outline': [self.buildings_outline],
            'geometry': [self.city_boundary]  # Primary geometry
        }, crs=self.buildings_gdf.crs)
        
        city_props_gdf.to_file(gpkg_path, layer='city_properties', driver='GPKG')
```

### 4. Update from_geopackage Method

```python
@classmethod
def from_geopackage(cls, gpkg_path, edges_path: str = None):
    b_gdf = gpd.read_file(gpkg_path, layer='buildings')
    s_gdf = gpd.read_file(gpkg_path, layer='streets')
    
    # Try to load city properties
    try:
        props_gdf = gpd.read_file(gpkg_path, layer='city_properties')
        if not props_gdf.empty:
            props = props_gdf.iloc[0]
            name = props.get('name', 'Garden City')
            block_side_length = props.get('block_side_length', 15.0)
            web_mercator_origin_x = props.get('web_mercator_origin_x', -4265699.0)
            web_mercator_origin_y = props.get('web_mercator_origin_y', 4392976.0)
            city_boundary = props.get('city_boundary', None)
            buildings_outline = props.get('buildings_outline', None)
        else:
            # Defaults
            name = 'Garden City'
            block_side_length = 15.0
            web_mercator_origin_x = -4265699.0
            web_mercator_origin_y = 4392976.0
            city_boundary = None
            buildings_outline = None
    except Exception:
        # Layer doesn't exist, use defaults for backwards compatibility
        name = 'Garden City'
        block_side_length = 15.0
        web_mercator_origin_x = -4265699.0
        web_mercator_origin_y = 4392976.0
        city_boundary = None
        buildings_outline = None
    
    # ... load blocks, edges ...
    
    city = cls.from_geodataframes(b_gdf, s_gdf, bl_gdf, edges_df)
    
    # Set properties
    city.name = name
    city.block_side_length = block_side_length
    city.web_mercator_origin_x = web_mercator_origin_x
    city.web_mercator_origin_y = web_mercator_origin_y
    
    # Override city_boundary and buildings_outline if loaded from properties
    if city_boundary is not None:
        city.city_boundary = city_boundary
    if buildings_outline is not None:
        city.buildings_outline = buildings_outline
    
    return city
```

### 5. Benefits

1. **Self-documenting**: City files contain all metadata needed for transformations
2. **Backwards compatible**: Old geopackages without properties layer still work with defaults
3. **Flexible**: Different cities can have different block sizes and origins (e.g., Philly with 10m blocks)
4. **Standardized**: Consolidates the scattered transformation constants into city metadata
5. **Complete**: Persists both scalar properties AND geometry properties (boundary, outline)

### 6. Migration Path

1. Update existing `garden-city.gpkg` to include properties layer
2. Update notebooks to read properties from city instead of hardcoding
3. Create utility function on City class:
   ```python
   def get_mercator_transform_params(self):
       """Get parameters for transforming block coords to Web Mercator."""
       return {
           'block_size': self.block_side_length,
           'false_easting': self.web_mercator_origin_x,
           'false_northing': self.web_mercator_origin_y
       }
   ```
4. Potentially add convenience method:
   ```python
   def to_mercator(self, x, y, ha=None):
       """Transform block coordinates to Web Mercator."""
       result_x = self.block_side_length * x + self.web_mercator_origin_x
       result_y = self.block_side_length * y + self.web_mercator_origin_y
       if ha is not None:
           result_ha = self.block_side_length * ha
           return result_x, result_y, result_ha
       return result_x, result_y
   ```

## Testing Requirements

1. Test saving/loading city with properties
2. Test backwards compatibility (loading old geopackages)
3. Test default values
4. Test different parameter values (non-Garden City)
5. Test geometry persistence (city_boundary, buildings_outline)

