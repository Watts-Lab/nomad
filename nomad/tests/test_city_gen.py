import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon, shape
from shapely import wkt
from pathlib import Path

from nomad.city_gen import City, RandomCityGenerator, RasterCity
from nomad.map_utils import blocks_to_mercator, mercator_to_blocks

def _load_fixture():
    """Load the test fixture from nomad/data/city_fixture.gpkg"""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    gpkg = data_dir / "city_fixture.gpkg"
    assert gpkg.exists(), f"Missing test fixture: {gpkg}. Run 'python nomad/data/generate_fixture.py' to create it."
    buildings = gpd.read_file(gpkg, layer="buildings")
    streets = gpd.read_file(gpkg, layer="streets")
    boundary = gpd.read_file(gpkg, layer="boundary")
    return buildings, streets, boundary.geometry.iloc[0]

def test_city_to_geodataframes_and_persist(tmp_path):
    # Small deterministic city
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=5, seed=1)
    city = rcg.generate_city()

    # Ensure buildings exist
    b_gdf = city.buildings_gdf
    s_gdf = city.streets_gdf
    assert len(b_gdf) > 0

    # Verify GeoDataFrames
    assert isinstance(b_gdf, gpd.GeoDataFrame)
    assert isinstance(s_gdf, gpd.GeoDataFrame)
    assert 'geometry' in b_gdf.columns and 'geometry' in s_gdf.columns

    # Persist silently
    b_path = tmp_path / 'buildings.geojson'
    s_path = tmp_path / 'streets.geojson'
    city.to_file(buildings_path=str(b_path), streets_path=str(s_path))

    # Read back
    b_back = gpd.read_file(b_path)
    s_back = gpd.read_file(s_path)
    assert len(b_back) == len(b_gdf)
    assert len(s_back) == len(s_gdf)

    # Extra checks: door columns present and a silent plot call
    assert 'door_point' in b_gdf.columns
    assert 'door_cell_x' in b_gdf.columns and 'door_cell_y' in b_gdf.columns


def test_geopackage_roundtrip(tmp_path):
    rcg = RandomCityGenerator(width=6, height=6, street_spacing=3, seed=1)
    city = rcg.generate_city()
    gpkg = tmp_path / 'city.gpkg'
    city.save_geopackage(str(gpkg))
    city2 = City.from_geopackage(str(gpkg))

    assert len(city.buildings_gdf) == len(city2.buildings_gdf)
    assert len(city.streets_gdf) == len(city2.streets_gdf)


def test_street_graph_connectivity():
    rcg = RandomCityGenerator(width=8, height=8, street_spacing=4, seed=2)
    city = rcg.generate_city()
    G = city.get_street_graph()
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0


def test_from_geodataframes_roundtrip_nonsquare(tmp_path):
    rcg = RandomCityGenerator(width=10, height=6, street_spacing=5, seed=3)
    city = rcg.generate_city()
    gpkg = tmp_path / 'ns_city.gpkg'
    city.save_geopackage(str(gpkg))
    city2 = City.from_geopackage(str(gpkg))
    assert len(city.buildings_gdf) == len(city2.buildings_gdf)
    assert len(city.streets_gdf) == len(city2.streets_gdf)


def test_shortest_path():
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=5, seed=1)
    city = rcg.generate_city()
    city.compute_shortest_paths(callable_only=False)
    
    # Get two street coordinates
    streets_temp = city.streets_gdf.reset_index(drop=True)
    street_coords = []
    for _, row in streets_temp.iterrows():
        coord = (int(row['coord_x']), int(row['coord_y']))
        if coord in city.streets_gdf.index:
            street_coords.append(coord)
        if len(street_coords) >= 2:
            break
    
    start_coord = street_coords[0]
    end_coord = street_coords[1]
    path = city.get_shortest_path(start_coord, end_coord)
    assert isinstance(path, list)
    assert len(path) > 0
    assert path[0] == start_coord
    assert path[-1] == end_coord
    
    # Test with non-street coordinates (should raise ValueError)
    try:
        invalid_coord = (100, 100)
        city.get_shortest_path(start_coord, invalid_coord)
        assert False, "Expected ValueError for out-of-bounds coordinates"
    except ValueError:
        pass

    # Test with non-street block (building door cell should be a street)
    first_building = city.buildings_gdf.iloc[0]
    building_door = (int(first_building['door_cell_x']), int(first_building['door_cell_y']))
    assert building_door in city.blocks_gdf.index, "Building door must be in blocks_gdf"
    assert city.blocks_gdf.loc[building_door, 'building_type'] == 'street', "Building door must be on a street"
    
    # Find an actual building block to test routing error
    building_blocks = city.blocks_gdf[(city.blocks_gdf['building_type'].notna()) & (city.blocks_gdf['building_type'] != 'street')]
    assert not building_blocks.empty, "RandomCityGenerator must create building blocks"
    building_block = building_blocks.index[0]
    try:
        city.get_shortest_path(start_coord, building_block)
        assert False, "Expected ValueError for non-street block"
    except ValueError:
        pass


def test_add_building_with_gdf_row():
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=2, seed=42)
    city = rcg.generate_city()
    # Create a building adjacent to a street cell, not overlapping the door
    geom = Polygon([(1, 0), (1, 1), (2, 1), (2, 0)])
    gdf_row = gpd.GeoDataFrame({
        'id': ['test-building'],
        'building_type': ['test'],
        'geometry': [geom],
        'door_cell_x': [0],
        'door_cell_y': [0]
    })
    city.add_building(building_type='test', door=(0, 0), gdf_row=gdf_row)
    assert not city.buildings_gdf[city.buildings_gdf['id'] == 'test-building'].empty

def test_add_buildings_from_gdf():
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=2, seed=42)
    city = rcg.generate_city()
    # Create two non-overlapping buildings, each adjacent to a different street cell
    # Building 1: adjacent to street (0,0), occupies cells (1,0)-(2,1)
    geom1 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0)])
    # Building 2: adjacent to street (0,2), occupies cells (1,2)-(2,3)
    geom2 = Polygon([(1, 2), (1, 3), (2, 3), (2, 2)])
    gdf = gpd.GeoDataFrame({
        'id': ['test1', 'test2'],
        'building_type': ['test', 'test'],
        'geometry': [geom1, geom2],
        'door_cell_x': [0, 0],
        'door_cell_y': [0, 2]
    })
    city.add_buildings_from_gdf(gdf)
    assert len(city.buildings_gdf[city.buildings_gdf['id'].isin(['test1', 'test2'])]) == 2


def test_add_building_overlap_error():
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=2, seed=42)
    city = rcg.generate_city()
    # Add first building
    geom1 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0)])
    city.add_building(building_type='test', door=(0, 0), geom=geom1)
    # Try to add overlapping building
    geom2 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0)])  # Same location
    try:
        city.add_building(building_type='test', door=(0, 0), geom=geom2)
        assert False, "Expected ValueError for overlapping building"
    except ValueError as e:
        assert "overlaps" in str(e).lower()


def test_coordinate_roundtrip(tmp_path):
    """
    Test coordinate transformation roundtrip: blocks -> mercator -> blocks.
    Ensures that transforming coordinates and transforming back yields the original values.
    """
    # Create test data in block coordinates
    df = pd.DataFrame({
        'x': [0.0, 1.5, 10.0, 25.3],
        'y': [0.0, 2.5, 15.0, 30.7],
        'ha': [0.5, 0.75, 1.0, 1.5],
        'other_col': ['a', 'b', 'c', 'd']
    })
    
    # Test with default Garden City parameters
    mercator_df = blocks_to_mercator(df)
    
    # Check transformation happened
    assert mercator_df['x'].iloc[0] == -4265699.0
    assert mercator_df['y'].iloc[0] == 4392976.0
    assert mercator_df['ha'].iloc[0] == 7.5  # 0.5 * 15
    
    # Check other columns preserved
    assert mercator_df['other_col'].tolist() == ['a', 'b', 'c', 'd']
    
    # Transform back
    blocks_df = mercator_to_blocks(mercator_df)
    
    # Check roundtrip accuracy (within floating point precision)
    np.testing.assert_allclose(blocks_df['x'].values, df['x'].values, rtol=1e-10)
    np.testing.assert_allclose(blocks_df['y'].values, df['y'].values, rtol=1e-10)
    np.testing.assert_allclose(blocks_df['ha'].values, df['ha'].values, rtol=1e-10)
    
    # Test with custom parameters (e.g., Philly with 10m blocks)
    mercator_df2 = blocks_to_mercator(df, block_size=10.0, false_easting=-8000000, false_northing=4800000)
    blocks_df2 = mercator_to_blocks(mercator_df2, block_size=10.0, false_easting=-8000000, false_northing=4800000)
    
    np.testing.assert_allclose(blocks_df2['x'].values, df['x'].values, rtol=1e-10)
    np.testing.assert_allclose(blocks_df2['y'].values, df['y'].values, rtol=1e-10)


def test_city_persistence_with_properties(tmp_path):
    """
    Test saving and loading a city with properties layer.
    Ensures city metadata (name, block_side_length, mercator origins) are persisted and restored.
    """
    # Create a city with custom properties
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=5, seed=42)
    city = rcg.generate_city()
    
    # Set custom properties
    city.name = "Test City"
    city.block_side_length = 12.5
    city.web_mercator_origin_x = -5000000.0
    city.web_mercator_origin_y = 5000000.0
    
    # Save to geopackage
    gpkg_path = tmp_path / "test_city.gpkg"
    city.save_geopackage(str(gpkg_path), persist_city_properties=True)
    
    # Load back
    city2 = City.from_geopackage(str(gpkg_path))
    
    # Verify properties were preserved
    assert city2.name == "Test City"
    assert city2.block_side_length == 12.5
    assert city2.web_mercator_origin_x == -5000000.0
    assert city2.web_mercator_origin_y == 5000000.0
    
    # Verify geometries were preserved (compare bounds since geometries may be loaded differently)
    # Handle case where geometry might be WKT string
    if isinstance(city2.city_boundary, str):
        city2_boundary = wkt.loads(city2.city_boundary)
    else:
        city2_boundary = city2.city_boundary
    
    if isinstance(city2.buildings_outline, str):
        city2_outline = wkt.loads(city2.buildings_outline)
    else:
        city2_outline = city2.buildings_outline
    
    # Compare geometries (using equals_exact for floating point tolerance)
    assert city2_boundary.equals_exact(city.city_boundary, tolerance=1e-6)
    assert city2_outline.equals_exact(city.buildings_outline, tolerance=1e-6)
    
    # Verify buildings and streets were preserved
    assert len(city2.buildings_gdf) == len(city.buildings_gdf)
    assert len(city2.streets_gdf) == len(city.streets_gdf)
    
    # Test City.to_mercator and City.from_mercator methods
    test_data = pd.DataFrame({
        'x': [0.0, 1.0, 2.0],
        'y': [0.0, 1.0, 2.0]
    })
    
    mercator_data = city2.to_mercator(test_data)
    
    # Should use city's custom parameters
    assert mercator_data['x'].iloc[0] == -5000000.0  # 12.5 * 0 + (-5000000)
    assert mercator_data['y'].iloc[0] == 5000000.0   # 12.5 * 0 + 5000000
    assert mercator_data['x'].iloc[1] == -5000000.0 + 12.5  # 12.5 * 1 + (-5000000)
    
    # Test roundtrip through city methods
    blocks_data = city2.from_mercator(mercator_data)
    np.testing.assert_allclose(blocks_data['x'].values, test_data['x'].values, rtol=1e-10)
    np.testing.assert_allclose(blocks_data['y'].values, test_data['y'].values, rtol=1e-10)
    
    # Test backwards compatibility: save without properties, load with defaults
    gpkg_path2 = tmp_path / "test_city_no_props.gpkg"
    city.save_geopackage(str(gpkg_path2), persist_city_properties=False)
    
    city3 = City.from_geopackage(str(gpkg_path2))
    
    # Should have default values
    assert city3.name == "Garden City"
    assert city3.block_side_length == 15.0
    assert city3.web_mercator_origin_x == -4265699.0
    assert city3.web_mercator_origin_y == 4392976.0


def test_compute_gravity():
    buildings, streets, boundary = _load_fixture()
    buildings = buildings.head(100)
    
    city = RasterCity(boundary, streets, buildings, block_side_length=15.0)
    
    city._build_hub_network(hub_size=16)
    city.compute_gravity(exponent=2.0)
    
    n_buildings = len(city.buildings_gdf)
    assert city.grav.shape == (n_buildings, n_buildings)
    
    mask = ~np.eye(n_buildings, dtype=bool)
    assert (city.grav.values[mask] > 0).all()


def test_compute_gravity_callable():
    buildings, streets, boundary = _load_fixture()
    buildings = buildings.head(100)
    
    city_dense = RasterCity(boundary, streets, buildings, block_side_length=15.0)
    city_dense._build_hub_network(hub_size=16)
    city_dense.compute_gravity(exponent=2.0, callable_only=False)
    
    city_callable = RasterCity(boundary, streets, buildings, block_side_length=15.0)
    city_callable._build_hub_network(hub_size=16)
    city_callable.compute_gravity(exponent=2.0, callable_only=True)
    
    assert callable(city_callable.grav)
    
    rng = np.random.default_rng(42)
    test_buildings = rng.choice(city_dense.buildings_gdf['id'].values, size=5, replace=False)
    
    for bid in test_buildings:
        dense_row = city_dense.grav.loc[bid].values
        callable_row = city_callable.grav(bid).values
        assert np.allclose(dense_row, callable_row, atol=1e-5)


def test_resolve_overlaps():
    buildings, streets, boundary = _load_fixture()
    buildings = buildings.head(100)
    
    city_default = RasterCity(boundary, streets, buildings, block_side_length=15.0, resolve_overlaps=False)
    city_resolved = RasterCity(boundary, streets, buildings, block_side_length=15.0, resolve_overlaps=True)
    
    n_default = len(city_default.buildings_gdf)
    n_resolved = len(city_resolved.buildings_gdf)
    
    assert n_resolved >= n_default
    assert n_resolved > 0
    assert n_default > 0


def test_gravity_persistence_with_load(tmp_path):
    """
    Test saving and loading gravity infrastructure with load_gravity=True.
    """
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=5, seed=42)
    city = rcg.generate_city()
    
    city._build_hub_network(hub_size=50)
    city.compute_gravity(exponent=2.0, callable_only=True)
    
    gpkg_path = tmp_path / "test_city_gravity.gpkg"
    city.save_geopackage(str(gpkg_path), persist_gravity_data=True)
    
    city2 = City.from_geopackage(str(gpkg_path), load_gravity=True)
    
    assert hasattr(city2, 'hubs')
    assert hasattr(city2, 'hub_df')
    assert hasattr(city2, 'grav_hub_info')
    assert hasattr(city2, 'mh_dist_nearby_doors')
    assert hasattr(city2, 'grav')
    assert callable(city2.grav)
    
    assert len(city2.hubs) == len(city.hubs)
    assert city2.hub_df.shape == city.hub_df.shape
    assert len(city2.grav_hub_info) == len(city.grav_hub_info)
    assert len(city2.mh_dist_nearby_doors) == len(city.mh_dist_nearby_doors)
    
    building_id = city.buildings_gdf['id'].iloc[0]
    grav_row_orig = city.grav(building_id)
    grav_row_loaded = city2.grav(building_id)
    
    pd.testing.assert_series_equal(grav_row_orig, grav_row_loaded)


def test_gravity_persistence_without_load(tmp_path):
    """
    Test saving gravity infrastructure but loading without it (load_gravity=False).
    """
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=5, seed=42)
    city = rcg.generate_city()
    
    city._build_hub_network(hub_size=50)
    city.compute_gravity(exponent=2.0, callable_only=True)
    
    gpkg_path = tmp_path / "test_city_no_gravity.gpkg"
    city.save_geopackage(str(gpkg_path), persist_gravity_data=True)
    
    city2 = City.from_geopackage(str(gpkg_path), load_gravity=False)
    
    assert not hasattr(city2, 'grav') or city2.grav is None or not callable(city2.grav)
    assert not hasattr(city2, 'hubs') or city2.hubs is None
    assert not hasattr(city2, 'hub_df') or city2.hub_df is None
    assert not hasattr(city2, 'grav_hub_info') or city2.grav_hub_info is None


def test_gravity_persistence_no_data(tmp_path):
    """
    Test loading a geopackage without gravity data.
    """
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=5, seed=42)
    city = rcg.generate_city()
    
    gpkg_path = tmp_path / "test_city_no_grav_data.gpkg"
    city.save_geopackage(str(gpkg_path), persist_gravity_data=False)
    
    city2 = City.from_geopackage(str(gpkg_path), load_gravity=True)
    
    assert not hasattr(city2, 'grav') or city2.grav is None or not callable(city2.grav)


def test_unique_building_ids():
    """
    Regression test: RasterCity must generate unique building IDs.
    Building IDs are based on an internal building block adjacent to the door,
    ensuring uniqueness since buildings cannot overlap.
    """
    buildings_gdf, streets_gdf, boundary = _load_fixture()
    
    city = RasterCity(
        boundary_polygon=boundary,
        streets_gdf=streets_gdf,
        buildings_gdf=buildings_gdf,
        block_side_length=10.0,
        resolve_overlaps=True,
        verbose=False
    )
    
    building_ids = city.buildings_gdf['id'].tolist()
    unique_ids = set(building_ids)
    
    assert len(building_ids) == len(unique_ids), \
        f"Found {len(building_ids) - len(unique_ids)} duplicate building IDs"
    assert city.buildings_gdf.index.is_unique, "buildings_gdf index contains duplicates"


def test_gravity_same_door_buildings():
    """
    Test that gravity computation handles buildings with the same door correctly.
    Buildings sharing a door have Manhattan distance 0, which should be set to 1 to avoid inf gravity.
    """
    buildings_gdf, streets_gdf, boundary = _load_fixture()
    
    city = RasterCity(boundary, streets_gdf, buildings_gdf, block_side_length=15.0)
    
    # First, try to find two buildings that naturally share a door
    door_counts = city.buildings_gdf.groupby(['door_cell_x', 'door_cell_y']).size()
    shared_doors = door_counts[door_counts > 1]
    
    if len(shared_doors) > 0:
        # Found buildings with shared door
        door_x, door_y = shared_doors.index[0]
        same_door_buildings = city.buildings_gdf[
            (city.buildings_gdf['door_cell_x'] == door_x) & 
            (city.buildings_gdf['door_cell_y'] == door_y)
        ]
        bids = same_door_buildings['id'].tolist()[:2]
    else:
        # Create two buildings with the same door by finding a street block adjacent to two buildings
        if len(city.buildings_gdf) < 2:
            return  # Not enough buildings to test
        
        # Find a street block adjacent to at least two different buildings
        street_coords = set(zip(city.streets_gdf['coord_x'], city.streets_gdf['coord_y']))
        building_coords = {}
        for idx, row in city.buildings_gdf.iterrows():
            geom = row['geometry']
            minx, miny, maxx, maxy = geom.bounds
            coords = [(int(x), int(y)) for x in range(int(minx), int(maxx)+1) 
                      for y in range(int(miny), int(maxy)+1)]
            building_coords[row['id']] = set(coords)
        
        # Find a street block adjacent to multiple buildings
        shared_door = None
        adjacent_buildings = []
        for sx, sy in street_coords:
            neighbors = [(sx+1, sy), (sx-1, sy), (sx, sy+1), (sx, sy-1)]
            adj_bids = []
            for bid, coords in building_coords.items():
                if any(n in coords for n in neighbors):
                    adj_bids.append(bid)
            if len(adj_bids) >= 2:
                shared_door = (sx, sy)
                adjacent_buildings = adj_bids[:2]
                break
        
        if shared_door is None:
            return  # Could not create test scenario
        
        # Assign the shared door to both buildings
        bids = adjacent_buildings
        for bid in bids:
            city.buildings_gdf.loc[bid, 'door_cell_x'] = shared_door[0]
            city.buildings_gdf.loc[bid, 'door_cell_y'] = shared_door[1]
            city.buildings_gdf.loc[bid, 'door_point'] = Point(shared_door[0] + 0.5, shared_door[1] + 0.5)
    
    city._build_hub_network(hub_size=16)
    
    # Test callable gravity
    city.compute_gravity(exponent=2.0, callable_only=True)
    grav_row = city.grav(bids[0])
    assert not np.isinf(grav_row).any(), "Callable gravity contains inf values"
    assert not np.isnan(grav_row).any(), "Callable gravity contains NaN values"
    assert grav_row.loc[bids[1]] > 0, "Gravity between same-door buildings should be positive"
    
    # Test dense gravity
    city.compute_gravity(exponent=2.0, callable_only=False)
    assert not np.isinf(city.grav.values).any(), "Dense gravity contains inf values"
    assert not np.isnan(city.grav.values).any(), "Dense gravity contains NaN values"
    assert city.grav.loc[bids[0], bids[1]] > 0, "Gravity between same-door buildings should be positive"


def test_to_file_reverse_affine_transformation(tmp_path):
    """Test that to_file correctly reverses affine transformations."""
    from nomad.city_gen import RandomCityGenerator
    
    # Create a simple deterministic city
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=5, seed=42)
    city = rcg.generate_city()
    
    # Set known offset and rotation for testing
    city.offset_x = 100
    city.offset_y = 200
    city.rotation_deg = 15.0
    
    # Export without reverse transformation (garden city units)
    buildings_gc = tmp_path / "buildings_gc.gpkg"
    city.to_file(buildings_path=str(buildings_gc), driver='GPKG')
    
    # Read back
    buildings_loaded_gc = gpd.read_file(buildings_gc)
    
    # Verify garden city coordinates are preserved
    assert len(buildings_loaded_gc) > 0, "Should have buildings"
    # Note: door_cell_x/y columns exist
    
    # Export with reverse transformation
    buildings_wm = tmp_path / "buildings_wm.gpkg"
    streets_wm = tmp_path / "streets_wm.gpkg"
    
    city.to_file(
        buildings_path=str(buildings_wm),
        streets_path=str(streets_wm),
        driver='GPKG',
        reverse_affine=True
    )
    
    # Read back
    buildings_loaded_wm = gpd.read_file(buildings_wm)
    streets_loaded_wm = gpd.read_file(streets_wm)
    
    # Verify transformations
    assert len(buildings_loaded_wm) == len(city.buildings_gdf), "Building count mismatch"
    assert len(streets_loaded_wm) > 0, "Streets should be exported"
    
    # Verify garden city columns were dropped
    assert 'door_cell_x' not in buildings_loaded_wm.columns, "door_cell_x should be dropped"
    assert 'door_cell_y' not in buildings_loaded_wm.columns, "door_cell_y should be dropped"
    
    # Verify CRS is Web Mercator
    assert buildings_loaded_wm.crs.to_epsg() == 3857, "Buildings should be in Web Mercator"
    assert streets_loaded_wm.crs.to_epsg() == 3857, "Streets should be in Web Mercator"
    
    # Verify geometries are scaled and translated
    # A building at (2, 3) in garden city should be at:
    # (2 * 15 + 100 * 15, 3 * 15 + 200 * 15) = (1530, 3045) in Web Mercator
    # Just verify the scale factor is applied
    gc_area = city.buildings_gdf.geometry.iloc[0].area  # in block units^2
    wm_area = buildings_loaded_wm.geometry.iloc[0].area  # in meters^2
    expected_area = gc_area * (city.block_side_length ** 2)
    
    assert abs(wm_area - expected_area) < 1.0, \
        f"Area scaling incorrect: {wm_area} vs expected {expected_area}"


