import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon, shape
from shapely import wkt
from pathlib import Path

from nomad.city_gen import City, RandomCityGenerator, RasterCity
from nomad.map_utils import blocks_to_mercator, mercator_to_blocks

def _load_sandbox():
    repo_root = Path(__file__).resolve().parents[2]
    gpkg = repo_root / "examples" / "research" / "virtual_philadelphia" / "sandbox" / "sandbox_data.gpkg"
    assert gpkg.exists(), f"Missing sandbox geopackage: {gpkg}"
    buildings = gpd.read_file(gpkg, layer="buildings")
    streets = gpd.read_file(gpkg, layer="streets")
    boundary = gpd.read_file(gpkg, layer="boundary")
    return buildings, streets, boundary.geometry.iloc[0]

def test_city_to_geodataframes_and_persist(tmp_path):
    # Small deterministic city
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=5, seed=1)
    city = rcg.generate_city()

    # Ensure buildings exist
    b_gdf, s_gdf = city.to_geodataframes()
    assert len(b_gdf) > 0

    # Convert to GeoDataFrames
    b_gdf, s_gdf = city.to_geodataframes()
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

    b1, s1 = city.to_geodataframes()
    b2, s2 = city2.to_geodataframes()
    assert len(b1) == len(b2)
    assert len(s1) == len(s2)


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
    b1, s1 = city.to_geodataframes()
    b2, s2 = city2.to_geodataframes()
    assert len(b1) == len(b2)
    assert len(s1) == len(s2)


def test_shortest_path():
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=5, seed=1)
    city = rcg.generate_city()
    
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
    assert city.blocks_gdf.loc[building_door, 'kind'] == 'street', "Building door must be on a street"
    
    # Find an actual building block to test routing error
    building_blocks = city.blocks_gdf[city.blocks_gdf['kind'] == 'building']
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
    buildings, streets, boundary = _load_sandbox()
    buildings = buildings.head(100)
    
    city = RasterCity(boundary, streets, buildings, block_size=15.0)
    
    city._build_hub_network(hub_size=16)
    city.compute_gravity(exponent=2.0)
    
    n_buildings = len(city.buildings_gdf)
    assert city.grav.shape == (n_buildings, n_buildings)
    
    mask = ~np.eye(n_buildings, dtype=bool)
    assert (city.grav.values[mask] > 0).all()


