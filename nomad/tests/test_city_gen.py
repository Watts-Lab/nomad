import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from nomad.city_gen import City, RandomCityGenerator

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

    # Silent plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 2))
    city.plot_city(ax, doors=False, address=False)
    plt.close(fig)

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


def test_street_adjacency_edges_smoke():
    rcg = RandomCityGenerator(width=8, height=8, street_spacing=4, seed=2)
    city = rcg.generate_city()
    edges = city.street_adjacency_edges()
    # Basic properties: DataFrame with u and v columns; non-negative count
    assert hasattr(edges, 'columns')
    assert len(edges.columns) > 0  # Check if there are any columns
    assert len(edges) > 0  # Ensure there are edges in the DataFrame


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
    
    # Test with valid street coordinates
    if not city.streets_gdf.empty:
        # Temporarily reset index to access coord_x and coord_y as columns
        streets_temp = city.streets_gdf.reset_index(drop=True)
        street_coords = []
        for _, row in streets_temp.iterrows():
            coord = (int(row['coord_x']), int(row['coord_y']))
            # Check if this coordinate tuple is in the index of streets_gdf
            if coord in city.streets_gdf.index:
                street_coords.append(coord)
            if len(street_coords) >= 2:
                break
        if len(street_coords) >= 2:
            start_coord = street_coords[0]
            end_coord = street_coords[1]
            path = city.get_shortest_path(start_coord, end_coord)
            assert isinstance(path, list)
            assert len(path) > 0
            assert path[0] == start_coord
            assert path[-1] == end_coord
        else:
            print("Skipping shortest path test: Not enough street coordinates found in index")
            return
    else:
        print("Skipping shortest path test: No streets available")
        return
    
    # Test with non-street coordinates (should raise ValueError)
    try:
        invalid_coord = (100, 100)  # Out of bounds
        city.get_shortest_path(start_coord, invalid_coord)
        assert False, "Expected ValueError for out-of-bounds coordinates"
    except ValueError:
        pass

    # Test with non-street block (building)
    building_coords = None
    for idx, row in city.buildings_gdf.iterrows():
        building_coords = (int(row['door_cell_x']), int(row['door_cell_y']))
        break
    if building_coords and building_coords in city.blocks_gdf.index and city.blocks_gdf.loc[building_coords, 'kind'] == 'building':
        try:
            city.get_shortest_path(start_coord, building_coords)
            assert False, "Expected ValueError for non-street block"
        except ValueError:
            pass


def test_add_building_with_gdf_row():
    city = City()
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=2, seed=42)
    city = rcg.generate_city()
    # Create a sample GeoDataFrame row for a building
    from shapely.geometry import Polygon
    geom = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    gdf_row = gpd.GeoDataFrame({
        'id': ['test-building'],
        'type': ['test'],
        'geometry': [geom],
        'door_cell_x': [0],
        'door_cell_y': [0]
    })
    city.add_building(building_type='test', door=(0, 0), gdf_row=gdf_row)
    assert not city.buildings_gdf[city.buildings_gdf['id'] == 'test-building'].empty

def test_add_buildings_from_gdf():
    city = City()
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=2, seed=42)
    city = rcg.generate_city()
    # Create a sample GeoDataFrame with multiple buildings
    from shapely.geometry import Polygon
    # Dynamically select door coordinates from streets_gdf
    street_coords = [(row['coord_x'], row['coord_y']) for _, row in city.streets_gdf.head(2).iterrows()]
    if len(street_coords) < 2:
        print("Not enough street coordinates to test multiple buildings, adding streets")
        for x in range(0, 10, 2):
            for y in range(0, 10, 2):
                city.add_street((x, y))
        city.streets_gdf = city._derive_streets_from_blocks()
        street_coords = [(row['coord_x'], row['coord_y']) for _, row in city.streets_gdf.head(2).iterrows()]
    if len(street_coords) < 2:
        print("Still not enough street coordinates, test may fail")
    geom1 = Polygon([(street_coords[0][0], street_coords[0][1]), (street_coords[0][0], street_coords[0][1]+1), (street_coords[0][0]+1, street_coords[0][1]+1), (street_coords[0][0]+1, street_coords[0][1])])
    geom2 = Polygon([(street_coords[1][0], street_coords[1][1]), (street_coords[1][0], street_coords[1][1]+1), (street_coords[1][0]+1, street_coords[1][1]+1), (street_coords[1][0]+1, street_coords[1][1])])
    gdf = gpd.GeoDataFrame({
        'id': ['test1', 'test2'],
        'type': ['test', 'test'],
        'geometry': [geom1, geom2],
        'door_cell_x': [street_coords[0][0], street_coords[1][0]],
        'door_cell_y': [street_coords[0][1], street_coords[1][1]]
    })
    city.add_buildings_from_gdf(gdf)
    assert len(city.buildings_gdf[city.buildings_gdf['id'].isin(['test1', 'test2'])]) == 2


