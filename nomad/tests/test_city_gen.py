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
    assert 'door_x' in b_gdf.columns and 'door_y' in b_gdf.columns
    assert 'door_point' in b_gdf.columns

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
    assert set(['u','v']).issubset(edges.columns)
    assert len(edges) >= 0


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
    start_coord = (0, 0)
    end_coord = (5, 5)
    path = city.get_shortest_path(start_coord, end_coord)
    assert isinstance(path, list)
    assert len(path) > 0
    assert path[0] == start_coord
    assert path[-1] == end_coord
    
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
    if building_coords:
        try:
            city.get_shortest_path(start_coord, building_coords)
            assert False, "Expected ValueError for non-street block"
        except ValueError:
            pass


