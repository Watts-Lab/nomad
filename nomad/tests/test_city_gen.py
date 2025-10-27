import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from nomad.city_gen import City, RandomCityGenerator


def test_city_to_geodataframes_and_persist(tmp_path):
    # Small deterministic city
    rcg = RandomCityGenerator(width=10, height=10, street_spacing=5, seed=1)
    city = rcg.generate_city()

    # Ensure buildings exist
    assert isinstance(city.buildings, dict)
    assert len(city.buildings) > 0

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


