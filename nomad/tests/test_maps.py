"""
Tests for the map_utils module - OpenStreetMap data utilities.

Tests verify:
1. Buildings can be downloaded from OSM
2. Categorization works with multiple schemas (garden_city, geolife_plus)
3. Output includes OSM type, subtype, and category
4. Geometries are proper polygons
5. Rotation and explosion of geometries works correctly
"""

import pytest
import geopandas as gpd
from shapely.geometry import Polygon, LineString, MultiPolygon, MultiLineString
import numpy as np

from nomad.map_utils import (
    download_osm_buildings,
    download_osm_streets,
    get_category_summary,
    get_subtype_summary,
    get_osm_type_summary,
    rotate_and_explode
)


@pytest.fixture
def philly_bbox():
    """Bounding box for a section of Philadelphia."""
    return (
    -75.16620602,  # west
    39.941158234,  # south
    -75.14565573,  # east
    39.955720193   # north
)


def test_download_osm_buildings_garden_city(philly_bbox):
    """
    Test the main workflow with garden_city schema.
    
    Verifies that:
    - We can download real building data from OpenStreetMap
    - Buildings are returned as proper polygon geometries
    - Buildings have osm_type, subtype, and category columns
    """
    buildings = download_osm_buildings(philly_bbox, schema='garden_city')

    # Should return a GeoDataFrame
    assert isinstance(buildings, gpd.GeoDataFrame)

    # Should have all three classification columns
    assert 'osm_type' in buildings.columns
    assert 'subtype' in buildings.columns
    assert 'category' in buildings.columns
    
    # Should have found some buildings in this area
    assert len(buildings) > 0, "No buildings found in the test area"
    
    # All geometries should be polygons (building footprints)
    geometry_types = buildings.geometry.geom_type.unique()
    assert all(
        geom_type in ['Polygon', 'MultiPolygon'] 
        for geom_type in geometry_types
    ), f"Found non-polygon geometries: {geometry_types}"
    
    # All categories should be valid garden city types
    valid_categories = {'residential', 'retail', 'workplace', 'park', 'other'}
    actual_categories = set(buildings['category'].unique())
    assert actual_categories.issubset(valid_categories), (
        f"Found invalid categories: {actual_categories - valid_categories}"
    )
    
    # OSM types and subtypes should exist and not be null
    assert buildings['osm_type'].notna().all(), "Found null osm_types"
    assert buildings['subtype'].notna().all(), "Found null subtypes"
    
    # Should have at least residential in Philadelphia
    assert 'residential' in actual_categories, (
        "Expected to find residential buildings in Philadelphia"
    )


def test_download_osm_buildings_geolife(philly_bbox):
    """Test with geolife_plus category schema."""
    buildings = download_osm_buildings(philly_bbox, schema='geolife_plus')

    # Should return a GeoDataFrame
    assert isinstance(buildings, gpd.GeoDataFrame)

    # Should have all three classification columns
    assert 'osm_type' in buildings.columns
    assert 'subtype' in buildings.columns
    assert 'category' in buildings.columns

    # All categories should be valid geolife_plus types
    valid_categories = {'unknown', 'residential', 'commercial', 'school'}
    actual_categories = set(buildings['category'].unique())
    assert actual_categories.issubset(valid_categories), (
        f"Found invalid categories: {actual_categories - valid_categories}"
    )


def test_download_osm_buildings_includes_parks(philly_bbox):
    """
    Test that download_osm_buildings includes both buildings and parks.
    
    Verifies that buildings and parks are combined into
    a single GeoDataFrame with consistent categorization.
    """
    all_features = download_osm_buildings(philly_bbox, schema='garden_city')

    # Should return a GeoDataFrame
    assert isinstance(all_features, gpd.GeoDataFrame)

    # Should have all three columns
    assert 'osm_type' in all_features.columns
    assert 'subtype' in all_features.columns
    assert 'category' in all_features.columns
    
    # Should have found features
    assert len(all_features) > 0

    # All categories should be valid
    valid_categories = {'residential', 'retail', 'workplace', 'park', 'other'}
    actual_categories = set(all_features['category'].unique())
    assert actual_categories.issubset(valid_categories)


def test_category_summary():
    """Test the category summary utility function."""
    # Create a simple test GeoDataFrame
    test_data = gpd.GeoDataFrame({
        'geometry': [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),
        ],
        'osm_type': ['house', 'supermarket', 'house'],
        'subtype': ['residential', 'commercial', 'residential'],
        'category': ['residential', 'retail', 'residential']
    }, crs="EPSG:4326")
    
    summary = get_category_summary(test_data)
    
    # Should return a dictionary with counts
    assert isinstance(summary, dict)
    assert summary == {'residential': 2, 'retail': 1}
    
    # Should raise error for uncategorized data
    uncategorized = gpd.GeoDataFrame({
        'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        'building': ['house']
    }, crs="EPSG:4326")

    with pytest.raises(ValueError, match="Features must be categorized"):
        get_category_summary(uncategorized)


def test_subtype_summary():
    """Test the subtype summary utility function."""
    test_data = gpd.GeoDataFrame({
        'geometry': [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),
        ],
        'osm_type': ['house', 'supermarket', 'hospital'],
        'subtype': ['residential', 'commercial', 'medical'],
        'category': ['residential', 'retail', 'workplace']
    }, crs="EPSG:4326")
    
    summary = get_subtype_summary(test_data)
    
    # Should return a dictionary with counts
    assert isinstance(summary, dict)
    assert summary == {'residential': 1, 'commercial': 1, 'medical': 1}


def test_osm_type_summary():
    """Test the OSM type summary utility function (most granular)."""
    test_data = gpd.GeoDataFrame({
        'geometry': [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),
        ],
        'osm_type': ['house', 'supermarket', 'house'],
        'subtype': ['residential', 'commercial', 'residential'],
        'category': ['residential', 'retail', 'residential']
    }, crs="EPSG:4326")
    
    summary = get_osm_type_summary(test_data)
    
    # Should return a dictionary with counts
    assert isinstance(summary, dict)
    assert summary == {'house': 2, 'supermarket': 1}


def test_categorization_logic_garden_city():
    """Test that categorization returns all three levels with garden_city schema."""
    from nomad.map_utils import _classify_building
    import pandas as pd
    
    # Test building tag priority - should return (osm_type, subtype, category)
    assert _classify_building(pd.Series({'building': 'house'}), 'garden_city') == ('house', 'residential', 'residential')
    assert _classify_building(pd.Series({'building': 'hospital'}), 'garden_city') == ('hospital', 'medical', 'workplace')
    assert _classify_building(pd.Series({'building': 'supermarket'}), 'garden_city') == ('supermarket', 'commercial', 'retail')
    
    # Test amenity tag (when no building tag)
    assert _classify_building(pd.Series({'amenity': 'school'}), 'garden_city') == ('school', 'education', 'workplace')
    assert _classify_building(pd.Series({'amenity': 'restaurant'}), 'garden_city') == ('restaurant', 'commercial', 'retail')
    
    # Test place of worship → retail
    assert _classify_building(pd.Series({'amenity': 'place_of_worship'}), 'garden_city') == ('place_of_worship', 'religious', 'retail')
    
    # Test parking → park
    assert _classify_building(pd.Series({'amenity': 'parking'}), 'garden_city') == ('parking', 'parking', 'park')
    
    # Test shop tag (preserves shop type in osm_type)
    result = _classify_building(pd.Series({'shop': 'bakery'}), 'garden_city')
    assert result[0] == 'commercial_bakery'  # osm_type preserves shop type
    assert result[1] == 'commercial'  # subtype
    assert result[2] == 'retail'  # category


def test_categorization_logic_geolife():
    """Test that categorization works correctly with geolife_plus schema."""
    from nomad.map_utils import _classify_building
    import pandas as pd
    
    # Residential stays residential
    assert _classify_building(pd.Series({'building': 'house'}), 'geolife_plus') == ('house', 'residential', 'residential')
    
    # Schools map to 'school' category
    assert _classify_building(pd.Series({'amenity': 'school'}), 'geolife_plus') == ('school', 'education', 'school')
    
    # Commercial buildings map to 'commercial'
    assert _classify_building(pd.Series({'building': 'supermarket'}), 'geolife_plus') == ('supermarket', 'commercial', 'commercial')
    
    # Medical buildings map to 'unknown'
    assert _classify_building(pd.Series({'building': 'hospital'}), 'geolife_plus') == ('hospital', 'medical', 'unknown')
    
    # Unknown types map to 'unknown'
    assert _classify_building(pd.Series({'unknown': 'value'}), 'geolife_plus') == ('unknown', 'unknown', 'unknown')


def test_priority_order():
    """Test that building tag takes priority over other tags."""
    from nomad.map_utils import _classify_building
    import pandas as pd
    
    # Building tag should win over amenity
    result = _classify_building(pd.Series({
        'building': 'residential',
        'amenity': 'school'
    }), 'garden_city')
    assert result == ('residential', 'residential', 'residential')  # building tag wins


def test_crs_handling(philly_bbox):
    """Test that coordinate reference systems are handled correctly."""
    # Default should be EPSG:4326 (WGS84 lat/lon)
    buildings_default = download_osm_buildings(philly_bbox)
    assert buildings_default.crs.to_string() == "EPSG:4326"
    
    # Should support custom CRS
    buildings_mercator = download_osm_buildings(philly_bbox, crs="EPSG:3857")
    assert buildings_mercator.crs.to_string() == "EPSG:3857"


def test_download_osm_streets(philly_bbox):
    """
    Test downloading street network from OSM.
    
    Verifies that:
    - Streets can be downloaded from OpenStreetMap
    - Streets are returned as LineString geometries
    - Excluded types (parking aisles, tunnels, etc.) are filtered out
    """
    streets = download_osm_streets(philly_bbox)
    
    # Should return a GeoDataFrame
    assert isinstance(streets, gpd.GeoDataFrame)
    
    # Should have geometry column
    assert 'geometry' in streets.columns
    
    # Should have found some streets in this area
    assert len(streets) > 0, "No streets found in the test area"
    
    # All geometries should be LineStrings or MultiLineStrings
    geometry_types = streets.geometry.geom_type.unique()
    assert all(
        geom_type in ['LineString', 'MultiLineString'] 
        for geom_type in geometry_types
    ), f"Found non-line geometries: {geometry_types}"
    
    # Should have highway column with valid types
    assert 'highway' in streets.columns
    valid_highway_types = {
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
        'unclassified', 'residential', 'living_street', 'service'
    }
    actual_highway_types = set(streets['highway'].unique())
    assert actual_highway_types.issubset(valid_highway_types), (
        f"Found unexpected highway types: {actual_highway_types - valid_highway_types}"
    )
    
    # Should not have excluded service types
    if 'service' in streets.columns:
        excluded_services = streets['service'].isin(['parking_aisle', 'driveway'])
        assert not excluded_services.any(), "Found excluded service types (parking_aisle, driveway)"
    
    # Should not have tunnels or covered ways
    if 'tunnel' in streets.columns:
        assert not (streets['tunnel'] == 'yes').any(), "Found tunnel streets"
    if 'covered' in streets.columns:
        assert not (streets['covered'] == 'yes').any(), "Found covered streets"


def test_rotate_and_explode_buildings():
    """Test rotating and exploding building geometries."""
    # Create test data with multipolygons
    poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    poly2 = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])
    multipoly = MultiPolygon([poly1, poly2])
    
    test_gdf = gpd.GeoDataFrame({
        'id': [1, 2],
        'geometry': [multipoly, poly1]
    }, crs="EPSG:4326")
    
    # Test without rotation
    result = rotate_and_explode(test_gdf, rotation_deg=0)
    
    # Should explode multipolygon into separate polygons
    assert len(result) == 3, "Should have 3 polygons after explosion"
    assert all(result.geometry.geom_type == 'Polygon'), "All should be Polygons"
    
    # Test with rotation
    result_rotated = rotate_and_explode(test_gdf, rotation_deg=45)
    
    # Should still have 3 polygons
    assert len(result_rotated) == 3
    assert all(result_rotated.geometry.geom_type == 'Polygon')
    
    # Geometries should be different after rotation
    assert not result.geometry.iloc[0].equals(result_rotated.geometry.iloc[0])


def test_rotate_and_explode_streets():
    """Test rotating and exploding street geometries."""
    # Create test data with multilinestrings
    line1 = LineString([(0, 0), (10, 0)])
    line2 = LineString([(0, 10), (10, 10)])
    multiline = MultiLineString([line1, line2])
    
    test_gdf = gpd.GeoDataFrame({
        'id': [1, 2],
        'geometry': [multiline, line1]
    }, crs="EPSG:4326")
    
    # Test without rotation
    result = rotate_and_explode(test_gdf, rotation_deg=0)
    
    # Should explode multilinestring into separate linestrings
    assert len(result) == 3, "Should have 3 linestrings after explosion"
    assert all(result.geometry.geom_type == 'LineString'), "All should be LineStrings"
    
    # Test with rotation
    result_rotated = rotate_and_explode(test_gdf, rotation_deg=90)
    
    # Should still have 3 linestrings
    assert len(result_rotated) == 3
    assert all(result_rotated.geometry.geom_type == 'LineString')
    
    # Geometries should be different after rotation
    assert not result.geometry.iloc[0].equals(result_rotated.geometry.iloc[0])


def test_rotate_and_explode_custom_origin():
    """Test rotation with custom origin point."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    test_gdf = gpd.GeoDataFrame({
        'id': [1],
        'geometry': [poly]
    }, crs="EPSG:4326")
    
    # Rotate around a point that is NOT the centroid
    origin_point = (0, 0)
    result = rotate_and_explode(test_gdf, rotation_deg=90, origin=origin_point)
    
    # Should have rotated around the specified point
    assert len(result) == 1
    assert result.geometry.geom_type.iloc[0] == 'Polygon'
    
    # The centroid should have changed position
    # Original centroid is at (5, 5)
    # After 90 degree rotation around (0,0), it should move to (-5, 5)
    original_centroid = poly.centroid
    rotated_centroid = result.geometry.iloc[0].centroid
    
    # Centroid should have moved
    assert not np.isclose(original_centroid.x, rotated_centroid.x)
    # After rotation around origin, x ≈ -5, y ≈ 5
    assert np.isclose(rotated_centroid.x, -5, atol=0.1)
    assert np.isclose(rotated_centroid.y, 5, atol=0.1)
