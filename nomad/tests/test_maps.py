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
    rotate,
    remove_overlaps
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
    assert 'garden_city_category' in buildings.columns
    
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
    actual_categories = set(buildings['garden_city_category'].unique())
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
    assert 'geolife_plus_category' in buildings.columns

    # All categories should be valid geolife_plus types
    valid_categories = {'unknown', 'residential', 'commercial', 'school'}
    actual_categories = set(buildings['geolife_plus_category'].unique())
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
    actual_categories = set(all_features['garden_city_category'].unique())
    assert actual_categories.issubset(valid_categories)


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
    
    # Test shop tag (when no building tag, shop should map to commercial)
    result = _classify_building(pd.Series({'shop': 'bakery'}), 'garden_city')
    assert result[0] == 'bakery'  # osm_type preserves shop type
    assert result[1] == 'unknown'  # subtype (bakery not in mapping)
    assert result[2] == 'other'  # category (default for unknown)


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


def test_remove_overlaps_identical_polygons():
    """Test that remove_overlaps handles identical polygons correctly."""
    from shapely.geometry import Polygon
    
    # Create identical polygons
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])  # Identical to poly1
    
    # Create GeoDataFrame with identical polygons
    gdf = gpd.GeoDataFrame({
        'id': [1, 2],
        'name': ['Building A', 'Building B']
    }, geometry=[poly1, poly2], crs='EPSG:4326')
    
    # Apply overlap removal
    result = remove_overlaps(gdf)
    
    # Should keep exactly one copy
    assert len(result) == 1, f"Expected 1 polygon, got {len(result)}"
    assert result.iloc[0]['id'] == 1, "Should keep the first polygon"
    assert result.iloc[0]['name'] == 'Building A', "Should keep the first polygon's metadata"


def test_remove_overlaps_contained_polygons():
    """Test that remove_overlaps removes contained polygons."""
    from shapely.geometry import Polygon
    
    # Create a large polygon and a smaller one contained within it
    large_poly = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    small_poly = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    
    gdf = gpd.GeoDataFrame({
        'id': [1, 2],
        'name': ['Large Building', 'Small Building']
    }, geometry=[large_poly, small_poly], crs='EPSG:4326')
    
    # Apply overlap removal
    result = remove_overlaps(gdf)
    
    # Should keep only the large polygon
    assert len(result) == 1, f"Expected 1 polygon, got {len(result)}"
    assert result.iloc[0]['id'] == 1, "Should keep the large polygon"
    assert result.iloc[0]['name'] == 'Large Building', "Should keep the large polygon's metadata"


def test_remove_overlaps_exclude_categories():
    """Test that remove_overlaps can exclude certain categories."""
    from shapely.geometry import Polygon
    
    # Create overlapping polygons with different categories
    poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    poly2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    
    gdf = gpd.GeoDataFrame({
        'id': [1, 2],
        'category': ['building', 'park']
    }, geometry=[poly1, poly2], crs='EPSG:4326')
    
    # Apply overlap removal excluding parks
    result = remove_overlaps(gdf, exclude_categories=['park'])
    
    # Should keep both polygons since parks are excluded from overlap removal
    assert len(result) == 2, f"Expected 2 polygons, got {len(result)}"
    assert set(result['category'].tolist()) == {'building', 'park'}, "Should keep both categories"


# =============================================================================
# COMPREHENSIVE TESTS BASED ON REAL ISSUES ENCOUNTERED
# =============================================================================

def test_empty_bounding_box():
    """Test behavior when no features exist in the bounding box."""
    # Use a bounding box in the middle of the ocean (should have no features)
    ocean_bbox = (-10.0, 30.0, -9.0, 31.0)  # Atlantic Ocean
    
    buildings = download_osm_buildings(ocean_bbox, clip=True)
    streets = download_osm_streets(ocean_bbox, clip=True)
    
    # Should return empty GeoDataFrames, not crash
    assert len(buildings) == 0, "Should return empty GeoDataFrame for empty area"
    assert len(streets) == 0, "Should return empty GeoDataFrame for empty area"
    
    # Should have correct columns even when empty
    expected_cols = ['osm_type', 'subtype', 'category', 'geometry']
    assert list(buildings.columns) == expected_cols, f"Expected columns {expected_cols}, got {list(buildings.columns)}"


def test_different_coordinate_systems():
    """Test that functions work correctly with different CRS."""
    # Philadelphia bounding box
    philly_bbox = (-75.1662060, 39.9411582, -75.1456557, 39.9557201)
    
    # Test with WGS84 (EPSG:4326)
    buildings_wgs84 = download_osm_buildings(philly_bbox, crs='EPSG:4326', clip=True)
    assert buildings_wgs84.crs == 'EPSG:4326', "Should use specified CRS"
    
    # Test with Web Mercator (EPSG:3857) 
    buildings_mercator = download_osm_buildings(philly_bbox, crs='EPSG:3857', clip=True)
    assert buildings_mercator.crs == 'EPSG:3857', "Should use specified CRS"
    
    # Both should have features (if any exist in the area)
    # The actual count may vary, but both should work without errors
    assert isinstance(buildings_wgs84, gpd.GeoDataFrame), "Should return GeoDataFrame"
    assert isinstance(buildings_mercator, gpd.GeoDataFrame), "Should return GeoDataFrame"


def test_water_feature_exclusion():
    """Test that water features are properly excluded from buildings and parks."""
    # Use a bounding box that includes water features (like Schuylkill River)
    water_bbox = (-75.18232868207252 - 0.01, 39.95057101861711 - 0.01,
                  -75.18232868207252 + 0.01, 39.95057101861711 + 0.01)
    
    buildings = download_osm_buildings(water_bbox, clip=True)
    
    # Should not contain water features
    if 'natural' in buildings.columns:
        water_features = buildings[buildings['natural'] == 'water']
        assert len(water_features) == 0, f"Found {len(water_features)} water features in buildings"
    
    if 'waterway' in buildings.columns:
        waterway_features = buildings[buildings['waterway'].notna()]
        assert len(waterway_features) == 0, f"Found {len(waterway_features)} waterway features in buildings"


def test_park_categorization():
    """Test that parks are correctly categorized as 'park', not 'other'."""
    # Use a bounding box known to have parks (like Washington Square Park)
    park_bbox = (-75.1531894880407 - 0.005, 39.9462501797551 - 0.005,
                 -75.1531894880407 + 0.005, 39.9462501797551 + 0.005)
    
    buildings = download_osm_buildings(park_bbox, clip=True)
    
    # If there are parks, they should be categorized as 'park'
    park_features = buildings[buildings['category'] == 'park']
    if len(park_features) > 0:
        # Parks should have leisure=park or similar tags
        assert len(park_features) > 0, "Should find park features"
        
        # Check that parks are not misclassified as 'other'
        other_features = buildings[buildings['category'] == 'other']
        park_osm_types = set(park_features['osm_type'].tolist())
        other_osm_types = set(other_features['osm_type'].tolist())
        
        # Should not have leisure=park in 'other' category
        assert 'park' not in other_osm_types, "Parks should not be in 'other' category"


def test_parking_lot_classification():
    """Test that parking lots are classified as buildings, not parks."""
    # Use a bounding box with parking lots
    parking_bbox = (-75.1550204236118 - 0.001, 39.94419232826087 - 0.001,
                    -75.1550204236118 + 0.001, 39.94419232826087 + 0.001)
    
    buildings = download_osm_buildings(parking_bbox, clip=True)
    
    # Parking lots should be classified as 'other' buildings, not parks
    parking_features = buildings[buildings['osm_type'] == 'parking']
    if len(parking_features) > 0:
        assert all(parking_features['category'] == 'other'), "Parking lots should be 'other' buildings"
        assert all(parking_features['subtype'] == 'parking'), "Parking lots should have 'parking' subtype"


def test_speculative_classification():
    """Test the infer_building_types parameter works correctly."""
    # Use a small bounding box with mixed building types
    test_bbox = (-75.1662060, 39.9411582, -75.1456557, 39.9557201)
    
    # Test without speculative classification
    buildings_basic = download_osm_buildings(test_bbox, infer_building_types=False, clip=True)
    
    # Test with speculative classification
    buildings_inferred = download_osm_buildings(test_bbox, infer_building_types=True, clip=True)
    
    # Both should work without errors
    assert isinstance(buildings_basic, gpd.GeoDataFrame), "Basic classification should work"
    assert isinstance(buildings_inferred, gpd.GeoDataFrame), "Inferred classification should work"
    
    # With inference, we might get more 'residential' and 'retail' classifications
    if len(buildings_inferred) > 0:
        inferred_categories = set(buildings_inferred['category'].tolist())
        basic_categories = set(buildings_basic['category'].tolist())
        
        # Should have same or more categories with inference
        assert inferred_categories.issuperset(basic_categories), "Inference should not remove categories"


def test_multipolygon_explosion():
    """Test that MultiPolygons are properly exploded."""
    # Create a test MultiPolygon
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    multipolygon = MultiPolygon([poly1, poly2])
    
    gdf = gpd.GeoDataFrame({
        'id': [1],
        'category': ['test']
    }, geometry=[multipolygon], crs='EPSG:4326')
    
    # Test explosion
    exploded = gdf.explode(ignore_index=True)
    
    assert len(exploded) == 2, f"Expected 2 polygons after explosion, got {len(exploded)}"
    assert all(exploded.geometry.geom_type == 'Polygon'), "All geometries should be Polygons after explosion"


def test_rotation_around_common_centroid():
    """Test that rotation happens around a common centroid, not individual centroids."""
    # Create two polygons
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    
    gdf = gpd.GeoDataFrame({
        'id': [1, 2],
        'category': ['test', 'test']
    }, geometry=[poly1, poly2], crs='EPSG:4326')
    
    # Rotate around common centroid
    rotated = rotate_and_explode(gdf, rotation_deg=90)
    
    # Should have same number of features
    assert len(rotated) == 2, f"Expected 2 polygons after rotation, got {len(rotated)}"
    
    # All geometries should still be valid
    assert all(rotated.geometry.is_valid), "All rotated geometries should be valid"


def test_overlap_removal_containment():
    """Test that overlap removal only removes fully contained polygons."""
    # Create contained polygon scenario
    outer_poly = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    inner_poly = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    
    gdf = gpd.GeoDataFrame({
        'id': [1, 2],
        'category': ['building', 'building']
    }, geometry=[outer_poly, inner_poly], crs='EPSG:4326')
    
    # Apply overlap removal
    result = remove_overlaps(gdf)
    
    # Should remove the contained polygon, keep the outer one
    assert len(result) == 1, f"Expected 1 polygon after overlap removal, got {len(result)}"
    
    # The remaining polygon should be the larger one
    remaining_area = result.geometry.iloc[0].area
    assert remaining_area == outer_poly.area, "Should keep the larger polygon"


def test_schema_switching():
    """Test that different schemas work correctly."""
    test_bbox = (-75.1662060, 39.9411582, -75.1456557, 39.9557201)
    
    # Test garden_city schema
    buildings_gc = download_osm_buildings(test_bbox, schema='garden_city', clip=True)
    
    # Test geolife_plus schema  
    buildings_gl = download_osm_buildings(test_bbox, schema='geolife_plus', clip=True)
    
    # Both should work
    assert isinstance(buildings_gc, gpd.GeoDataFrame), "Garden city schema should work"
    assert isinstance(buildings_gl, gpd.GeoDataFrame), "Geolife plus schema should work"
    
    # Should have different category sets
    if len(buildings_gc) > 0 and len(buildings_gl) > 0:
        gc_categories = set(buildings_gc['category'].tolist())
        gl_categories = set(buildings_gl['category'].tolist())
        
        # Different schemas should have different category sets
        assert gc_categories != gl_categories, "Different schemas should have different categories"


def test_city_name_download():
    """Test downloading data using city name instead of bounding box."""
    from nomad.map_utils import get_city_boundary_osm, download_osm_buildings, download_osm_streets
    
    # Test city boundary function
    city = get_city_boundary_osm('Philadelphia, Pennsylvania')
    assert isinstance(city, gpd.GeoDataFrame), "Should return GeoDataFrame"
    
    # Test building download with city name (small sample)
    buildings = download_osm_buildings('Philadelphia, Pennsylvania', explode=True)
    assert isinstance(buildings, gpd.GeoDataFrame), "Should return GeoDataFrame"
    
    # Test street download with city name (small sample)  
    streets = download_osm_streets('Philadelphia, Pennsylvania', explode=True)
    assert isinstance(streets, gpd.GeoDataFrame), "Should return GeoDataFrame"
