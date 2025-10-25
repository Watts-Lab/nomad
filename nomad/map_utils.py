"""
OpenStreetMap data utilities for the NOMAD library.

Simple functions to download and categorize buildings (including parks) and streets
from OpenStreetMap using a flexible two-step categorization:
1. OSM tags → detailed subtypes
2. Subtypes → category schemas (garden_city, geolife_plus, etc.)

Based on: https://wiki.openstreetmap.org/wiki/Map_features
"""

import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.affinity import rotate
from shapely.geometry import box
from nomad.constants import (
    OSM_BUILDING_TO_SUBTYPE, OSM_AMENITY_TO_SUBTYPE, OSM_TOURISM_TO_SUBTYPE,
    PARK_TAGS, DEFAULT_CRS, DEFAULT_CATEGORY_SCHEMA, CATEGORY_SCHEMAS,
    STREET_HIGHWAY_TYPES, STREET_EXCLUDED_SERVICE_TYPES,
    STREET_EXCLUDE_COVERED, STREET_EXCLUDE_TUNNELS, STREET_EXCLUDED_SURFACES
)


def get_category_for_subtype(subtype, schema='garden_city'):
    """
    Get the category for a subtype in a specific schema.
    
    Parameters
    ----------
    subtype : str
        The building subtype
    schema : str
        The category schema to use ('garden_city', 'geolife_plus', etc.)
    
    Returns
    -------
    str
        The category in the specified schema, or 'other'/'unknown' if not found
    """
    if schema not in CATEGORY_SCHEMAS:
        raise ValueError(f"Unknown category schema: {schema}. Available: {list(CATEGORY_SCHEMAS.keys())}")
    
    mapping = CATEGORY_SCHEMAS[schema]
    default = 'unknown' if schema == 'geolife_plus' else 'other'
    return mapping.get(subtype, default)


def download_osm_buildings(bbox, crs=DEFAULT_CRS, schema=DEFAULT_CATEGORY_SCHEMA, clip=False):
    """
    Download and categorize buildings and parks from OpenStreetMap.
    
    This is the main user-facing function for downloading all building features
    including parks, with automatic categorization.
    
    Parameters
    ----------
    bbox : tuple
        Bounding box as (west, south, east, north) in WGS84 coordinates
    crs : str, default "EPSG:4326"
        Target CRS for the downloaded geometries
    schema : str, default 'garden_city'
        Category schema to use: 'garden_city', 'geolife_plus', etc.
    clip : bool, default False
        If True, clip geometries to the exact bounding box. If False, keep
        complete buildings that intersect the bbox (may extend beyond it).
    
    Returns
    -------
    gpd.GeoDataFrame
        All buildings and parks with 'osm_type', 'subtype', and 'category' columns
    """
    # Download buildings
    buildings = _download_buildings(bbox, crs, clip=clip)
    buildings = _categorize_buildings(buildings, schema)
    
    # Download parks
    parks = _download_parks(bbox, crs, schema, clip=clip)
    
    # Combine everything
    all_features = gpd.GeoDataFrame(
        pd.concat([buildings, parks], ignore_index=True),
        crs=crs
    )
    
    return all_features


def download_osm_streets(bbox, crs=DEFAULT_CRS, clip=True, clip_to_gdf=None):
    """
    Download street network from OpenStreetMap within a bounding box.
    
    Downloads major streets excluding parking aisles, driveways, tunnels,
    covered ways, and paving stone surfaces.
    
    Parameters
    ----------
    bbox : tuple
        Bounding box as (west, south, east, north) in WGS84 coordinates
    crs : str, default "EPSG:4326"
        Target CRS for the downloaded geometries
    clip : bool, default True
        If True, clip streets to a bounding box
    clip_to_gdf : gpd.GeoDataFrame, optional
        If provided, clip streets to the bounding box of this GeoDataFrame
        instead of the input bbox. Useful for clipping streets to building extent.
    
    Returns
    -------
    gpd.GeoDataFrame
        Street network with geometry and OSM tags
    """
    
    # Download streets from OSM using defined highway types
    tags = {'highway': STREET_HIGHWAY_TYPES}
    streets = ox.features_from_bbox(bbox=bbox, tags=tags)
    
    # Filter to LineString geometries only
    streets = streets[
        streets.geometry.apply(lambda geom: geom.geom_type in ['LineString', 'MultiLineString'])
    ]
    
    # Apply exclusion filters
    if 'service' in streets.columns:
        streets = streets[~streets['service'].isin(STREET_EXCLUDED_SERVICE_TYPES)]
    
    if STREET_EXCLUDE_COVERED and 'covered' in streets.columns:
        streets = streets[streets['covered'] != 'yes']
    
    if STREET_EXCLUDE_TUNNELS and 'tunnel' in streets.columns:
        streets = streets[streets['tunnel'] != 'yes']
    
    if 'surface' in streets.columns:
        streets = streets[~streets['surface'].isin(STREET_EXCLUDED_SURFACES)]
    
    streets = streets.to_crs(crs)
    
    if clip:
        # Determine clip box
        if clip_to_gdf is not None:
            # Clip to the bounds of another GeoDataFrame (e.g., buildings)
            clip_box = box(*clip_to_gdf.total_bounds)
        else:
            # Clip to the original bbox
            clip_box = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(crs).iloc[0]
        
        # Clip streets to the bounding box
        streets['geometry'] = streets.geometry.intersection(clip_box)
        streets = streets[~streets.geometry.is_empty]
        
        # Re-filter after clipping (intersection may create other geometry types)
        streets = streets[
            streets.geometry.apply(lambda geom: geom.geom_type in ['LineString', 'MultiLineString'])
        ]
    
    return streets


def remove_overlaps(gdf):
    """
    Remove geometries that are entirely contained within other geometries.
    
    This approach only removes geometries that are completely inside other geometries,
    avoiding complex overlap scenarios and issues with rotated city grids.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with potentially overlapping geometries
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with contained geometries removed
    """
    if len(gdf) == 0:
        return gdf.copy()
    
    result = gdf.copy().reset_index(drop=True)
    
    # Use spatial join to find containment relationships
    # sjoin with 'within' predicate finds geometries that are contained within others
    temp_left = result[['geometry']].copy()
    temp_right = result[['geometry']].copy()
    
    # Find geometries that are contained within other geometries
    contained = gpd.sjoin(temp_left, temp_right, predicate='within', how='inner')
    
    # Remove self-containment (where index == index_right)
    contained = contained[contained.index != contained.index_right]
    
    if len(contained) == 0:
        # No contained geometries found
        return result
    
    # Get indices of geometries that are contained within others
    contained_indices = contained.index.unique()
    
    # Remove contained geometries
    result = result[~result.index.isin(contained_indices)]
    
    return result


def rotate_and_explode(gdf, rotation_deg=0.0, origin='centroid', clip_to_original_bounds=False):
    """
    Rotate and explode geometries for spatial analysis.
    
    This function rotates geometries around a specified origin and explodes
    multi-geometries into single geometries. Optionally clips back to original bounds.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with geometries to rotate and explode
    rotation_deg : float, default 0.0
        Rotation angle in degrees (counter-clockwise)
    origin : str or tuple, default 'centroid'
        Rotation origin. If 'centroid', rotates around the centroid of all geometries.
        Otherwise, provide a tuple (x, y) for a specific point.
    clip_to_original_bounds : bool, default False
        If True, clip rotated geometries back to the original bounding box (rotated)
    
    Returns
    -------
    gpd.GeoDataFrame
        Transformed GeoDataFrame with rotated and exploded geometries
    """
    
    df = gdf.copy()
    
    # Store original bounds
    original_bounds = df.total_bounds
    
    # Determine rotation origin
    if origin == 'centroid':
        cx = (original_bounds[0] + original_bounds[2]) / 2
        cy = (original_bounds[1] + original_bounds[3]) / 2
        origin_point = (cx, cy)
    else:
        origin_point = origin
    
    # Rotate geometries if rotation angle is non-zero
    if rotation_deg != 0.0:
        df['geometry'] = df.geometry.map(
            lambda g: rotate(g, rotation_deg, origin=origin_point)
        )
        
        # Optionally clip to rotated bounding box
        if clip_to_original_bounds:
            # Create and rotate the original bounding box
            original_box = box(*original_bounds)
            rotated_box = rotate(original_box, rotation_deg, origin=origin_point)
            
            # Clip geometries to rotated box
            df['geometry'] = df.geometry.intersection(rotated_box)
    
    # Explode multi-geometries into single geometries
    df = df.explode(ignore_index=True)
    
    # Filter to single geometry types (LineString for streets, Polygon for buildings)
    if len(df) > 0:
        geom_types = df.geometry.geom_type.unique()
        if 'MultiLineString' in geom_types or 'LineString' in geom_types:
            # For street data, keep only LineStrings
            df = df[df.geometry.geom_type == 'LineString']
        elif 'MultiPolygon' in geom_types or 'Polygon' in geom_types:
            # For building/polygon data, keep only Polygons
            df = df[df.geometry.geom_type == 'Polygon']
    
    # Remove empty geometries that may result from clipping
    df = df[~df.geometry.is_empty]
    
    return df.reset_index(drop=True)


def verify_bbox_bounds(gdf, expected_bbox, tolerance=1e-6):
    """
    Verify that all geometries in a GeoDataFrame are within expected bounds.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to check
    expected_bbox : tuple
        Expected bounding box as (minx, miny, maxx, maxy)
    tolerance : float, default 1e-6
        Tolerance for floating point comparison
    
    Returns
    -------
    dict
        Dictionary with 'within_bounds' (bool), 'actual_bounds' (tuple),
        'expected_bounds' (tuple), and 'exceeds_by' (dict with minx, miny, maxx, maxy)
    """
    actual = gdf.total_bounds
    expected = expected_bbox
    
    exceeds = {
        'minx': max(0, expected[0] - actual[0]),
        'miny': max(0, expected[1] - actual[1]),
        'maxx': max(0, actual[2] - expected[2]),
        'maxy': max(0, actual[3] - expected[3])
    }
    
    within_bounds = all(v <= tolerance for v in exceeds.values())
    
    return {
        'within_bounds': within_bounds,
        'actual_bounds': tuple(actual),
        'expected_bounds': expected,
        'exceeds_by': exceeds
    }


# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================

def _download_buildings(bbox, crs, clip=False):
    """Download raw building footprints from OSM, optionally clip to bbox."""
    
    buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
    
    # Filter to area geometries only
    buildings = buildings[
        buildings.geometry.apply(lambda geom: geom.geom_type in ['Polygon', 'MultiPolygon'])
    ]
    
    buildings = buildings.to_crs(crs)
    
    if clip:
        # Create clip box in WGS84, then convert to target CRS
        clip_box = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(crs).iloc[0]
        
        # Clip to the requested bounding box (OSM returns features that extend beyond bbox)
        buildings['geometry'] = buildings.geometry.intersection(clip_box)
        buildings = buildings[~buildings.geometry.is_empty]
        
        # Re-filter after clipping (intersection may create other geometry types)
        buildings = buildings[
            buildings.geometry.apply(lambda geom: geom.geom_type in ['Polygon', 'MultiPolygon'])
        ]
    
    return buildings


def _download_parks(bbox, crs, schema, clip=False):
    """Download park and open space areas from OSM, optionally clip to bbox."""
    
    park_tags = {}
    for tag_key, tag_values in PARK_TAGS.items():
        park_tags[tag_key] = True if isinstance(tag_values, list) else tag_values
    
    parks = ox.features_from_bbox(bbox=bbox, tags=park_tags)
    
    # Filter to area geometries only (same as buildings)
    parks = parks[
        parks.geometry.apply(lambda geom: geom.geom_type in ['Polygon', 'MultiPolygon'])
    ]
    
    parks = parks.to_crs(crs)
    
    if clip:
        # Create clip box in WGS84, then convert to target CRS
        clip_box = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(crs).iloc[0]
        
        # Clip to the requested bounding box (OSM returns features that extend beyond bbox)
        parks['geometry'] = parks.geometry.intersection(clip_box)
        parks = parks[~parks.geometry.is_empty]
        
        # Re-filter after clipping
        parks = parks[
            parks.geometry.apply(lambda geom: geom.geom_type in ['Polygon', 'MultiPolygon'])
        ]
    
    # Add classification columns
    parks['osm_type'] = 'park'
    parks['subtype'] = 'park'
    parks['category'] = get_category_for_subtype('park', schema)
    
    return parks


def _categorize_buildings(buildings_gdf, schema):
    """Apply categorization to buildings."""
    buildings = buildings_gdf.copy()
    
    classifications = buildings.apply(lambda row: _classify_building(row, schema), axis=1)
    buildings['osm_type'] = classifications.apply(lambda x: x[0])
    buildings['subtype'] = classifications.apply(lambda x: x[1])
    buildings['category'] = classifications.apply(lambda x: x[2])
    
    return buildings


def _classify_building(row, schema):
    """
    Classify a single building based on its OSM tags.
    
    Priority order: amenity → building → tourism → shop → office → healthcare → craft
    Returns: (osm_type, subtype, category)
    """
    # Priority 1: amenity tag (e.g., townhall takes priority over building=civic)
    if "amenity" in row and pd.notna(row["amenity"]):
        osm_value = str(row["amenity"]).lower().strip()
        if osm_value in OSM_AMENITY_TO_SUBTYPE:
            subtype = OSM_AMENITY_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
    
    # Priority 2: building tag
    if "building" in row and pd.notna(row["building"]):
        osm_value = str(row["building"]).lower().strip()
        if osm_value != 'yes' and osm_value in OSM_BUILDING_TO_SUBTYPE:
            subtype = OSM_BUILDING_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
    
    # Priority 3: tourism tag
    if "tourism" in row and pd.notna(row["tourism"]):
        osm_value = str(row["tourism"]).lower().strip()
        if osm_value in OSM_TOURISM_TO_SUBTYPE:
            subtype = OSM_TOURISM_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
    
    # Priority 4: shop tag
    if "shop" in row and pd.notna(row["shop"]):
        osm_value = str(row["shop"]).lower().strip()
        category = get_category_for_subtype('commercial', schema)
        return (f'shop_{osm_value}', 'commercial', category)
    
    # Priority 5: office tag
    if "office" in row and pd.notna(row["office"]):
        osm_value = str(row["office"]).lower().strip()
        category = get_category_for_subtype('office', schema)
        return (f'office_{osm_value}', 'office', category)
    
    # Priority 6: healthcare tag
    if "healthcare" in row and pd.notna(row["healthcare"]):
        osm_value = str(row["healthcare"]).lower().strip()
        category = get_category_for_subtype('medical', schema)
        return (osm_value, 'medical', category)
    
    # Priority 7: craft tag
    if "craft" in row and pd.notna(row["craft"]):
        osm_value = str(row["craft"]).lower().strip()
        category = get_category_for_subtype('commercial', schema)
        return (osm_value, 'commercial', category)
    
    # Default: unknown/other
    default_category = 'unknown' if schema == 'geolife_plus' else 'other'
    return ('unknown', 'unknown', default_category)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_category_summary(features_gdf):
    """Get a summary of features by category."""
    if 'category' not in features_gdf.columns:
        raise ValueError("Features must be categorized (missing 'category' column)")
    return features_gdf['category'].value_counts().to_dict()


def get_subtype_summary(features_gdf):
    """Get a summary of features by detailed subtype."""
    if 'subtype' not in features_gdf.columns:
        raise ValueError("Features must be categorized (missing 'subtype' column)")
    return features_gdf['subtype'].value_counts().to_dict()


def get_osm_type_summary(features_gdf):
    """Get a summary of features by OSM type (most granular)."""
    if 'osm_type' not in features_gdf.columns:
        raise ValueError("Features must be categorized (missing 'osm_type' column)")
    return features_gdf['osm_type'].value_counts().to_dict()
