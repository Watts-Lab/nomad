"""
OpenStreetMap data utilities for the NOMAD library.

This module provides clean, professional functions to download and process
geospatial data from OpenStreetMap, with flexible categorization schemas.

Key Features:
- Download buildings, parks, and streets from OSM
- Flexible two-step categorization (OSM tags → subtypes → schemas)
- Efficient overlap removal and geometry processing
- Professional error handling and validation

Author: NOMAD Development Team
"""

import geopandas as gpd
import pandas as pd
import osmnx as ox
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import box
from shapely.affinity import rotate as shapely_rotate
from typing import Tuple, List, Optional, Union
import warnings

from nomad.constants import (
    OSM_BUILDING_TO_SUBTYPE, OSM_AMENITY_TO_SUBTYPE, OSM_TOURISM_TO_SUBTYPE,
    PARK_TAGS, DEFAULT_CRS, DEFAULT_CATEGORY_SCHEMA, CATEGORY_SCHEMAS,
    STREET_HIGHWAY_TYPES, STREET_EXCLUDED_SERVICE_TYPES,
    STREET_EXCLUDE_COVERED, STREET_EXCLUDE_TUNNELS, STREET_EXCLUDED_SURFACES
)


def _clip_to_bbox(gdf: gpd.GeoDataFrame, bbox: Tuple[float, float, float, float], 
                  crs: str = DEFAULT_CRS) -> gpd.GeoDataFrame:
    """Clip geometries to bounding box."""
    if len(gdf) == 0:
        return gdf.copy()
    
    # Create clip box
    west, south, east, north = bbox
    clip_box = box(west, south, east, north)
    clip_gdf = gpd.GeoDataFrame([1], geometry=[clip_box], crs=crs)
    
    # Convert to same CRS if needed
    if gdf.crs != crs:
        gdf_clipped = gdf.to_crs(crs)
    else:
        gdf_clipped = gdf.copy()
    
    # Perform intersection
    clipped = gdf_clipped.intersection(clip_gdf.geometry.iloc[0])
    
    # Filter out empty geometries
    valid_mask = ~clipped.is_empty
    result = gdf_clipped[valid_mask].copy()
    result.geometry = clipped[valid_mask]
    
    return result


# =============================================================================
# CATEGORIZATION FUNCTIONS
# =============================================================================

def get_category_for_subtype(subtype: str, schema: str = DEFAULT_CATEGORY_SCHEMA) -> str:
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
        
    Raises
    ------
    ValueError
        If schema is not recognized
    """
    
    mapping = CATEGORY_SCHEMAS[schema]
    default = 'unknown' if schema == 'geolife_plus' else 'other'
    return mapping.get(subtype, default)


def _classify_building(row: pd.Series, schema: str) -> Tuple[str, str, str]:
    """
    Classify a single building based on its OSM tags.
    
    This is an alias for _classify_feature for backward compatibility.
    """
    return _classify_feature(row, schema)


def _classify_feature(row: pd.Series, schema: str, infer_building_types: bool = False) -> Tuple[str, str, str]:
    """
    Classify a single feature based on its OSM tags.
    
    Priority order: building → amenity → tourism → shop → office → healthcare → craft
    
    Parameters
    ----------
    row : pd.Series
        Feature row with OSM tags
    schema : str
        Category schema to use
        
    Returns
    -------
    Tuple[str, str, str]
        (osm_type, subtype, category)
    """
    # Priority 1: building tag (but only if it's a specific building type, not 'yes')
    if "building" in row and pd.notna(row["building"]):
        osm_value = str(row["building"]).lower().strip()
        if osm_value != 'yes' and osm_value in OSM_BUILDING_TO_SUBTYPE:
            subtype = OSM_BUILDING_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
        # For building=yes, continue to check other tags (amenity, leisure, etc.)
    
    # Priority 2: amenity tag (including parking lots)
    if "amenity" in row and pd.notna(row["amenity"]):
        osm_value = str(row["amenity"]).lower().strip()
        if osm_value == 'parking':
            # Parking lots are always buildings of type "other"
            return ('parking', 'parking', 'other')
        elif osm_value in OSM_AMENITY_TO_SUBTYPE:
            subtype = OSM_AMENITY_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
    
    # Priority 3: leisure tag (for parks and recreational areas)
    if "leisure" in row and pd.notna(row["leisure"]):
        osm_value = str(row["leisure"]).lower().strip()
        if osm_value in OSM_BUILDING_TO_SUBTYPE:
            subtype = OSM_BUILDING_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
        else:
            # Leisure tag exists but not in mapping - preserve the leisure type
            default_category = 'unknown' if schema == 'geolife_plus' else 'other'
            return (osm_value, 'unknown', default_category)
    
    # Priority 4: landuse tag (for land use classifications)
    if "landuse" in row and pd.notna(row["landuse"]):
        osm_value = str(row["landuse"]).lower().strip()
        if osm_value in OSM_BUILDING_TO_SUBTYPE:
            subtype = OSM_BUILDING_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
        else:
            # Landuse tag exists but not in mapping - preserve the landuse type
            default_category = 'unknown' if schema == 'geolife_plus' else 'other'
            return (osm_value, 'unknown', default_category)
    
    # Priority 5: tourism tag
    if "tourism" in row and pd.notna(row["tourism"]):
        osm_value = str(row["tourism"]).lower().strip()
        if osm_value in OSM_TOURISM_TO_SUBTYPE:
            subtype = OSM_TOURISM_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
    
    # Priority 6: shop tag
    if "shop" in row and pd.notna(row["shop"]):
        osm_value = str(row["shop"]).lower().strip()
        if osm_value in OSM_BUILDING_TO_SUBTYPE:
            subtype = OSM_BUILDING_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
        else:
            # Shop tag exists but not in mapping - preserve the shop type
            default_category = 'unknown' if schema == 'geolife_plus' else 'other'
            return (osm_value, 'unknown', default_category)
    
    # Priority 7: office tag
    if "office" in row and pd.notna(row["office"]):
        osm_value = str(row["office"]).lower().strip()
        if osm_value in OSM_BUILDING_TO_SUBTYPE:
            subtype = OSM_BUILDING_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
    
    # Priority 8: healthcare tag
    if "healthcare" in row and pd.notna(row["healthcare"]):
        osm_value = str(row["healthcare"]).lower().strip()
        if osm_value in OSM_BUILDING_TO_SUBTYPE:
            subtype = OSM_BUILDING_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
    
    # Priority 9: craft tag
    if "craft" in row and pd.notna(row["craft"]):
        osm_value = str(row["craft"]).lower().strip()
        if osm_value in OSM_BUILDING_TO_SUBTYPE:
            subtype = OSM_BUILDING_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
    
    # Default case: unknown building
    default_category = 'unknown' if schema == 'geolife_plus' else 'other'
    
    # Optional speculative classification for building=yes cases
    if infer_building_types and "building" in row and pd.notna(row["building"]):
        building_value = str(row["building"]).lower().strip()
        if building_value == 'yes':
            # Check if it has amenity tag
            has_amenity = "amenity" in row and pd.notna(row["amenity"])
            
            # Check height (if available)
            height = None
            if "height" in row and pd.notna(row["height"]):
                try:
                    height = float(str(row["height"]))
                except (ValueError, TypeError):
                    height = None
            
            # Speculative classification logic
            if has_amenity:
                # Building with amenity -> retail
                return ('yes', 'commercial', 'retail')
            elif height is None or height < 20:
                # Low/no height building without amenity -> residential
                return ('yes', 'residential', 'residential')
    
    return ('unknown', 'unknown', default_category)


def _categorize_features(gdf: gpd.GeoDataFrame, schema: str, infer_building_types: bool = False) -> gpd.GeoDataFrame:
    """Apply categorization to all features in a GeoDataFrame."""
    if len(gdf) == 0:
        return gdf.copy()
    
    result = gdf.copy()
    classifications = result.apply(lambda row: _classify_feature(row, schema, infer_building_types), axis=1)
    
    result['osm_type'] = classifications.apply(lambda x: x[0])
    result['subtype'] = classifications.apply(lambda x: x[1])
    result['category'] = classifications.apply(lambda x: x[2])
    
    return result


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def _download_osm_features(bbox: Tuple[float, float, float, float], 
                          tags: dict, 
                          crs: str = DEFAULT_CRS,
                          clip: bool = False) -> gpd.GeoDataFrame:
    """
    Download features from OSM with given tags.
    
    Parameters
    ----------
    bbox : Tuple[float, float, float, float]
        Bounding box (west, south, east, north)
    tags : dict
        OSM tags to search for
    crs : str
        Coordinate reference system
    clip : bool
        Whether to clip features to bounding box
    
    Returns
    -------
    gpd.GeoDataFrame
        Downloaded features
    """
    
    try:
        features = ox.features_from_bbox(bbox=bbox, tags=tags)
    except Exception as e:
        raise RuntimeError(f"Failed to download OSM features: {e}")
    
    if len(features) == 0:
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    
    # Filter to area geometries only
    features = features[
        features.geometry.apply(lambda geom: geom.geom_type in ['Polygon', 'MultiPolygon'])
    ]
    
    # Convert to target CRS
    features = features.to_crs(crs)
    
    # Clip if requested
    if clip:
        features = _clip_to_bbox(features, bbox, crs)
    
    return features


def download_osm_buildings(bbox: Tuple[float, float, float, float], 
                          crs: str = DEFAULT_CRS, 
                          schema: str = DEFAULT_CATEGORY_SCHEMA, 
                          clip: bool = False,
                          infer_building_types: bool = False,
                          explode: bool = False) -> gpd.GeoDataFrame:
    """
    Download and categorize buildings and parks from OpenStreetMap.
    
    Parameters
    ----------
    bbox : Tuple[float, float, float, float]
        Bounding box (west, south, east, north) in WGS84
    crs : str, default='EPSG:4326'
        Coordinate reference system for output
    schema : str, default='garden_city'
        Category schema ('garden_city', 'geolife_plus', etc.)
    clip : bool, default=False
        Whether to clip features to exact bounding box
    infer_building_types : bool, default=False
        Whether to apply heuristics for building=yes cases
    explode : bool, default=False
        Whether to explode MultiPolygons/MultiLineStrings
    
    Returns
    -------
    gpd.GeoDataFrame
        Buildings and parks with columns: osm_type, subtype, category
    """
    
    # Download buildings (including parking lots with building tags)
    building_tags = {"building": True}
    buildings = _download_osm_features(bbox, building_tags, crs, clip)
    
    # Download parking lots (even without building tags) - they should be buildings, not parks
    parking_tags = {"amenity": ["parking"]}
    parking_lots = _download_osm_features(bbox, parking_tags, crs, clip)
    
    # Filter out underground parking from parking lots
    if 'parking' in parking_lots.columns:
        parking_lots = parking_lots[parking_lots['parking'] != 'underground']
    if 'layer' in parking_lots.columns:
        parking_lots = parking_lots[parking_lots['layer'] != '-1']
    
    # Combine buildings and parking lots
    if len(parking_lots) > 0:
        if len(buildings) > 0:
            buildings = gpd.GeoDataFrame(
                pd.concat([buildings, parking_lots], ignore_index=True),
                crs=crs
            )
        else:
            buildings = parking_lots.copy()
    
    # EXCLUDE WATER FEATURES - they should not be buildings
    if 'natural' in buildings.columns:
        buildings = buildings[buildings['natural'] != 'water']
    if 'waterway' in buildings.columns:
        buildings = buildings[buildings['waterway'].isna()]
    
    # Download parks and green spaces (NO PARKING LOTS, NO WATER - only leisure, natural)
    park_tags = {}
    for tag_key, tag_values in PARK_TAGS.items():
        if tag_key not in ['landuse', 'amenity']:  # Remove landuse and amenity (no parking)
            park_tags[tag_key] = tag_values
    
    parks = _download_osm_features(bbox, park_tags, crs, clip)
    
    # EXCLUDE WATER FEATURES FROM PARKS - they should not be parks either
    if 'natural' in parks.columns:
        parks = parks[parks['natural'] != 'water']
    if 'waterway' in parks.columns:
        parks = parks[parks['waterway'].isna()]
    
    # Categorize buildings and parks separately to avoid cross-contamination
    categorized_buildings = _categorize_features(buildings, schema, infer_building_types) if len(buildings) > 0 else gpd.GeoDataFrame(columns=['osm_type', 'subtype', 'category', 'geometry'], crs=crs)
    categorized_parks = _categorize_features(parks, schema, infer_building_types) if len(parks) > 0 else gpd.GeoDataFrame(columns=['osm_type', 'subtype', 'category', 'geometry'], crs=crs)
    
    # Combine only AFTER categorization to prevent parks from being misclassified as buildings
    if len(categorized_buildings) == 0 and len(categorized_parks) == 0:
        result = gpd.GeoDataFrame(columns=['osm_type', 'subtype', 'category', 'geometry'], crs=crs)
    elif len(categorized_buildings) == 0:
        result = categorized_parks
    elif len(categorized_parks) == 0:
        result = categorized_buildings
    else:
        result = gpd.GeoDataFrame(
            pd.concat([categorized_buildings, categorized_parks], ignore_index=True),
            crs=crs
        )
    
    # Explode if requested
    if explode and len(result) > 0:
        result = result.explode(ignore_index=True)
    
    return result


def download_osm_streets(bbox: Tuple[float, float, float, float], 
                        crs: str = DEFAULT_CRS, 
                        clip: bool = True,
                        clip_to_gdf: Optional[gpd.GeoDataFrame] = None,
                        explode: bool = False) -> gpd.GeoDataFrame:
    """Download street network from OpenStreetMap."""
    
    try:
        # Download street network
        G = ox.graph_from_bbox(bbox=bbox)
        
        # Convert to GeoDataFrame
        streets = ox.graph_to_gdfs(G, edges=True, nodes=False)
        
    except Exception as e:
        raise RuntimeError(f"Failed to download street network: {e}")
    
    if len(streets) == 0:
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    
    # Convert to target CRS
    streets = streets.to_crs(crs)
    
    # Filter highway types
    streets = streets[streets['highway'].isin(STREET_HIGHWAY_TYPES)]
    
    # Filter service roads
    if 'service' in streets['highway'].values:
        service_mask = streets['highway'] != 'service'
        if 'service' in streets.columns:
            service_mask |= ~streets['service'].isin(STREET_EXCLUDED_SERVICE_TYPES)
        streets = streets[service_mask]
    
    # Filter covered roads
    if STREET_EXCLUDE_COVERED and 'tunnel' in streets.columns:
        streets = streets[streets['tunnel'] != 'yes']
    
    # Filter tunnels
    if STREET_EXCLUDE_TUNNELS and 'tunnel' in streets.columns:
        streets = streets[streets['tunnel'] != 'yes']
    
    # Filter surface types
    if 'surface' in streets.columns:
        streets = streets[~streets['surface'].isin(STREET_EXCLUDED_SURFACES)]
    
    # Clip if requested
    if clip_to_gdf is not None and len(clip_to_gdf) > 0:
        # Clip to bounds of another GeoDataFrame
        clip_bounds = clip_to_gdf.total_bounds
        clip_bbox = (clip_bounds[0], clip_bounds[1], clip_bounds[2], clip_bounds[3])
        streets = _clip_to_bbox(streets, clip_bbox, crs)
    elif clip:
        # Clip to original bounding box
        streets = _clip_to_bbox(streets, bbox, crs)
    
    # Explode if requested
    if explode and len(streets) > 0:
        streets = streets.explode(ignore_index=True)
    
    return streets


# =============================================================================
# GEOMETRY PROCESSING FUNCTIONS
# =============================================================================

def remove_overlaps(gdf: gpd.GeoDataFrame, exclude_categories: Optional[List[str]] = None) -> gpd.GeoDataFrame:
    """
    Remove geometries that are entirely contained within other geometries.
    
    This function handles both identical polygons (keeps one copy) and truly
    contained polygons (removes the contained one). It uses spatial indexing
    for efficient processing.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with potentially overlapping geometries
    exclude_categories : Optional[List[str]]
        List of categories to exclude from overlap removal (e.g., ['park'])
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with contained geometries removed
        
    Notes
    -----
    - Only removes geometries that are completely inside other geometries
    - Preserves at least one copy of identical polygons
    - Uses spatial indexing for efficiency
    """
    if len(gdf) == 0:
        return gdf.copy()
    
    result = gdf.copy().reset_index(drop=True)
    
    # Remove duplicates first (keep one copy of identical geometries)
    result = result.drop_duplicates(subset=['geometry'])
    
    # If exclude_categories is specified, only process geometries not in those categories
    if exclude_categories is not None and 'category' in result.columns:
        to_process = result[~result['category'].isin(exclude_categories)].copy()
        excluded = result[result['category'].isin(exclude_categories)].copy()
        
        if len(to_process) == 0:
            return result  # Nothing to process
        
        # Process only the non-excluded geometries
        processed = _remove_overlaps_internal(to_process)
        
        # Combine processed and excluded geometries
        final_result = gpd.GeoDataFrame(
            pd.concat([processed, excluded], ignore_index=True),
            crs=result.crs
        )
        return final_result
    else:
        return _remove_overlaps_internal(result)


def _remove_overlaps_internal(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Internal function to perform overlap removal.
    
    Handles identical polygons by keeping one copy, and removes geometries
    that are entirely contained within different geometries.
    """
    result = gdf.copy().reset_index(drop=True)
    
    # Step 1: Handle identical polygons
    to_remove = []
    geometry_groups = {}
    
    for idx, geom in enumerate(result.geometry):
        geom_wkt = geom.wkt  # Use WKT string as hashable key
        if geom_wkt not in geometry_groups:
            geometry_groups[geom_wkt] = []
        geometry_groups[geom_wkt].append(idx)
    
    # For each group of identical geometries, keep only the first one
    for geom_wkt, indices in geometry_groups.items():
        if len(indices) > 1:
            to_remove.extend(indices[1:])  # Keep first, remove rest
    
    # Step 2: Handle true containment (different geometries only)
    unique_geometries = set()
    for geom_wkt, indices in geometry_groups.items():
        if len(indices) == 1:  # Only unique geometries
            unique_geometries.add(indices[0])
    
    if len(unique_geometries) > 1:
        # Use spatial join to find containment relationships among unique geometries
        unique_gdf = result.iloc[list(unique_geometries)].copy()
        temp_left = unique_gdf[['geometry']].copy()
        temp_right = unique_gdf[['geometry']].copy()
        
        # Find geometries that are contained within other geometries
        contained = gpd.sjoin(temp_left, temp_right, predicate='within', how='inner')
        
        # Remove self-containment
        contained = contained[contained.index != contained.index_right]
        
        # Process containment for unique geometries
        for idx in contained.index.unique():
            if idx not in to_remove:  # Don't double-remove
                containing_geoms = contained[contained.index == idx]['index_right'].values
                different_containers = [i for i in containing_geoms if i != idx]
                
                if different_containers:
                    to_remove.append(idx)
    
    # Remove geometries that are duplicates or contained within different geometries
    result = result[~result.index.isin(to_remove)]
    
    return result


def rotate(gdf: gpd.GeoDataFrame, rotation_deg: float = 0.0, origin: Union[str, Tuple[float, float]] = 'centroid') -> gpd.GeoDataFrame:
    """
    Rotate geometries around a single point.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input geometries
    rotation_deg : float, default=0.0
        Rotation angle in degrees (positive = counterclockwise)
    origin : Union[str, Tuple[float, float]], default='centroid'
        Rotation origin point. Can be 'centroid' or (x, y) coordinates
        
    Returns
    -------
    gpd.GeoDataFrame
        Rotated geometries
    """
    if len(gdf) == 0:
        return gdf.copy()
    
    result = gdf.copy()
    
    # Determine rotation origin
    if origin == 'centroid':
        # Calculate centroid of all geometries combined
        all_geoms = result.geometry.union_all()
        origin_point = all_geoms.centroid
        origin_coords = (origin_point.x, origin_point.y)
    else:
        origin_coords = origin
    
    # Rotate all geometries around the same point
    if rotation_deg != 0:
        result.geometry = result.geometry.apply(
            lambda geom: shapely_rotate(geom, rotation_deg, origin=origin_coords)
        )
    
    return result


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def get_category_summary(gdf: gpd.GeoDataFrame) -> dict:
    """Get summary of categories in the dataset."""
    if 'category' not in gdf.columns or len(gdf) == 0:
        raise ValueError("Features must be categorized")
    return gdf['category'].value_counts().to_dict()


def get_subtype_summary(gdf: gpd.GeoDataFrame) -> dict:
    """Get summary of subtypes in the dataset."""
    if 'subtype' not in gdf.columns or len(gdf) == 0:
        return {}
    return gdf['subtype'].value_counts().to_dict()


def get_prominent_streets(streets_gdf: gpd.GeoDataFrame, k: int = 10) -> gpd.GeoDataFrame:
    """
    Select the most prominent streets based on length and highway type priority.
    
    Parameters
    ----------
    streets_gdf : gpd.GeoDataFrame
        Street network GeoDataFrame
    k : int, default=10
        Number of prominent streets to return
        
    Returns
    -------
    gpd.GeoDataFrame
        Top k prominent streets
    """
    if len(streets_gdf) == 0:
        return streets_gdf.copy()
    
    result = streets_gdf.copy()
    
    # Calculate street lengths
    result['length'] = result.geometry.length
    
    # Highway type priority (higher = more prominent)
    highway_priority = {
        'motorway': 10, 'trunk': 9, 'primary': 8, 'secondary': 7,
        'tertiary': 6, 'unclassified': 5, 'residential': 4, 'service': 3
    }
    
    if 'highway' in result.columns:
        result['priority'] = result['highway'].map(highway_priority).fillna(1)
    else:
        result['priority'] = 1
    
    # Calculate prominence score (length * priority)
    result['prominence_score'] = result['length'] * result['priority']
    
    # Get top k streets
    top_streets = result.nlargest(k, 'prominence_score')
    
    # Drop temporary columns
    columns_to_drop = ['priority', 'prominence_score']
    top_streets = top_streets.drop(columns=[col for col in columns_to_drop if col in top_streets.columns])
    
    return top_streets


def rotate_streets_to_align(streets_gdf: gpd.GeoDataFrame, k: int = 200) -> Tuple[gpd.GeoDataFrame, float]:
    if len(streets_gdf) == 0:
        return streets_gdf.copy(), 0.0
    
    # Get random non-highway streets
    non_highway = streets_gdf[~streets_gdf['highway'].isin(['motorway', 'trunk', 'primary'])]
    if len(non_highway) > k:
        sample_streets = non_highway.sample(n=k)
    else:
        sample_streets = non_highway
    
    # Extract all segment angles efficiently
    all_coords = []
    for geom in sample_streets.geometry:
        if hasattr(geom, 'coords'):
            all_coords.extend(list(geom.coords))
    
    if len(all_coords) < 2:
        return streets_gdf.copy(), 0.0
    
    # Vectorized angle calculation
    coords_array = np.array(all_coords)
    dx = coords_array[1:, 0] - coords_array[:-1, 0]
    dy = coords_array[1:, 1] - coords_array[:-1, 1]
    mask = (dx != 0) | (dy != 0)
    
    if not np.any(mask):
        return streets_gdf.copy(), 0.0
    
    angles = np.arctan2(dy[mask], dx[mask])
    angles = ((angles + np.pi/2) % np.pi) - np.pi/2
    
    # Calculate rotation
    A, B = np.sum(np.cos(4 * angles)), np.sum(np.sin(4 * angles))
    rotation_rad = -0.25 * np.arctan2(B, A)
    rotation_deg = np.degrees(rotation_rad)
    
    # Rotate all streets around common centroid
    all_geoms = streets_gdf.geometry.union_all()
    origin_coords = (all_geoms.centroid.x, all_geoms.centroid.y)
    
    rotated_streets = streets_gdf.copy()
    rotated_streets.geometry = rotated_streets.geometry.apply(
        lambda geom: shapely_rotate(geom, rotation_deg, origin=origin_coords)
    )
    
    return rotated_streets, rotation_deg
