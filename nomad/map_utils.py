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
import warnings
from osmnx._errors import InsufficientResponseError
import tempfile
import os

from nomad.constants import (
    OSM_BUILDING_TO_SUBTYPE, OSM_AMENITY_TO_SUBTYPE, OSM_TOURISM_TO_SUBTYPE,
    PARK_TAGS, DEFAULT_CRS, DEFAULT_CATEGORY_SCHEMA, CATEGORY_SCHEMAS,
    STREET_HIGHWAY_TYPES, STREET_EXCLUDED_SERVICE_TYPES,
    STREET_EXCLUDE_COVERED, STREET_EXCLUDE_TUNNELS, STREET_EXCLUDED_SURFACES
)




from contextlib import contextmanager

@contextmanager
def _osmnx_temp_cache_dir():
    """Use a temporary OSMnx cache folder for the duration of the call, then delete it.

    This provides the performance/robustness of caching within a single call
    but leaves no persistent files on disk afterward.
    """
    prev_use_cache = getattr(ox.settings, 'use_cache', None)
    prev_cache_folder = getattr(ox.settings, 'cache_folder', None)
    with tempfile.TemporaryDirectory(prefix="osmnx-cache-") as tmpdir:
        try:
            ox.settings.use_cache = True
            ox.settings.cache_folder = tmpdir
        except Exception:
            # Best-effort: some environments may not expose these settings
            pass
        try:
            yield
        finally:
            # Restore settings
            try:
                if prev_use_cache is not None:
                    ox.settings.use_cache = prev_use_cache
                if prev_cache_folder is not None:
                    ox.settings.cache_folder = prev_cache_folder
            except Exception:
                pass


# Cache mode control
DEFAULT_OSMNX_CACHE_MODE = os.environ.get('NOMAD_OSMNX_CACHE_MODE', 'persistent')  # 'temp' | 'persistent' | 'off'

@contextmanager
def _osmnx_persistent_cache():
    prev_use_cache = getattr(ox.settings, 'use_cache', None)
    prev_cache_folder = getattr(ox.settings, 'cache_folder', None)
    try:
        ox.settings.use_cache = True
        # Default to ./cache under current working directory
        cache_dir = os.environ.get('NOMAD_OSMNX_CACHE_DIR', os.path.abspath('cache'))
        os.makedirs(cache_dir, exist_ok=True)
        ox.settings.cache_folder = cache_dir
        yield
    finally:
        try:
            if prev_use_cache is not None:
                ox.settings.use_cache = prev_use_cache
            if prev_cache_folder is not None:
                ox.settings.cache_folder = prev_cache_folder
        except Exception:
            pass


@contextmanager
def _osmnx_cache_off():
    prev_use_cache = getattr(ox.settings, 'use_cache', None)
    try:
        ox.settings.use_cache = False
        yield
    finally:
        try:
            if prev_use_cache is not None:
                ox.settings.use_cache = prev_use_cache
        except Exception:
            pass


def set_osmnx_cache_mode(mode):
    """Globally set default OSMnx cache mode: 'temp', 'persistent', or 'off'."""
    global DEFAULT_OSMNX_CACHE_MODE
    if mode not in ('temp', 'persistent', 'off'):
        raise ValueError("cache mode must be one of: 'temp', 'persistent', 'off'")
    DEFAULT_OSMNX_CACHE_MODE = mode


@contextmanager
def _osmnx_cache_context(mode=None):
    mode = (mode or DEFAULT_OSMNX_CACHE_MODE)
    if mode == 'temp':
        with _osmnx_temp_cache_dir():
            yield
    elif mode == 'persistent':
        with _osmnx_persistent_cache():
            yield
    elif mode == 'off':
        with _osmnx_cache_off():
            yield
    else:
        raise ValueError("Unknown cache mode")
def _clip_to_bbox(gdf, bbox, crs=DEFAULT_CRS):
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

def get_category_for_subtype(subtype, schema= DEFAULT_CATEGORY_SCHEMA):
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


def _classify_building(row, schema):
    """
    Classify a single building based on its OSM tags.
    
    This is an alias for _classify_feature for backward compatibility.
    """
    return _classify_feature(row, schema)


def _parse_tag_values(value):
    """Normalize OSM tag value(s) to a list of lowercased strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    # Lists/arrays/tuples of values
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = list(value)
    else:
        # Split common multi-value delimiters used in OSM tags
        text = str(value)
        # Replace pipes and commas with semicolons for unified splitting
        for delim in ['|', ',']:
            text = text.replace(delim, ';')
        values = [v for v in (part.strip() for part in text.split(';')) if v]
    return [str(v).lower().strip() for v in values]


def _choose_amenity_classification(row, schema):
    """
    Pick the primary amenity-based classification, preferring non-parking amenities
    when multiple amenities exist. Returns a (osm_value, subtype, category) tuple
    or None if no amenity-based classification can be determined.
    """
    if 'amenity' not in row or pd.isna(row['amenity']):
        return None
    amenities = _parse_tag_values(row['amenity'])
    if not amenities:
        return None

    mapped = [(a, OSM_AMENITY_TO_SUBTYPE[a]) for a in amenities if a in OSM_AMENITY_TO_SUBTYPE]
    # Prefer the first non-parking amenity if present
    for amenity_value, subtype in mapped:
        if subtype != 'parking':
            category = get_category_for_subtype(subtype, schema)
            return (amenity_value, subtype, category)
    # Otherwise, if parking exists (and no better amenity), choose parking
    for a in amenities:
        if a == 'parking':
            subtype = 'parking'
            category = get_category_for_subtype(subtype, schema)
            return ('parking', subtype, category)
    return None


def _classify_feature(row, schema, infer_building_types=False):
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
        if osm_value == 'parking':
            # Building tag takes precedence: building=parking means parking.
            subtype = 'parking'
            category = get_category_for_subtype(subtype, schema)
            return ('parking', subtype, category)
        if osm_value != 'yes' and osm_value in OSM_BUILDING_TO_SUBTYPE:
            subtype = OSM_BUILDING_TO_SUBTYPE[osm_value]
            category = get_category_for_subtype(subtype, schema)
            return (osm_value, subtype, category)
        # For building=yes, continue to check other tags (amenity, leisure, etc.)
    
    # Priority 2: amenity tag (support multiple amenities; prefer non-parking)
    amenity_choice = _choose_amenity_classification(row, schema)
    if amenity_choice is not None:
        return amenity_choice
    
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


def _categorize_features(gdf, schema, infer_building_types= False):
    """Apply categorization to all features in a GeoDataFrame."""
    if len(gdf) == 0:
        return gdf.copy()
    
    result = gdf.copy()
    classifications = result.apply(lambda row: _classify_feature(row, schema, infer_building_types), axis=1)
    
    result['osm_type'] = classifications.apply(lambda x: x[0])
    result['subtype'] = classifications.apply(lambda x: x[1])
    result['category'] = classifications.apply(lambda x: x[2])
    
    return result


# Derive secondary amenity-based subtypes when multiple amenities exist
def _derive_secondary_subtypes(row, schema, primary_subtype):
    amenities = _parse_tag_values(row['amenity']) if 'amenity' in row and pd.notna(row['amenity']) else []
    secondary = []
    for a in amenities:
        if a in OSM_AMENITY_TO_SUBTYPE:
            subtype = OSM_AMENITY_TO_SUBTYPE[a]
            if subtype != 'parking' and subtype != primary_subtype and subtype not in secondary:
                secondary.append(subtype)
    return secondary[:2]


def _add_secondary_subtypes_columns(gdf, schema):
    if len(gdf) == 0:
        return gdf
    sec = gdf.apply(lambda row: _derive_secondary_subtypes(row, schema, row.get('subtype')), axis=1)
    gdf = gdf.copy()
    gdf['subtype_2'] = sec.apply(lambda arr: arr[0] if isinstance(arr, list) and len(arr) > 0 else pd.NA)
    gdf['subtype_3'] = sec.apply(lambda arr: arr[1] if isinstance(arr, list) and len(arr) > 1 else pd.NA)
    return gdf

# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================
#rotate_and_explode SEEMS LIKE AN UNNECESSARY WRAPPER
def _download_osm_features(bbox, tags, crs=DEFAULT_CRS, clip=False, cache_mode=None):
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
        with _osmnx_cache_context(cache_mode):
            features = ox.features_from_bbox(bbox=bbox, tags=tags)
    except InsufficientResponseError:
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
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


def get_city_boundary_osm(name, simplify=True, crs="EPSG:4326"):
    """
    Get city boundary from OSM using city name.
    
    Returns
    -------
    tuple
        (multipolygon, center_coords, population)
        - multipolygon: city boundary geometry
        - center_coords: (lon, lat) tuple
        - population: integer or None
        
    Example
    -------
    >>> boundary, center, pop = get_city_boundary_osm('Philadelphia, Pennsylvania')
    """
    try:
        city_gdf = ox.geocode_to_gdf(name)
        city_gdf = city_gdf.to_crs(crs)
        
        if len(city_gdf) == 0:
            return None, None, None
            
        geometry = city_gdf.geometry.iloc[0]
        if simplify:
            geometry = geometry.simplify(tolerance=0.001)
            
        # Get center coordinates
        center = geometry.centroid
        center_coords = (center.x, center.y)
        
        # Get population
        population = None
        if 'population' in city_gdf.columns:
            pop_val = city_gdf['population'].iloc[0]
            if not pd.isna(pop_val):
                population = int(pop_val)
                
        return geometry, center_coords, population
        
    except Exception as e:
        warnings.warn(f"Could not fetch city boundary for '{name}': {e}")
        return None, None, None


def download_osm_buildings(bbox_or_city, 
                          crs=DEFAULT_CRS, 
                          schema=DEFAULT_CATEGORY_SCHEMA, 
                          clip=False,
                          infer_building_types=False,
                          explode=False,
                          by_chunks=False,
                          chunk_miles=1.0,
                          cache_mode=None):
    """
    Download and categorize buildings and parks from OpenStreetMap.
    
    Parameters
    ----------
    bbox_or_city : Union[Tuple[float, float, float, float], str]
        Bounding box (west, south, east, north) in WGS84 or city name
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
    
    # Determine if input is bbox or city name
    if isinstance(bbox_or_city, str):
        boundary, center, population = get_city_boundary_osm(bbox_or_city)
        if boundary is None:
            return gpd.GeoDataFrame()
        # Use city boundary directly instead of converting to bbox
        city_polygon = boundary
    else:
        bbox = bbox_or_city
        city_polygon = None

    # Chunked path
    if by_chunks:
        return _download_buildings_by_chunks(bbox if city_polygon is None else city_polygon,
                                            crs=crs,
                                            schema=schema,
                                            clip=True,
                                            infer_building_types=infer_building_types,
                                            explode=explode,
                                            chunk_miles=chunk_miles,
                                            cache_mode=cache_mode)
    
    # Download buildings (including parking lots with building tags)
    building_tags = {"building": True}
    if city_polygon is not None:
        with _osmnx_cache_context(cache_mode):
            buildings = ox.features_from_polygon(city_polygon, building_tags)
    else:
        buildings = _download_osm_features(bbox, building_tags, crs, clip, cache_mode)
    
    # Download parking lots (even without building tags) - they should be buildings, not parks
    parking_tags = {"amenity": ["parking"]}
    if city_polygon is not None:
        with _osmnx_cache_context(cache_mode):
            parking_lots = ox.features_from_polygon(city_polygon, parking_tags)
    else:
        parking_lots = _download_osm_features(bbox, parking_tags, crs, clip, cache_mode)
    
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
    
    if city_polygon is not None:
        with _osmnx_cache_context(cache_mode):
            parks = ox.features_from_polygon(city_polygon, park_tags)
    else:
        parks = _download_osm_features(bbox, park_tags, crs, clip, cache_mode)
    
    # EXCLUDE WATER FEATURES FROM PARKS - they should not be parks either
    if 'natural' in parks.columns:
        parks = parks[parks['natural'] != 'water']
    if 'waterway' in parks.columns:
        parks = parks[parks['waterway'].isna()]
    
    # Essential columns to keep
    essential_cols = ['geometry', 'osm_type', 'subtype', 'subtype_2', 'subtype_3',
                     f'{schema}_category', 'category',
                     'addr:street', 'addr:city', 'addr:state', 'addr:housenumber', 'addr:postcode']
    
    # Categorize buildings
    if len(buildings) > 0:
        classifications = buildings.apply(lambda row: _classify_feature(row, schema, infer_building_types), axis=1)
        buildings['osm_type'] = classifications.apply(lambda x: x[0])
        buildings['subtype'] = classifications.apply(lambda x: x[1])
        buildings[f'{schema}_category'] = classifications.apply(lambda x: x[2])
        # Backward-compatibility: also provide generic 'category' column
        buildings['category'] = buildings[f'{schema}_category']
        buildings = _add_secondary_subtypes_columns(buildings, schema)
        buildings = buildings[[col for col in essential_cols if col in buildings.columns]]
    
    # Categorize parks
    if len(parks) > 0:
        classifications = parks.apply(lambda row: _classify_feature(row, schema, infer_building_types), axis=1)
        parks['osm_type'] = classifications.apply(lambda x: x[0])
        parks['subtype'] = classifications.apply(lambda x: x[1])
        parks[f'{schema}_category'] = classifications.apply(lambda x: x[2])
        # Backward-compatibility: also provide generic 'category' column
        parks['category'] = parks[f'{schema}_category']
        parks = _add_secondary_subtypes_columns(parks, schema)
        parks = parks[[col for col in essential_cols if col in parks.columns]]
    
    # Combine only AFTER categorization to prevent parks from being misclassified as buildings
    if len(buildings) == 0 and len(parks) == 0:
        # For empty areas, return the generic columns expected by tests
        result = gpd.GeoDataFrame(columns=['osm_type', 'subtype', 'category', 'geometry'], crs=crs)
    elif len(buildings) == 0:
        result = parks
    elif len(parks) == 0:
        result = buildings
    else:
        result = gpd.GeoDataFrame(
            pd.concat([buildings, parks], ignore_index=True),
            crs=crs
        )
    
    # Explode if requested
    if explode and len(result) > 0:
        result = result.explode(ignore_index=True)
    
    return result


def download_osm_streets(bbox_or_city, 
                        crs=DEFAULT_CRS, 
                        clip=True,
                        clip_to_gdf=None,
                        explode=False,
                        by_chunks=False,
                        chunk_miles=1.0,
                        cache_mode=None):
    """Download street network from OpenStreetMap using OSMnx query-time filters.

    This implementation pushes highway/service/tunnel/covered/surface filters into
    the Overpass query via OSMnx's custom_filter and truncates at the graph level
    before converting to GeoDataFrames to avoid unnecessary post-processing.
    """

    # Determine if input is bbox or city name
    if isinstance(bbox_or_city, str):
        boundary, center, population = get_city_boundary_osm(bbox_or_city)
        if boundary is None:
            return gpd.GeoDataFrame()
        city_polygon = boundary
        bbox = None
    else:
        bbox = bbox_or_city
        city_polygon = None

    # Build custom Overpass filter (exact match on allowed highway types)
    highway_types = "|".join(STREET_HIGHWAY_TYPES)
    parts = [f'["highway"~"^({highway_types})$"]']
    if STREET_EXCLUDED_SERVICE_TYPES:
        excluded_services = "|".join(STREET_EXCLUDED_SERVICE_TYPES)
        parts.append(f'["service"!~"{excluded_services}"]')
    if STREET_EXCLUDE_TUNNELS:
        parts.append('["tunnel"!="yes"]')
    if STREET_EXCLUDE_COVERED:
        parts.append('["covered"!="yes"]')
    if STREET_EXCLUDED_SURFACES:
        excluded_surfaces = "|".join(STREET_EXCLUDED_SURFACES)
        parts.append(f'["surface"!~"{excluded_surfaces}"]')
    custom_filter = "".join(parts)

    # Chunked path
    if by_chunks:
        target = bbox if city_polygon is None else city_polygon
        return _download_streets_by_chunks(target,
                                           crs=crs,
                                           clip=True,
                                           clip_to_gdf=clip_to_gdf,
                                           explode=explode,
                                           chunk_miles=chunk_miles,
                                           cache_mode=cache_mode)

    try:
        with _osmnx_cache_context(cache_mode):
            # Construct graph with query-time filtering and graph-level truncation
            if city_polygon is not None:
                G = ox.graph_from_polygon(city_polygon, custom_filter=custom_filter, simplify=True)
            else:
                G = ox.graph_from_bbox(bbox=bbox, custom_filter=custom_filter, truncate_by_edge=bool(clip), simplify=True)

        # Optional additional truncation by another GDF's bounds
        if clip_to_gdf is not None and len(clip_to_gdf) > 0:
            cb = clip_to_gdf.total_bounds  # (minx, miny, maxx, maxy)
            west2, south2, east2, north2 = cb[0], cb[1], cb[2], cb[3]
            G = ox.truncate.truncate_graph_bbox(G, north2, south2, east2, west2, truncate_by_edge=True)

        # Convert to GeoDataFrame
        streets = ox.graph_to_gdfs(G, edges=True, nodes=False)

    except InsufficientResponseError:
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    except Exception as e:
        raise RuntimeError(f"Failed to download street network: {e}")

    if len(streets) == 0:
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)

    # Convert to target CRS
    streets = streets.to_crs(crs)

    # Normalize highway values that may be lists into simple strings for downstream tests
    if 'highway' in streets.columns:
        streets['highway'] = streets['highway'].apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else v)

    # Explode if requested
    if explode and len(streets) > 0:
        streets = streets.explode(ignore_index=True)

    return streets


# =============================================================================
# CHUNKED DOWNLOAD HELPERS
# =============================================================================

def _degrees_for_miles_at_lat(miles, lat_deg):
    # Approximate conversion: 1 deg lat ~ 69 miles; 1 deg lon ~ 69*cos(lat) miles
    if miles <= 0:
        return 0.0, 0.0
    deg_lat = miles / 69.0
    # Avoid division by zero at poles
    lat_rad = np.radians(max(min(lat_deg, 89.9), -89.9))
    miles_per_deg_lon = 69.0 * np.cos(lat_rad)
    if miles_per_deg_lon <= 0:
        deg_lon = deg_lat  # fallback
    else:
        deg_lon = miles / miles_per_deg_lon
    return deg_lon, deg_lat


def _chunk_target_to_bboxes(target, chunk_miles):
    # target can be bbox tuple (west, south, east, north) or a shapely polygon
    if isinstance(target, tuple) and len(target) == 4:
        west, south, east, north = target
        polygon = box(west, south, east, north)
    else:
        polygon = target
        west, south, east, north = polygon.bounds

    mean_lat = (south + north) / 2.0
    step_lon, step_lat = _degrees_for_miles_at_lat(chunk_miles, mean_lat)
    if step_lon <= 0 or step_lat <= 0:
        return [(west, south, east, north)]

    bboxes = []
    y = south
    while y < north:
        y2 = min(y + step_lat, north)
        x = west
        while x < east:
            x2 = min(x + step_lon, east)
            tile = box(x, y, x2, y2)
            if tile.intersects(polygon):
                bboxes.append((x, y, x2, y2))
            x = x2
        y = y2
    return bboxes


def _dedupe_geometries(gdf):
    if len(gdf) == 0:
        return gdf
    return gdf.drop_duplicates(subset=['geometry'])


def _download_buildings_by_chunks(target, crs, schema, clip, infer_building_types, explode, chunk_miles, cache_mode=None):
    tiles = _chunk_target_to_bboxes(target, chunk_miles)
    parts = []
    for tile in tiles:
        part = download_osm_buildings(tile if isinstance(target, tuple) else tile,
                                      crs=crs, schema=schema, clip=True,
                                      infer_building_types=infer_building_types,
                                      explode=explode, by_chunks=False,
                                      cache_mode=cache_mode)
        if len(part) > 0:
            parts.append(part)
    if not parts:
        return gpd.GeoDataFrame(columns=['osm_type', 'subtype', f'{schema}_category', 'category', 'geometry'], crs=crs)
    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs)
    combined = _dedupe_geometries(combined)
    # Final clip to target polygon if provided
    if not isinstance(target, tuple):
        combined = gpd.clip(combined, gpd.GeoDataFrame(geometry=[target], crs=crs))
    return combined


def _download_streets_by_chunks(target, crs, clip, clip_to_gdf, explode, chunk_miles, cache_mode=None):
    tiles = _chunk_target_to_bboxes(target, chunk_miles)
    parts = []
    for tile in tiles:
        part = download_osm_streets(tile if isinstance(target, tuple) else tile,
                                    crs=crs, clip=True,
                                    clip_to_gdf=clip_to_gdf,
                                    explode=explode, by_chunks=False,
                                    cache_mode=cache_mode)
        if len(part) > 0:
            parts.append(part)
    if not parts:
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs)
    combined = _dedupe_geometries(combined)
    # Final clip to target polygon if provided
    if not isinstance(target, tuple):
        combined = gpd.clip(combined, gpd.GeoDataFrame(geometry=[target], crs=crs))
    return combined


# =============================================================================
# GEOMETRY PROCESSING FUNCTIONS
# =============================================================================

def remove_overlaps(gdf, exclude_categories= None):
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


def _remove_overlaps_internal(gdf):
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


def rotate(gdf, rotation_deg=0.0, origin='centroid'):
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

def get_category_summary(gdf):
    """Get summary of categories in the dataset."""
    if 'category' not in gdf.columns or len(gdf) == 0:
        raise ValueError("Features must be categorized")
    return gdf['category'].value_counts().to_dict()


def get_subtype_summary(gdf):
    """Get summary of subtypes in the dataset."""
    if 'subtype' not in gdf.columns or len(gdf) == 0:
        return {}
    return gdf['subtype'].value_counts().to_dict()


def get_prominent_streets(streets_gdf, k= 10):
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


def rotate_streets_to_align(streets_gdf, k=200):
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
