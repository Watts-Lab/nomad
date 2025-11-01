"""
OpenStreetMap data utilities used to download, process, rotate, and persist
geospatial layers for the NOMAD library.
"""

import geopandas as gpd
import pandas as pd
import osmnx as ox
import numpy as np
from shapely.geometry import box
from shapely.affinity import rotate as shapely_rotate
import warnings
from osmnx._errors import InsufficientResponseError
import os

from nomad.constants import (
    OSM_BUILDING_TO_SUBTYPE, OSM_AMENITY_TO_SUBTYPE, OSM_TOURISM_TO_SUBTYPE,
    PARK_TAGS, DEFAULT_CRS, DEFAULT_CATEGORY_SCHEMA, CATEGORY_SCHEMAS,
    STREET_HIGHWAY_TYPES, STREET_EXCLUDED_SERVICE_TYPES,
    STREET_EXCLUDE_COVERED, STREET_EXCLUDE_TUNNELS, STREET_EXCLUDED_SURFACES,
    INTERSECTION_CONSOLIDATION_TOLERANCE_M, STREET_MIN_LENGTH_M
)

from contextlib import contextmanager
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
    """Return the category name for a subtype in the given schema.

    If the subtype is not present, returns 'unknown' for 'geolife_plus' or
    'other' for all other schemas.
    """
    mapping = CATEGORY_SCHEMAS[schema]
    default = 'unknown' if schema == 'geolife_plus' else 'other'
    return mapping.get(subtype, default)


def _classify_building(row, schema):
    # Alias for _classify_feature for backward compatibility.
    return _classify_feature(row, schema)


def _parse_tag_values(value):
    # Normalize OSM tag value(s) to a list of lowercase strings.
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
    # Choose amenity-based classification; prefer non-parking when multiple exist.
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
    # Classify a feature based on OSM tags; building > amenity > other tags.
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
    
    # Optional inference for building=yes: after all other tags,
    # if no amenity and height < 20, label as residential
    if infer_building_types and "building" in row and pd.notna(row["building"]):
        building_value = str(row["building"]).lower().strip()
        if building_value == 'yes':
            has_amenity = ("amenity" in row and pd.notna(row["amenity"]))
            height = None
            if "height" in row and pd.notna(row["height"]):
                try:
                    height = float(str(row["height"]))
                except (ValueError, TypeError):
                    height = None
            # Require an additional residential signal to avoid mass overclassification
            landuse_val = str(row.get('landuse', '')).lower().strip() if 'landuse' in row and pd.notna(row['landuse']) else ''
            addr_fields = ['addr:housenumber', 'addr:housename', 'addr:unit']
            has_addr = any((c in row and pd.notna(row[c])) for c in addr_fields)
            building_use = str(row.get('building:use', '')).lower().strip() if 'building:use' in row and pd.notna(row['building:use']) else ''

            is_residential_signal = (
                landuse_val == 'residential' or
                ('residential' in building_use if building_use else False) or
                has_addr
            )

            if (not has_amenity) and (height is not None and height < 20) and is_residential_signal:
                subtype = 'residential'
                category = get_category_for_subtype(subtype, schema)
                return ('yes', subtype, category)
            # Otherwise, keep unknown/other
            return ('yes', 'unknown', default_category)
    
    return ('unknown', 'unknown', default_category)


def _categorize_features(gdf, schema, infer_building_types= False):
    # Apply categorization to all rows of a GeoDataFrame.
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
def _download_osm_features(bbox, tags, crs=DEFAULT_CRS):
    """Fetch OSM features for bbox/tags and return polygons in target CRS."""
    try:
        features = ox.features_from_bbox(bbox=bbox, tags=tags)
    except InsufficientResponseError:
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    if len(features) == 0:
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    # Keep only area geometries
    features = features[features.geometry.apply(lambda g: g.geom_type in ['Polygon', 'MultiPolygon'])]
    return features.to_crs(crs)


def get_city_boundary_osm(name, simplify=True, crs="EPSG:4326"):
    """Fetch a city's boundary from OSM.

    Returns a tuple of (boundary_multipolygon, center_coordinates, population).
    If a boundary cannot be retrieved, returns (None, None, None).
    """
    try:
        city_gdf = ox.geocode_to_gdf(name)
        if crs is not None:
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
                          explode=False):
    """Download buildings + parks from OSM and categorize them.

    Parameters
    - bbox_or_city: bbox tuple, city name, or shapely polygon
    - crs: output CRS
    - schema: category schema name
    - clip: retained for API parity (bbox path uses Overpass bbox)
    - infer_building_types: heuristics for building="yes"
    - explode: explode MultiPolygons into Polygons
    """
    
    # Determine if input is bbox, city name, or shapely polygon
    if isinstance(bbox_or_city, str):
        boundary, center, population = get_city_boundary_osm(bbox_or_city)
        if boundary is None:
            return gpd.GeoDataFrame()
        # Use city boundary directly instead of converting to bbox
        city_polygon = boundary
    elif hasattr(bbox_or_city, 'geom_type'):
        city_polygon = bbox_or_city
        bbox = None
    else:
        bbox = bbox_or_city
        city_polygon = None

    # by_chunks parameter is deprecated and ignored for simplicity and performance
    
    # Download buildings (including parking lots with building tags)
    building_tags = {"building": True}
    if city_polygon is not None:
        try:
            buildings = ox.features_from_polygon(city_polygon, building_tags)
            buildings = buildings.to_crs(crs)
        except InsufficientResponseError:
            buildings = gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    else:
        buildings = _download_osm_features(bbox, building_tags, crs)
    
    # Download parking lots (even without building tags) - they should be buildings, not parks
    parking_tags = {"amenity": ["parking"]}
    if city_polygon is not None:
        try:
            parking_lots = ox.features_from_polygon(city_polygon, parking_tags)
            parking_lots = parking_lots.to_crs(crs)
        except InsufficientResponseError:
            parking_lots = gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    else:
        parking_lots = _download_osm_features(bbox, parking_tags, crs)
    
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
        try:
            parks = ox.features_from_polygon(city_polygon, park_tags)
            parks = parks.to_crs(crs)
        except InsufficientResponseError:
            parks = gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    else:
        parks = _download_osm_features(bbox, park_tags, crs)
    
    # EXCLUDE WATER FEATURES FROM PARKS - they should not be parks either
    if 'natural' in parks.columns:
        parks = parks[parks['natural'] != 'water']
    if 'waterway' in parks.columns:
        parks = parks[parks['waterway'].isna()]
    
    # Essential columns to keep
    essential_cols = ['geometry', 'osm_type', 'subtype', 'subtype_2', 'subtype_3',
                     'building_type', 'osm_category',
                     'addr:street', 'addr:city', 'addr:state', 'addr:housenumber', 'addr:postcode']
    
    # Categorize buildings
    if len(buildings) > 0:
        classifications = buildings.apply(lambda row: _classify_feature(row, schema, infer_building_types), axis=1)
        buildings['osm_type'] = classifications.apply(lambda x: x[0])
        buildings['subtype'] = classifications.apply(lambda x: x[1])
        buildings['building_type'] = classifications.apply(lambda x: x[2])
        # Keep schema-specific category name for reference (osm_category)
        buildings['osm_category'] = buildings['building_type']
        buildings = _add_secondary_subtypes_columns(buildings, schema)
        buildings = buildings[[col for col in essential_cols if col in buildings.columns]]
    
    # Categorize parks
    if len(parks) > 0:
        classifications = parks.apply(lambda row: _classify_feature(row, schema, infer_building_types), axis=1)
        parks['osm_type'] = classifications.apply(lambda x: x[0])
        parks['subtype'] = classifications.apply(lambda x: x[1])
        parks['building_type'] = classifications.apply(lambda x: x[2])
        # Keep schema-specific category name for reference (osm_category)
        parks['osm_category'] = parks['building_type']
        parks = _add_secondary_subtypes_columns(parks, schema)
        parks = parks[[col for col in essential_cols if col in parks.columns]]
    
    # Combine only AFTER categorization to prevent parks from being misclassified as buildings
    if len(buildings) == 0 and len(parks) == 0:
        result = gpd.GeoDataFrame(columns=['osm_type', 'subtype', 'building_type', 'geometry'], crs=crs)
    elif len(buildings) == 0:
        result = parks
    elif len(parks) == 0:
        result = buildings
    else:
        result = gpd.GeoDataFrame(pd.concat([buildings, parks], ignore_index=True), crs=crs)

    # If requested, ensure all geometries are strictly inside the boundary by exploding then clipping
    if clip and len(result) > 0:
        # Build mask polygon (target CRS)
        if city_polygon is not None:
            mask = gpd.GeoDataFrame(geometry=[city_polygon], crs="EPSG:4326").to_crs(crs)
        else:
            mask = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326").to_crs(crs)
        # explode first, then clip
        result = result.explode(ignore_index=True)
        result = gpd.clip(result, mask)
        result = result[result.geometry.notna() & ~result.geometry.is_empty].reset_index(drop=True)

    # Optional explode for callers who want exploded geometries without clipping
    if (not clip) and explode and len(result) > 0:
        result = result.explode(ignore_index=True)

    return result


def download_osm_streets(bbox_or_city,
                        crs=DEFAULT_CRS,
                        clip=True,
                        clip_to_gdf=None,
                        explode=False,
                        graphml_path=None):
    """Download filtered street network from OSM and return edges as GeoDataFrame."""

    # Determine if input is bbox, city name, or shapely polygon
    if isinstance(bbox_or_city, str):
        boundary, center, population = get_city_boundary_osm(bbox_or_city)
        if boundary is None:
            return gpd.GeoDataFrame()
        city_polygon = boundary
        bbox = None
    elif hasattr(bbox_or_city, 'geom_type'):
        city_polygon = bbox_or_city
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

    # by_chunks parameter is deprecated and ignored for simplicity and performance

    try:
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

        # Project for metric tolerance/lengths and consolidate intersections
        Gp = ox.project_graph(G)
        try:
            Gc = ox.simplification.consolidate_intersections(
                Gp,
                tolerance=INTERSECTION_CONSOLIDATION_TOLERANCE_M,
                rebuild_graph=True
            )
        except Exception:
            Gc = Gp

        # Optionally persist the consolidated OSMnx graph as GraphML (projected CRS)
        if graphml_path:
            try:
                # OSMnx API: save_graphml at top-level
                ox.save_graphml(Gc, filepath=str(graphml_path))
            except Exception:
                pass

        # Ensure edge lengths exist (meters in projected CRS)
        try:
            ox.distance.add_edge_lengths(Gc)
        except Exception:
            pass

        # Extract nodes/edges, prune short edges, then return edges GDF
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(Gc)
        if 'length' in edges_gdf.columns:
            edges_gdf = edges_gdf[edges_gdf['length'] >= STREET_MIN_LENGTH_M]
        else:
            # Fallback by geometry length in projected CRS
            edges_gdf = edges_gdf[edges_gdf.geometry.length >= STREET_MIN_LENGTH_M]

        streets = edges_gdf

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
# SIMPLE I/O HELPERS (GeoJSON, GeoParquet, Shapefile, GeoPackage)
# =============================================================================

def save_geodata(gdf: gpd.GeoDataFrame, path: str, layer: str = None):
    """Persist a GeoDataFrame to disk based on file extension.

    Supported formats:
    - .geojson/.json -> GeoJSON
    - .parquet/.geoparquet -> GeoParquet
    - .shp -> ESRI Shapefile (multiple files created alongside)
    - .gpkg -> GeoPackage (layer optional, defaults to 'data')
    """
    if gdf is None:
        raise ValueError("gdf cannot be None")
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if ext in ('.geojson', '.json'):
        gdf.to_file(path, driver='GeoJSON')
    elif ext in ('.parquet', '.geoparquet'):
        gdf.to_parquet(path)
    elif ext == '.shp':
        gdf.to_file(path, driver='ESRI Shapefile')
    elif ext == '.gpkg':
        gdf.to_file(path, layer=(layer or 'data'), driver='GPKG')
    else:
        raise ValueError(f"Unsupported geodata format: {ext}")


def load_geodata(path: str, layer: str = None) -> gpd.GeoDataFrame:
    """Load a GeoDataFrame from disk based on file extension.

    Supports .geojson/.json, .parquet/.geoparquet, .shp, .gpkg
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.geojson', '.json', '.shp', '.gpkg'):
        kwargs = {}
        if ext == '.gpkg' and layer is not None:
            kwargs['layer'] = layer
        return gpd.read_file(path, **kwargs)
    elif ext in ('.parquet', '.geoparquet'):
        return gpd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported geodata format: {ext}")


# =============================================================================
# GEOMETRY PROCESSING FUNCTIONS
# =============================================================================

def remove_overlaps(gdf, exclude_categories= None):
    """Remove polygons fully contained within others, keeping one of any identical shapes.

    If exclude_categories is provided and a 'category' column exists, rows in
    those categories are not considered for removal. Uses spatial indexing for
    efficiency.
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
    # Perform overlap removal on a GeoDataFrame (helper for remove_overlaps).
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
    """Rotate all geometries around a single origin.

    Parameters
    - gdf: input GeoDataFrame
    - rotation_deg: degrees counterclockwise
    - origin: 'centroid' or a tuple of (x, y)
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
    """Return a dict of building_type counts."""
    return gdf['building_type'].value_counts().to_dict()


def get_subtype_summary(gdf):
    """Return a dict of subtype counts."""
    if 'subtype' not in gdf.columns or len(gdf) == 0:
        return {}
    return gdf['subtype'].value_counts().to_dict()


# Removed unused prominence helper to keep module focused on core use cases


def rotate_streets_to_align(streets_gdf, k=200):
    """Estimate grid alignment from street bearings and rotate the network.

    Returns a tuple of (rotated_streets_gdf, rotation_degrees).
    """
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


# =============================================================================
# GARDEN CITY COORDINATE TRANSFORMATION UTILITIES
# =============================================================================

def blocks_to_mercator(data, block_size=15.0, false_easting=-4265699.0, false_northing=4392976.0):
    """
    Convert city block coordinates to Web Mercator coordinates.
    
    This function applies an affine transformation to convert abstract city block
    coordinates (in units of blocks) to Web Mercator projection coordinates (EPSG:3857)
    in meters. The transformation is: x_mercator = block_size * x_block + false_easting
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'x', 'y' columns in city block coordinates
    block_size : float, default 15.0
        Size of one city block in meters
    false_easting : float, default -4265699.0
        False easting offset (x-origin) in Web Mercator meters
    false_northing : float, default 4392976.0
        False northing offset (y-origin) in Web Mercator meters
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'x', 'y' columns updated to Web Mercator coordinates.
        If 'ha' column exists, it is also scaled by block_size.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2]})
    >>> result = blocks_to_mercator(df, block_size=15, false_easting=-4265699, false_northing=4392976)
    >>> result['x'].tolist()
    [-4265699.0, -4265684.0, -4265669.0]
    """
    # Validate required columns
    if 'x' not in data.columns or 'y' not in data.columns:
        raise ValueError("DataFrame must contain 'x' and 'y' columns")
    
    # Create a copy to avoid modifying original
    result = data.copy()
    
    # Apply affine transformation: mercator = block_size * block + origin
    result['x'] = block_size * result['x'] + false_easting
    result['y'] = block_size * result['y'] + false_northing
    
    # Scale horizontal accuracy if present
    if 'ha' in result.columns:
        result['ha'] = block_size * result['ha']
    
    return result


def mercator_to_blocks(data, block_size=15.0, false_easting=-4265699.0, false_northing=4392976.0):
    """
    Convert Web Mercator coordinates back to city block coordinates.
    
    This function applies the inverse affine transformation to convert Web Mercator
    projection coordinates (EPSG:3857) in meters back to abstract city block
    coordinates. The transformation is: x_block = (x_mercator - false_easting) / block_size
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'x', 'y' columns in Web Mercator coordinates
    block_size : float, default 15.0
        Size of one city block in meters
    false_easting : float, default -4265699.0
        False easting offset (x-origin) in Web Mercator meters
    false_northing : float, default 4392976.0
        False northing offset (y-origin) in Web Mercator meters
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'x', 'y' columns updated to city block coordinates.
        If 'ha' column exists, it is also scaled back by dividing by block_size.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [-4265699.0, -4265684.0], 'y': [4392976.0, 4392991.0]})
    >>> result = mercator_to_blocks(df, block_size=15, false_easting=-4265699, false_northing=4392976)
    >>> result['x'].tolist()
    [0.0, 1.0]
    """
    # Validate required columns
    if 'x' not in data.columns or 'y' not in data.columns:
        raise ValueError("DataFrame must contain 'x' and 'y' columns")
    
    # Create a copy to avoid modifying original
    result = data.copy()
    
    # Apply inverse affine transformation: block = (mercator - origin) / block_size
    result['x'] = (result['x'] - false_easting) / block_size
    result['y'] = (result['y'] - false_northing) / block_size
    
    # Scale horizontal accuracy back if present
    if 'ha' in result.columns:
        result['ha'] = result['ha'] / block_size
    
    return result
