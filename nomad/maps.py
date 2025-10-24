"""
OpenStreetMap data utilities for the NOMAD library.

This module provides comprehensive functionality to download and process OpenStreetMap data,
including all buildings, walkways, and parks. It categorizes features into location types
with detailed business classification based on the official OSM wiki.

Based on: https://wiki.openstreetmap.org/wiki/Map_features
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import osmnx as ox
from nomad.map_constants import (
    LOCATION_TYPE_MAPPING, BUILDING_CATEGORIES, BUSINESS_CLASSIFICATIONS,
    DEFAULT_CRS
)


class OSMDataDownloader:
    """
    A comprehensive class to download and categorize OpenStreetMap data.

    This class downloads ALL buildings, walkways, and parks from OpenStreetMap and
    categorizes them into location types with detailed business classification.

    Main Features:
    - Downloads ALL buildings (not just categorized ones)
    - Categorizes into 5 location types: residential, retail, workplace, park, walkway
    - Provides building category and subcategory classification
    - Includes business classification with NAICS code equivalents
    - Supports flexible custom category mapping
    - Handles all geometry types (Point, Line, Polygon, MultiPolygon)

    Location Types:
    - residential: Places where people live (houses, apartments, hotels)
    - retail: Places where people shop or acquire services (shops, restaurants, services)
    - workplace: Places where people work or study (offices, schools, hospitals)
    - park: Open spaces and recreational areas (parks, forests, recreation grounds)
    - walkway: Roads, paths, and transportation infrastructure
    - other: Features that don't fit main categories

    Business Classification:
    - Provides NAICS code equivalents for business types
    - Detailed categorization using shop=*, amenity=*, office=* tags
    - Supports retail, workplace, and service classifications

    Based on: https://wiki.openstreetmap.org/wiki/Map_features
    """

    def __init__(self, bbox, crs=DEFAULT_CRS):
        """
        Initialize the OSM data downloader.

        Parameters
        ----------
        bbox : tuple
            Bounding box as (west, south, east, north) in WGS84 coordinates
        crs : str, default "EPSG:3857"
            Target CRS for the downloaded geometries
        """
        self.bbox = bbox
        self.crs = crs

    def _classify_location_type(self, row):
        """
        Classify a feature into a location type based on its OSM tags.

        Uses the comprehensive LOCATION_TYPE_MAPPING from map_constants.py

        Parameters
        ----------
        row : pd.Series
            Row from a GeoDataFrame with OSM tags

        Returns
        -------
        str
            Location type: residential, retail, workplace, park, walkway, or other
        """
        # Check each location type in priority order
        for location_type in LOCATION_TYPE_MAPPING.keys():
            tag_groups = LOCATION_TYPE_MAPPING[location_type]

            for tag_key, tag_values in tag_groups.items():
                if tag_key in row and pd.notna(row[tag_key]):
                    if tag_values is True:  # Any value matches (e.g., any shop)
                        return location_type
                    elif isinstance(tag_values, list) and row[tag_key] in tag_values:
                        return location_type

        return "other"

    def _get_building_category_info(self, row):
        """
        Get building category and subcategory information.

        Parameters
        ----------
        row : pd.Series
            Row from a GeoDataFrame with OSM tags

        Returns
        -------
        tuple
            (category, subcategory) strings or (None, None) if not found
        """
        # Check building tag first
        if "building" in row and pd.notna(row["building"]):
            building_value = row["building"]
            if building_value in BUILDING_CATEGORIES:
                info = BUILDING_CATEGORIES[building_value]
                return info["category"], info["subcategory"]

        # Check amenity tag
        if "amenity" in row and pd.notna(row["amenity"]):
            amenity_value = row["amenity"]
            if amenity_value in BUILDING_CATEGORIES:
                info = BUILDING_CATEGORIES[amenity_value]
                return info["category"], info["subcategory"]

        return None, None

    def _get_business_classification_info(self, row):
        """
        Get detailed business classification including NAICS equivalent.

        Parameters
        ----------
        row : pd.Series
            Row from a GeoDataFrame with OSM tags

        Returns
        -------
        tuple
            (category, subcategory, naics_equivalent) or (None, None, None) if not found
        """
        # Check shop tag first (most specific)
        if "shop" in row and pd.notna(row["shop"]):
            shop_value = row["shop"]
            if shop_value in BUSINESS_CLASSIFICATIONS:
                info = BUSINESS_CLASSIFICATIONS[shop_value]
                return info["category"], info["subcategory"], info.get("naics_equivalent")

        # Check amenity tag
        if "amenity" in row and pd.notna(row["amenity"]):
            amenity_value = row["amenity"]
            if amenity_value in BUSINESS_CLASSIFICATIONS:
                info = BUSINESS_CLASSIFICATIONS[amenity_value]
                return info["category"], info["subcategory"], info.get("naics_equivalent")

        # Check office tag
        if "office" in row and pd.notna(row["office"]):
            office_value = row["office"]
            if office_value in BUSINESS_CLASSIFICATIONS:
                info = BUSINESS_CLASSIFICATIONS[office_value]
                return info["category"], info["subcategory"], info.get("naics_equivalent")

        # Check building tag as fallback
        if "building" in row and pd.notna(row["building"]):
            building_value = row["building"]
            if building_value in BUILDING_CATEGORIES:
                info = BUILDING_CATEGORIES[building_value]
                return info["category"], info["subcategory"], None

        return None, None, None

    def download_all_buildings(self, include_all_geometries=True):
        """
        Download ALL buildings from OpenStreetMap and categorize them.

        This method downloads all features with building=* and categorizes them
        into location types with detailed classification.

        Parameters
        ----------
        include_all_geometries : bool, default True
            Whether to include all geometry types (Point, Line, Polygon, MultiPolygon)
            or just areas (Polygon, MultiPolygon)

        Returns
        -------
        gpd.GeoDataFrame
            All buildings with location_type, building_category, building_subcategory columns
        """
        # Download ALL buildings (not just categorized ones)
        buildings = ox.features_from_bbox(bbox=self.bbox, tags={"building": True})
        buildings = buildings.to_crs(self.crs)

        if not include_all_geometries:
            # Filter to only area geometries (Polygon, MultiPolygon)
            buildings = buildings[
                buildings.geometry.apply(lambda geom: geom.geom_type in ['Polygon', 'MultiPolygon'])
            ]

        # Add location type classification
        buildings['location_type'] = buildings.apply(self._classify_location_type, axis=1)

        # Add building category and subcategory
        category_info = buildings.apply(self._get_building_category_info, axis=1, result_type='expand')
        buildings['building_category'] = category_info[0]
        buildings['building_subcategory'] = category_info[1]

        # Add detailed business classification (including NAICS equivalents)
        business_info = buildings.apply(self._get_business_classification_info, axis=1, result_type='expand')
        buildings['business_category'] = business_info[0]
        buildings['business_subcategory'] = business_info[1]
        buildings['naics_equivalent'] = business_info[2]

        # Fill missing categories with location_type for consistency
        buildings['building_category'] = buildings['building_category'].fillna(buildings['location_type'])
        buildings['business_category'] = buildings['business_category'].fillna(buildings['building_category'])

        return buildings

    def download_walkways(self):
        """
        Download all walkways, roads, and paths from OpenStreetMap.

        Returns
        -------
        gpd.GeoDataFrame
            All walkway features with location_type="walkway"
        """
        # Download all highways (roads, paths, etc.)
        walkways = ox.features_from_bbox(bbox=self.bbox, tags={"highway": True})
        walkways = walkways.to_crs(self.crs)

        # Add location type
        walkways['location_type'] = 'walkway'

        return walkways

    def download_parks(self):
        """
        Download all park and open space areas from OpenStreetMap.

        This includes leisure=park, landuse=park, natural areas, etc.

        Returns
        -------
        gpd.GeoDataFrame
            All park features with location_type="park"
        """
        # Build comprehensive park tags by flattening the nested structure
        park_tags = {}
        for tag_key, tag_values in LOCATION_TYPE_MAPPING["park"].items():
            if isinstance(tag_values, list):
                # For multiple values of the same tag, use True to match any
                park_tags[tag_key] = True
            else:
                # For boolean values, just use the key
                park_tags[tag_key] = tag_values

        parks = ox.features_from_bbox(bbox=self.bbox, tags=park_tags)
        parks = parks.to_crs(self.crs)

        # Add location type
        parks['location_type'] = 'park'

        return parks

    def download_by_location_type(self, location_type):
        """
        Download features of a specific location type.

        Parameters
        ----------
        location_type : str
            One of: "residential", "retail", "workplace", "park", "walkway"

        Returns
        -------
        gpd.GeoDataFrame
            Features of the specified location type
        """
        if location_type == "walkway":
            return self.download_walkways()
        elif location_type == "park":
            return self.download_parks()
        elif location_type in ["residential", "retail", "workplace"]:
            # For building types, use the comprehensive download and filter
            all_buildings = self.download_all_buildings()
            return all_buildings[all_buildings['location_type'] == location_type]
        else:
            raise ValueError("Unknown location type: {}".format(location_type))

    def download_all_data(self):
        """
        Download all features (buildings, walkways, parks) with unified classification.

        This is the main method that provides a comprehensive dataset with:
        - All buildings (categorized and uncategorized)
        - All walkways (roads, paths)
        - All parks (open spaces)

        Returns
        -------
        gpd.GeoDataFrame
            All features with location_type, building_category, building_subcategory columns
        """
        # Download each feature type
        buildings = self.download_all_buildings()
        walkways = self.download_walkways()
        parks = self.download_parks()

        # Combine all features
        all_features = gpd.GeoDataFrame(
            pd.concat([buildings, walkways, parks], ignore_index=True),
            crs=self.crs
        )

        return all_features

    def get_location_type_summary(self):
        """
        Get a summary of features by location type.

        Returns
        -------
        dict
            Count of features by location type
        """
        all_data = self.download_all_data()
        return all_data['location_type'].value_counts().to_dict()

    def add_custom_location_type(self, name, tags):
        """
        Add a custom location type mapping.

        Parameters
        ----------
        name : str
            Name of the new location type
        tags : dict
            OSM tags mapping (e.g., {"building": ["hospital"], "amenity": ["clinic"]})
        """
        LOCATION_TYPE_MAPPING[name] = tags

    def get_business_classification_summary(self):
        """
        Get a summary of business classifications (NAICS equivalents).

        Returns
        -------
        dict
            Count of features by NAICS equivalent code
        """
        all_buildings = self.download_all_buildings()
        naics_counts = all_buildings['naics_equivalent'].value_counts()
        return naics_counts[naics_counts.index.notna()].to_dict()

    def get_detailed_classification_summary(self):
        """
        Get detailed summary of all classification types.

        Returns
        -------
        dict
            Nested dictionary with counts by location_type, building_category, business_category
        """
        all_data = self.download_all_data()

        summary = {
            'location_type': all_data['location_type'].value_counts().to_dict(),
            'building_category': all_data['building_category'].value_counts().to_dict(),
            'business_category': all_data['business_category'].value_counts().to_dict()
        }

        # Add NAICS breakdown
        naics_data = all_data[all_data['naics_equivalent'].notna()]
        if not naics_data.empty:
            summary['naics_equivalent'] = naics_data['naics_equivalent'].value_counts().to_dict()

        return summary

