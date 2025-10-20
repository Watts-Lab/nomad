"""
OpenStreetMap data utilities for the NOMAD library.

This module provides functionality to download and process OpenStreetMap data,
particularly for building geometries and street networks. It includes utilities
for categorizing buildings and downloading data from OpenStreetMap.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import osmnx as ox


class OSMDataDownloader:
    """
    A class to download and categorize OpenStreetMap building data.
    
    This class provides methods to download building geometries from OpenStreetMap
    using OSMnx and categorize them into different types (residential, retail,
    workplace, etc.).
    """
    
    def __init__(self, bbox, crs="EPSG:3857"):
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
        self._building_tags = self._get_building_tags()
        
    def _get_building_tags(self):
        """Define building type categories and their corresponding OSM tags."""
        return {
            "residential": {
                "building": [
                    "residential", "house", "hotel", "detached", 
                    "semidetached_house", "terrace", "apartments", 
                    "bungalow", "dormitory", "cabin", "farm"
                ]
            },
            "retail": {
                "building": [
                    "retail", "shop", "supermarket", "mall", 
                    "department_store", "kiosk", "restaurant"
                ]
            },
            "workplace": {
                "building": [
                    "office", "industrial", "commercial", "public", 
                    "hospital", "school", "university", "government", 
                    "civic", "warehouse", "factory", "workshop"
                ]
            }
        }
    
    def download_parks(self):
        """
        Download park geometries from OpenStreetMap.
        
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing park geometries
        """
        parks = ox.features_from_bbox(bbox=self.bbox, tags={"leisure": "park"})
        return parks.to_crs(self.crs)
    
    def download_buildings_by_type(self, building_type):
        """
        Download buildings of a specific type from OpenStreetMap.
        
        Parameters
        ----------
        building_type : str
            Type of building to download ("residential", "retail", "workplace")
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing building geometries of the specified type
        """
        if building_type not in self._building_tags:
            raise ValueError(f"Unknown building type: {building_type}")
        
        tags = self._building_tags[building_type]
        buildings = ox.features_from_bbox(bbox=self.bbox, tags=tags)
        buildings = buildings.to_crs(self.crs)
        
        # Filter to only include Polygon and MultiPolygon geometries
        buildings = buildings[
            buildings.geometry.apply(lambda geom: geom.geom_type in ['Polygon', 'MultiPolygon'])
        ]
        
        return buildings
    
    def download_all_buildings(self):
        """
        Download all categorized buildings from OpenStreetMap.
        
        Returns
        -------
        dict
            Dictionary with building type as key and GeoDataFrame as value
        """
        results = {}
        
        # Download categorized buildings
        for building_type in self._building_tags.keys():
            results[building_type] = self.download_buildings_by_type(building_type)
        
        # Download all buildings to find uncategorized ones
        all_buildings = ox.features_from_bbox(bbox=self.bbox, tags={"building": True})
        all_buildings = all_buildings.to_crs(self.crs)
        all_buildings = all_buildings[
            all_buildings.geometry.apply(lambda geom: geom.geom_type in ['Polygon', 'MultiPolygon'])
        ]
        
        # Find uncategorized buildings
        classified_osmids = set()
        for gdf in results.values():
            if not gdf.empty:
                classified_osmids.update(gdf.index.unique())
        
        other_buildings = all_buildings[~all_buildings.index.isin(classified_osmids)]
        results["other"] = other_buildings
        
        return results
    
    def download_all_data(self):
        """
        Download all data (parks and buildings) from OpenStreetMap.
        
        Returns
        -------
        dict
            Dictionary containing all downloaded geometries
        """
        data = {"parks": self.download_parks()}
        data.update(self.download_all_buildings())
        return data

