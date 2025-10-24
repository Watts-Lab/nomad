#!/usr/bin/env python3
"""
Test script for nomad.maps OSMDataDownloader
Demonstrates downloading and categorizing buildings and parks from OpenStreetMap.

Usage:
    python test_nomad_maps.py

This will download data for a small area in Philadelphia and show the results.
"""

import nomad.maps as nm

def main():
    """Test the OSMDataDownloader functionality."""

    # Define bounding box for University of Pennsylvania area (guaranteed to have data)
    bbox = (
        -75.1939,  # west
        39.9500,  # south
        -75.1720, # east
        39.9600   # north
    )

    print("🗺️  Testing NOMAD Maps - OSM Data Downloader")
    print(f"📍 Location: University of Pennsylvania area, Philadelphia")
    print(f"📦 Bounding Box: {bbox}")
    print()

    # Initialize the downloader
    downloader = nm.OSMDataDownloader(bbox=bbox, crs="EPSG:3857")

    # Test 1: Download all categorized buildings
    print("🏢 1. Downloading all buildings by category...")
    all_buildings = downloader.download_all_buildings()

    total_buildings = 0
    for category, gdf in all_buildings.items():
        count = len(gdf)
        total_buildings += count
        print(f"   • {category}: {count} buildings")

    print(f"   📊 Total: {total_buildings} buildings")
    print()

    # Test 2: Download parks
    print("🌳 2. Downloading parks...")
    parks = downloader.download_parks()
    print(f"   • Found {len(parks)} parks")
    print()

    # Test 3: Download everything
    print("📥 3. Downloading all data (parks + buildings)...")
    all_data = downloader.download_all_data()

    total_features = 0
    for data_type, gdf in all_data.items():
        count = len(gdf)
        total_features += count
        print(f"   • {data_type}: {count} features")

    print(f"   📊 Grand Total: {total_features} features")
    print()

    # Show sample data structure
    print("🔍 Sample data structure:")
    if len(all_buildings['residential']) > 0:
        print(f"   • Residential buildings columns: {list(all_buildings['residential'].columns)}")
        print(f"   • Sample residential building OSM ID: {all_buildings['residential'].index[0]}")

    if len(parks) > 0:
        print(f"   • Parks columns: {list(parks.columns)}")
        print(f"   • Sample park: {parks.iloc[0].get('name', 'Unnamed')}")

    print()
    print("✅ Test completed successfully!")
    print("💡 The downloaded GeoDataFrames are ready for spatial analysis and visualization.")

if __name__ == "__main__":
    main()
