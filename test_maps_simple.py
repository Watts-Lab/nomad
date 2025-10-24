#!/usr/bin/env python3
"""
Simple test script for nomad.maps OSMDataDownloader functionality.
Tests with a small but populated area in Philadelphia.
"""

import nomad.maps as nm

def test_simple():
    """Simple test of OSMDataDownloader functionality."""

    # Use a slightly larger bbox to ensure we get some buildings
    # This covers University of Pennsylvania area which should have buildings
    bbox = (
        -75.1939,  # west
        39.9500,  # south
        -75.1720, # east
        39.9600   # north
    )

    print(f"Testing with bbox: {bbox}")
    print("University of Pennsylvania area in Philadelphia\n")

    # Initialize downloader
    downloader = nm.OSMDataDownloader(bbox=bbox, crs="EPSG:3857")

    # Test 1: Download all buildings first (most reliable)
    print("1. Testing download_all_buildings()...")
    try:
        all_buildings = downloader.download_all_buildings()
        total = sum(len(gdf) for gdf in all_buildings.values())
        print(f"   ✓ Found {total} total buildings across {len(all_buildings)} categories:")

        for category, gdf in all_buildings.items():
            print(f"     - {category}: {len(gdf)} buildings")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 2: Download parks
    print("\n2. Testing download_parks()...")
    try:
        parks = downloader.download_parks()
        print(f"   ✓ Found {len(parks)} parks")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 3: Download all data
    print("\n3. Testing download_all_data()...")
    try:
        all_data = downloader.download_all_data()
        total_features = sum(len(gdf) for gdf in all_data.values())
        print(f"   ✓ Found {total_features} total features:")

        for data_type, gdf in all_data.items():
            print(f"     - {data_type}: {len(gdf)} features")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n✓ Test completed!")

if __name__ == "__main__":
    test_simple()
