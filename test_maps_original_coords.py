#!/usr/bin/env python3
"""
Test script using the exact coordinates provided by the user.
Demonstrates the functionality even with a very small area.
"""

import nomad.maps as nm

def test_original_coordinates():
    """Test using the exact coordinates provided by the user."""

    # Use the exact coordinates from the user's request
    # Point 1: 39.92149038069586, -75.18063830113564 (north-west)
    # Point 2: 39.90035534162599, -75.15709001589359 (south-east)
    bbox = (
        -75.18063830113564,  # west
        39.90035534162599,   # south
        -75.15709001589359,  # east
        39.92149038069586    # north
    )

    print(f"Testing with original coordinates bbox: {bbox}")
    print("This is a very small area in Philadelphia (~2.3km x 2.4km)\n")

    downloader = nm.OSMDataDownloader(bbox=bbox, crs="EPSG:3857")

    # Test downloading all buildings (most comprehensive)
    print("Testing download_all_buildings()...")
    all_buildings = downloader.download_all_buildings()

    print("Results by category:")
    total = 0
    for category, gdf in all_buildings.items():
        count = len(gdf)
        total += count
        print(f"  {category}: {count} buildings")

        # Show a sample if available
        if count > 0 and count <= 3:
            print(f"    Sample OSM IDs: {list(gdf.index[:3])}")

    print(f"\nTotal buildings found: {total}")

    # Test parks
    print("\nTesting parks...")
    parks = downloader.download_parks()
    print(f"Parks found: {len(parks)}")

    return all_buildings

if __name__ == "__main__":
    test_original_coordinates()
