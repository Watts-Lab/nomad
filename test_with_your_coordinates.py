#!/usr/bin/env python3
"""
Test script using your exact coordinates for a minimal test.
"""

import nomad.maps as nm

def test_your_coordinates():
    """Test with your exact Philadelphia coordinates."""

    # Your exact coordinates
    # Point 1: 39.92149038069586, -75.18063830113564 (north-west)
    # Point 2: 39.90035534162599, -75.15709001589359 (south-east)

    bbox = (
        -75.18063830113564,  # west
        39.90035534162599,   # south
        -75.15709001589359,  # east
        39.92149038069586    # north
    )

    print("🗺️  Testing with your exact coordinates")
    print(f"📦 Bounding Box: {bbox}")
    print()

    downloader = nm.OSMDataDownloader(bbox=bbox, crs="EPSG:3857")

    # Test individual categories with error handling
    print("🏢 Buildings by category:")
    for category in ["residential", "retail", "workplace"]:
        try:
            buildings = downloader.download_buildings_by_type(category)
            print(f"   • {category}: {len(buildings)} buildings")
        except:
            print(f"   • {category}: 0 buildings (no data in this small area)")

    # Download all buildings (more reliable)
    print("\n🏢 All buildings (including uncategorized):")
    try:
        all_buildings = downloader.download_all_buildings()
        for category, gdf in all_buildings.items():
            print(f"   • {category}: {len(gdf)} buildings")
    except Exception as e:
        print(f"   ⚠ Some categories failed: {str(e)[:50]}...")

        # Fallback to direct download
        import osmnx as ox
        direct_buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
        direct_buildings = direct_buildings.to_crs(downloader.crs)
        direct_buildings = direct_buildings[
            direct_buildings.geometry.apply(lambda geom: geom.geom_type in ['Polygon', 'MultiPolygon'])
        ]
        print(f"   • Direct download: {len(direct_buildings)} total buildings")

    # Test parks
    print("\n🌳 Parks:")
    try:
        parks = downloader.download_parks()
        print(f"   • Found {len(parks)} parks")
    except:
        print("   • 0 parks found")

    print("\n✅ Test completed!")
    print("💡 Small areas may not have all building types - this is normal!")

if __name__ == "__main__":
    test_your_coordinates()
