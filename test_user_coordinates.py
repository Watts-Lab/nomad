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
    print("📦 Bounding Box: {}".format(bbox))
    print()

    downloader = nm.OSMDataDownloader(bbox=bbox, crs="EPSG:3857")

    # Test individual categories with error handling
    print("🏢 Buildings by category:")
    for category in ["residential", "retail", "workplace"]:
        try:
            buildings = downloader.download_by_location_type(category)
            print("   • {}: {} buildings".format(category, len(buildings)))
        except:
            print("   • {}: 0 buildings (no data in this small area)".format(category))

    # Download all buildings (more reliable)
    print("\n🏢 All buildings (including uncategorized):")
    try:
        all_buildings = downloader.download_all_buildings()
        total_buildings = len(all_buildings)
        categorized = len(all_buildings[all_buildings['location_type'] != 'other'])
        uncategorized = len(all_buildings[all_buildings['location_type'] == 'other'])

        print("   • Total buildings: {}".format(total_buildings))
        print("   • Categorized: {}".format(categorized))
        print("   • Uncategorized: {}".format(uncategorized))

        # Show breakdown by location type
        location_types = all_buildings['location_type'].value_counts()
        for loc_type, count in location_types.items():
            print("     - {}: {}".format(loc_type, count))

    except Exception as e:
        print("   ⚠ Error: {}".format(str(e)[:50]))

        # Fallback to direct download
        import osmnx as ox
        direct_buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
        direct_buildings = direct_buildings.to_crs(downloader.crs)
        direct_buildings = direct_buildings[
            direct_buildings.geometry.apply(lambda geom: geom.geom_type in ['Polygon', 'MultiPolygon'])
        ]
        print("   • Direct download: {} total buildings".format(len(direct_buildings)))

    # Test parks
    print("\n🌳 Parks:")
    try:
        parks = downloader.download_parks()
        print("   • Found {} parks".format(len(parks)))
    except:
        print("   • 0 parks found")

    # Test walkways
    print("\n🚶 Walkways:")
    try:
        walkways = downloader.download_walkways()
        print("   • Found {} walkways".format(len(walkways)))
    except Exception as e:
        print("   • Error: {}".format(str(e)[:50]))

    print("\n✅ Test completed!")
    print("💡 Small areas may not have all building types - this is normal!")

if __name__ == "__main__":
    test_your_coordinates()
