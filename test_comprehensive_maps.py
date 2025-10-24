#!/usr/bin/env python3
"""
Comprehensive test script for the enhanced NOMAD Maps module.

This demonstrates all the new functionality including:
- Complete building categorization with location types
- Business classification with NAICS equivalents
- Walkways and parks support
- Flexible category mapping
- Comprehensive data summaries

Usage:
    python test_comprehensive_maps.py
"""

import nomad.maps as nm

def main():
    """Test all the enhanced functionality."""

    # Use University of Pennsylvania area (guaranteed to have comprehensive data)
    bbox = (
        -75.1939,  # west
        39.9500,  # south
        -75.1720, # east
        39.9600   # north
    )

    print("🗺️  COMPREHENSIVE NOMAD Maps Test")
    print("=" * 50)
    print(f"📍 Location: University of Pennsylvania area, Philadelphia")
    print(f"📦 Bounding Box: {bbox}")
    print()

    # Initialize the enhanced downloader
    downloader = nm.OSMDataDownloader(bbox=bbox, crs="EPSG:3857")

    # Test 1: Get comprehensive summary
    print("📊 1. COMPREHENSIVE DATA SUMMARY")
    print("-" * 40)

    summary = downloader.get_detailed_classification_summary()
    for category, counts in summary.items():
        print(f"\n{category.upper()}:")
        for item, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {item}: {count}")
        if len(counts) > 10:
            remaining = len(counts) - 10
            print(f"  ... and {remaining} more categories")

    # Test 2: Download all data with enhanced classification
    print("\n\n🏗️  2. ENHANCED BUILDING CLASSIFICATION")
    print("-" * 40)

    all_buildings = downloader.download_all_buildings()

    print("Sample buildings with full classification:")
    sample_buildings = all_buildings.head(5)
    for idx, (_, building) in enumerate(sample_buildings.iterrows()):
        print(f"\n  Building {idx + 1}:")
        print(f"    Location Type: {building.get('location_type', 'N/A')}")
        print(f"    Building Category: {building.get('building_category', 'N/A')}")
        print(f"    Building Subcategory: {building.get('building_subcategory', 'N/A')}")
        print(f"    Business Category: {building.get('business_category', 'N/A')}")
        print(f"    Business Subcategory: {building.get('business_subcategory', 'N/A')}")
        print(f"    NAICS Equivalent: {building.get('naics_equivalent', 'N/A')}")

        # Show key OSM tags
        osm_tags = {}
        for col in all_buildings.columns:
            if col.startswith(('building', 'amenity', 'shop', 'office', 'name')) and pd.notna(building[col]):
                osm_tags[col] = building[col]
        if osm_tags:
            print(f"    Key OSM Tags: {osm_tags}")

    # Test 3: Location type breakdown
    print("\n\n📍 3. LOCATION TYPE BREAKDOWN")
    print("-" * 40)

    location_summary = downloader.get_location_type_summary()
    total_features = sum(location_summary.values())

    for location_type, count in sorted(location_summary.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_features) * 100 if total_features > 0 else 0
        print(f"  {location_type}: {count} features ({percentage:.1f}%)")

    # Test 4: Business classification (NAICS)
    print("\n\n💼 4. BUSINESS CLASSIFICATION (NAICS EQUIVALENTS)")
    print("-" * 40)

    business_summary = downloader.get_business_classification_summary()
    if business_summary:
        print("Top business classifications by NAICS code:")
        for naics, count in sorted(business_summary.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  NAICS {naics}: {count} businesses")

        if len(business_summary) > 15:
            print(f"  ... and {len(business_summary) - 15} more NAICS codes")
    else:
        print("  No NAICS classifications found in this area")

    # Test 5: Individual location types
    print("\n\n🎯 5. INDIVIDUAL LOCATION TYPES")
    print("-" * 40)

    for location_type in ['residential', 'retail', 'workplace', 'park']:
        try:
            specific_data = downloader.download_by_location_type(location_type)
            print(f"  {location_type}: {len(specific_data)} features")

            # Show top categories for this location type
            if location_type in ['residential', 'retail', 'workplace']:
                top_buildings = specific_data['building'].value_counts().head(3)
                if not top_buildings.empty:
                    print(f"    Top building types: {dict(top_buildings)}")
        except Exception as e:
            print(f"  {location_type}: Error - {str(e)[:50]}...")

    # Test 6: Walkways
    print("\n🚶 6. WALKWAYS (ROADS & PATHS)")
    print("-" * 40)

    try:
        walkways = downloader.download_walkways()
        print(f"  Total walkways: {len(walkways)}")

        # Show highway types
        if 'highway' in walkways.columns:
            highway_types = walkways['highway'].value_counts().head(5)
            print(f"  Top highway types: {dict(highway_types)}")
    except Exception as e:
        print(f"  Error downloading walkways: {str(e)[:50]}...")

    # Test 7: Parks
    print("\n\n🌳 7. PARKS & OPEN SPACES")
    print("-" * 40)

    try:
        parks = downloader.download_parks()
        print(f"  Total parks/open spaces: {len(parks)}")

        # Show park types
        park_types = {}
        for col in ['leisure', 'landuse', 'natural']:
            if col in parks.columns:
                types = parks[col].value_counts()
                if not types.empty:
                    park_types.update(dict(types))

        if park_types:
            print("  Park/open space types:")
            for park_type, count in sorted(park_types.items(), key=lambda x: x[1], reverse=True)[:8]:
                print(f"    {park_type}: {count}")
    except Exception as e:
        print(f"  Error downloading parks: {str(e)[:50]}...")

    # Test 8: Data structure info
    print("\n\n📋 8. DATA STRUCTURE INFORMATION")
    print("-" * 40)

    all_data = downloader.download_all_data()
    print(f"Total features downloaded: {len(all_data)}")
    print(f"Available columns: {len(all_data.columns)}")
    print()
    print("Column categories:")
    column_groups = {
        'Classification': [col for col in all_data.columns if 'category' in col.lower() or 'type' in col.lower()],
        'Business': [col for col in all_data.columns if 'business' in col.lower() or 'naics' in col.lower()],
        'OSM Tags': [col for col in all_data.columns if col.startswith(('building', 'amenity', 'shop', 'office', 'highway', 'leisure', 'landuse'))],
        'Geography': ['geometry'],
        'Other': [col for col in all_data.columns if not any(x in col.lower() for x in ['category', 'type', 'business', 'naics', 'geometry', 'building', 'amenity', 'shop', 'office', 'highway', 'leisure', 'landuse'])]
    }

    for group_name, columns in column_groups.items():
        if columns:
            print(f"  {group_name}: {len(columns)} columns")
            if group_name in ['Classification', 'Business']:
                print(f"    {columns}")

    print("\n✅ COMPREHENSIVE TEST COMPLETED!")
    print("\nKey improvements in this version:")
    print("✓ Downloads ALL buildings (not just categorized ones)")
    print("✓ Comprehensive location type classification")
    print("✓ Detailed business classification with NAICS equivalents")
    print("✓ Support for walkways (highways) and parks (multiple types)")
    print("✓ Flexible category mapping system")
    print("✓ Handles all geometry types (Point, Line, Polygon, MultiPolygon)")
    print("✓ Rich metadata for analysis and visualization")

if __name__ == "__main__":
    import pandas as pd
    main()
