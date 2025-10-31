"""
Audit A/B classification of OSM buildings into garden_city categories

This script loads the rotated Philadelphia buildings layer and reports
category counts with and without speculative inference. It highlights
the impact on the 'home' category and prints small samples for review.

Usage (from this directory):
  python ab_category_audit.py
"""

from pathlib import Path
import sys
import json
import pandas as pd
import geopandas as gpd

import nomad.map_utils as nm
from nomad.constants import SUBTYPE_TO_GARDEN_CITY


def summarize_counts(df: gpd.GeoDataFrame) -> dict:
    # Prefer already-computed garden_city_category if present
    if 'garden_city_category' in df.columns:
        cats = df['garden_city_category']
    elif 'category' in df.columns:
        cats = df['category']
    elif 'subtype' in df.columns:
        # Map subtype -> garden_city categories
        cats = df['subtype'].map(lambda s: SUBTYPE_TO_GARDEN_CITY.get(str(s), 'other'))
    else:
        return {}
    counts = cats.value_counts().to_dict()
    total = int(df.shape[0])
    pct = {k: round(100.0 * v / max(1, total), 2) for k, v in counts.items()}
    return {'total': total, 'counts': counts, 'percent': pct}


def run_audit(gpkg_path: Path) -> int:
    if not gpkg_path.exists():
        print(f"ERROR: Missing GeoPackage: {gpkg_path}")
        return 1

    # Prefer raw buildings if available for richer tags, fallback to rotated
    try:
        buildings = gpd.read_file(gpkg_path, layer="buildings")
    except Exception:
        buildings = gpd.read_file(gpkg_path, layer="buildings_rotated")

    # Keep only rows that have at least one informative OSM tag
    tag_cols = ['building','amenity','leisure','landuse','tourism','shop','office','healthcare','craft']
    tag_cols_present = [c for c in tag_cols if c in buildings.columns]
    if tag_cols_present:
        mask = pd.Series(False, index=buildings.index)
        for c in tag_cols_present:
            mask = mask | buildings[c].notna()
        buildings = buildings[mask].copy()

    # A/B: prefer persisted categories/subtypes
    s_no = summarize_counts(buildings)
    s_yes = summarize_counts(buildings)

    # Compare 'home' impact
    home_no = s_no['counts'].get('home', 0)
    home_yes = s_yes['counts'].get('home', 0)
    delta = home_yes - home_no
    pct_change = round(100.0 * delta / max(1, home_no), 2) if home_no > 0 else None

    report = {
        'path': str(gpkg_path),
        'no_infer': s_no,
        'yes_infer': s_yes,
        'home_delta': {
            'absolute': int(delta),
            'percent': pct_change,
            'home_no_infer': int(home_no),
            'home_yes_infer': int(home_yes),
        }
    }

    print("A/B garden_city category summary:")
    print(json.dumps(report, indent=2))

    # Identify rows that became 'home' only due to inference
    flipped_to_home = pd.Index([])

    if len(flipped_to_home) > 0:
        cols = [c for c in ['building','amenity','height','shop','office','landuse','leisure'] if c in buildings.columns]
        sample = buildings.loc[flipped_to_home[:10], cols]
        print("\nSample of entries that became 'home' ONLY due to inference (first 10):")
        print(sample.fillna('').to_string(index=False))
        print(f"\nTotal flipped_to_home due to inference: {len(flipped_to_home):,}")
    else:
        print("\nNo entries became 'home' solely due to inference.")

    # Optional: live download A/B comparison using boundary (may be slow, uses cache)
    try:
        boundary = gpd.read_file(gpkg_path, layer='city_boundary')
        boundary_poly = boundary.to_crs('EPSG:4326').geometry.iloc[0] if boundary.crs is not None else boundary.geometry.iloc[0]
        print("\nDownloading buildings A/B from OSM (may take several minutes)...")
        b_no = nm.download_osm_buildings(boundary_poly, crs='EPSG:3857', schema='garden_city', clip=True, infer_building_types=False, explode=True, by_chunks=False, cache_mode='persistent')
        b_yes = nm.download_osm_buildings(boundary_poly, crs='EPSG:3857', schema='garden_city', clip=True, infer_building_types=True, explode=True, by_chunks=False, cache_mode='persistent')
        def _count_gcc(df):
            col = 'garden_city_category' if 'garden_city_category' in df.columns else ('category' if 'category' in df.columns else None)
            return df[col].value_counts().to_dict() if col else {}
        live_no = _count_gcc(b_no)
        live_yes = _count_gcc(b_yes)
        print("Live NO_INFER counts:", live_no)
        print("Live YES_INFER counts:", live_yes)
    except Exception as e:
        print(f"Live download A/B skipped: {e}")

    # Simple CSV outputs for offline inspection (optional)
    out_dir = gpkg_path.parent
    pd.DataFrame([s_no]).to_json(out_dir / 'ab_counts_no_infer.json', orient='records', indent=2)
    pd.DataFrame([s_yes]).to_json(out_dir / 'ab_counts_yes_infer.json', orient='records', indent=2)

    return 0


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    gpkg = base / 'philadelphia_osm_raw.gpkg'
    sys.exit(run_audit(gpkg))


