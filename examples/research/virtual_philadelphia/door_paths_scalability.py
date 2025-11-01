"""
Door path scalability benchmark:

- Loads sandbox GeoPackage (buildings, streets, boundary)
- Scales boundary by factors (e.g., 1.0, 1.25, 1.5), crops layers to each
- Rasterizes via RasterCityGenerator
- Builds lazy street graph (with shortcut hubs)
- Selects up to 100 building door street blocks as seeds
- For each seed, computes door-to-all distances (in blocks) using City.get_distance_fast

Usage:
  python door_paths_scalability.py --gpkg sandbox/sandbox_data.gpkg --block-size 15 --factors 1.0 1.25 1.5 --seeds 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
from shapely.affinity import scale as shp_scale

from nomad.city_gen import RasterCityGenerator


def crop_layers(boundary_poly, buildings: gpd.GeoDataFrame, streets: gpd.GeoDataFrame,
                crs: str = "EPSG:3857"):
    """Clip buildings and streets to boundary polygon in a single CRS."""
    if buildings.crs is None:
        buildings = buildings.set_crs(crs)
    if streets.crs is None:
        streets = streets.set_crs(crs)
    if buildings.crs != crs:
        buildings = buildings.to_crs(crs)
    if streets.crs != crs:
        streets = streets.to_crs(crs)

    b_clipped = gpd.clip(buildings, gpd.GeoDataFrame(geometry=[boundary_poly], crs=crs))
    s_clipped = gpd.clip(streets, gpd.GeoDataFrame(geometry=[boundary_poly], crs=crs))
    b_clipped = b_clipped[b_clipped.geometry.notna() & ~b_clipped.geometry.is_empty].reset_index(drop=True)
    s_clipped = s_clipped[s_clipped.geometry.notna() & ~s_clipped.geometry.is_empty].reset_index(drop=True)
    return b_clipped, s_clipped


def run_one_factor(boundary_poly, buildings, streets, block_size: float, seeds: int):
    gen = RasterCityGenerator(boundary_poly, streets, buildings, block_size=block_size)
    t0 = time.time()
    city = gen.generate_city()
    t_city = time.time() - t0

    t1 = time.time()
    G = city.get_street_graph(lazy=True)
    t_graph = time.time() - t1

    door_series = city.id_to_door_cell().dropna()
    # Keep only doors that are valid street nodes (paranoia)
    valid = [(cx, cy) for (cx, cy) in door_series.values if (cx, cy) in G.nodes]
    if not valid:
        return {
            'buildings': 0,
            'streets': len(G.nodes),
            'blocks': len(city.blocks_gdf),
            't_city': t_city,
            't_graph': t_graph,
            'pairs': 0,
            't_paths': 0.0,
            'throughput_pairs_per_s': 0.0,
            'dist_summary': {}
        }

    rng = np.random.default_rng(42)
    seeds_list = rng.choice(len(valid), size=min(seeds, len(valid)), replace=False)
    seed_nodes = [valid[i] for i in seeds_list]

    # Compute distances from each seed to all other doors
    all_nodes = valid
    t2 = time.time()
    dists = []
    total_pairs = 0
    for s in seed_nodes:
        for t in all_nodes:
            if s == t:
                continue
            d = city.get_distance_fast(s, t)
            dists.append(d)
            total_pairs += 1
    t_paths = time.time() - t2

    # Summaries
    finite = [x for x in dists if np.isfinite(x)]
    dist_summary = {}
    if finite:
        arr = np.array(finite)
        dist_summary = {
            'min': float(np.min(arr)),
            'p25': float(np.percentile(arr, 25)),
            'median': float(np.median(arr)),
            'p75': float(np.percentile(arr, 75)),
            'max': float(np.max(arr))
        }

    return {
        'buildings': int(len(city.buildings_gdf)),
        'streets': int(len(G.nodes)),
        'blocks': int(len(city.blocks_gdf)),
        't_city': float(t_city),
        't_graph': float(t_graph),
        'pairs': int(total_pairs),
        't_paths': float(t_paths),
        'throughput_pairs_per_s': float(total_pairs / t_paths) if t_paths > 0 else 0.0,
        'dist_summary': dist_summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpkg', type=str, default='sandbox/sandbox_data.gpkg')
    ap.add_argument('--factors', type=float, nargs='+', default=[1.0, 1.25, 1.5])
    ap.add_argument('--block-size', type=float, default=15.0)
    ap.add_argument('--seeds', type=int, default=100)
    args = ap.parse_args()

    gpkg = Path(args.gpkg)
    if not gpkg.exists():
        print(f"ERROR: Missing GeoPackage: {gpkg}")
        return 1

    buildings = gpd.read_file(gpkg, layer='buildings')
    streets = gpd.read_file(gpkg, layer='streets')
    boundary = gpd.read_file(gpkg, layer='boundary')
    crs = buildings.crs or streets.crs or 'EPSG:3857'
    boundary_poly = boundary.to_crs(crs).geometry.iloc[0]

    print(f"Loaded buildings={len(buildings):,}, streets={len(streets):,}")

    for f in args.factors:
        expanded = shp_scale(boundary_poly, xfact=f, yfact=f, origin=boundary_poly.centroid)
        b_clip, s_clip = crop_layers(expanded, buildings, streets, crs=crs)

        print(f"\n=== Factor {f:.2f} ===")
        print(f"Cropped buildings={len(b_clip):,}, streets={len(s_clip):,}")

        stats = run_one_factor(expanded, b_clip, s_clip, block_size=args.block_size, seeds=args.seeds)
        print(f"Rasterization: {stats['t_city']:.2f}s; Graph build: {stats['t_graph']:.2f}s")
        print(f"Blocks={stats['blocks']:,}; Streets(nodes)={stats['streets']:,}; Buildings={stats['buildings']:,}")
        print(f"Door pairs computed={stats['pairs']:,} in {stats['t_paths']:.2f}s; throughput={stats['throughput_pairs_per_s']:.1f} pairs/s")
        if stats['dist_summary']:
            ds = stats['dist_summary']
            print(f"Distance (blocks) summary: min={ds['min']:.0f}, p25={ds['p25']:.0f}, median={ds['median']:.0f}, p75={ds['p75']:.0f}, max={ds['max']:.0f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


