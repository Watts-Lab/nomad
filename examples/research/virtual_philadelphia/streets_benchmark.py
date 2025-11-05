"""
Benchmark streets-only rasterization and street-graph build on rotated Philadelphia

Usage:
  python streets_benchmark.py
"""

from pathlib import Path
import time
import geopandas as gpd
from nomad.city_gen import RasterCity


def main():
    gpkg = Path('philadelphia_osm_raw.gpkg')  # Relative path for Jupyter compatibility
    if not gpkg.exists():
        print(f"ERROR: Missing {gpkg}")
        return 1

    streets = gpd.read_file(gpkg, layer='streets_rotated')
    boundary = gpd.read_file(gpkg, layer='city_boundary_rotated')
    if streets.crs is None:
        # rotated layers should already be in EPSG:3857
        streets.set_crs('EPSG:3857', inplace=True)
    rot_buildings = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=streets.crs)

    boundary_geom = boundary.geometry.iloc[0]
    block_size = 15.0

    t0 = time.time()
    city = RasterCity(boundary_geom, streets, rot_buildings, block_size=block_size)
    t_raster = time.time() - t0
    print(f"Streets-only rasterization: {t_raster:.2f}s; streets={len(city.streets_gdf):,}; blocks={len(city.blocks_gdf):,}")

    t1 = time.time()
    G = city.get_street_graph(lazy=True)
    t_graph = time.time() - t1
    print(f"Street graph build (lazy+shortcuts): {t_graph:.2f}s; nodes={len(G.nodes):,}; edges={len(G.edges):,}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


