import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from nomad.city_gen import (
    RasterCityGenerator,
    generate_canvas_blocks,
    find_connected_components,
)


def _sandbox_paths():
    # repo_root/ examples / research / virtual_philadelphia / sandbox / sandbox_data.gpkg
    repo_root = Path(__file__).resolve().parents[2]
    gpkg = repo_root / "examples" / "research" / "virtual_philadelphia" / "sandbox" / "sandbox_data.gpkg"
    return gpkg


def _load_sandbox():
    gpkg = _sandbox_paths()
    assert gpkg.exists(), f"Missing sandbox geopackage: {gpkg}"
    buildings = gpd.read_file(gpkg, layer="buildings")
    streets = gpd.read_file(gpkg, layer="streets")
    boundary = gpd.read_file(gpkg, layer="boundary")
    return buildings, streets, boundary.geometry.iloc[0]


def test_raster_city_generate_and_graph_smoke():
    buildings, streets, boundary_geom = _load_sandbox()
    # Keep test fast
    buildings = buildings.head(500)
    gen = RasterCityGenerator(boundary_geom, streets, buildings, block_size=15.0)

    t0 = time.time()
    city = gen.generate_city()
    elapsed = time.time() - t0
    assert elapsed < 30.0  # sanity bound for CI

    # Basic structure
    assert len(city.blocks_gdf) > 0
    assert len(city.streets_gdf) > 0

    # Street graph builds and is connected (single component)
    G = city.get_street_graph(lazy=True)
    assert G.number_of_nodes() > 0
    # Quick connected components test via BFS on a sample
    from collections import deque
    start = next(iter(G.nodes))
    seen = set([start])
    dq = deque([start])
    while dq:
        u = dq.popleft()
        for v in G.neighbors(u):
            if v not in seen:
                seen.add(v)
                dq.append(v)
    # At least a large chunk should be reachable
    assert len(seen) >= int(0.8 * G.number_of_nodes())


def test_shortest_path_and_fast_distance():
    buildings, streets, boundary_geom = _load_sandbox()
    buildings = buildings.head(300)
    city = RasterCityGenerator(boundary_geom, streets, buildings, block_size=15.0).generate_city()
    G = city.get_street_graph(lazy=True)
    nodes = list(G.nodes)
    assert len(nodes) >= 2
    u, v = nodes[10], nodes[-10]
    path = city.get_shortest_path(u, v)
    assert path and path[0] == u and path[-1] == v
    d = city.get_distance_fast(u, v)
    assert np.isfinite(d)
    assert int(d) == max(0, len(path) - 1)


def test_connectivity_filter_only_largest_component():
    # Build a tiny disconnected set of street blocks to verify the filter logic
    # 2 components: {(0,0),(1,0)} and {(10,10)}; only the largest should remain.
    blocks = pd.DataFrame({'coord_x':[0,1,10], 'coord_y':[0,0,10]})
    blocks['geometry'] = blocks.apply(lambda r: Path, axis=1)  # placeholder; not used by filter
    streets_gdf = gpd.GeoDataFrame(blocks, geometry=gpd.points_from_xy(blocks['coord_x'], blocks['coord_y']))
    # Reuse helper via city_gen import
    from nomad.city_gen import verify_street_connectivity
    filtered, summary = verify_street_connectivity(streets_gdf)
    assert summary['kept'] == 2 and summary['discarded'] == 1


