from pathlib import Path
import time
import geopandas as gpd
import pandas as pd
import numpy as np

from nomad.city_gen import RasterCity

SANDBOX_GPKG = Path("sandbox/sandbox_data_large.gpkg")
HUB_SIZES = [200, 400, 600]
N_CALLS = 500
RANDOM_SEED = 42

print("="*60)
print("HUB SIZE BENCHMARK - GRAVITY CALL PERFORMANCE")
print("="*60)
print(f"Testing {N_CALLS} random gravity calls per hub size")
print()

print("Loading data...")
buildings = gpd.read_file(SANDBOX_GPKG, layer="buildings")
streets = gpd.read_file(SANDBOX_GPKG, layer="streets")
boundary = gpd.read_file(SANDBOX_GPKG, layer="boundary")
print(f"Loaded {len(buildings):,} buildings, {len(streets):,} streets")
print()

results = []

for i, hub_size in enumerate(HUB_SIZES, 1):
    print(f"[{i}/{len(HUB_SIZES)}] Hub Size: {hub_size}")
    print("-"*60)
    
    print(f"  Building city...")
    t0 = time.time()
    city = RasterCity(
        boundary.geometry.iloc[0],
        streets,
        buildings,
        block_side_length=10.0,
        resolve_overlaps=True,
        other_building_behavior="filter",
        verbose=False
    )
    gen_time = time.time() - t0
    print(f"  City generation:     {gen_time:>6.2f}s")
    
    print(f"  Building street graph...")
    t1 = time.time()
    city.get_street_graph()
    graph_time = time.time() - t1
    print(f"  Street graph:        {graph_time:>6.2f}s")
    
    print(f"  Building hub network (size={hub_size})...")
    t2 = time.time()
    city._build_hub_network(hub_size=hub_size)
    hub_time = time.time() - t2
    print(f"  Hub network:         {hub_time:>6.2f}s")
    
    print(f"  Computing gravity...")
    t3 = time.time()
    city.compute_gravity(exponent=2.0, callable_only=True)
    grav_time = time.time() - t3
    print(f"  Gravity computation: {grav_time:>6.2f}s")
    
    print(f"  Sampling {N_CALLS} random buildings...")
    rng = np.random.default_rng(RANDOM_SEED)
    sample_bids = rng.choice(city.buildings_gdf['id'].values, size=N_CALLS, replace=True)
    
    print(f"  Running {N_CALLS} gravity calls...")
    t_calls = time.time()
    for j, bid in enumerate(sample_bids):
        if (j + 1) % 100 == 0:
            elapsed = time.time() - t_calls
            rate = (j + 1) / elapsed
            print(f"    {j+1}/{N_CALLS} calls ({rate:.1f} calls/s, {elapsed:.1f}s elapsed)")
        city.grav(bid)
    calls_time = time.time() - t_calls
    avg_call_time_ms = (calls_time / N_CALLS) * 1000
    
    print(f"  {N_CALLS} gravity calls:  {calls_time:>6.2f}s")
    print(f"  Avg per call:        {avg_call_time_ms:>6.2f}ms")
    print()
    
    results.append({
        'Hub Size': hub_size,
        'Hub Network (s)': f'{hub_time:.2f}',
        'Gravity Comp (s)': f'{grav_time:.2f}',
        f'{N_CALLS} Calls (s)': f'{calls_time:.2f}',
        'Avg Call (ms)': f'{avg_call_time_ms:.2f}',
        'Hub Matrix': f'{hub_size}×{hub_size}',
        'Hub MB': f'{(hub_size * hub_size * 4) / 1024**2:.2f}'
    })

print("="*60)
print("SUMMARY")
print("="*60)
df = pd.DataFrame(results)
print(df.to_string(index=False))
print()
print("Key Insight: Average call time shows the runtime cost during")
print("trajectory generation, where gravity is queried repeatedly.")

