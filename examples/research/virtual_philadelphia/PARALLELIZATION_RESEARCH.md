# Parallelization Research: Serializing City Objects with Callable Gravity

## Summary

**YES, the City object with callable gravity CAN be serialized for parallel processing**, but you need to use `dill` or `cloudpickle` instead of standard `pickle`.

## Test Results

### Simple Closure Test
- **pickle**: ❌ FAIL (AttributeError)
- **dill**: ✅ SUCCESS
- **cloudpickle**: ✅ SUCCESS

### Complex Closure (numpy/pandas)
- **pickle**: ❌ FAIL (AttributeError)
- **dill**: ✅ SUCCESS
- **cloudpickle**: ✅ SUCCESS

### Actual RasterCity Object (25k buildings, hub_size=100)
- **pickle**: ❌ FAIL ("Can't pickle local object 'City.compute_gravity.<locals>.compute_gravity_row'")
- **dill**: ✅ SUCCESS (72.66 MB)
- **cloudpickle**: ✅ SUCCESS (72.65 MB)

## Why Standard Pickle Fails

Python's `pickle` module cannot serialize:
1. **Nested/local functions** (like `compute_gravity_row` defined inside `compute_gravity`)
2. **Closures** that capture external variables
3. **Lambda functions**

The `city.grav` callable is a closure that captures:
- `building_ids` (numpy array)
- `bid_to_idx` (dict)
- `hub_to_hub` (numpy array)
- `closest_hub_idx` (numpy array)
- `dist_to_closest_hub` (numpy array)
- `self.mh_dist_nearby_doors` (pandas Series with MultiIndex)

## Solutions for Parallelization

### Option 1: Use `dill` or `cloudpickle` (Recommended)

Both libraries extend pickle to handle closures and nested functions.

**Example from existing codebase** (`examples/generate_synthetic_trajectories.py`):

```python
from joblib import Parallel, delayed

def generate_agent_trajectory(args):
    identifier, home, work, seed = args
    
    # Load city in each worker process
    city = City.from_geopackage('city.gpkg')
    agent = Agent(identifier=identifier, city=city, home=home, workplace=work)
    
    agent.generate_dest_diary(
        end_time=pd.Timestamp("2024-01-08 00:00-05:00"),
        epr_time_res=15,
        rho=0.4,
        gamma=0.3,
        seed=seed
    )
    
    return agent.destination_diary

# Parallel execution
results = Parallel(n_jobs=-1, backend='loky')(
    delayed(generate_agent_trajectory)(params) for params in agent_params
)
```

**Note**: `joblib` with `backend='loky'` uses `cloudpickle` by default!

### Option 2: Load City in Each Worker (Current Pattern)

The existing `generate_synthetic_trajectories.py` loads the city from geopackage in each worker:

```python
def generate_agent_trajectory(args):
    city = City.from_geopackage('garden-city.gpkg')  # Load in worker
    # ... rest of logic
```

**Pros**:
- Works with standard pickle
- Each worker has independent city object
- No serialization overhead for large city

**Cons**:
- Repeated I/O (loading geopackage N times)
- Higher memory per worker (each has full city)

### Option 3: Serialize City Once, Distribute to Workers

With `dill`/`cloudpickle`, you can serialize the city once and pass it to workers:

```python
import cloudpickle
from multiprocessing import Pool

# Serialize city once
city_bytes = cloudpickle.dumps(city)

def generate_agent_trajectory(city_bytes, agent_params):
    city = cloudpickle.loads(city_bytes)
    # ... rest of logic

with Pool() as pool:
    results = pool.starmap(generate_agent_trajectory, 
                          [(city_bytes, params) for params in agent_params])
```

**Pros**:
- Single I/O operation
- Faster startup per worker

**Cons**:
- 72 MB serialized size per worker (for large box with 25k buildings)
- Memory overhead if many workers

## Memory Analysis

For the large box (25k buildings, hub_size=100):
- **Serialized size**: ~73 MB
- **In-memory size**: Larger (includes Python object overhead)
- **Key components**:
  - `buildings_gdf`: ~6 MB
  - `streets_gdf`: ~4 MB
  - `hub_df` (100×100): 0.04 MB
  - `grav_hub_info`: ~1.4 MB
  - `mh_dist_nearby_doors`: ~0.0 MB (sparse)
  - Closure overhead: ~60 MB (captured arrays, dicts)

## Recommendations for Full Philadelphia

### For 10-100 agents:
**Use Option 1 (joblib with loky backend)**
- Simple, works out of the box
- `cloudpickle` handles serialization automatically
- Good balance of speed and simplicity

```python
from joblib import Parallel, delayed

def generate_agent_diary(city, agent_params):
    identifier, home, work, datetime, seed = agent_params
    agent = Agent(identifier=identifier, city=city, home=home, workplace=work, datetime=datetime)
    agent.generate_dest_diary(end_time=end_time, seed=seed, ...)
    return agent.destination_diary

results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
    delayed(generate_agent_diary)(city, params) for params in agent_params
)
```

### For 100+ agents on a cluster:
**Use Option 2 (load city per worker)**
- More memory efficient
- Better for distributed systems (Dask, Ray)
- Avoids large serialization overhead

```python
def generate_agent_diary(geopackage_path, agent_params):
    city = RasterCity.from_geopackage(geopackage_path, load_gravity=True)
    # ... rest of logic

# Can use with Dask/Ray for true distributed computing
```

### For maximum performance:
**Pre-save city with gravity to geopackage**
- Use `city.save_geopackage(path, persist_gravity_data=True)`
- Workers load from geopackage (fast, ~2-3s)
- Avoids serialization entirely
- Each worker gets independent copy

## Key Insight

The callable gravity function is **fully serializable** with `dill`/`cloudpickle`. The 73 MB size is reasonable for parallelization across 10-50 workers. For larger scale (100+ workers), loading from geopackage per worker is more memory-efficient.

## Testing Serialization

Run `test_serialization.py` to verify serialization works for your specific city configuration.

