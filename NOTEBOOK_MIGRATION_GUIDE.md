# Notebook Migration Guide: compute_shortest_paths()

## Summary of Changes

The `City` class now requires explicit computation of shortest paths before trajectory generation, mirroring the pattern used for `compute_gravity()`.

### What Changed

**Before**:
```python
city = City.from_geopackage('garden-city.gpkg')
agent.generate_trajectory(...)  # Just worked
```

**After**:
```python
city = City.from_geopackage('garden-city.gpkg')
city.compute_shortest_paths(callable_only=False)  # Required for small cities
agent.generate_trajectory(...)  # Now works
```

---

## Notebooks Requiring Updates

### Priority 1: Core Examples (Need Immediate Fixes)

#### 1. `examples/random_city.ipynb`
**Location**: Line ~105 (after `city_generator.generate_city()`)  
**Add**:
```python
clustered_city.compute_shortest_paths(callable_only=False)
```

#### 2. `examples/research/robustness-of-algorithms/prescribed_trajectory_exp_1.py`
**Location**: Line 38 (after `city = cg.City.from_geopackage(city_file)`)  
**Add**:
```python
city.compute_shortest_paths(callable_only=False)
```

**Location**: Line 145 (after `city = City.from_geopackage(config["city_file"])`)  
**Add**:
```python
city.compute_shortest_paths(callable_only=False)
```

#### 3. `examples/research/garden-city-paper-code.py` / `.ipynb`
**Multiple locations** after city loading  
**Add**:
```python
city.compute_shortest_paths(callable_only=False)
```

---

### Priority 2: Research Notebooks (May Be Experimental)

#### 4. `examples/research/virtual_philly.py` / `.ipynb`
**Check if still used** - May be outdated  
**Add** after city creation:
```python
city.compute_shortest_paths(callable_only=False)
```

#### 5. `examples/generate_synthetic_trajectories.py` / `.ipynb`
**Check** if it uses `generate_trajectory()`  
**Add** if needed:
```python
city.compute_shortest_paths(callable_only=False)
```

#### 6. `examples/research/robustness-of-algos-paper.py` / `.ipynb`
**Check** if it uses `generate_trajectory()`  
**Add** if needed:
```python
city.compute_shortest_paths(callable_only=False)
```

---

## Why Dense Mode for These Notebooks?

All these notebooks use **small cities** (garden-city.gpkg or RandomCityGenerator with dimensions ~100x100):
- **Garden city**: ~36 buildings, ~400 street blocks
- **Random city (101x101)**: ~400 street blocks

Dense storage (`callable_only=False`) is:
- ✅ **Fast**: O(1) path lookups
- ✅ **Safe**: Memory footprint negligible for < 10k nodes
- ✅ **Simple**: No placeholder limitations

---

## For Large Cities (Future Work)

For **RasterCity** with large bounding boxes or full Philadelphia:
```python
city = RasterCity(boundary, streets, buildings, ...)
city.get_street_graph()
city._build_hub_network(hub_size=200)  # Optional: for gravity
city.compute_shortest_paths(callable_only=True)  # On-demand paths
```

**Note**: `callable_only=True` currently uses `nx.shortest_path()` on demand, which is a **PLACEHOLDER**. Production use requires hub-based routing implementation.

---

## Testing the Changes

### Updated Tests
- ✅ `test_shortest_path()` in `test_city_gen.py` - Now calls `compute_shortest_paths()`

### Manual Testing Checklist
1. Run `random_city.ipynb` after adding the line
2. Run `prescribed_trajectory_exp_1.py` after adding the lines
3. Verify trajectories generate successfully
4. Check for no runtime errors related to `shortest_paths`

---

## Error Messages to Watch For

### If you forget to call `compute_shortest_paths()`:
```
ValueError: Must call compute_shortest_paths() before using get_shortest_path().
```
**Fix**: Add `city.compute_shortest_paths(callable_only=False)` after city creation.

### If coordinate is not a street block:
```
ValueError: Start coordinate (x, y) must be a street block.
```
**This is a different issue** - coordinate validation, not related to the refactor.

---

## Next Steps

1. **Fix Priority 1 notebooks** (random_city, prescribed_trajectory_exp_1, garden-city-paper-code)
2. **Test each notebook** to verify trajectories generate correctly
3. **Commit fixes** with message: `"fix: add compute_shortest_paths() calls to notebooks"`
4. **Verify all tests pass** (`pytest nomad/tests/test_city_gen.py::test_shortest_path`)

