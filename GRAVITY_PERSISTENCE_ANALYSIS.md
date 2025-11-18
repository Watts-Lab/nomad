# Gravity Persistence and Usage Analysis

## Question 1: Does save_geopackage persist gravity?

### Current Implementation

**`save_geopackage` (lines 979-1127) with `persist_gravity_data=True` saves:**
- `hubs`: List of ~100 street block coordinates
- `hub_df`: 100×100 hub-to-hub distance matrix
- `grav_hub_info`: Per-building closest hub info (index, distance)
- `mh_dist_nearby_doors`: Nearby door pairs with exact Manhattan distances

**`from_geopackage` (lines 1217-1352) with `load_gravity=True` restores:**
- All the above structures
- **Reconstructs `city.grav` as a callable** (lines 1328-1347) that computes gravity on-the-fly using hub approximation

### Answer: Yes and No

**Hub approximation mode (`use_proxy_hub_distance=True`):**
✅ **YES** - Fully persistable and restorable. The callable `city.grav` works immediately after loading.

**True distance mode (`use_proxy_hub_distance=False`):**
❌ **NO** - Nothing is persisted. Must call `compute_gravity(use_proxy_hub_distance=False)` after loading.

### Why True Distance Mode Isn't Persisted

True distance mode (lines 788-822):
- Does not use or set `self.hubs`, `self.hub_df`, `self.grav_hub_info`, or `self.mh_dist_nearby_doors`
- Only sets `self.grav` as a DataFrame or callable wrapping the DataFrame
- The DataFrame is not saved to geopackage

---

## Question 2: Should We Persist True Distance Mode?

### Analysis

**For small cities (<200 buildings) where true distance mode is intended:**
- Distance matrix is N×N ≈ 40K entries for N=200
- Compute time is N × |E| graph operations
- For a grid with N=200, E ≈ 4N (grid graph), so ~800 shortest path queries
- Typical compute time: <1 second

**Arguments AGAINST persisting:**
1. Recomputation is fast for the small cities where true distance is used
2. Adding persistence adds complexity to already comprehensive save/load code
3. Violates "avoid bloating" principle (Principle #1)
4. Would require adding new geopackage layers or significantly expanding existing ones

**Arguments FOR persisting:**
1. Consistency - hub mode is persistable, why not true mode?
2. Workflow simplification - no need to remember to recompute

### Recommendation: DO NOT add persistence for true distance mode

**Rationale:**
- Recomputation cost is negligible for intended use case
- Saving N×N matrix to geopackage adds complexity and file size
- Follows Principle #1 (avoid bloating)
- Document clearly in docstring and examples that true mode must be recomputed

---

## Question 3: Garden City Usage

Looking at usage in other scripts:

**`synthetic_philadelphia.py` (lines 213-226):**
```python
G = city.get_street_graph()
city._build_hub_network(hub_size=config["hub_size"])
city.compute_gravity(exponent=2.0, callable_only=True)
city.save_geopackage(gpkg_path)
```

**`profile_trajectory.py` (lines 24-27):**
```python
city.get_street_graph()
city._build_hub_network(hub_size=100)
city.compute_gravity(exponent=2.0, callable_only=True)
city.compute_shortest_paths(callable_only=True)
```

**Garden City script (lines 170-181):**
```python
city.get_street_graph()
city.save_geopackage('../garden-city.gpkg')

# Later:
city = City.from_geopackage('../garden-city.gpkg')
#city.compute_gravity()  # COMMENTED OUT
```

### What Garden City Needs

Garden City has ~80 buildings, ideal for true distance mode.

**Option A: Compute before save (doesn't persist, but establishes workflow):**
```python
city.get_street_graph()
city.compute_gravity(exponent=2.0, callable_only=True, use_proxy_hub_distance=False)
city.save_geopackage('../garden-city.gpkg')  # Only saves graph, not gravity
```

**Option B: Compute after load (recommended for true distance mode):**
```python
city = City.from_geopackage('../garden-city.gpkg')
# For small cities, use exact distances
city.compute_gravity(exponent=2.0, callable_only=True, use_proxy_hub_distance=False)
```

**Option C: Use hub approximation (persists automatically):**
```python
city.get_street_graph()
city._build_hub_network(hub_size=10)  # Small hub network for small city
city.compute_gravity(exponent=2.0, callable_only=True)  # Uses hub approximation
city.save_geopackage('../garden-city.gpkg')  # Persists hub infrastructure

# Later:
city = City.from_geopackage('../garden-city.gpkg')
# city.grav already works!
```

**Recommendation: Option B or C depending on whether exact distances matter for the paper.**

---

## Question 4: Can We Unify by Using All Doors as Hubs?

### Mathematical Analysis

If `hubs = all_door_coordinates`:
- Each building's closest hub = its own door (distance = 0)
- Hub approximation becomes: `dist(i,j) = 0 + hub_to_hub[i,j] + 0 = hub_to_hub[i,j]`
- This gives exact distances!

### Problems This Creates

#### Problem 1: Violates Coding Principle #6 (Double Loops)

Current `_build_hub_network` (lines 578-616) and true distance mode (lines 793-799) both have:
```python
for origin in hubs:  # N iterations
    distances = nx.single_source_shortest_path_length(G, origin)
    for dest in hubs:  # N iterations  
        rows.append(...)  # Double loop
```

For N=200 buildings, this is 40,000 iterations just to build the rows list. While the graph shortest path is only called N times (efficient), the row building is a double loop which Principle #6 says to avoid.

**Counter-argument:** This is unavoidable for building an N×N matrix. The current true distance implementation also has this structure. But the principle says to avoid double loops when pandas/numpy can do the job. Here, we're building edge lists for graph operations, so vectorization isn't applicable.

**Verdict:** This is a necessary double loop, not a violation. But adding a new method with a double loop needs justification.

#### Problem 2: Persistence Layer Confusion

`save_geopackage` persists hub infrastructure, `from_geopackage` reconstructs a hub-based callable.

If we use all-doors-as-hubs for true distance:
- `hub_df` becomes N×N (larger but manageable)
- Saves correctly (flattening works)
- Loads correctly (reshaping works)
- Reconstructs callable using hub approximation formula (lines 1328-1345)
- **BUT** the callable does extra work: computes `dist_to_hub + hub_to_hub + dist_to_hub` where both hub distances are 0
- **AND** loops through `mh_dist_nearby_doors` which would be empty/redundant

This works but violates the abstraction. The persistence layer thinks it's saving hub approximation infrastructure, but it's actually saving exact distances disguised as hubs.

#### Problem 3: Redundant Nearby Doors Computation

Hub mode computes `mh_dist_nearby_doors` (lines 698-737) for pairs where Manhattan distance is small enough that exact distance is better than hub approximation.

With all-doors-as-hubs:
- Hub approximation IS exact
- `mh_dist_nearby_doors` is redundant (all pairs are "correct" via hub_df)
- Computing it wastes time (chunked Manhattan distance calculation)
- Should skip this entirely for true distance mode

But the code structure makes this hard - the nearby doors logic is embedded in the hub mode branch.

#### Problem 4: Code Doesn't Actually Simplify

**Current structure:**
- Hub mode: ~110 lines (682-787)
- True mode: ~35 lines (788-822)
- Total: ~145 lines
- Clean separation

**Proposed unification:**
- Extract helper for hub building: ~30 lines
- Modify `_build_hub_network` to use helper: ~5 lines
- Modify hub mode to call helper with either selected hubs or all doors: ~5 lines
- Conditional logic to skip nearby doors for true mode: ~10 lines
- Total: ~160 lines
- **Net increase: +15 lines**

This violates Principle #1 (avoid bloating) and Principle #7 (plan first and explain why it adheres to principles).

### Conclusion: Unification is NOT Worth It

**Reasons:**
1. ❌ Doesn't reduce code size (increases by ~15 lines)
2. ❌ Makes persistence layer confusing (saves "hubs" that aren't really hubs)
3. ❌ Requires conditional logic to skip nearby doors optimization
4. ❌ Reconstructed callable does redundant work
5. ✅ Mathematical equivalence is correct (only positive)

**The current separation is better:**
- Hub mode for large cities (>200 buildings), persistable
- True mode for small cities (<200 buildings), not persistable but fast to recompute
- Clear separation of concerns
- Each mode optimized for its use case

---

## Recommended Actions

### 1. Garden City Example (Required Fix)

**File:** `examples/research/garden-city-paper-code.py`
**Line 181:** Uncomment and modify:

```python
city = City.from_geopackage('../garden-city.gpkg')
# Garden City is small (~80 buildings), use exact distances
city.compute_gravity(exponent=2.0, callable_only=True, use_proxy_hub_distance=False)
```

**Rationale:** The commented-out line indicates gravity is needed. For Garden City's size, true distance mode is appropriate.

### 2. Documentation (Important)

**File:** `nomad/city_gen.py`
**Method:** `compute_gravity` docstring (lines 656-676)

**Add to Notes section:**
```
Persistence:
- Hub mode (use_proxy_hub_distance=True): Gravity infrastructure is saved by 
  save_geopackage() and restored by from_geopackage(load_gravity=True). The 
  city.grav callable works immediately after loading.
- True distance mode (use_proxy_hub_distance=False): Not persisted. Must call 
  compute_gravity(..., use_proxy_hub_distance=False) after loading. This is 
  fast for small cities (<200 buildings) where this mode is intended.
```

### 3. No Code Changes Needed

Current implementation is clean and follows coding principles:
- ✅ No bloat (Principle #1)
- ✅ Not blind coded (clear structure)
- ✅ No ad-hoc patches (Principle #3)
- ✅ Vectorized where possible (Principle #6)

Adding persistence or unification would violate Principle #1.

---

## Summary

**Q: Does save_geopackage persist gravity?**
A: Yes for hub mode, no for true distance mode.

**Q: Should we fix this?**
A: No. Recomputation is fast for small cities. Adding persistence adds bloat.

**Q: Should we unify using all-doors-as-hubs?**
A: No. It doesn't simplify code, confuses abstractions, and adds complexity.

**Q: What should Garden City example do?**
A: Call `compute_gravity(..., use_proxy_hub_distance=False)` after loading.

**Q: What about other scripts?**
A: They use hub mode (large cities), which already persists correctly.

