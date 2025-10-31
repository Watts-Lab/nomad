# Streets, Trajectories, and Scalability: From Raster Grids to Native OSM Graphs (Philadelphia)

## Executive summary
- **How streets are used in `traj_gen`**: exclusively as sequences of discrete street blocks for routing between a source door and a destination door, plus a polygonal corridor to constrain motion integration. Paths are obtained via `city.get_shortest_path((x,y),(x,y))` on a grid graph and converted to a centerline `MultiLineString` for stepping.
- **How streets are used in `city_gen` (non–data-driven)**: to ensure building doors connect to adjacent street blocks; to build a street graph; optionally to precompute all-pairs shortest paths (legacy and not scalable); and to provide a block-based city raster for plotting and simple pathfinding.
- **Scalability pain points**: city-scale rasterization (hundreds of millions of blocks at fine resolution), unioning many polygons per step bound, and any all-pairs shortest path precomputation. Sampling over hundreds of thousands of buildings per agent also needs careful data structures and numerics.
- **Recommendation**: replace grid/raster streets with a vector-first approach using the OSM road graph directly (centerlines as polylines). Keep buildings as polygons with one “door” snapped to the nearest edge/node. Route on the OSM graph (A*/ALT landmarks + caches). Build route geometries as polylines and use buffered lines for step corridors. Use hierarchical and alias sampling for destinations with stable log-probabilities.

---

## Part A — Exactly how streets are used in `traj_gen`

`traj_gen` relies on the city’s street network as a graph of discrete grid cells (street blocks). When an agent travels between buildings, it:
- Determines the door cell of the starting building and the door cell of the destination building.
- Calls `city.get_shortest_path(start_cell, dest_cell)` to get a list of street blocks that form the path.
- Converts that path to a `MultiLineString` of center points and integrates motion along it.
- Creates a polygonal “bound” by unioning street geometries (and possibly the start geometry) to keep steps inside a corridor.

Code references:

```459:501:nomad/traj_gen.py
        # Shortest path between street blocks (door cells)
        try:
            street_path = city.get_shortest_path(start_node, dest_cell)
        except ValueError:
            street_path = []
        if not street_path:
            # No path; step straight towards destination door centroid
            direction = np.asarray([dest_door_centroid.x, dest_door_centroid.y]) - start_point_arr
            norm = np.linalg.norm(direction)
            if norm == 0:
                return start_point_arr, None
            step = (direction / norm) * (0.5 * dt)
            coord = start_point_arr + step
            return coord, location

        # Build continuous path through block centers, include start/end segments
        path = [(x + 0.5, y + 0.5) for (x, y) in street_path]
        path = start_segment + path + [(dest_door_centroid.x, dest_door_centroid.y)]
        path_ml = MultiLineString([path])
        street_geom = unary_union([city.get_block(b)['geometry'] for b in street_path])
        bound_poly = unary_union([start_info['geometry'], street_geom]) if start_info['geometry'] is not None else street_geom

        # Transformed coordinates of current position along the path
        path_coord = _path_coords(path_ml, start_point_arr)

        heading_drift = 3.33 * dt
        sigma = 0.5 * dt / 1.96

        while True:
            # Step in transformed (path-based) space
            step = rng.normal(loc=[heading_drift, 0], scale=sigma * np.sqrt(dt), size=2)
            path_coord = (path_coord[0] + step[0], 0.7 * path_coord[1] + step[1])

            if path_coord[0] > path_ml.length:
                coord = np.array([dest_geom.centroid.x, dest_geom.centroid.y])
                break

            coord = _cartesian_coords(path_ml, *path_coord)

            if bound_poly.contains(Point(coord)):
                break

        return coord, location
```

Implication for scalability: streets are only needed to (1) provide a route between doors and (2) provide a narrow corridor for the stepping integrator. Neither requires a full city raster if a vector path (polyline) corridor can be used instead.

---

## Part B — Role of streets in `city_gen` (non–data‑driven)

The non–data-driven city generation uses streets at grid granularity mainly to assign doors and to build a graph. Streets and buildings live on a block grid; buildings must be adjacent to at least one street block to get a door.

Key city attributes include a grid-based street graph and optional shortest-paths store:

```84:99:nomad/city_gen.py
    street_graph : dict
        A dictionary representing the graph of streets with their neighbors.
    shortest_paths : dict
        A dictionary containing the shortest paths between all pairs of streets.
```

Graph construction and shortest paths logic (note the `lazy=True` path that avoids all-pairs):

```512:597:nomad/city_gen.py
    def get_street_graph(self, lazy=True):
        ...
        G = nx.Graph()
        G.add_nodes_from([(int(x), int(y)) for x, y in nodes_df[['coord_x', 'coord_y']].itertuples(index=False)])
        if not edges_df.empty:
            edge_list = [((int(r.x), int(r.y)), (int(r.nx), int(r.ny))) for r in edges_df.itertuples(index=False)]
            G.add_edges_from(edge_list, weight=1)

        self.street_graph = G
        
        # Lazy mode: don't precompute all paths (saves memory)
        if lazy:
            self.shortest_paths = {}
            self.gravity = None
        else:
            # Legacy mode: precompute all pairs (infeasible city-scale)
            self.shortest_paths = dict(nx.all_pairs_shortest_path(G))
            ...
        # Always build the shortcut ("highway") network for fast queries
        self._build_shortcut_network()
        return self.street_graph
```

Door/adjacency logic in the random generator assigns a building’s door to an adjacent street block:

```1483:1501:nomad/city_gen.py
    def get_adjacent_street(self, location):
        ...
        possible_streets = np.array([(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]])
        ...
        # treat mask as street if present in streets_gdf
        if hasattr(self.city, 'streets_gdf') and not self.city.streets_gdf.empty:
            street_mask = np.array([
                ((self.city.streets_gdf['coord_x'] == sx) & (self.city.streets_gdf['coord_y'] == sy)).any()
                for sx, sy in valid_streets
            ])
            valid_streets = valid_streets[street_mask]
        return tuple(valid_streets[0].tolist()) if valid_streets.size > 0 else None
```

Implication: in the synthetic case the use of streets is narrow—adjacency for doors, a grid graph for routing, and optional plotting. This supports replacing grid streets with a more natural street centerline graph while preserving the door concept.

---

## Part C — Rasterization and shortest paths: where and why it hurts

The rasterization pathway converts OSM polygons/lines into blocks. It is clear and compatible with the grid city, but it does not scale to full Philadelphia at fine block sizes.

```1672:1711:nomad/city_gen.py
class RasterCityGenerator:
    def generate_city(self) -> 'City':
        print("Generating canvas blocks...")
        self.blocks_gdf = generate_canvas_blocks(self.boundary_polygon, self.block_size, self.crs)
        ...
        typed_blocks = assign_block_types(self.blocks_gdf, self.streets_gdf, buildings_by_type, self.block_size)
        ...
        connected_streets, summary = verify_street_connectivity(street_blocks)
        ...
        city = City(..., manual_streets=True, ...)
        city.blocks_gdf = typed_blocks.copy()
        ...
        city.streets_gdf = gpd.GeoDataFrame(connected_streets[[...]], ...)
```

Primary scalability issues:
- Grid explosion: memory and I/O scale with the number of blocks intersecting the boundary; with 0.5–1 m cells this becomes massive.
- Union operations per route step: `unary_union` over many block polygons for a bound corridor is expensive; grows with path length and polygon count.
- All-pairs shortest paths: legacy branch is O(n²) space; must be avoided at city scale (the code already implements `lazy=True` and shortcut routing to mitigate this).

---

## Part D — OSM’s native street graph: structure and implications

- **Elements**:
  - Nodes: OSM “nodes” with geographic coordinates.
  - Ways: ordered lists of nodes (street centerlines), with tags (`highway`, `name`, `oneway`, `lanes`, `maxspeed`, `width`, etc.).
  - Relations: collections for complex structures (e.g., routes).
- **In OSMnx** (used in this repo via `nomad.map_utils.download_osm_streets`):
  - `ox.graph_from_polygon(...)`/`graph_from_bbox(...)` returns a NetworkX `MultiDiGraph`.
  - Nodes carry `osmid`, projected `x`/`y` coordinates, etc.
  - Edges carry `u`, `v`, `key`, `length` (meters), `highway`, optional `geometry` as a shapely `LineString` polyline (when an edge is curved/segmented), and other tags.
  - Graphs are usually simplified (degree-2 nodes contracted) but still centerline-based.
  - There is no native “width” in geometry; width is an optional tag (`width=*`) or implied via `lanes`/`highway` class. If a corridor is needed, it is created by buffering the centerline with an assumed or tag-derived width.

Code touchpoint in this repo that acquires OSM data as a graph and converts to GeoDataFrames:

```664:691:nomad/map_utils.py
with _osmnx_cache_context(cache_mode):
    if city_polygon is not None:
        G = ox.graph_from_polygon(city_polygon, custom_filter=custom_filter, simplify=True)
    else:
        G = ox.graph_from_bbox(bbox=bbox, custom_filter=custom_filter, truncate_by_edge=bool(clip), simplify=True)
...
streets = ox.graph_to_gdfs(G, edges=True, nodes=False)
streets = streets.to_crs(crs)
if 'highway' in streets.columns:
    streets['highway'] = streets['highway'].apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else v)
```

Implication: the natural unit for routing is an edge/node graph with centerline polylines and lengths. Width is a rendering/semantics concern and can be approximated or looked up from tags.

---

## Part E — A scalable, vector-first plan

Replace the grid streets with the OSM street graph and keep buildings as polygons. The door remains essential, but becomes a snapped point to the nearest edge/node.

1) Street network and routing
- Use OSMnx to build a walking graph (undirected or bidirectional edges with proper `length`).
- Preprocess: simplify the graph; optionally keep only the largest connected component; optional weak filtering of service/track/private roads depending on needs.
- Routing strategy:
  - Default: A* with Euclidean heuristic in projected CRS.
  - Accelerate with ALT landmarks: precompute distances to 16–32 landmarks; cache per-node potentials to bound A* expansions.
  - Maintain a small LRU cache of recent `node→node` shortest paths and lengths.
  - Optional: build a sparse “hub” overlay (already conceptually present in `City._build_shortcut_network`) to bound query time without O(n²) memory.

2) Doors and building connectivity
- For each building polygon, compute a door point:
  - Project the building centroid or an exterior point to the nearest street edge’s polyline (perpendicular projection) and snap to the closest edge node if within a small threshold; otherwise record a continuous position along the edge (parametric distance) as a “virtual access point”.
  - Store: `building_id → (u, v, edge_key, snapped_point, snapped_to_node?: bool)`.
  - For routing, convert a door to a node: if snapped to a node, route `node→node`; if not, pick nearer endpoint node `(u or v)` and add the projected offset as an initial/terminal along-edge segment in the route geometry.
- Do not insert 500k access nodes into the street graph; treat access points as geometry-only segments attached to the closest edge endpoint for routing purposes.

3) Route geometry and stepping
- Replace block-center polylines with concatenated edge geometries returned by the path `(u0→u1→...→uk)`; prepend/append the short access segments from/to door points.
- Replace polygon union corridors with a single `LineString`/`MultiLineString` buffer: `route_geom.buffer(w)` where `w` is an assumed or tag-derived half-width (e.g., 4–6 m for sidewalks) plus a small tolerance. This is dramatically faster than unioning many block polygons per step.

4) Destination sampling at scale (30k users)
- Use numerically stable log-weights: compute `p ∝ exp(logit - max_logit)` with double precision; normalize by sum in float64.
- Precompute alias tables for any static distribution (e.g., exploration base over all buildings or per-category pools). Sampling becomes O(1) per draw after O(n) preprocess.
- Preferential return: maintain each agent’s top-K visited buildings with a separate alias table; mix with exploration via a Bernoulli switch; K≈50–200 keeps memory bounded.
- Spatial hierarchy: cluster buildings (e.g., H3 or k-means) and sample cluster→building to reduce vector sizes; cache cluster-level alias tables.
- Sparsity: when applicable, store weights sparsely and avoid dense n-length vectors for each user.

5) Memory and throughput expectations (Philadelphia scale)
- OSMnx graphs for a large city are typically 50k–200k nodes and 100k–500k edges depending on filters; with A*+ALT and caching, single-source queries are milliseconds.
- For 30k users with ~100 trips/day, expect a few million route queries. With caching, repeated OD pairs (e.g., home↔work) amortize well. Landmark potentials are shared and memory is O(n·L) for L landmarks.

6) Compatibility layer to `traj_gen`
- Provide a `CityLike` interface exposing:
  - `get_shortest_path_door_to_door(bid_src, bid_dst) → route geometry (LineString/MultiLineString) + node sequence`
  - `get_distance_fast(node_or_door_a, node_or_door_b)` (existing name can be preserved)
  - `get_door(building_id) → snapped access descriptor`
- Update `_sample_step` to accept a polyline route and a buffered corridor instead of grid-block unions. The stepping math already operates in path-length coordinates and adapts cleanly to polylines.

---

## Part G — Compact hybrid design to try now

We adopt a hybrid structure that keeps rasterized buildings (for doors and simple adjacency) and a vector street graph (for routing), plus a light “shortcut” backbone:

- Doors: rasterize buildings and assign one door per connected component. Store the door point and its snapped edge/node in the street graph. Cache a short “stub” segment from door point to the graph access point.
- Street graph: undirected, simplified OSMnx graph with `length`. Sample well-distributed “shortcut hubs” by uniform sampling across the city bbox (or use grid/Poisson-disk), snap samples to nearest nodes, and deduplicate. Optionally stratify by density.
- Cached partial routes: for each door, cache door→nearest-hub path (node list) and door→nearest-k neighbor doors (within a radius) as `MultiLineString`s. For hubs, precompute hub→hub next-hop (ALT backbones or hub BFS table). Query assembly: A→hu + hu→hv + hv→B, with door/buffered stub segments at ends.
- Execution: Use a buffered `LineString` corridor per route for the step integrator; dispense with heavy polygon unions.

This yields short, local per-door caches and small O(H²) hub tables. Memory remains bounded and independent of O(N²) all-pairs.

---

## Part H — Answers to key design questions

1) Are rectangular/paraaxial building polygons essential?
- Not essential. Current stepping uses the destination centroid and (earlier) a bounding polygon mainly to constrain noise. With a polyline corridor buffer, building geometry need not be rectangles nor aligned. Doors remain points; buildings can be arbitrary polygons (from OSM) and the corridor keeps the walker on route. Rasterized rectangles are convenient for synthetic demos, not a hard requirement.

2) Is there already a “stub” for door→street? Can we cache it?
- Yes. The existing trajectory code already includes start/end segments around doors when building a path (start segment + street path + end segment). Generalizing this to vector streets: store, per building, a cached short segment from the door point to the nearest edge/node (the “stub”). On route assembly, prepend/append these stubs and buffer the whole polyline into the corridor. This is efficient and reusable across trips.

3) Do we need road thickness? What about the drift/orthogonal coordinates?
- The drift logic works in path-length coordinates; it does not require a full-width street polygon. A small buffer around the route polyline suffices as a corridor to keep positions near the line and out of buildings. Replace big `unary_union` street polygons with `route.buffer(w)` where `w` is a small width (e.g., ~3–5 m), and optionally check against building polygons to avoid intrusions. This simplifies and speeds up stepping without losing realism.

4) Efficiently getting planar faces/neighborhoods from the street graph?
- Raw OSM graphs are often non-planar (bridges/viaducts). Two practical options:
  - Vector approach: polygonize edges. Filter out bridges (e.g., `bridge!=yes`) and similar, then `merged = unary_union(edges)` and `faces = polygonize(merged)` from Shapely. This is fast, parallelizable by tiles, and robust for neighborhoods.
  - Graph approach: after planarization, `networkx.algorithms.planarity.check_planarity` returns an embedding from which faces can be enumerated quickly. You must first split at intersections (OSMnx already consolidates intersections; additional planarization may be needed). For city-scale, prefer tile-based polygonize for performance and memory stability.

References: OSMnx user reference (graph/convert/simplify/routing) and Shapely polygonize utilities. See OSMnx docs: https://osmnx.readthedocs.io/en/stable/user-reference.html

---

## Part F — Migration plan (incremental, low risk)

- Phase 1: Keep raster cities for tests and small demos. Introduce `OSMStreetNetwork` component and a `VectorCityAdapter` that wraps buildings GDF + OSM graph and implements the `CityLike` interface methods used by `traj_gen`.
- Phase 2: Add door snapping and OD routing on OSM graph. Implement buffered-route corridors and switch `_sample_step` to use them when available.
- Phase 3: Add ALT landmarks and caching; add alias-based sampling for exploration; add top‑K preferential return memory.
- Phase 4: Remove dependency on raster streets for large, real cities; keep raster for toy examples.

---

## Appendix — Additional code references

- `RandomCityGenerator` calls lazy graph build and assigns doors for a block city:

```1512:1517:nomad/city_gen.py
    def generate_city(self):
        """Generates a systematically structured city where blocks are fully occupied with buildings."""
        self.place_buildings_in_blocks()
        self.city.get_street_graph(lazy=True)  # Use lazy mode to avoid memory issues
        return self.city
```

- Distance fast-path using shortcut network and a small cache:

```1354:1405:nomad/city_gen.py
    def get_distance_fast(self, start_coord: tuple, end_coord: tuple) -> float:
        ...
        if key in self._distance_cache:
            return self._distance_cache[key]
        try:
            path = self.get_shortest_path(start_coord, end_coord)
            if not path:
                d = float('inf')
            else:
                d = float(max(0, len(path) - 1))
        except Exception:
            d = float('inf')
        ...
        return d
```

- OSM download and conversion to GDFs (edges as polylines with tags):

```664:697:nomad/map_utils.py
with _osmnx_cache_context(cache_mode):
    if city_polygon is not None:
        G = ox.graph_from_polygon(city_polygon, custom_filter=custom_filter, simplify=True)
    else:
        G = ox.graph_from_bbox(bbox=bbox, custom_filter=custom_filter, truncate_by_edge=bool(clip), simplify=True)
...
streets = ox.graph_to_gdfs(G, edges=True, nodes=False)
streets = streets.to_crs(crs)
if 'highway' in streets.columns:
    streets['highway'] = streets['highway'].apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else v)
```

---

## Closing note on removing rasterization
- For production-scale cities, rasterization should be optional and limited to toy demos. Streets are naturally centerlines; routing naturally happens on a polyline graph. A vector-first approach retains realism, removes grid blowup, and makes route integration faster by buffering a single route polyline instead of unioning many block polygons.
