## A scalable way to use real streets for NOMAD trajectories

Philadelphia’s data volume pushes the original raster‑first design past its limits. The old approach divided the world into small grid cells, treated every street as adjacent street blocks, and formed a “corridor” by unioning those blocks along a path. That made sense for tiny demos, but at city scale the costs pile up: the grid itself is large, repeated polygon unions are expensive, and any attempt to precompute many shortest paths is quadratic in the number of locations. With 500k buildings and a comparable number of street segments, this becomes the dominant runtime and memory sink.

The good news is that traj_gen doesn’t need any of that machinery. It needs three things only: a path from one door to another along streets, a narrow corridor that keeps simulated steps near that path and out of buildings, and a coordinate system along the path to apply drift. A single polyline plus a small buffer satisfies all three. The polyline comes from routing on a vector street graph. The buffer provides the “tube” to constrain movement. The polyline’s cumulative length provides the path‑based coordinate system that traj_gen already uses. In other words, we can keep traj_gen’s behavior intact while removing the grid and the heavy polygon unions that make it slow.

```474:480:nomad/traj_gen.py
# Build continuous path through block centers, include start/end segments
path = [(x + 0.5, y + 0.5) for (x, y) in street_path]
path = start_segment + path + [(dest_door_centroid.x, dest_door_centroid.y)]
path_ml = MultiLineString([path])
street_geom = unary_union([city.get_block(b)['geometry'] for b in street_path])
```

```1019:1036:nomad/traj_gen.py
def _path_coords(multilines, point, eps=0.001):
    """
    Given a MultiLineString and a cartesian point, returns the transformed coordinates:
    distance along the path and signed perpendicular offset.
    """
    if not isinstance(point, Point):
        point = Point(point)

    distance = multilines.project(point)
    point_on_path = multilines.interpolate(distance)
    offset_point = multilines.interpolate(distance - eps)

    p = np.array([point_on_path.x, point_on_path.y])
    q = np.array([offset_point.x, offset_point.y])
    direction = p - q
    unit_direction = direction / np.linalg.norm(direction)

    # Rotate 90° counter-clockwise to get the normal vector
    normal = np.flip(unit_direction) * np.array([-1, 1])

    delta = np.array([point.x - p[0], point.y - p[1]])
    offset = np.dot(delta, normal)

    return distance, offset
```

In the non–data‑driven generator, streets mainly exist so that each building can have a door “on a street.” Today that means placing a door next to a street block. With a vector street graph, the idea is the same, just cleaner: each building has a door point and a tiny “stub” segment that snaps that door to the nearest street access (a node or a point along an edge). The stub is a short LineString you compute once and cache per building. It replaces repeated nearest‑street lookups and gives you a consistent place to enter and exit the network.

```312:331:nomad/city_gen.py
# Compute door centroid via intersection with target street
door_poly = box(door[0], door[1], door[0]+1, door[1]+1)
door_line = geom.intersection(door_poly)
if door_line.is_empty:
    # ... fallbacks elided ...
    door_centroid = (door[0] + 0.5, door[1] + 0.5)
else:
    door_centroid = (door_line.centroid.x, door_line.centroid.y)
```

```1510:1521:nomad/city_gen.py
def get_adjacent_street(self, location):
    """Finds the closest predefined street to assign as the door, ensuring it is within bounds."""
    if not location or not isinstance(location, tuple):
        return None
    x, y = location
    possible_streets = np.array([(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]])
    
    valid_mask = (possible_streets[:, 0] >= 0) & (possible_streets[:, 0] < self.width) & 
                 (possible_streets[:, 1] >= 0) & (possible_streets[:, 1] < self.height)
```

OpenStreetMap, via OSMnx, already gives us the right abstraction for streets. We build a MultiDiGraph where edges represent street centerlines as polylines with attributes. After simplification and intersection consolidation, nodes correspond to true intersections and dead‑ends and edges correspond to the street segments between them. We project the graph to a metric CRS, ensure every edge has a length, and treat it as undirected for pedestrian‑style routing. OSM does not encode curb‑to‑curb width in this graph, and that’s fine: width is a simulation concern, not a map concern. We supply it by buffering the route polyline when we need a corridor. In the codebase we enable intersection consolidation by default and drop edges shorter than 20 meters to avoid driveways and tiny artifacts; filtering tunnels or covered segments and keeping the largest connected component further improves quality.

```104:107:nomad/constants.py
INTERSECTION_CONSOLIDATION_TOLERANCE_M = 12.0  # merge clustered nodes into a single intersection
STREET_MIN_LENGTH_M = 20.0  # drop micro segments like short driveways/service stubs
```

```524:532:nomad/map_utils.py
Gp = ox.project_graph(G)
try:
    Gc = ox.simplification.consolidate_intersections(
        Gp,
        tolerance=INTERSECTION_CONSOLIDATION_TOLERANCE_M,
        rebuild_graph=True
    )
except Exception:
    Gc = Gp
```

```549:556:nomad/map_utils.py
nodes_gdf, edges_gdf = ox.graph_to_gdfs(Gc)
if 'length' in edges_gdf.columns:
    edges_gdf = edges_gdf[edges_gdf['length'] >= STREET_MIN_LENGTH_M]
else:
    # Fallback by geometry length in projected CRS
    edges_gdf = edges_gdf[edges_gdf.geometry.length >= STREET_MIN_LENGTH_M]
```

Routing then becomes straightforward. To go from building A to building B, we assemble a path as: A’s door stub, then the shortest path on the street graph, then B’s door stub. Concatenate those segments and you have a single polyline. Buffer it a few meters to create the corridor. During simulation we step along the polyline in path‑length coordinates, apply the usual drift, and reject samples that fall outside the corridor. The effect is the same “follow a street with a bit of randomness” that we want, but the corridor is now cheap to construct and does not depend on raster streets.

This design scales because it avoids global precomputation and leans on small, reusable pieces. There is no all‑pairs shortest path table. We compute only what we need and cache what is reused. Two caches do most of the work. First, door stubs are per‑building and never change; compute them once. Second, a sparse “shortcut” backbone gives fast long‑range routing: sample a modest number of hubs across the city (for example, by a grid or Poisson‑disk), snap them to graph nodes, and precompute routes or next hops among hubs. For a long trip, route from the door to the nearest hub, traverse hub‑to‑hub, then hub to the destination door. For short trips within a local radius, route directly and cache the result the first time you see it. With this mix, query time is effectively constant while the cache only grows with actual use, not with the square of the number of buildings.

```604:641:nomad/city_gen.py
def _build_shortcut_network(self, target_num_hubs: int = 256) -> None:
    """
    Build a sparse shortcut routing structure that enables near-instant shortest
    path queries with low memory:

    - Select a well-distributed subset of street blocks as hubs
    - Precompute hub-to-hub next-hop table (on the hub graph)
    - Precompute, for every node, the next step toward its nearest hub (multi-source BFS)
    """
    if not hasattr(self, 'street_graph') or self.street_graph is None:
        return
    
    G = self.street_graph
    if G.number_of_nodes() == 0:
        return
    
    # Select hubs
    hubs = self._select_hubs(target_num_hubs)
```

Some clarifications help connect this to the existing code. Building polygons do not need to be rectangular or axis‑aligned. That constraint was an artifact of the raster demo. For routing, a building needs a door point and, optionally, its polygon for plotting or occasional collision checks. The corridor is defined by buffering the route polyline, not by the shape of the building. The “stub” segment is not a new concept either: traj_gen already builds short segments from a building’s interior to its door; we are just formalizing the persistent piece that connects the door to the street network. As for road thickness, we don’t need a special map attribute for it. We only need thickness when sampling, so a constant‑width buffer around the route suffices. If you want extra safety, subtract building polygons wherever the buffer overlaps them; this yields the same tube the drift logic expects, without global unions.

It is also feasible to derive “neighborhoods” or “blocks” from the street graph when you need them for clustering or local sampling. If you drop non‑planar edges such as tunnels and bridges and work in a projected CRS, you can dissolve the street centerlines and polygonize them to get faces. Do it by tiles if the dataset is large. This uses standard vector operations and is fast at scale. A purely graph‑theoretic approach via planar embeddings is possible too, but OSM networks are not strictly planar, so preprocessing is required; in practice, polygonizing lines is the more robust path for our purposes.

On persistence, we save the consolidated street network as GraphML. That is the right format to reload into OSMnx/NetworkX without loss of attributes. Buildings, doors, and any derived polygons can be saved and loaded as GeoJSON, GeoParquet, Shapefile, or GeoPackage. These formats keep the pipeline easy to inspect and efficient to move through development.

```535:542:nomad/map_utils.py
# Optionally persist the consolidated OSMnx graph as GraphML (projected CRS)
if graphml_path:
    try:
        # OSMnx API: save_graphml at top-level
        ox.save_graphml(Gc, filepath=str(graphml_path))
    except Exception:
        pass
```

```992:1008:nomad/city_gen.py
# Optional GraphML persistence of the internal grid street graph
if street_graphml_path:
    try:
        if not hasattr(self, 'street_graph') or self.street_graph is None:
            self.get_street_graph(lazy=True)
        G = self.street_graph
        # Remap tuple nodes to string ids and attach coordinates as attributes
        H = nx.Graph()
        for (x, y) in G.nodes:
            node_id = f"{int(x)}_{int(y)}"
            H.add_node(node_id, x=int(x), y=int(y))
        for u, v, data in G.edges(data=True):
            uid = f"{int(u[0])}_{int(u[1])}"
            vid = f"{int(v[0])}_{int(v[1])}"
            w = data.get('weight', 1)
            H.add_edge(uid, vid, weight=int(w))
        nx.write_graphml(H, street_graphml_path)
    except Exception:
        # Silently ignore GraphML write issues to avoid breaking primary persistence
        pass
```

What changes in the code is modest but high‑impact. The map utilities already consolidate intersections by default and filter out sub‑20‑meter edges. The I/O functions read and write the standard geospatial formats. The next small refactors are to replace union‑of‑blocks corridors with buffered route polylines and to add a small index that stores, for each building, its door point, the snapped street access, and its precomputed stub. A lightweight route assembler can then produce a MultiLineString on demand and feed it into the existing path‑based coordinate logic without further changes.

The core idea is deliberately simple: keep buildings for doors, keep streets as a vector graph, and connect the two with short, cached stubs. Build routes as polylines and turn them into thin tubes with a buffer. This preserves realism, stabilizes performance, and cleans up the code paths so they remain understandable and maintainable at city scale.
