# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Report on scalability of methods to decide between building routes in `traj_gen.py`
#
# There are two Critical Bottlenecks
#
# - 1. All-Pairs Shortest Path Computation (Memory), used in `nomad/city_gen.py:1190`
#
# ```python
# city.shortest_paths = dict(nx.all_pairs_shortest_path(G))
# ```
# - 2. Geometric Path Construction in `_sample_step` (Performance)
#
# **The Expensive Operations** (lines 424-433):
#
# ```python
# # 1. Get shortest path (this part is OK with hub network)
# street_path = city.get_shortest_path(start_node, dest_cell)
#
# # 2. Build MultiLineString from path (EXPENSIVE)
# path = [(x + 0.5, y + 0.5) for (x, y) in street_path]
# path = start_segment + path + [(dest_door_centroid.x, dest_door_centroid.y)]
# path_ml = MultiLineString([path])
#
# # 3. Union all geometries along path (VERY EXPENSIVE)
# street_geom = unary_union([city.get_block(b)['geometry'] for b in street_path])
# bound_poly = unary_union([start_info['geometry'], street_geom]) if start_info['geometry'] is not None else street_geom
# ```
#
# **Purpose**: Agent moves along street path with:
# - Forward drift (`heading_drift`)
# - Random walk in path-perpendicular direction
# - Constrained to stay within street corridor (`bound_poly`)
#
#

# %%
