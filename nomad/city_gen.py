from shapely.geometry import box, Polygon, LineString, MultiLineString, Point, MultiPolygon
from shapely import wkt
from shapely.affinity import scale, translate, rotate as shapely_rotate
from shapely.ops import unary_union
from shapely.prepared import prep
import pickle
import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import networkx as nx
import geopandas as gpd
import os
import time
import pdb
from typing import List, Tuple, Set
from pyproj import CRS

from nomad.map_utils import blocks_to_mercator, mercator_to_blocks
from nomad.constants import TYPE_PRIORITY

# =============================================================================
# STREET CLASS
# =============================================================================

class Street:
    """
    Class for street block in the city through which individuals move from building to building.

    Attributes
    ----------
    coordinates : tuple
        A tuple representing the (x, y) coordinates of the street block.
    geometry : shapely.geometry.polygon.Polygon
        A polygon representing the geometry of the street block.
    id : str
        A unique identifier for the street block, formatted as 's-x{coordinates[0]}-y{coordinates[1]}'.

    Methods
    -------
    (none)
    """

    def __init__(self, coordinates: tuple = None, geometry: Polygon = None):
        if coordinates is not None:
            self.coordinates = coordinates
            self.geometry = box(coordinates[0], coordinates[1], coordinates[0] + 1, coordinates[1] + 1)
        elif geometry is not None:
            if not geometry.is_valid:
                raise ValueError("Invalid street geometry")
            bounds = geometry.bounds
            if abs(bounds[2] - bounds[0] - 1) > 0.01 or abs(bounds[3] - bounds[1] - 1) > 0.01:
                raise ValueError("Street geometry must represent a 1x1 block")
            self.coordinates = (int(bounds[0]), int(bounds[1]))
            self.geometry = geometry
        else:
            raise ValueError("Either coordinates or geometry must be provided")

        self.id = f's-x{self.coordinates[0]}-y{self.coordinates[1]}'


# =============================================================================
# CITY CLASS
# =============================================================================


class City:
    """
    Class for representing a city containing buildings, streets, and methods for city management.

    Attributes
    ----------
    buildings : dict
        A dictionary of Building objects with their IDs as keys.
    streets : dict
        A dictionary of Street objects with their coordinates as keys.
    buildings_outline : shapely.geometry.polygon.Polygon
        A polygon representing the combined geometry of all buildings in the city.
    city_boundary : shapely.geometry.polygon.Polygon
        A polygon representing the boundary of the city.
    dimensions : tuple
        A tuple representing the dimensions of the city (width, height).
    street_graph : dict
        A dictionary representing the graph of streets with their neighbors.
    shortest_paths : dict
        A dictionary containing the shortest paths between all pairs of streets.
    gravity : pandas.DataFrame
        A DataFrame containing the gravity values between all pairs of streets.

    Methods
    -------
    add_building
        Adds a building to the city.
    get_block
        Returns the block (Street or Building) at the given coordinates.
    get_street_graph
        Constructs the graph of streets and calculates the shortest paths between all pairs of streets.
    save
        Saves the city object to a file.
    plot_city
        Plots the city on a given matplotlib axis.
    """

    def __init__(self,
                 dimensions: tuple = (0,0),
                 manual_streets: bool = False,
                 name: str = "Garden City",
                 block_side_length: float = 15.0,
                 web_mercator_origin_x: float = -4265699.0,
                 web_mercator_origin_y: float = 4392976.0,
                 rotation_deg: float = 0.0,
                 offset_x: int = 0,
                 offset_y: int = 0):
        
        """
        Initializes the City object with given dimensions and optionally manual streets.

        Parameters
        ----------
        dimensions : tuple
            A tuple representing the dimensions of the city (width, height).
        manual_streets : bool
            If True, streets must be manually added to the city. 
            If False, all blocks are initialized as streets until it is populated with a building.
        name : str, optional
            Name of the city (default: "Garden City").
        block_side_length : float, optional
            Side length of city blocks in meters (default: 15.0).
        web_mercator_origin_x : float, optional
            False easting for Web Mercator projection (default: -4265699.0).
        web_mercator_origin_y : float, optional
            False northing for Web Mercator projection (default: 4392976.0).
        rotation_deg : float, optional
            Rotation applied to input data in degrees counterclockwise (default: 0.0).
        offset_x : int, optional
            Grid offset in block units along x-axis (default: 0).
        offset_y : int, optional
            Grid offset in block units along y-axis (default: 0).
        """

        self.name = name
        self.block_side_length = block_side_length
        self.web_mercator_origin_x = web_mercator_origin_x
        self.web_mercator_origin_y = web_mercator_origin_y
        self.rotation_deg = rotation_deg
        self.offset_x = offset_x
        self.offset_y = offset_y

        self.buildings_outline = Polygon()

        self.manual_streets = manual_streets

        if not (isinstance(dimensions, tuple) and len(dimensions) == 2
                and all(isinstance(d, int) for d in dimensions)):
            raise ValueError("Dimensions must be a tuple of two integers.")
        self.city_boundary = box(0, 0, dimensions[0], dimensions[1])

        self.dimensions = dimensions

        self.buildings_gdf = gpd.GeoDataFrame(
            columns=['id','building_type','door_cell_x','door_cell_y','door_point','size','geometry'],
            geometry='geometry', crs=None
        )
        self.buildings_gdf.set_index('id', inplace=True, drop=False)
        self.buildings_gdf.index.name = None
        
        if manual_streets:
            self.blocks_gdf = gpd.GeoDataFrame(
                columns=['coord_x','coord_y','kind','building_id','building_type','geometry'],
                geometry='geometry', crs=None
            )
            self.blocks_gdf.set_index(['coord_x', 'coord_y'], inplace=True, drop=False)
            self.blocks_gdf.index.names = [None, None]
            self.streets_gdf = gpd.GeoDataFrame(
                columns=['coord_x','coord_y','id','geometry'],
                geometry='geometry', crs=None
            )
            self.streets_gdf.set_index(['coord_x', 'coord_y'], inplace=True, drop=False)
            self.streets_gdf.index.names = [None, None]
        else:
            self.blocks_gdf = self._init_blocks_gdf()
            self.streets_gdf = self._derive_streets_from_blocks()
        # Convenience properties are defined below for GDF-first access
        # Precomputed building-to-building gravity (optional, built on demand)
        self.grav = None
        self.door_dist = None

    def _derive_streets_from_blocks(self):
        """
        Derives streets GeoDataFrame from blocks_gdf where kind is 'street'.
        """
        if not hasattr(self, 'blocks_gdf') or self.blocks_gdf.empty:
            self.blocks_gdf = self._init_blocks_gdf()
        streets = self.blocks_gdf[self.blocks_gdf['building_type'] == 'street'].copy()
        if not streets.empty:
            streets['id'] = streets['coord_x'].astype(int).astype(str) + '-y' + streets['coord_y'].astype(int).astype(str)
            streets['id'] = 's-x' + streets['id']
        streets_gdf = gpd.GeoDataFrame(streets, geometry='geometry', crs=self.blocks_gdf.crs)
        streets_gdf.set_index(['coord_x', 'coord_y'], inplace=True, drop=False)
        streets_gdf.index.names = [None, None]
        return streets_gdf

    def _init_blocks_gdf(self):
        """Initialize grid of unit blocks covering current dimensions."""
        width, height = self.dimensions
        if width <= 0 or height <= 0:
            return gpd.GeoDataFrame(columns=['coord_x','coord_y','kind','building_id','building_type','geometry'], geometry='geometry', crs=None)

        x_coords = np.arange(width) # must be in city_block units
        y_coords = np.arange(height) # must be in city_block units
        
        # Create a MultiIndex from the product of x and y coordinates
        multi_index = pd.MultiIndex.from_product([x_coords, y_coords], names=['coord_x', 'coord_y'])
        
        # Create a DataFrame from the MultiIndex
        df = pd.DataFrame(index=multi_index).reset_index()
        
        # Add other columns
        df['building_type'] = 'street'
        df['building_id'] = None
        df['geometry'] = df.apply(lambda row: box(row['coord_x'], row['coord_y'], row['coord_x']+1, row['coord_y']+1), axis=1)
        
        blocks_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=None)
        blocks_gdf.set_index(['coord_x', 'coord_y'], inplace=True, drop=False)
        blocks_gdf.index.names = [None, None]
        return blocks_gdf

    @property
    def buildings_df(self):
        return self.buildings_gdf

    @property
    def streets_df(self):
        return self.streets_gdf

    def add_street(self, coords):
        """
        Adds a street to the city at the specified (x, y) coordinates.
        """
        x, y = coords
        # ensure blocks_gdf marks as street
        if hasattr(self, 'blocks_gdf'):
            mask = (self.blocks_gdf['coord_x'] == x) & (self.blocks_gdf['coord_y'] == y)
            if mask.any():
                self.blocks_gdf.loc[mask, ['kind','building_id','building_type']] = ['street', None, None]
            else:
                self.blocks_gdf = pd.concat([self.blocks_gdf, gpd.GeoDataFrame([{
                    'coord_x': x,
                    'coord_y': y,
                    'kind': 'street',
                    'building_id': None,
                    'building_type': None,
                    'geometry': box(x, y, x+1, y+1)
                }], geometry='geometry', crs=self.blocks_gdf.crs)], ignore_index=True)
        # update streets_gdf
        if hasattr(self, 'streets_gdf'):
            sid = f's-x{x}-y{y}'
            row = {'coord_x': x, 'coord_y': y, 'id': sid, 'geometry': box(x, y, x+1, y+1)}
            self.streets_gdf = pd.concat([self.streets_gdf, gpd.GeoDataFrame([row], geometry='geometry', crs=self.streets_gdf.crs)], ignore_index=True)

    def add_building(self, building_type, door, geom=None, blocks=None, gdf_row=None):
        """
        Adds a building to the city with the specified type, door location, and geometry.

        Parameters
        ----------
        building_type : str
            The type of the building ('home', 'workplace', 'retail', 'park').
        door : tuple
            A tuple representing the (x, y) coordinates of the door of the building.
        geom : shapely.geometry.polygon.Polygon, optional
            The geometry of the building (can be a box or MultiPolygon).
        blocks : list of tuples, optional
            A list of (x, y) coordinates representing the blocks occupied by the building.
        gdf_row : geopandas.GeoDataFrame or pandas.Series, optional
            A single row GeoDataFrame or Series containing building information.

        Raises
        ------
        ValueError
            If the door is not on an existing street, if the building overlaps with existing buildings,
            or if the geometry/blocks do not align with the grid.
        """

        if gdf_row is not None:
            if isinstance(gdf_row, pd.Series):
                gdf_row = gpd.GeoDataFrame([gdf_row], geometry='geometry')
            elif not isinstance(gdf_row, gpd.GeoDataFrame) or len(gdf_row) != 1:
                raise ValueError("gdf_row must be a GeoDataFrame with exactly one row or a pandas Series.")
            # Use explicit column if present; otherwise rely on provided argument
            building_type = gdf_row.iloc[0]['building_type']
            door = (gdf_row.iloc[0]['door_cell_x'], gdf_row.iloc[0]['door_cell_y'])
            geom = gdf_row.iloc[0]['geometry'] if 'geometry' in gdf_row.columns else geom

        # If geom is a list of blocks, reassign it to blocks
        if isinstance(geom, list):
            blocks = geom
            geom = None

        # If geom is None and blocks are provided, construct geometry from blocks
        if geom is None and blocks is not None and len(blocks) > 0:
            block_polys = [box(x, y, x+1, y+1) for x, y in blocks]
            geom = MultiPolygon(block_polys).envelope if len(block_polys) > 1 else block_polys[0]
            # Validate that blocks align with grid
            for x, y in blocks:
                if (x, y) not in self.blocks_gdf.index:
                    raise ValueError(f"Block ({x}, {y}) does not align with city grid.")
        elif geom is not None and blocks is None:
            # Validate that geometry aligns with grid by checking bounds
            minx, miny, maxx, maxy = geom.bounds
            if not (minx.is_integer() and miny.is_integer() and maxx.is_integer() and maxy.is_integer()):
                raise ValueError(f"Geometry bounds {geom.bounds} do not align with integer grid.")
            # Derive blocks from geometry bounds for validation
            blocks = [(int(x), int(y)) for x in range(int(minx), int(maxx))
                      for y in range(int(miny), int(maxy))
                      if minx <= x < maxx and miny <= y < maxy]
            for x, y in blocks:
                if (x, y) not in self.blocks_gdf.index:
                    raise ValueError(f"Derived block ({x}, {y}) from geometry does not align with city grid.")
        elif geom is None and (blocks is None or len(blocks) == 0):
            raise ValueError("Either geom or blocks must be provided.")

        # Compute door centroid via intersection with target street
        door_poly = box(door[0], door[1], door[0]+1, door[1]+1)
        door_line = geom.intersection(door_poly)
        # Require true line adjacency (not area overlap) to the door cell
        if door_line.is_empty or not isinstance(door_line, LineString):
            raise ValueError(f"Door {door} must be adjacent to new building.")
        else:
            door_centroid = (door_line.centroid.x, door_line.centroid.y)

        # Note: street network adjacency is no longer required. Door adjacency
        # is validated via door_line above.
        
        if blocks is not None and len(blocks) > 0:
            building_blocks_set = set(blocks)
            existing_building_blocks = set()
            if hasattr(self, 'blocks_gdf') and not self.blocks_gdf.empty:
                existing_building_blocks = set(
                    self.blocks_gdf[(self.blocks_gdf['building_type'].notna()) & (self.blocks_gdf['building_type'] != 'street')].index.tolist()
                )
            
            if building_blocks_set & existing_building_blocks:
                raise ValueError(
                    "New building overlaps with existing buildings."
                )
        else:
            geom_only_overlaps = self.buildings_outline.contains(geom) or self.buildings_outline.overlaps(geom)
            if geom_only_overlaps:
                raise ValueError(
                    "New building overlaps with existing buildings."
                )

        # Derive blocks from geom if not provided
        if blocks is None:
            minx, miny, maxx, maxy = geom.bounds
            blocks = [(int(x), int(y)) for x in range(int(minx), int(maxx) + 1)
                      for y in range(int(miny), int(maxy) + 1)
                      if minx <= x < maxx and miny <= y < maxy]

        # add building

        if gdf_row is not None and isinstance(gdf_row, gpd.GeoDataFrame) and 'id' in gdf_row.columns:
            building_id = str(gdf_row.iloc[0]['id'])
        else:
            # pick building block adjacent to door using adjacency helper
            candidate = None
            if blocks is not None and len(blocks) > 0:
                mask = self.check_adjacent(blocks, door)  # list[bool]
                idx = next((i for i, m in enumerate(mask) if m), None)
                candidate = blocks[idx] if idx is not None else blocks[0]
            if candidate is None:
                candidate = door
            building_id = f"{building_type[0]}-x{int(candidate[0])}-y{int(candidate[1])}"
        self.buildings_outline = unary_union([self.buildings_outline, geom])
        # Append to buildings_gdf
        dpt = Point(door_centroid[0], door_centroid[1])
        size_blocks = len(blocks) if blocks is not None else 0
        new_row = gpd.GeoDataFrame([
            {
                'id': building_id,
                'building_type': building_type,
                'door_cell_x': door[0],
                'door_cell_y': door[1],
                'door_point': dpt,
                'size': size_blocks,
                'geometry': geom
            }
        ], geometry='geometry', crs=self.buildings_gdf.crs)
        new_row.set_index('id', inplace=True, drop=False)
        new_row.index.name = None
        # Avoid FutureWarning by assigning directly when empty
        if self.buildings_gdf.empty:
            self.buildings_gdf = new_row
        else:
            # Preserve id index across concatenations
            self.buildings_gdf = pd.concat([self.buildings_gdf, new_row], axis=0, ignore_index=False)

        # blocks are no longer streets
        for block in blocks:
            # update blocks_gdf record
            if hasattr(self, 'blocks_gdf') and not self.blocks_gdf.empty:
                cx, cy = block
                self.blocks_gdf.loc[(cx, cy), ['kind','building_id','building_type']] = ['building', building_id, building_type]
            # remove from streets_gdf if present
            if hasattr(self, 'streets_gdf') and not self.streets_gdf.empty and (cx, cy) in self.streets_gdf.index:
                self.streets_gdf = self.streets_gdf.drop(index=(cx, cy))

        # expand city boundary if necessary
        minx, miny, maxx, maxy = geom.bounds
        if maxx > self.dimensions[0] or maxy > self.dimensions[1]:
            new_width = max(self.dimensions[0], int(np.ceil(maxx)))
            new_height = max(self.dimensions[1], int(np.ceil(maxy)))
            self.dimensions = (new_width, new_height)
            self.city_boundary = box(0, 0, new_width, new_height)
            self.blocks_gdf = self._init_blocks_gdf()
            self.streets_gdf = self._derive_streets_from_blocks()

    def add_buildings_from_gdf(self, buildings_gdf):
        """
        Initialize or add multiple buildings from a GeoDataFrame.

        Parameters
        ----------
        buildings_gdf : geopandas.GeoDataFrame
            A GeoDataFrame containing building information with columns for type, door coordinates, and geometry.

        Raises
        ------
        ValueError
            If the GeoDataFrame lacks required columns or contains invalid data.
        """
        required_columns = ['building_type', 'door_cell_x', 'door_cell_y', 'geometry']
        if not all(col in buildings_gdf.columns for col in required_columns):
            raise ValueError(f"GeoDataFrame must contain columns: {required_columns}")

        for _, row in buildings_gdf.iterrows():
            self.add_building(
                building_type=row['building_type'],
                door=(row['door_cell_x'], row['door_cell_y']),
                geom=row['geometry'],
                gdf_row=buildings_gdf.loc[[row.name]]
            )

    def get_block(self, coordinates):
        """
        Returns information about the block at the given coordinates.

        Parameters
        ----------
        coordinates : tuple
            A tuple representing the (x, y) coordinates of the block.

        Returns
        -------
        dict
            A dictionary containing the kind of block ('street' or 'building'), the ID of the building if
            applicable, the type of building if applicable, and the geometry of the block.
        """
        if not isinstance(coordinates, tuple) or len(coordinates) != 2:
            raise ValueError("Coordinates must be a tuple of (x, y).")
        x, y = coordinates
        if not (0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1]):
            raise ValueError(f"Coordinates {coordinates} out of city bounds {self.dimensions}")
        
        # Direct index access using tuple
        if (x, y) in self.blocks_gdf.index:
            row = self.blocks_gdf.loc[(x, y)]
            return {
                'building_type': row['building_type'],
                'building_id': row['building_id'],
                'geometry': row['geometry']
            }
        return {'building_type': None, 'building_id': None, 'geometry': None}

    def check_adjacent(self, geom1, geom2, graph=None):
        """Adjacency on a grid of 1x1 blocks. Returns list[bool]."""
        # Case 1: geom2 is a block cell (x, y)
        if isinstance(geom2, tuple):
            bx, by = int(geom2[0]), int(geom2[1])
            if graph is not None:
                nb = set(graph.neighbors((bx, by)))
            else:
                nb = {(bx+1, by), (bx-1, by), (bx, by+1), (bx, by-1)}

            if isinstance(geom1, tuple):
                cx, cy = int(geom1[0]), int(geom1[1])
                return [((cx, cy) in nb)]
            if isinstance(geom1, list):
                return [((int(b[0]), int(b[1])) in nb) for b in geom1]
            if isinstance(geom1, (Polygon, MultiPolygon)):
                cell_poly = box(bx, by, bx+1, by+1)
                return [isinstance(geom1.intersection(cell_poly), LineString)]
            raise TypeError("geom1 must be a block tuple, list of block tuples, or shapely (Multi)Polygon when geom2 is a block")

        # Case 2: geom2 is a geometry
        if hasattr(geom2, 'intersection'):
            if isinstance(geom1, tuple):
                cx, cy = int(geom1[0]), int(geom1[1])
                poly = box(cx, cy, cx+1, cy+1)
                return [isinstance(poly.intersection(geom2), LineString)]
            if isinstance(geom1, list):
                return [isinstance(box(int(b[0]), int(b[1]), int(b[0])+1, int(b[1])+1).intersection(geom2), LineString) for b in geom1]
            return [isinstance(geom1.intersection(geom2), LineString)]

        raise TypeError("Unsupported types: geom1 must be block tuple/list or shapely geometry; geom2 must be block tuple or shapely geometry")

    def get_street_graph(self):
        """Build the street graph needed for routing.

        Constructs a grid-adjacency graph where each street block is a node
        and edges connect cardinally adjacent blocks.
        """
        if hasattr(self, 'street_graph') and self.street_graph is not None:
            return self.street_graph

        # ---------------------------------------------------------------------
        # Build street block adjacency graph
        # ---------------------------------------------------------------------
        # Nodes
        nodes_df = self.streets_gdf[['coord_x', 'coord_y']].copy()

        # Build edge list via vectorized neighbor matches
        edge_list = []
        for dx, dy in [(1, 0), (0, 1)]:  # undirected; only positive directions
            shifted = nodes_df.copy()
            shifted['nx'] = nodes_df['coord_x'] + dx
            shifted['ny'] = nodes_df['coord_y'] + dy
            # Merge to find neighbor pairs
            merged = shifted.merge(
                nodes_df.rename(columns={'coord_x': 'nx', 'coord_y': 'ny'}),
                on=['nx', 'ny'], how='inner'
            )

            edge_list.append(
                merged[['coord_x', 'coord_y', 'nx', 'ny']].rename(columns={'coord_x': 'x', 'coord_y': 'y'})
            )

        edges_df = pd.concat(edge_list, ignore_index=True)

        # Create graph
        G = nx.Graph()
        G.add_nodes_from([(int(x), int(y)) for x, y in nodes_df.values])
        edge_list = [((int(r.x), int(r.y)), (int(r.nx), int(r.ny))) for r in edges_df.itertuples(index=False)]
        G.add_edges_from(edge_list, weight=1)
        self.street_graph = G
        
        return self.street_graph

    # ---------------------------------------------------------------------
    # Shortcut ("highway") network for fast on-demand routing
    # ---------------------------------------------------------------------
    def _build_hub_network(self, hub_size=100):
        """
        Build a sparse shortcut routing structure that enables near-instant shortest
        path queries with low memory:

        - Select a well-distributed subset of street blocks as hubs
        - Precompute hub-to-hub distances in street_graph
        """
        G = self.get_street_graph()
        hubs = self._select_hubs(hub_size)
        
        # Build edge list with hub tuples as indices
        rows = []
        for origin in hubs:
            distances = nx.single_source_shortest_path_length(G, origin)
            for dest in hubs:
                if origin == dest:
                    continue
                rows.append({'origin': origin, 'dest': dest, 'distance': distances[dest]})
        
        edge_list = pd.DataFrame(rows)
        # Pivot to dense adjacency matrix with hub tuples as row/column indices
        hub_df = edge_list.pivot(index='origin', columns='dest', values='distance').fillna(0).astype(int)
        
        self.hubs = hubs
        self.hub_df = hub_df
        
        return hub_df

    def _select_hubs(self, hub_size = 100):
        """
        Compute evenly spaced street blocks as hubs by partitioning coordinates into quantile buckets
        and selecting one hub from each bucket pair.
        """
        n_buckets = int(np.sqrt(hub_size))
        
        # Create quantile buckets (not added to the dataframe)
        x_buckets = pd.qcut(self.streets_gdf['coord_x'], n_buckets, labels=False, duplicates='drop')
        y_buckets = pd.qcut(self.streets_gdf['coord_y'], n_buckets, labels=False, duplicates='drop')
        
        # Select one hub from each (x_bucket, y_bucket) pair
        hubs = []
        for x_bucket in range(n_buckets):
            for y_bucket in range(n_buckets):
                subset = self.streets_gdf[(x_buckets == x_bucket) & (y_buckets == y_bucket)]
                if not subset.empty:
                    hubs.append(subset.index[0])
        
        return hubs

    def _compute_nearest_hub_and_next_step(self, hubs: set):
        """
        Multi-source BFS from all hubs.
        For each node v, compute:
          - nearest_hub[v]: the hub assigned to v
          - next_to_hub[v]: the next node to step to in order to reach nearest_hub[v]
        """
        from collections import deque

        G = self.street_graph
        nearest_hub = {}
        next_to_hub = {}
        visited = set()

        dq = deque()
        for h in hubs:
            dq.append(h)
            nearest_hub[h] = h
            next_to_hub[h] = h  # self pointer for hubs
            visited.add(h)

        while dq:
            u = dq.popleft()
            for w in G.neighbors(u):
                if w in visited:
                    continue
                visited.add(w)
                nearest_hub[w] = nearest_hub[u]
                # The next step to hub for w is u (the node we came from)
                next_to_hub[w] = u
                dq.append(w)

        return nearest_hub, next_to_hub

    def _compute_hub_next_hop(self, hubs: set):
        """
        For each hub s, run a BFS and assign for every other hub t the first step to take
        from s along the shortest path to t. Stops early when all hubs are discovered.
        Returns: dict-of-dicts next_hop[s][t] -> neighbor node (first step from s).
        """
        from collections import deque

        G = self.street_graph
        hubs_list = list(hubs)
        hub_set = set(hubs_list)
        next_hop = {h: {} for h in hubs_list}

        for s in hubs_list:
            seen = set([s])
            first_step = {s: s}
            dq = deque()
            # Initialize with immediate neighbors of s; their first_step is themselves
            for n in G.neighbors(s):
                if n not in seen:
                    seen.add(n)
                    first_step[n] = n
                    dq.append(n)

            remaining = hub_set - {s}
            while dq and remaining:
                u = dq.popleft()
                if u in hub_set and u in remaining:
                    next_hop[s][u] = first_step[u]
                    remaining.remove(u)
                    # Note: continue exploring to potentially find other hubs
                for w in G.neighbors(u):
                    if w in seen:
                        continue
                    seen.add(w)
                    first_step[w] = first_step[u]
                    dq.append(w)

            # ensure self-case exists
            next_hop[s][s] = s

        return next_hop

    # ---------------------------------------------------------------------
    # Building-to-building gravity
    # ---------------------------------------------------------------------
    def _pairwise_manhattan(self, x1, y1, x2, y2):
        """
        Compute pairwise Manhattan distances between two sets of coordinates.
        """
        dx = np.abs(x1[:, None] - x2[None, :])
        dy = np.abs(y1[:, None] - y2[None, :])
        return dx + dy

    def compute_gravity(self, exponent=2.0, callable_only=False, n_chunks=10):
        """Precompute building-to-building gravity using Manhattan distances and hub shortcuts.
                
        Uses only self.streets_gdf and self.hub_df to compute gravity matrix.
        For each pair of buildings, estimates distance using hub-based shortcuts:
          dist(i,j) = manhattan(door_i, hub_i) + hub_distance(hub_i, hub_j) + manhattan(door_j, hub_j)
          grav(i,j) = 1 / dist(i,j)^exponent  (0 on diagonal)
        
        Parameters
        ----------
        exponent : float
            The gravity decay exponent (default 2.0)
        callable_only : bool
            If True, store callable function instead of dense matrix (default False)
        
        Stores result in self.grav as DataFrame (callable_only=False) or callable (callable_only=True)
        """
        # Part 1: Compute lean distance structures
        building_ids = self.buildings_gdf['id'].to_numpy()
        door_x = self.buildings_gdf['door_cell_x'].astype(int).to_numpy()
        door_y = self.buildings_gdf['door_cell_y'].astype(int).to_numpy()
        
        hubs = np.array(self.hubs)
        door_to_hub_dist = self._pairwise_manhattan(door_x, door_y, hubs[:, 0], hubs[:, 1])
        closest_hub_idx = door_to_hub_dist.argmin(axis=1).astype(np.int32)
        dist_to_closest_hub = door_to_hub_dist.min(axis=1).astype(np.int32)
        
        self.grav_hub_info = pd.DataFrame({
            'closest_hub_idx': closest_hub_idx,
            'dist_to_hub': dist_to_closest_hub
        }, index=building_ids)
        
        # Compute close pairs in chunks to avoid memory issues
        n = len(building_ids)
        chunk_size = max(1, n // n_chunks)
        bid_i_list = []
        bid_j_list = []
        dist_list = []
        
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            
            chunk_dist = self._pairwise_manhattan(
                door_x[i_start:i_end], door_y[i_start:i_end],
                door_x, door_y
            ).astype(np.int32)
            
            min_hub_dist = np.minimum(
                dist_to_closest_hub[i_start:i_end, None],
                dist_to_closest_hub[None, :]
            )
            is_close = chunk_dist <= min_hub_dist
            
            i_local, j_global = np.where(is_close)
            i_global = i_local + i_start
            upper_mask = i_global < j_global
            
            if upper_mask.any():
                bid_i_list.append(building_ids[i_global[upper_mask]])
                bid_j_list.append(building_ids[j_global[upper_mask]])
                dist_list.append(chunk_dist[i_local[upper_mask], j_global[upper_mask]])
        
        if bid_i_list:
            self.mh_dist_nearby_doors = pd.Series(
                np.concatenate(dist_list),
                index=pd.MultiIndex.from_arrays([
                    np.concatenate(bid_i_list),
                    np.concatenate(bid_j_list)
                ])
            )
        else:
            self.mh_dist_nearby_doors = pd.Series([], dtype=np.int32, index=pd.MultiIndex.from_arrays([[], []]))
        
        # Fix: Buildings sharing same door have Manhattan distance 0, set to 1 to avoid inf gravity
        # Note: Future improvement - use Manhattan distance between door centroids for true uniqueness
        door_groups = self.buildings_gdf.groupby(['door_cell_x', 'door_cell_y'])['id']
        same_door_mask = door_groups.transform('size') > 1
        same_door_buildings = set(self.buildings_gdf[same_door_mask]['id'].values)
        
        if same_door_buildings and len(self.mh_dist_nearby_doors) > 0:
            zero_dist_mask = (self.mh_dist_nearby_doors == 0)
            for (bid_i, bid_j) in self.mh_dist_nearby_doors[zero_dist_mask].index:
                if bid_i in same_door_buildings and bid_j in same_door_buildings:
                    self.mh_dist_nearby_doors.loc[(bid_i, bid_j)] = 1
        
        # Part 2: Create dense matrix or callable
        if callable_only:
            bid_to_idx = {bid: i for i, bid in enumerate(building_ids)}
            hub_to_hub = self.hub_df.values
            
            def compute_gravity_row(building_id):
                idx = bid_to_idx[building_id]
                hub_i = closest_hub_idx[idx]
                dist_to_hub_i = dist_to_closest_hub[idx]
                
                distances = dist_to_hub_i + hub_to_hub[hub_i, closest_hub_idx] + dist_to_closest_hub
                
                for (bid_i, bid_j), d in self.mh_dist_nearby_doors.items():
                    if bid_i == building_id:
                        distances[bid_to_idx[bid_j]] = d
                    elif bid_j == building_id:
                        distances[bid_to_idx[bid_i]] = d
                
                distances[idx] = 1  # Temporary non-zero to avoid divide-by-zero warning
                gravity_row = 1.0 / (distances ** exponent)
                gravity_row[idx] = 0.0  # Self-gravity is always 0
                
                return pd.Series(gravity_row, index=building_ids)
            
            self.grav = compute_gravity_row
        else:
            hub_to_hub = self.hub_df.values
            dist_matrix = dist_to_closest_hub[:, None] + hub_to_hub[closest_hub_idx[:, None], closest_hub_idx[None, :]] + dist_to_closest_hub[None, :]
            
            bid_to_idx = {bid: i for i, bid in enumerate(building_ids)}
            for (bid_i, bid_j), d in self.mh_dist_nearby_doors.items():
                i, j = bid_to_idx[bid_i], bid_to_idx[bid_j]
                dist_matrix[i, j] = dist_matrix[j, i] = d
            
            np.fill_diagonal(dist_matrix, 1)  # Temporary non-zero to avoid divide-by-zero warning
            gravity = 1.0 / (dist_matrix ** exponent)
            np.fill_diagonal(gravity, 0.0)  # Self-gravity is always 0
            
            self.grav = pd.DataFrame(gravity, index=building_ids, columns=building_ids)

    def save(self, filename):
        """
        Saves the city object to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the city object to.
        """

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def plot_city(self, ax, doors=True, address=True, zorder=1, heatmap_agent=None, colors=None, alpha=1):
        """
        Plots the city on a given matplotlib axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the city.
        doors : bool
            Whether to plot doors of buildings.
        address : bool
            Whether to plot the address of buildings.
        zorder : int
            The z-order of the plot.
        heatmap_agent : Agent
            The agent for which to plot a heatmap of time spent in each building.
        colors : dict
            A dictionary mapping building types to colors for plotting.
        """

        # Draw city boundary
        x, y = self.city_boundary.exterior.xy
        ax.plot(np.array(x), np.array(y), linewidth=2, color='black')

        # Colors
        if colors is None:
            colors = {
                'street': 'white',
                'home': 'skyblue',
                'workplace': '#C9A0DC',
                'retail': 'lightgrey',
                'park': 'lightgreen',
                'default': 'lightcoral'
            }

        # Streets: fill each street cell
        if hasattr(self, 'streets_gdf') and not self.streets_gdf.empty:
            for _, s in self.streets_gdf.iterrows():
                sx, sy = s.geometry.exterior.xy
                ax.fill(sx, sy, facecolor=colors['street'], linewidth=0.5, zorder=zorder)

        # Buildings
        if heatmap_agent is not None and not self.buildings_gdf.empty:
            weights = heatmap_agent.diary.groupby('location').duration.sum()
            norm = Normalize(vmin=0, vmax=max(weights)/2 if len(weights) else 1)
            base_color = np.array([1, 0, 0])

            for _, b in self.buildings_gdf.iterrows():
                geom = b.geometry
                weight = weights.get(b['id'], 0)
                a = norm(weight) if weight > 0 else 0
                if isinstance(geom, (Polygon, MultiPolygon)):
                    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
                    for poly in polys:
                        bx, by = poly.exterior.xy
                        ax.fill(bx, by, facecolor=base_color, alpha=a,
                        edgecolor='black', linewidth=0.5,
                                zorder=zorder)
                        ax.plot(bx, by, color='black', alpha=1, linewidth=0.5, zorder=zorder + 1)
                        for interior_ring in poly.interiors:
                            x_int, y_int = interior_ring.xy
                            ax.plot(x_int, y_int, color='black', linewidth=0.5, zorder=zorder + 1)
                            ax.fill(x_int, y_int, facecolor='white', zorder=zorder + 1)

                if doors:
                    door_x = float(b['door_cell_x'])
                    door_y = float(b['door_cell_y'])
                    door_line = geom.intersection(box(door_x, door_y, door_x+1, door_y+1))
                    scaled = scale(door_line, xfact=0.25, yfact=0.25, origin=door_line.centroid)
                    if isinstance(scaled, LineString):
                        dx, dy = scaled.xy
                    ax.plot(dx, dy, linewidth=2, color='white', zorder=zorder)

                if address:
                    door_coord = (int(b['door_cell_x']), int(b['door_cell_y']))
                    bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
                    axes_width_in_inches = bbox.width
                    axes_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                    fontsize = (axes_width_in_inches / axes_data_range) * 13
                    ax.text(door_coord[0] + 0.15, door_coord[1] + 0.15,
                            f"{door_coord[0]}, {door_coord[1]}",
                            ha='left', va='bottom', fontsize=fontsize, color='black')

            sm = ScalarMappable(cmap=cm.Reds, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.02).set_label('Minutes Spent')

        elif not self.buildings_gdf.empty:
            for _, b in self.buildings_gdf.iterrows():
                geom = b.geometry
                bcolor = colors.get(b['building_type'], colors['default'])
                if isinstance(geom, (Polygon, MultiPolygon)):
                    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
                    for poly in polys:
                        bx, by = poly.exterior.xy
                        ax.fill(bx, by, facecolor=bcolor,
                            edgecolor='black', linewidth=0.5, alpha=alpha,
                            zorder=zorder)
                        for interior_ring in poly.interiors:
                            x_int, y_int = interior_ring.xy
                            ax.plot(x_int, y_int, color='black', linewidth=0.5, zorder=zorder + 1)
                            ax.fill(x_int, y_int, facecolor='white', zorder=zorder + 1)

                if doors:
                    door_x = float(b['door_cell_x'])
                    door_y = float(b['door_cell_y'])
                    door_line = geom.intersection(box(door_x, door_y, door_x+1, door_y+1))
                    scaled = scale(door_line, xfact=0.25, yfact=0.25, origin=door_line.centroid)
                    if isinstance(scaled, LineString):
                        dx, dy = scaled.xy
                        ax.plot(dx, dy, linewidth=2, color='white', zorder=zorder + 2)
                    elif isinstance(scaled, MultiLineString):
                        for single_line in scaled.geoms:
                            dx, dy = single_line.xy
                            ax.plot(dx, dy, linewidth=2, color='white', zorder=zorder + 2)

                if address:
                    door_coord = (int(b['door_cell_x']), int(b['door_cell_y']))
                    bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
                    axes_width_in_inches = bbox.width
                    axes_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                    fontsize = (axes_width_in_inches / axes_data_range) * 13
                    ax.text(door_coord[0] + 0.15, door_coord[1] + 0.15,
                            f"{door_coord[0]}, {door_coord[1]}",
                            ha='left', va='bottom', fontsize=fontsize, color='black')

        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def to_geodataframes(self):
        """
        Return buildings and streets as GeoDataFrames without altering existing structures.

        Returns
        -------
        (gpd.GeoDataFrame, gpd.GeoDataFrame)
            buildings_gdf with columns: id, type, door_point, door_cell_x, door_cell_y, size, geometry
            streets_gdf with columns: coord_x, coord_y, id, geometry
        """
        # Return current primary stores
        return self.buildings_gdf.copy(), self.streets_gdf.copy()

    # Removed id_to_door_cell: unnecessary fallback logic; use buildings_gdf['door_cell_x','door_cell_y'] directly

    def to_file(self, buildings_path=None, streets_path=None, driver='GeoJSON'):
        """
        Persist city layers to disk. Writes only layers whose path is provided.
        """
        b_gdf, s_gdf = self.to_geodataframes()
        if buildings_path:
            b_gdf.to_file(buildings_path, driver=driver)
        if streets_path:
            s_gdf.to_file(streets_path, driver=driver)

    def save_geopackage(self, gpkg_path, persist_blocks: bool = False, 
                        persist_city_properties: bool = True, persist_gravity_data: bool = True,
                        edges_path: str = None, street_graphml_path: str = None):
        """Save buildings/streets/properties to a GeoPackage; optionally persist blocks, edges,
        and write the internal street graph to GraphML.

        Parameters
        ----------
        gpkg_path : str
            Path to GeoPackage (.gpkg) to write `buildings` and `streets` layers.
        persist_blocks : bool, default False
            If True and `blocks_gdf` is available, persist as `blocks` layer.
        persist_city_properties : bool, default True
            If True, persist city properties (name, block_side_length, web_mercator_origin,
            city_boundary, buildings_outline) as `city_properties` layer.
        persist_gravity_data : bool, default True
            If True and gravity infrastructure exists, persist hubs, hub distances, and nearby doors.
        edges_path : str, optional
            If provided and gravity is available, writes an edges parquet with columns
            [ox, oy, dx, dy, distance, gravity].
        street_graphml_path : str, optional
            If provided, writes the city's internal street graph (grid adjacency) to GraphML.
            Nodes are labeled as "x_y" with integer attributes x and y; edges retain 'weight'.
        """
        b_gdf, s_gdf = self.to_geodataframes()
        b_gdf.to_file(gpkg_path, layer='buildings', driver='GPKG')
        s_gdf.to_file(gpkg_path, layer='streets', driver='GPKG')
        if persist_blocks and hasattr(self, 'blocks_gdf') and not self.blocks_gdf.empty:
            self.blocks_gdf.to_file(gpkg_path, layer='blocks', driver='GPKG')
        
        # Persist city properties
        if persist_city_properties:
            city_props_gdf = gpd.GeoDataFrame({
                'name': [self.name],
                'block_side_length': [self.block_side_length],
                'web_mercator_origin_x': [self.web_mercator_origin_x],
                'web_mercator_origin_y': [self.web_mercator_origin_y],
                'city_boundary': [self.city_boundary],
                'buildings_outline': [self.buildings_outline],
                'geometry': [self.city_boundary]  # Primary geometry column
            }, crs=self.buildings_gdf.crs)
            city_props_gdf.to_file(gpkg_path, layer='city_properties', driver='GPKG')
        
        # Persist gravity infrastructure
        if persist_gravity_data and hasattr(self, 'hubs') and self.hubs is not None:
            # Save hubs as DataFrame
            hubs_df = pd.DataFrame(self.hubs, columns=['hub_x', 'hub_y'])
            hubs_gdf = gpd.GeoDataFrame(
                hubs_df,
                geometry=gpd.points_from_xy(hubs_df['hub_x'], hubs_df['hub_y']),
                crs=self.buildings_gdf.crs
            )
            hubs_gdf.to_file(gpkg_path, layer='hubs', driver='GPKG')
            
            # Save hub_df (hub-to-hub distances) as flattened array
            if hasattr(self, 'hub_df') and self.hub_df is not None:
                n_hubs = len(self.hubs)
                hub_dist_flat = self.hub_df.values.flatten()
                hub_dist_df = pd.DataFrame({
                    'hub_dist_flat': hub_dist_flat,
                    'n_hubs': [n_hubs] * len(hub_dist_flat)
                })
                hub_dist_gdf = gpd.GeoDataFrame(hub_dist_df, geometry=[Point(0, 0)] * len(hub_dist_df), crs=self.buildings_gdf.crs)
                hub_dist_gdf.to_file(gpkg_path, layer='hub_distances', driver='GPKG')
            
            # Save grav_hub_info (building -> closest hub info)
            if hasattr(self, 'grav_hub_info') and self.grav_hub_info is not None:
                grav_hub_gdf = gpd.GeoDataFrame(
                    self.grav_hub_info.reset_index(),
                    geometry=[Point(0, 0)] * len(self.grav_hub_info),
                    crs=self.buildings_gdf.crs
                )
                grav_hub_gdf.to_file(gpkg_path, layer='grav_hub_info', driver='GPKG')
            
            # Save mh_dist_nearby_doors (nearby door pairs)
            if hasattr(self, 'mh_dist_nearby_doors') and self.mh_dist_nearby_doors is not None and len(self.mh_dist_nearby_doors) > 0:
                nearby_df = self.mh_dist_nearby_doors.reset_index()
                nearby_df.columns = ['bid_i', 'bid_j', 'dist']
                nearby_gdf = gpd.GeoDataFrame(
                    nearby_df,
                    geometry=[Point(0, 0)] * len(nearby_df),
                    crs=self.buildings_gdf.crs
                )
                nearby_gdf.to_file(gpkg_path, layer='nearby_doors', driver='GPKG')
        
        # Optional edges persistence (parquet)
        if edges_path and hasattr(self, 'gravity') and self.gravity is not None and len(self.gravity) > 0:
            edges_df = self.gravity.reset_index()
            # Split tuple origins/dests into integer columns
            edges_df[['ox', 'oy']] = pd.DataFrame(edges_df['origin'].tolist(), index=edges_df.index)
            edges_df[['dx', 'dy']] = pd.DataFrame(edges_df['dest'].tolist(), index=edges_df.index)
            edges_df = edges_df.drop(columns=['origin', 'dest'])
            edges_df.to_parquet(edges_path, index=False)

        # Optional GraphML persistence of the internal grid street graph
        if street_graphml_path:
            try:
                G = self.get_street_graph()
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

    @classmethod
    def from_geodataframes(cls, buildings_gdf, streets_gdf, blocks_gdf=None, edges_df: pd.DataFrame = None):
        """Construct a City from buildings and streets GeoDataFrames."""
        if buildings_gdf.empty:
            width, height = (0, 0) if (streets_gdf is None or streets_gdf.empty) else (
                int(streets_gdf['coord_x'].max() + 1), int(streets_gdf['coord_y'].max() + 1)
            )
        else:
            bounds = buildings_gdf.geometry.total_bounds
            width, height = int(np.ceil(bounds[2])), int(np.ceil(bounds[3]))
        if width <= 0 or height <= 0:
            width, height = (0,0)
        city = cls(dimensions=(width, height), manual_streets=True)

        # Adopt input GeoDataFrames with required columns
        city.buildings_gdf = gpd.GeoDataFrame(buildings_gdf, geometry='geometry', crs=buildings_gdf.crs)
        if 'building_type' not in city.buildings_gdf.columns:
            raise KeyError("buildings_gdf must contain 'building_type'.")
        missing_cols = set(['id','door_cell_x','door_cell_y','door_point','size']) - set(city.buildings_gdf.columns)
        for col in missing_cols:
            if col == 'size':
                city.buildings_gdf[col] = 0
            elif col == 'door_point':
                city.buildings_gdf[col] = city.buildings_gdf.geometry.centroid
            elif col in ['door_cell_x', 'door_cell_y']:
                pts = city.buildings_gdf.geometry.centroid
                if col == 'door_cell_x':
                    city.buildings_gdf['door_cell_x'] = np.floor(pts.x)
                else:
                    city.buildings_gdf['door_cell_y'] = np.floor(pts.y)
            else:
                city.buildings_gdf[col] = None
        # Ensure consistent index on id
        city.buildings_gdf.set_index('id', inplace=True, drop=False)
        city.buildings_gdf.index.name = None
        city.buildings_outline = unary_union(city.buildings_gdf.geometry.values) if not city.buildings_gdf.empty else Polygon()
        
        # Blocks/street layers
        if blocks_gdf is not None and not blocks_gdf.empty:
            city.blocks_gdf = gpd.GeoDataFrame(blocks_gdf, geometry='geometry', crs=blocks_gdf.crs)
            city.blocks_gdf.set_index(['coord_x', 'coord_y'], inplace=True, drop=False)
            city.blocks_gdf.index.names = [None, None]
            # Derive streets from blocks
            city.streets_gdf = city._derive_streets_from_blocks()
        else:
            # streets_gdf may be empty if manual_streets=True
            city.streets_gdf = gpd.GeoDataFrame(streets_gdf, geometry='geometry', crs=streets_gdf.crs) if (streets_gdf is not None and not streets_gdf.empty) else gpd.GeoDataFrame(columns=['coord_x','coord_y','id','geometry'], geometry='geometry', crs=None)
            # Initialize blocks grid if missing
            city.blocks_gdf = city._init_blocks_gdf()
        missing_cols = set(['coord_x','coord_y','id']) - set(city.streets_gdf.columns)
        for col in missing_cols:
            if col == 'id':
                city.streets_gdf[col] = city.streets_gdf.apply(lambda r: f"s-x{int(r['coord_x'])}-y{int(r['coord_y'])}", axis=1)
            else:
                city.streets_gdf[col] = 0
        
        city.blocks_gdf = city._init_blocks_gdf()
        # Vectorized update of blocks_gdf for building blocks
        if not city.buildings_gdf.empty:
            building_bounds = city.buildings_gdf.geometry.bounds
            building_ids = city.buildings_gdf['id'].values
            building_types = city.buildings_gdf['building_type'].values
            for i, (bid, btype) in enumerate(zip(building_ids, building_types)):
                minx, miny, maxx, maxy = building_bounds.iloc[i]
                mask = (city.blocks_gdf['coord_x'] >= int(np.floor(minx))) & (city.blocks_gdf['coord_x'] < int(np.ceil(maxx))) & \
                       (city.blocks_gdf['coord_y'] >= int(np.floor(miny))) & (city.blocks_gdf['coord_y'] < int(np.ceil(maxy)))
                city.blocks_gdf.loc[mask, ['kind', 'building_id', 'building_type']] = ['building', bid, btype]

        # Recompute derived attributes, or preload from edges if provided
        city.buildings_outline = unary_union(city.buildings_gdf.geometry.values) if not city.buildings_gdf.empty else Polygon()
        if edges_df is not None and not edges_df.empty:
            # Expect columns: ox, oy, dx, dy, distance, gravity
            if all(c in edges_df.columns for c in ['ox','oy','dx','dy']):
                # Set gravity from edges
                tmp = edges_df.copy()
                tmp['origin'] = list(zip(tmp['ox'].astype(int), tmp['oy'].astype(int)))
                tmp['dest'] = list(zip(tmp['dx'].astype(int), tmp['dy'].astype(int)))
                cols = ['origin','dest'] + [c for c in ['distance','gravity'] if c in tmp.columns]
                city.gravity = tmp[cols].set_index(['origin','dest'])
                # Build graph from edges
                G = nx.Graph()
                for r in tmp[['ox','oy','dx','dy']].itertuples(index=False):
                    G.add_edge((int(r.ox), int(r.oy)), (int(r.dx), int(r.dy)), weight=1)
                city.street_graph = G
                city.shortest_paths = dict(nx.all_pairs_shortest_path(G))
            else:
                city.get_street_graph()
        else:
            city.get_street_graph()
        return city

    @classmethod
    def from_geopackage(cls, gpkg_path, edges_path: str = None, poi_cols: dict | None = None, load_gravity: bool = True):
        b_gdf = gpd.read_file(gpkg_path, layer='buildings')
        # Optional explicit renaming via poi_cols (e.g., {'building_type':'type'})
        if poi_cols:
            rename_map = {}
            for target, src in poi_cols.items():
                if target not in b_gdf.columns:
                    if src in b_gdf.columns:
                        rename_map[src] = target
                    else:
                        raise KeyError(f"poi_cols refers to missing source column '{src}' for target '{target}'.")
            if rename_map:
                b_gdf = b_gdf.rename(columns=rename_map)
        # Enforce required columns strictly
        required_building_cols = ['id', 'building_type', 'door_cell_x', 'door_cell_y', 'geometry']
        missing = [c for c in required_building_cols if c not in b_gdf.columns]
        if missing:
            raise KeyError(f"'buildings' layer missing required columns {missing}. Provide explicit mappings via column_map kwargs, e.g., building_type='type'.")
        s_gdf = gpd.read_file(gpkg_path, layer='streets')
        
        bl_gdf = None
        props_gdf = None
        try:
            bl_gdf = gpd.read_file(gpkg_path, layer='blocks')
        except Exception:
            pass
        
        try:
            props_gdf = gpd.read_file(gpkg_path, layer='city_properties')
        except Exception:
            pass
        
        city_props = {}
        if props_gdf is not None and not props_gdf.empty:
            props = props_gdf.iloc[0]
            city_props['name'] = props.get('name', 'Garden City')
            city_props['block_side_length'] = props.get('block_side_length', 15.0)
            city_props['web_mercator_origin_x'] = props.get('web_mercator_origin_x', -4265699.0)
            city_props['web_mercator_origin_y'] = props.get('web_mercator_origin_y', 4392976.0)
            
            boundary = props.get('city_boundary')
            if boundary is not None:
                city_props['city_boundary'] = wkt.loads(boundary) if isinstance(boundary, str) else boundary
                
            outline = props.get('buildings_outline')
            if outline is not None:
                city_props['buildings_outline'] = wkt.loads(outline) if isinstance(outline, str) else outline
        
        edges_df = pd.read_parquet(edges_path) if edges_path and os.path.exists(edges_path) else None
        
        city = cls.from_geodataframes(b_gdf, s_gdf, bl_gdf, edges_df)
        
        for key, value in city_props.items():
            setattr(city, key, value)
        
        # Load gravity infrastructure if requested
        if load_gravity:
            try:
                hubs_gdf = gpd.read_file(gpkg_path, layer='hubs')
                city.hubs = list(zip(hubs_gdf['hub_x'].astype(int), hubs_gdf['hub_y'].astype(int)))
                
                hub_dist_gdf = gpd.read_file(gpkg_path, layer='hub_distances')
                hub_dist_gdf = hub_dist_gdf.drop(columns=['geometry'])
                n_hubs = int(hub_dist_gdf['n_hubs'].iloc[0])
                hub_dist_flat = hub_dist_gdf['hub_dist_flat'].values.astype(np.int32)
                hub_dist_matrix = hub_dist_flat.reshape((n_hubs, n_hubs))
                city.hub_df = pd.DataFrame(hub_dist_matrix, index=city.hubs, columns=city.hubs)
                
                grav_hub_gdf = gpd.read_file(gpkg_path, layer='grav_hub_info')
                grav_hub_gdf = grav_hub_gdf.drop(columns=['geometry'])
                city.grav_hub_info = grav_hub_gdf.set_index('index')
                city.grav_hub_info.index.name = None
                city.grav_hub_info['closest_hub_idx'] = city.grav_hub_info['closest_hub_idx'].astype(np.int32)
                city.grav_hub_info['dist_to_hub'] = city.grav_hub_info['dist_to_hub'].astype(np.int32)
                
                try:
                    nearby_gdf = gpd.read_file(gpkg_path, layer='nearby_doors')
                    nearby_gdf = nearby_gdf.drop(columns=['geometry'])
                    city.mh_dist_nearby_doors = pd.Series(
                        nearby_gdf['dist'].values.astype(np.int32),
                        index=pd.MultiIndex.from_arrays([nearby_gdf['bid_i'], nearby_gdf['bid_j']])
                    )
                except Exception:
                    city.mh_dist_nearby_doors = pd.Series([], dtype=np.int32, index=pd.MultiIndex.from_arrays([[], []]))
                
                # Reconstruct callable gravity function
                building_ids = city.buildings_gdf['id'].to_numpy()
                bid_to_idx = {bid: i for i, bid in enumerate(building_ids)}
                hub_to_hub = city.hub_df.values
                closest_hub_idx = city.grav_hub_info['closest_hub_idx'].to_numpy()
                dist_to_closest_hub = city.grav_hub_info['dist_to_hub'].to_numpy()
                
                def compute_gravity_row(building_id, exponent=2.0):
                    idx = bid_to_idx[building_id]
                    hub_i = closest_hub_idx[idx]
                    dist_to_hub_i = dist_to_closest_hub[idx]
                    
                    distances = dist_to_hub_i + hub_to_hub[hub_i, closest_hub_idx] + dist_to_closest_hub
                    
                    for (bid_i, bid_j), d in city.mh_dist_nearby_doors.items():
                        if bid_i == building_id:
                            distances[bid_to_idx[bid_j]] = d
                        elif bid_j == building_id:
                            distances[bid_to_idx[bid_i]] = d
                    
                    distances[idx] = 1  # Temporary non-zero to avoid divide-by-zero warning
                    gravity_row = 1.0 / (distances ** exponent)
                    gravity_row[idx] = 0.0  # Self-gravity is always 0
                    
                    return pd.Series(gravity_row, index=building_ids)
                
                city.grav = compute_gravity_row
                
            except Exception:
                pass
        
        return city

    def get_building(self, identifier=None, door_coords=None, any_coords=None):
        """
        Retrieve a building by string ID, door coordinates, or any coordinates within the city's bounds.
        
        Parameters:
        -----------
        identifier : str, optional
            The string ID of the building (e.g., 'h-x8-y8').
        door_coords : tuple, optional
            The (x, y) coordinates of the building's door.
        any_coords : tuple, optional
            Any (x, y) coordinates within the city's bounds to find a building that contains this point.
        
        Returns:
        --------
        geopandas.GeoDataFrame or None
            A GeoDataFrame with a single row containing building details if found, None otherwise.
        """
        if identifier is not None:
            building = self.buildings_gdf[self.buildings_gdf['id'] == identifier]
            if not building.empty:
                return building
        elif door_coords is not None:
            # Match by door_cell_x/door_cell_y
            df = self.buildings_gdf
            mask = (df['door_cell_x'] == door_coords[0]) & (df['door_cell_y'] == door_coords[1])
            res = df[mask]
            if not res.empty:
                return res.iloc[[0]]
        elif any_coords is not None:
            point = Point(any_coords)
            building = self.buildings_gdf[self.buildings_gdf.geometry.contains(point)]
            if not building.empty:
                return building
        return None

    def get_building_coordinates(self):
        """
        Get building coordinates as a DataFrame with building_id, x, y, building_type, size columns.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: building_id, x, y, building_type, size
        """
        if self.buildings_gdf.empty:
            return pd.DataFrame(columns=['building_id','x','y','building_type','size'])
        centroids = self.buildings_gdf.geometry.centroid
        out = pd.DataFrame({
            'building_id': self.buildings_gdf['id'].values,
            'x': centroids.x.values,
            'y': centroids.y.values,
            'building_type': self.buildings_gdf['building_type'].values,
            'size': self.buildings_gdf['size'].values
        })
        return out
    
    def to_mercator(self, data):
        """
        Convert city block coordinates to Web Mercator coordinates using city's parameters.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'x', 'y' columns in city block coordinates
        
        Returns
        -------
        pd.DataFrame
            DataFrame with 'x', 'y' columns updated to Web Mercator coordinates.
            If 'ha' column exists, it is also scaled.
        """
        return blocks_to_mercator(
            data, 
            block_size=self.block_side_length,
            false_easting=self.web_mercator_origin_x,
            false_northing=self.web_mercator_origin_y
        )
    
    def from_mercator(self, data):
        """
        Convert Web Mercator coordinates back to city block coordinates using city's parameters.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'x', 'y' columns in Web Mercator coordinates
        
        Returns
        -------
        pd.DataFrame
            DataFrame with 'x', 'y' columns updated to city block coordinates.
            If 'ha' column exists, it is also scaled back.
        """
        return mercator_to_blocks(
            data,
            block_size=self.block_side_length,
            false_easting=self.web_mercator_origin_x,
            false_northing=self.web_mercator_origin_y
        )
    
    def to_file(self, buildings_path=None, streets_path=None, 
                street_graphml_path=None, driver='GeoJSON',
                to_crs=None, reverse_affine_transformation=False):
        """
        Export city layers to file.
        
        Coordinate System Flow
        ----------------------
        INTERNAL: Garden city block units (coord_x from 0, geometry in units)
          ↓ Optional reverse_affine_transformation
        OUTPUT: Web Mercator meters (scaled, translated, rotated back)
          ↓ Optional to_crs
        FINAL: Target CRS (e.g., EPSG:4326 for GeoJSON)
        
        Parameters
        ----------
        buildings_path : str, optional
            Path to save buildings layer
        streets_path : str, optional
            Path to save streets layer
        street_graphml_path : str, optional
            Path to save street graph as GraphML
        driver : str, default 'GeoJSON'
            Output format: 'GeoJSON', 'GPKG', 'Parquet', 'ESRI Shapefile'
        to_crs : str or CRS, optional
            Target CRS for reprojection (e.g., 'EPSG:4326')
        reverse_affine_transformation : bool, default False
            If True, converts geometries from garden city block units back to
            original Web Mercator coordinates by scaling, translating, and rotating.
            Garden city columns (coord_x, coord_y, door_cell_x, door_cell_y) are dropped.
        
        Notes
        -----
        Internal geometries are stored in garden city block units (1 block = 1 unit).
        Use reverse_affine_transformation=True to convert back to meters.
        For GeoJSON output, final CRS must be EPSG:4326 (enforced).
        """
        # GraphML: no transformations
        if street_graphml_path:
            G = self.get_street_graph()
            nx.write_graphml(G, street_graphml_path)
        
        def transform_gdf(gdf):
            gdf = gdf.copy()
            
            if reverse_affine_transformation:
                # Drop garden city coordinate columns
                drop_cols = [c for c in ['coord_x', 'coord_y', 'door_cell_x', 'door_cell_y'] 
                            if c in gdf.columns]
                gdf = gdf.drop(columns=drop_cols)
                
                # Scale from garden city units to meters
                gdf['geometry'] = gdf['geometry'].apply(
                    lambda g: scale(g, xfact=self.block_side_length, 
                                  yfact=self.block_side_length, origin=(0, 0))
                )
                
                # Translate back to original position
                gdf['geometry'] = gdf['geometry'].translate(
                    xoff=self.offset_x * self.block_side_length,
                    yoff=self.offset_y * self.block_side_length
                )
                
                # Undo rotation (rotate by negative angle around current centroid)
                all_geoms = gdf.geometry.union_all()
                origin_point = all_geoms.centroid
                gdf['geometry'] = gdf['geometry'].apply(
                    lambda g: shapely_rotate(g, -self.rotation_deg, 
                                            origin=(origin_point.x, origin_point.y))
                )
                
                # Set CRS to Web Mercator after transformation
                gdf = gdf.set_crs('EPSG:3857', allow_override=True)
            
            # Reproject if requested
            if to_crs:
                gdf = gdf.to_crs(to_crs)
            
            # Validate GeoJSON CRS requirement
            if driver == 'GeoJSON':
                result_crs = CRS.from_user_input(gdf.crs) if gdf.crs else None
                if result_crs and not result_crs.equals(CRS.from_epsg(4326)):
                    raise ValueError(
                        f"GeoJSON requires EPSG:4326, got {result_crs}. "
                        f"Use to_crs='EPSG:4326' to reproject."
                    )
            
            return gdf
        
        if buildings_path:
            gdf = transform_gdf(self.buildings_gdf)
            gdf.to_file(buildings_path, driver=driver)
        
        if streets_path:
            gdf = transform_gdf(self.streets_gdf)
            gdf.to_file(streets_path, driver=driver)

    def get_shortest_path(self, start_coord: tuple, end_coord: tuple, plot: bool = False, ax=None):
        """Return a block path between two street cells.

        Parameters
        ----------
        start_coord : tuple[int, int]
            Starting street block (x, y).
        end_coord : tuple[int, int]
            Ending street block (x, y).
        plot : bool, optional
            Plot the resulting path on the city map when True.
        ax : matplotlib.axes.Axes, optional
            Target axis used when `plot=True`.

        Returns
        -------
        list[tuple[int, int]]
            Sequence of street blocks from start to end.

        Raises
        ------
        ValueError
            If either coordinate is not a valid street block.

        Notes
        -----
        - Uses the hub-based shortcut index for speed; hub-to-hub segments may
          be obtained via a local NetworkX shortest path when a direct next-hop
          entry is unavailable.
        """
        if not (isinstance(start_coord, tuple) and isinstance(end_coord, tuple)):
            raise ValueError("Coordinates must be tuples of (x, y).")
        
        # Check if coordinates are street blocks
        if not ((self.streets_gdf['coord_x'] == start_coord[0]) & (self.streets_gdf['coord_y'] == start_coord[1])).any():
            raise ValueError(f"Start coordinate {start_coord} must be a street block.")
        if not ((self.streets_gdf['coord_x'] == end_coord[0]) & (self.streets_gdf['coord_y'] == end_coord[1])).any():
            raise ValueError(f"End coordinate {end_coord} must be a street block.")

        # Ensure graph and shortcuts are built
        if not hasattr(self, 'street_graph') or self.street_graph is None:
            self.get_street_graph(lazy=True)
        if not hasattr(self, '_shortcut_hubs') or self._shortcut_hubs is None:
            try:
                self._build_hub_network()
            except Exception:
                pass

        # Prefer shortcut network if available; fall back to NetworkX if not
        if hasattr(self, '_shortcut_hubs') and self._shortcut_hubs:
            # Map arbitrary node to its hub via precomputed next_to_hub
            def path_to_hub(node):
                route = [node]
                # Prevent infinite loops; cap at city size
                for _ in range(max(1, len(self.streets_gdf))):
                    if node == self._nearest_hub[node]:
                        break
                    node = self._next_to_hub[node]
                    route.append(node)
                    if node == self._nearest_hub[node]:
                        break
                return route

            u = start_coord
            v = end_coord
            hu = self._nearest_hub[u]
            hv = self._nearest_hub[v]

            # u -> hu
            seg_u_hu = path_to_hub(u)
            # hubs path hu -> hv using next_hop table
            seg_hubs = [hu]
            cur = hu
            safe_cap = max(2, len(self._shortcut_hubs) * 4)
            for _ in range(safe_cap):
                if cur == hv:
                    break
                nh = self._hub_next_hop.get(cur, {}).get(hv)
                if nh is None:
                    # fallback to direct nx shortest path if hub-hub mapping missing
                    try:
                        nx_path = nx.shortest_path(self.get_street_graph(), cur, hv)
                    except nx.NetworkXNoPath:
                        return []
                    seg_hubs = nx_path
                    cur = hv
                    break
                seg_hubs.append(nh)
                cur = nh

            # hv -> v: walk from v to hv using next_to_hub, then reverse to get hv->v
            seg_v_to_hv = path_to_hub(v)
            # ensure last element is hv
            if seg_v_to_hv[-1] != hv:
                # fallback to nx shortest if something is off
                try:
                    nx_path = nx.shortest_path(self.get_street_graph(), hv, v)
                except nx.NetworkXNoPath:
                    return []
                seg_hv_to_v = nx_path
            else:
                seg_hv_to_v = list(reversed(seg_v_to_hv))

            # Concatenate segments, avoiding duplicate junctions
            path = seg_u_hu[:-1] + seg_hubs + seg_hv_to_v[1:]
        else:
            # Fallback: compute on-demand with NetworkX
            try:
                path = nx.shortest_path(self.get_street_graph(), start_coord, end_coord)
            except nx.NetworkXNoPath:
                return []

        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 10))
            else:
                fig = ax.figure

            # Plot city streets and buildings
            self.plot_city(ax=ax)

            # Plot the path as a MultiLineString through block centers
            path_centers = [(x + 0.5, y + 0.5) for x, y in path]
            path_lines = MultiLineString([LineString([(path_centers[i][0], path_centers[i][1]), (path_centers[i+1][0], path_centers[i+1][1])]) for i in range(len(path_centers)-1)])
            gpd.GeoSeries(path_lines).plot(ax=ax, color='red', linewidth=2, label='Shortest Path')

            # Highlight start and end points
            start_point = Point(path_centers[0])
            end_point = Point(path_centers[-1])
            gpd.GeoSeries(start_point).plot(ax=ax, color='green', markersize=100, label='Start')
            gpd.GeoSeries(end_point).plot(ax=ax, color='blue', markersize=100, label='End')

            ax.legend()
            ax.set_title(f'Shortest Path from {start_coord} to {end_coord}')
            plt.show(block=False)
            plt.close(fig)

        return path

    # ---------------------------------------------------------------------
    # Fast distance with small LRU-style cache (simple and effective)
    # ---------------------------------------------------------------------
    def get_distance_fast(self, start_coord: tuple, end_coord: tuple) -> float:
        """
        Return an approximate shortest-path distance in number of steps between two
        street blocks using the shortcut network for speed. Results are cached.

        Falls back to on-demand NetworkX shortest_path if shortcuts are unavailable.
        Returns np.inf if no path exists.
        """
        if not (isinstance(start_coord, tuple) and isinstance(end_coord, tuple)):
            raise ValueError("Coordinates must be tuples of (x, y).")

        # Initialize cache
        if not hasattr(self, '_distance_cache'):
            self._distance_cache = {}
        MAX_CACHE = 50000

        key = (start_coord, end_coord)
        if key in self._distance_cache:
            return self._distance_cache[key]

        # Compute via path length
        try:
            path = self.get_shortest_path(start_coord, end_coord)
            if not path:
                d = float('inf')
            else:
                d = float(max(0, len(path) - 1))
        except Exception:
            d = float('inf')

        # Cache with simple capacity control (drop half when full)
        try:
            self._distance_cache[key] = d
            if len(self._distance_cache) > MAX_CACHE:
                # drop roughly half oldest by arbitrary order
                for k in list(self._distance_cache.keys())[:MAX_CACHE // 2]:
                    self._distance_cache.pop(k, None)
        except Exception:
            pass

        return d


# =============================================================================
# RANDOM CITY CLASS
# =============================================================================


class RandomCityGenerator:
    def __init__(self, width, height, street_spacing=5, park_ratio=0.1, home_ratio=0.4, work_ratio=0.3, retail_ratio=0.2, seed=42, verbose=False):
        self.seed = seed
        self.width = width
        self.height = height
        self.street_spacing = street_spacing  # Determines regular intervals for streets
        self.park_ratio = park_ratio
        self.home_ratio = home_ratio
        self.work_ratio = work_ratio
        self.retail_ratio = retail_ratio
        self.verbose = verbose
        self.city = City(dimensions=(width, height))
        self.occupied = np.zeros((self.width, self.height), dtype=bool)  # NumPy array for efficiency
        self.streets = self.generate_streets()
        self.building_sizes = {
            'home': [(2, 2), (1, 2), (2, 1), (1, 1)],  # Mixed sizes
            'workplace': [(1, 3), (3, 1), (4, 4), (3, 3), (4, 2), (2, 4)],
            'retail': [(1, 3), (3, 1), (4, 4), (3, 3), (2, 4), (4, 2)],
            'park': [(6, 6), (5, 5), (4, 4)]
        }

    def generate_streets(self):
        """Predefine streets in a systematic grid pattern using a NumPy mask."""
        streets = np.zeros((self.width, self.height), dtype=bool)
        streets[::self.street_spacing, :] = True
        streets[:, ::self.street_spacing] = True
        return streets
    
    def get_block_type(self, x, y):
        """Dynamically assigns a block type instead of storing all in memory."""
        npr.seed(self.seed + x * self.width + y)  # Ensure consistency
        return npr.choice(['home', 'workplace', 'retail', 'park'], 
                          p=[self.home_ratio, self.work_ratio, self.retail_ratio, self.park_ratio])
    
    def fill_block(self, block_x, block_y, block_type):
        """Fills a block with a building of the specified type."""
        location = (block_x, block_y)
        door = self.get_adjacent_street(location)
        if door is None:
            if self.verbose:
                print(f"No adjacent street found for {location}")
            return
        
        # Define building footprint only on interior cells (preserve street corridors)
        # Streets are predefined along every `street_spacing` row/col; exclude those cells
        bx0 = block_x
        by0 = block_y
        bx1 = min(block_x + self.street_spacing, self.width)
        by1 = min(block_y + self.street_spacing, self.height)

        candidate_blocks = []
        for x in range(bx0, bx1):
            for y in range(by0, by1):
                # keep as building only if not a designated street cell
                if not self.streets[x, y]:
                    candidate_blocks.append((x, y))

        if not candidate_blocks:
            return

        block_polys = [box(x, y, x+1, y+1) for x, y in candidate_blocks]
        geom = unary_union(block_polys)
        blocks = candidate_blocks
        
        try:
            self.city.add_building(building_type=block_type, door=door, geom=geom, blocks=blocks)
        except ValueError as e:
            if self.verbose:
                print(f"Skipping building placement at {location} due to error: {e}")
    
    def get_adjacent_street(self, location):
        """Finds the closest predefined street to assign as the door, ensuring it is within bounds."""
        if not location or not isinstance(location, tuple):
            return None
        x, y = location
        possible_streets = np.array([(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]])
        
        valid_mask = (possible_streets[:, 0] >= 0) & (possible_streets[:, 0] < self.width) & \
                     (possible_streets[:, 1] >= 0) & (possible_streets[:, 1] < self.height)
        
        valid_streets = possible_streets[valid_mask]
        # treat mask as street if present in streets_gdf
        if hasattr(self.city, 'streets_gdf') and not self.city.streets_gdf.empty:
            street_mask = np.array([
                ((self.city.streets_gdf['coord_x'] == sx) & (self.city.streets_gdf['coord_y'] == sy)).any()
                for sx, sy in valid_streets
            ])
            valid_streets = valid_streets[street_mask]
        return tuple(valid_streets[0].tolist()) if valid_streets.size > 0 else None

    def place_buildings_in_blocks(self):
        """Fills each block completely with buildings using proportional distribution."""
        block_list = [(x, y) for x in range(0, self.width, self.street_spacing)
                      for y in range(0, self.height, self.street_spacing)]
        npr.shuffle(block_list)
        for block_x, block_y in block_list:
            block_type = self.get_block_type(block_x, block_y)
            self.fill_block(block_x, block_y, block_type)
    
    def generate_city(self):
        """Generates a systematically structured city where blocks are fully occupied with buildings."""
        self.place_buildings_in_blocks()
        self.city.get_street_graph()
        # Always return the city object, even if no buildings were added
        return self.city

# =============================================================================
# AUXILIARY METHODS
# =============================================================================

def load(filename):
    """Load a city object from a file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def save(city, filename):
    """Save the city object to a file."""
    with open(filename, 'wb') as file:
        pickle.dump(city, file)




# =============================================================================
# RASTERIZATION UTILITIES (integrated from rasterization.py)
# =============================================================================

def generate_canvas_blocks(boundary_polygon, block_size, crs="EPSG:3857", verbose=True, with_block_groups=False):
    _t0 = time.time()
    minx, miny, maxx, maxy = boundary_polygon.bounds
    x_min = int(minx // block_size)
    x_max = int(maxx // block_size) + 1
    y_min = int(miny // block_size)
    y_max = int(maxy // block_size) + 1
    blocks = []
    boundary_prep = prep(boundary_polygon)
    for x in range(x_min, x_max):
        x0 = x * block_size
        x1 = (x + 1) * block_size
        for y in range(y_min, y_max):
            y0 = y * block_size
            y1 = (y + 1) * block_size
            block_geom = box(x0, y0, x1, y1)
            if boundary_prep.intersects(block_geom):
                blocks.append({'coord_x': x, 'coord_y': y, 'geometry': block_geom})
    if not blocks:
        blocks_gdf = gpd.GeoDataFrame(columns=['coord_x','coord_y','geometry'], geometry='geometry', crs=crs)
    else:
        blocks_gdf = gpd.GeoDataFrame(blocks, geometry='geometry', crs=crs)
        blocks_gdf.set_index(['coord_x','coord_y'], inplace=True, drop=False)
        blocks_gdf.index.names = [None, None]
    
    blocks_gdf['building_type'] = None
    
    if with_block_groups:
        # Define block groups as connected components of blocks not intersecting streets, discarding huge ones
        pass
    
    if verbose:
        print(f"Generated {len(blocks_gdf):,} blocks (in {time.time()-_t0:.2f}s)")
    
    return blocks_gdf


def find_intersecting_blocks(geometries_gdf: gpd.GeoDataFrame, blocks_gdf: gpd.GeoDataFrame, predicate: str = 'intersects') -> pd.DataFrame:
    if len(geometries_gdf) == 0 or len(blocks_gdf) == 0:
        return pd.DataFrame(columns=['coord_x','coord_y','geometry_idx'])
    blocks_reset = blocks_gdf.reset_index()
    geometries_reset = geometries_gdf.reset_index()
    intersections = gpd.sjoin(blocks_reset, geometries_reset, how='inner', predicate=predicate)
    if len(intersections) == 0:
        return pd.DataFrame(columns=['coord_x','coord_y','geometry_idx'])
    if 'index_right' in intersections.columns:
        idx_col = 'index_right'
    else:
        idx_col = geometries_reset.index.name if geometries_reset.index.name else 'index'
        if idx_col not in intersections.columns:
            idx_col = intersections.columns[-1]
    result = intersections[['coord_x','coord_y', idx_col]].copy().rename(columns={idx_col:'geometry_idx'})
    return result


def assign_block_types(blocks_gdf, streets_gdf, buildings_gdf_input):
    street_intersections = find_intersecting_blocks(streets_gdf, blocks_gdf)
    street_coords = set(zip(street_intersections['coord_x'], street_intersections['coord_y']))
    blocks_gdf.loc[blocks_gdf.index.isin(street_coords), 'building_type'] = 'street'
    for building_type in ['park','workplace','home','retail','other']:
        subset = buildings_gdf_input[buildings_gdf_input['building_type'] == building_type]
        unassigned = blocks_gdf[blocks_gdf['building_type'].isna()]
        building_intersections = find_intersecting_blocks(subset, unassigned)
        building_coords = set(zip(building_intersections['coord_x'], building_intersections['coord_y']))
        blocks_gdf.loc[blocks_gdf.index.isin(building_coords), 'building_type'] = building_type


def find_connected_components(block_coords: List[Tuple[int,int]], connectivity: str = '4-connected') -> List[Set[Tuple[int,int]]]:
    if not block_coords:
        return []
    block_set = set(block_coords)
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)] if connectivity=='4-connected' else [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    components = []
    visited = set()
    for start in block_coords:
        if start in visited:
            continue
        component = set()
        queue = [start]
        visited.add(start)
        while queue:
            x,y = queue.pop(0)
            component.add((x,y))
            for dx,dy in neighbors:
                nb = (x+dx, y+dy)
                if nb in block_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(component)
    return components


def verify_street_connectivity(streets_gdf):
    if len(streets_gdf) == 0:
        return streets_gdf.copy(), {'total':0,'kept':0,'discarded':0}
    street_coords = list(zip(streets_gdf['coord_x'], streets_gdf['coord_y']))
    components = find_connected_components(street_coords, connectivity='4-connected')
    if not components:
        return streets_gdf.iloc[0:0].copy(), {'total':0,'kept':0,'discarded':0}
    largest = max(components, key=len)
    kept = set(largest)
    result = streets_gdf[streets_gdf.apply(lambda r: (r['coord_x'], r['coord_y']) in kept, axis=1)].copy()
    summary = {'total': len(streets_gdf), 'kept': len(result), 'discarded': len(streets_gdf)-len(result), 'n_components': len(components)}
    return result, summary


def assign_door_to_building(building_blocks, available_blocks, graph=None):
    """Find adjacent block that can serve as door (not occupied by buildings)."""
    if graph is not None:
        for x, y in building_blocks:
            for nb in graph.neighbors((x, y)):
                if nb in available_blocks:
                    return nb
    else:
        neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
        for x, y in building_blocks:
            for dx, dy in neighbors:
                nb = (x+dx, y+dy)
                if nb in available_blocks:
                    return nb
    return None


class RasterCity(City):
    def __init__(self, boundary_polygon, streets_gdf, buildings_gdf, block_side_length=15.0, crs="EPSG:3857", building_type_col='building_type', name="Rasterized City", resolve_overlaps=False, other_building_behavior="keep", rotation_deg=0.0, verbose=True):
        """
        Create a rasterized city from OSM data.
        
        Coordinate System Flow
        ----------------------
        INPUT: Web Mercator meters (EPSG:3857), possibly rotated
          ↓ Rotate by rotation_deg, translate by -offset to align grid at origin
        INTERNAL: Garden city block units (coord_x, coord_y from 0, geometries in units)
          ↓ All rasterization, spatial queries, and simulation logic
        OUTPUT: Transform back via reverse_affine_transformation in to_file()
        
        Parameters
        ----------
        boundary_polygon : shapely.geometry.Polygon
            Boundary polygon for the city
        streets_gdf : gpd.GeoDataFrame
            Streets GeoDataFrame
        buildings_gdf : gpd.GeoDataFrame
            Buildings GeoDataFrame
        block_side_length : float, optional
            Side length of each block in meters (default: 15.0)
        crs : str, optional
            Coordinate reference system (default: "EPSG:3857")
        building_type_col : str, optional
            Column name for building type (default: 'building_type')
        name : str, optional
            Name of the city (default: "Rasterized City")
        resolve_overlaps : bool, optional
            If True, resolve overlapping buildings by removing overlapping blocks (default: False)
        other_building_behavior : str, optional
            How to handle buildings with type 'other': 'keep', 'filter', or 'randomize' (default: 'keep')
        rotation_deg : float, optional
            Rotation applied to input geometries in degrees counterclockwise (default: 0.0)
        verbose : bool, optional
            Print progress messages (default: True)
        
        Note
        ----
        Input geometries are transformed to garden city block units. A canvas is
        generated to match rotated/shifted input data. Future: transform inputs
        to match a garden city canvas for clearer flow.
        """
        if other_building_behavior == "randomize":
            raise NotImplementedError("randomize option not yet implemented")
        
        minx, miny, maxx, maxy = boundary_polygon.bounds
        grid_min_x = int(minx // block_side_length)
        grid_min_y = int(miny // block_side_length)
        grid_max_x = int(maxx // block_side_length) + 1
        grid_max_y = int(maxy // block_side_length) + 1
        grid_width = grid_max_x - grid_min_x
        grid_height = grid_max_y - grid_min_y
        
        super().__init__(
            dimensions=(grid_width, grid_height),
            manual_streets=True,
            name=name,
            block_side_length=block_side_length,
            web_mercator_origin_x=minx,
            web_mercator_origin_y=miny,
            rotation_deg=rotation_deg,
            offset_x=0,
            offset_y=0
        )
        
        self.boundary_polygon = boundary_polygon
        self.crs = crs
        self.streets_gdf_input = streets_gdf.to_crs(crs)
        
        buildings_gdf = buildings_gdf.to_crs(crs)
        if building_type_col != 'building_type':
            buildings_gdf = buildings_gdf.rename(columns={building_type_col: 'building_type'})
        if 'building_type' not in buildings_gdf.columns:
            raise ValueError(f"buildings_gdf must have a 'building_type' column (or specify building_type_col parameter)")
        
        if other_building_behavior == "filter":
            buildings_gdf = buildings_gdf[buildings_gdf['building_type'] != 'other']
        
        self.buildings_gdf_input = buildings_gdf
        
        self.blocks_gdf = generate_canvas_blocks(self.boundary_polygon, self.block_side_length, self.crs, verbose=verbose)
        
        # Calculate offset and apply to all gdfs
        minx, miny, maxx, maxy = self.boundary_polygon.bounds
        self.offset_x = int(minx // self.block_side_length)
        self.offset_y = int(miny // self.block_side_length)
        offset_meters = (self.offset_x * self.block_side_length, self.offset_y * self.block_side_length)
        
        # Update Web Mercator origins to match actual offset
        self.web_mercator_origin_x = self.offset_x * self.block_side_length
        self.web_mercator_origin_y = self.offset_y * self.block_side_length
        
        self.blocks_gdf['coord_x'] = self.blocks_gdf['coord_x'] - self.offset_x
        self.blocks_gdf['coord_y'] = self.blocks_gdf['coord_y'] - self.offset_y
        self.blocks_gdf['building_id'] = None
        self.blocks_gdf['geometry'] = self.blocks_gdf['geometry'].translate(xoff=-offset_meters[0], yoff=-offset_meters[1])
        self.blocks_gdf.set_index(['coord_x','coord_y'], inplace=True, drop=False)
        self.blocks_gdf.index.names = [None, None]
        
        self.streets_gdf_input['geometry'] = self.streets_gdf_input['geometry'].translate(xoff=-offset_meters[0], yoff=-offset_meters[1])
        self.buildings_gdf_input['geometry'] = self.buildings_gdf_input['geometry'].translate(xoff=-offset_meters[0], yoff=-offset_meters[1])
        
        self.resolve_overlaps = resolve_overlaps
        self._rasterize(verbose=verbose)

    def _rasterize(self, verbose=True):
        # Assigning block types could arguably be its own method, but keeping it here for now
        # since it's a core part of the rasterization process that needs the input gdfs
        _t1 = time.time()
        if verbose:
            print("Assigning block types...")
        assign_block_types(self.blocks_gdf, self.streets_gdf_input, self.buildings_gdf_input)
        if verbose:
            print(f"Block types assigned (in {time.time()-_t1:.2f}s)")
        _t2 = time.time()
        if verbose:
            print("Assigning streets...")
        street_blocks = self.blocks_gdf[self.blocks_gdf['building_type'] == 'street'][['coord_x','coord_y','geometry']].copy()
        if verbose:
            print("Verifying street connectivity...")
        connected_streets, summary = verify_street_connectivity(street_blocks)
        if verbose:
            print(f"  Streets: {summary['kept']:,} kept, {summary['discarded']:,} discarded (in {time.time()-_t2:.2f}s)")
        
        # Streets: create geometry from block coordinates (garden city units)
        self.streets_gdf = gpd.GeoDataFrame(connected_streets[['coord_x','coord_y']].copy(), crs=None)
        self.streets_gdf['geometry'] = gpd.GeoSeries(
            [box(x, y, x+1, y+1) for x, y in zip(self.streets_gdf['coord_x'], self.streets_gdf['coord_y'])],
            crs=None
        )
        self.streets_gdf = self.streets_gdf.set_geometry('geometry')
        self.streets_gdf['id'] = ('s-x' + self.streets_gdf['coord_x'].astype(int).astype(str) + '-y' + self.streets_gdf['coord_y'].astype(int).astype(str))
        self.streets_gdf.set_index(['coord_x','coord_y'], inplace=True, drop=False)
        self.streets_gdf.index.names = [None, None]
        
        available_for_doors = self.blocks_gdf[self.blocks_gdf['building_type'] == 'street'].index
        
        _t3 = time.time()
        if verbose:
            print("Adding buildings to city...")
        added_building_blocks = set()
        skipped_overlap = 0
        resolved_overlap = 0
        new_building_rows = []
        block_updates = []
        
        building_types = [t for t in TYPE_PRIORITY if t in self.buildings_gdf_input['building_type'].values]
        
        for building_type in building_types:
            subset = self.buildings_gdf_input[self.buildings_gdf_input['building_type'] == building_type]
            tb_type = self.blocks_gdf[self.blocks_gdf['building_type'] == building_type]
            joins = find_intersecting_blocks(subset, tb_type)
            grouped = joins.groupby('geometry_idx')
            for bidx, grp in grouped:
                block_coords = list(zip(grp['coord_x'], grp['coord_y']))
                
                if self.resolve_overlaps:
                    block_coords_set = set(block_coords) - added_building_blocks
                    if not block_coords_set:
                        resolved_overlap += 1
                        continue
                    block_coords = list(block_coords_set)
                
                components = find_connected_components(block_coords, connectivity='4-connected')
                for component in components:
                    building_blocks = list(component)
                    door = assign_door_to_building(building_blocks, available_for_doors)
                    if door is None:
                        continue
                    building_blocks_set = set(building_blocks)
                    if building_blocks_set & added_building_blocks:
                        skipped_overlap += 1
                        continue
                    block_polys = [box(x, y, x+1, y+1) for x,y in building_blocks]
                    building_geom = block_polys[0] if len(block_polys) == 1 else unary_union(block_polys)
                    
                    # Pick building block adjacent to door for unique ID
                    mask = self.check_adjacent(building_blocks, door)
                    idx = next((i for i, m in enumerate(mask) if m), None)
                    candidate = building_blocks[idx] if idx is not None else building_blocks[0]
                    building_id = f"{building_type[0]}-x{int(candidate[0])}-y{int(candidate[1])}"
                    
                    dpt = Point(door[0] + 0.5, door[1] + 0.5)
                    new_building_rows.append({
                        'id': building_id,
                        'building_type': building_type,
                        'door_cell_x': door[0],
                        'door_cell_y': door[1],
                        'door_point': dpt,
                        'size': len(building_blocks),
                        'geometry': building_geom,
                    })
                    for (cx, cy) in building_blocks:
                        block_updates.append({'coord_x': cx, 'coord_y': cy, 'building_id': building_id, 'building_type': building_type})
                    added_building_blocks.update(building_blocks_set)

        if new_building_rows:
            nb_gdf = gpd.GeoDataFrame(new_building_rows, geometry='geometry', crs=None)
            nb_gdf.set_index('id', inplace=True, drop=False)
            nb_gdf.index.name = None
            if self.buildings_gdf.empty:
                self.buildings_gdf = nb_gdf
            else:
                self.buildings_gdf = pd.concat([self.buildings_gdf, nb_gdf], axis=0, ignore_index=False)

        if block_updates:
            upd_df = pd.DataFrame(block_updates)
            upd_df.set_index(['coord_x','coord_y'], inplace=True)
            common_idx = self.blocks_gdf.index.intersection(upd_df.index)
            if len(common_idx) > 0:
                cols = ['building_id','building_type']
                self.blocks_gdf.loc[common_idx, cols] = upd_df.loc[common_idx, cols].values
            if not self.streets_gdf.empty:
                drop_idx = self.streets_gdf.index.intersection(upd_df.index)
                if len(drop_idx) > 0:
                    self.streets_gdf = self.streets_gdf.drop(index=drop_idx)

        if verbose:
            overlap_count = resolved_overlap if self.resolve_overlaps else skipped_overlap
            print(f"  Added {len(new_building_rows)} buildings, skipped {overlap_count} due to overlap (adding took {time.time()-_t3:.2f}s)")
        
        if len(new_building_rows) == 0:
            raise ValueError(f"No buildings were added to city. Input had {len(self.buildings_gdf_input)} buildings.")
