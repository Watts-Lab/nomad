from shapely.geometry import box, Polygon, LineString, MultiLineString, Point, MultiPoint, MultiPolygon, GeometryCollection
from shapely import wkt
from math import floor, ceil
from shapely.affinity import scale
from shapely.ops import unary_union
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
import warnings
import geopandas as gpd
import os

from nomad.map_utils import blocks_to_mercator, mercator_to_blocks

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
                 web_mercator_origin_y: float = 4392976.0):
        
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
        """

        self.name = name
        self.block_side_length = block_side_length
        self.web_mercator_origin_x = web_mercator_origin_x
        self.web_mercator_origin_y = web_mercator_origin_y

        self.buildings_outline = Polygon()

        self.manual_streets = manual_streets

        if not (isinstance(dimensions, tuple) and len(dimensions) == 2
                and all(isinstance(d, int) for d in dimensions)):
            raise ValueError("Dimensions must be a tuple of two integers.")
        self.city_boundary = box(0, 0, dimensions[0], dimensions[1])

        self.dimensions = dimensions

        self.buildings_gdf = gpd.GeoDataFrame(
            columns=['id','type','door_cell_x','door_cell_y','door_point','size','geometry'],
            geometry='geometry', crs=None
        )
        self.buildings_gdf.set_index('id', inplace=True, drop=False)
        self.buildings_gdf.index.name = None
        self.blocks_gdf = self._init_blocks_gdf()
        self.streets_gdf = self._derive_streets_from_blocks()
        # Convenience properties are defined below for GDF-first access

    def _derive_streets_from_blocks(self):
        """
        Derives streets GeoDataFrame from blocks_gdf where kind is 'street'.
        """
        if not hasattr(self, 'blocks_gdf') or self.blocks_gdf.empty:
            self.blocks_gdf = self._init_blocks_gdf()
        streets = self.blocks_gdf[self.blocks_gdf['kind'] == 'street'].copy()
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

        x_coords = np.arange(width)
        y_coords = np.arange(height)
        
        # Create a MultiIndex from the product of x and y coordinates
        multi_index = pd.MultiIndex.from_product([x_coords, y_coords], names=['coord_x', 'coord_y'])
        
        # Create a DataFrame from the MultiIndex
        df = pd.DataFrame(index=multi_index).reset_index()
        
        # Add other columns
        df['kind'] = 'street'
        df['building_id'] = None
        df['building_type'] = None
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

    def add_building(self, building_type: str, door: tuple, geom: Polygon = None, blocks=None, gdf_row=None):
        """
        Adds a building to the city with the specified type, door location, and geometry.

        Parameters
        ----------
        building_type : str
            The type of the building ('home', 'work', 'retail', 'park').
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
        def check_adjacent(geom1, geom2):
            return geom1.touches(geom2) or geom1.intersects(geom2)

        if gdf_row is not None:
            if isinstance(gdf_row, pd.Series):
                gdf_row = gpd.GeoDataFrame([gdf_row], geometry='geometry')
            elif not isinstance(gdf_row, gpd.GeoDataFrame) or len(gdf_row) != 1:
                raise ValueError("gdf_row must be a GeoDataFrame with exactly one row or a pandas Series.")
            building_type = gdf_row.iloc[0]['type'] if 'type' in gdf_row.columns else building_type
            door = (gdf_row.iloc[0]['door_cell_x'], gdf_row.iloc[0]['door_cell_y']) if 'door_cell_x' in gdf_row.columns else door
            geom = gdf_row.iloc[0]['geometry'] if 'geometry' in gdf_row.columns else geom

        # If geom is actually a list of blocks, reassign it to blocks
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
        if door_line.is_empty:
            raise ValueError(f"Door {door} must be adjacent to new building.")
        door_centroid = (door_line.centroid.x, door_line.centroid.y)

        # Validate index presence without noisy prints
        in_index = (door[0], door[1]) in self.streets_gdf.index if isinstance(door, tuple) else door in self.streets_gdf.index

        # Validate adjacency using streets_gdf geometry at door
        srow = self.streets_gdf.loc[(door[0], door[1])] if (door[0], door[1]) in self.streets_gdf.index else None
        if srow is None or isinstance(srow, pd.DataFrame) and srow.empty:
            raise ValueError(f"Door {door} must be on an existing street cell.")
        street_geom = srow.geometry if isinstance(srow, pd.Series) else srow.iloc[0].geometry
        combined_plot = unary_union([geom, street_geom])
        if self.buildings_outline.contains(combined_plot) or self.buildings_outline.overlaps(combined_plot):
            raise ValueError(
                "New building or its door overlap with existing buildings."
            )

        if not check_adjacent(geom, street_geom):
            raise ValueError(f"Door {door} must be adjacent to new building.")

        # Derive blocks from geom if not provided
        if blocks is None:
            minx, miny, maxx, maxy = geom.bounds
            blocks = [(int(x), int(y)) for x in range(int(minx), int(maxx) + 1)
                      for y in range(int(miny), int(maxy) + 1)
                      if minx <= x < maxx and miny <= y < maxy]

        # add building
        # Prefer explicit id from gdf_row when provided
        if gdf_row is not None and isinstance(gdf_row, gpd.GeoDataFrame) and 'id' in gdf_row.columns:
            building_id = str(gdf_row.iloc[0]['id'])
        else:
            building_id = f"{building_type[0]}-x{door[0]}-y{door[1]}"
        self.buildings_outline = unary_union([self.buildings_outline, geom])
        # Append to buildings_gdf
        dpt = Point(door_centroid[0], door_centroid[1])
        size_blocks = len(blocks) if blocks is not None else 0
        new_row = gpd.GeoDataFrame([
            {
                'id': building_id,
                'type': building_type,
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
        required_columns = ['type', 'door_cell_x', 'door_cell_y', 'geometry']
        if not all(col in buildings_gdf.columns for col in required_columns):
            raise ValueError(f"GeoDataFrame must contain columns: {required_columns}")

        for _, row in buildings_gdf.iterrows():
            self.add_building(
                building_type=row['type'],
                door=(row['door_cell_x'], row['door_cell_y']),
                geom=row['geometry'],
                gdf_row=buildings_gdf.loc[[row.name]]
            )

    def street_adjacency_edges(self):
        """
        Returns a DataFrame of edges representing adjacency between street blocks.
        Each row is an edge (u,v) where u and v are street block IDs (or coordinates).
        """
        if not hasattr(self, 'gravity') or self.gravity is None or len(self.gravity) == 0:
            self.get_street_graph()
        # If gravity still empty (e.g., too few streets), derive edges from grid adjacency directly
        if self.gravity is None or len(self.gravity) == 0:
            if self.streets_gdf.empty:
                return pd.DataFrame(columns=['u', 'v'])
            nodes = set()
            edges_list = []
            for idx, row in self.streets_gdf.iterrows():
                x = int(row['coord_x']) if 'coord_x' in row else int(idx[0])
                y = int(row['coord_y']) if 'coord_y' in row else int(idx[1])
                nodes.add((x, y))
            for (x, y) in nodes:
                for dx, dy in [(1,0), (0,1)]:  # undirected, add each edge once
                    nb = (x+dx, y+dy)
                    if nb in nodes:
                        edges_list.append({'u': (x,y), 'v': nb})
            return pd.DataFrame(edges_list, columns=['u','v'])
        edges = self.gravity.reset_index()[['origin', 'dest']].rename(columns={'origin': 'u', 'dest': 'v'})
        return edges

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
            A dictionary containing the kind of block ('street' or 'building'), the ID of the building if applicable, the type of building if applicable, and the geometry of the block.
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
                'kind': row['kind'],
                'building_id': row['building_id'],
                'building_type': row['building_type'],
                'geometry': row['geometry']
            }
        return {'kind': 'unknown', 'building_id': None, 'building_type': None, 'geometry': None}

    def get_street_graph(self):
        """
        Constructs a graph of street blocks for shortest path calculations.
        Each street block is a node, and edges connect adjacent street blocks.
        Uses vectorized operations on `streets_gdf` to determine adjacency.
        """
        if hasattr(self, 'street_graph') and self.street_graph is not None:
            return self.street_graph

        if self.streets_gdf.empty:
            self.street_graph = nx.Graph()
            # Empty gravity as well
            self.gravity = pd.DataFrame(columns=['distance', 'gravity'])
            return self.street_graph

        # Nodes
        nodes_df = self.streets_gdf[['coord_x', 'coord_y']].copy()

        # Build edges via vectorized neighbor matches
        edges_frames = []
        for dx, dy in [(1, 0), (0, 1)]:  # undirected; only positive directions to avoid duplicates
            shifted = nodes_df.copy()
            shifted['nx'] = nodes_df['coord_x'] + dx
            shifted['ny'] = nodes_df['coord_y'] + dy
            # Merge to find neighbor pairs
            merged = shifted.merge(
                nodes_df.rename(columns={'coord_x': 'nx', 'coord_y': 'ny'}),
                on=['nx', 'ny'], how='inner'
            )
            if not merged.empty:
                edges_frames.append(
                    merged[['coord_x', 'coord_y', 'nx', 'ny']].rename(columns={'coord_x': 'x', 'coord_y': 'y'})
                )

        if edges_frames:
            edges_df = pd.concat(edges_frames, ignore_index=True)
        else:
            edges_df = pd.DataFrame(columns=['x', 'y', 'nx', 'ny'])

        # Create graph
        G = nx.Graph()
        G.add_nodes_from([(int(x), int(y)) for x, y in nodes_df[['coord_x', 'coord_y']].itertuples(index=False)])
        if not edges_df.empty:
            edge_list = [((int(r.x), int(r.y)), (int(r.nx), int(r.ny))) for r in edges_df.itertuples(index=False)]
            G.add_edges_from(edge_list, weight=1)

        # Compute shortest paths (for compatibility) and distances (for gravity)
        self.shortest_paths = dict(nx.all_pairs_shortest_path(G))
        # Distances
        dist_records = []
        for source, lengths in nx.all_pairs_shortest_path_length(G):
            for target, d in lengths.items():
                if source == target:
                    continue
                dist_records.append({'origin': source, 'dest': target, 'distance': int(d)})
        if dist_records:
            dist_df = pd.DataFrame(dist_records)
            dist_df['gravity'] = 1.0 / (dist_df['distance'] + 1)
            self.gravity = dist_df.set_index(['origin', 'dest'])
        else:
            self.gravity = pd.DataFrame(columns=['distance', 'gravity'])

        self.street_graph = G
        return self.street_graph

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
                'work': '#C9A0DC',
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
                bcolor = colors.get(b['type'], colors['default'])
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

    def id_to_door_cell(self):
        """Return Series mapping building id -> (door_cell_x, door_cell_y) with fallbacks."""
        if self.buildings_gdf.empty:
            return pd.Series(dtype=object)
        df = self.buildings_gdf.copy()
        if 'door_cell_x' in df.columns and 'door_cell_y' in df.columns:
            cx = df['door_cell_x']
            cy = df['door_cell_y']
        elif 'door_point' in df.columns:
            cx = np.floor(df['door_point'].x).astype('Int64')
            cy = np.floor(df['door_point'].y).astype('Int64')
        else:
            pts = df.geometry.centroid
            cx = np.floor(pts.x).astype('Int64')
            cy = np.floor(pts.y).astype('Int64')
        tuples = [ (int(x), int(y)) if (pd.notna(x) and pd.notna(y)) else None for x, y in zip(cx, cy) ]
        return pd.Series(tuples, index=df['id'])

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
                        persist_city_properties: bool = True, edges_path: str = None):
        """Save buildings/streets/properties to a GeoPackage; optionally persist blocks and edges.

        Parameters
        ----------
        gpkg_path : str
            Path to GeoPackage (.gpkg) to write `buildings` and `streets` layers.
        persist_blocks : bool, default False
            If True and `blocks_gdf` is available, persist as `blocks` layer.
        persist_city_properties : bool, default True
            If True, persist city properties (name, block_side_length, web_mercator_origin,
            city_boundary, buildings_outline) as `city_properties` layer.
        edges_path : str, optional
            If provided and gravity is available, writes an edges parquet with columns
            [ox, oy, dx, dy, distance, gravity].
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
        
        # Optional edges persistence (parquet)
        if edges_path and hasattr(self, 'gravity') and self.gravity is not None and len(self.gravity) > 0:
            edges_df = self.gravity.reset_index()
            # Split tuple origins/dests into integer columns
            edges_df[['ox', 'oy']] = pd.DataFrame(edges_df['origin'].tolist(), index=edges_df.index)
            edges_df[['dx', 'dy']] = pd.DataFrame(edges_df['dest'].tolist(), index=edges_df.index)
            edges_df = edges_df.drop(columns=['origin', 'dest'])
            edges_df.to_parquet(edges_path, index=False)

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
        missing_cols = set(['id','type','door_cell_x','door_cell_y','door_point','size']) - set(city.buildings_gdf.columns)
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
            building_types = city.buildings_gdf['type'].values if 'type' in city.buildings_gdf.columns else [None] * len(building_ids)
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
    def from_geopackage(cls, gpkg_path, edges_path: str = None):
        b_gdf = gpd.read_file(gpkg_path, layer='buildings')
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
            if 'door_cell_x' in df.columns and 'door_cell_y' in df.columns:
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
            'building_type': self.buildings_gdf['type'].values,
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

    def get_shortest_path(self, start_coord: tuple, end_coord: tuple, plot: bool = False, ax=None):
        """
        Retrieves the shortest path between two blocks using tuple coordinates and optionally plots it.

        Parameters
        ----------
        start_coord : tuple
            The starting block coordinates (x, y).
        end_coord : tuple
            The ending block coordinates (x, y).
        plot : bool, optional
            If True, plots the path on the provided or a new matplotlib axis.
        ax : matplotlib.axes.Axes, optional
            The axis to plot on. If None and plot is True, a new figure and axis are created.

        Returns
        -------
        list
            List of tuple coordinates representing the shortest path from start to end.
        Raises
        ------
        ValueError
            If the start or end coordinates are not street blocks or not in the city bounds.
        """
        if not (isinstance(start_coord, tuple) and isinstance(end_coord, tuple)):
            raise ValueError("Coordinates must be tuples of (x, y).")
        if start_coord not in self.streets_gdf.index or end_coord not in self.streets_gdf.index:
            raise ValueError("Start or end coordinates must be street blocks.")

        if not hasattr(self, 'shortest_paths'):
            self.get_street_graph()

        path = self.shortest_paths.get(start_coord, {}).get(end_coord, [])
        if not path:
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
            'work': [(1, 3), (3, 1), (4, 4), (3, 3), (4, 2), (2, 4)],
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
        return npr.choice(['home', 'work', 'retail', 'park'], 
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


def check_adjacent(geom1, geom2):
    intersection = geom1.intersection(geom2)
    if intersection.is_empty:
        return False
    return isinstance(intersection, (LineString, MultiLineString))
