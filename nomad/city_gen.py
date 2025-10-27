from shapely.geometry import box, Polygon, LineString, MultiLineString, Point, MultiPoint, MultiPolygon, GeometryCollection
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

    def __init__(self, 
                 coordinates: tuple):

        self.coordinates = coordinates
        self.geometry = box(coordinates[0], coordinates[1],
                            coordinates[0]+1, coordinates[1]+1)

        self.id = f's-x{coordinates[0]}-y{coordinates[1]}'


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
    address_book : dict
        A dictionary mapping coordinates to Building objects.
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
                 manual_streets: bool = False):
        
        """
        Initializes the City object with given dimensions and optionally manual streets.

        Parameters
        ----------
        dimensions : tuple
            A tuple representing the dimensions of the city (width, height).
        manual_streets : bool
            If True, streets must be manually added to the city. 
            If False, all blocks are initialized as streets until it is populated with a building.
        """

        self.buildings_outline = Polygon()

        self.manual_streets = manual_streets

        if not (isinstance(dimensions, tuple) and len(dimensions) == 2
                and all(isinstance(d, int) for d in dimensions)):
            raise ValueError("Dimensions must be a tuple of two integers.")
        self.city_boundary = box(0, 0, dimensions[0], dimensions[1])

        self.dimensions = dimensions

        # Primary GeoDataFrame stores (single source of truth)
        self.buildings_gdf = gpd.GeoDataFrame(
            columns=['id','type','door_x','door_y','door_cell_x','door_cell_y','door_point','size','geometry'],
            geometry='geometry', crs=None
        )
        self.buildings_gdf.set_index(['id', 'door_cell_x', 'door_cell_y'], inplace=True, drop=False)
        # Avoid index name collision on reset_index during GeoPandas to_file
        self.buildings_gdf.index.names = [None, None, None]
        # derived view can be taken directly from buildings_gdf when needed
        self.blocks_gdf = self._init_blocks_gdf()
        self.streets_gdf = self._derive_streets_from_blocks()
        # Convenience properties are defined below for GDF-first access

    def _derive_streets_from_blocks(self):
        """Build streets_gdf from blocks_gdf rows marked as 'street'."""
        if not hasattr(self, 'blocks_gdf') or self.blocks_gdf.empty:
            return gpd.GeoDataFrame(columns=['coord_x','coord_y','id','geometry'], geometry='geometry', crs=None)
        streets = self.blocks_gdf[self.blocks_gdf['kind'] == 'street'][['coord_x','coord_y','geometry']].copy()
        streets['id'] = streets.apply(lambda r: f"s-x{int(r['coord_x'])}-y{int(r['coord_y'])}", axis=1)
        streets = streets[['coord_x','coord_y','id','geometry']]
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

    def add_building(self, building_type: str, door: tuple, geom: Polygon, blocks=None):
        """
        Adds a building to the city with the specified type, door location, and geometry.

        Parameters
        ----------
        building_type : str
            The type of the building (e.g., 'home', 'work').
        door : tuple
            A tuple representing the (x, y) coordinates of the door of the building.
        geom : shapely.geometry.polygon.Polygon
            The geometry of the building.
        blocks : list of tuples, optional
            A list of (x, y) coordinates representing the blocks occupied by the building.

        Raises
        ------
        ValueError
            If the door is not on an existing street or if the building overlaps with existing buildings.
        """
        def check_adjacent(geom1, geom2):
            return geom1.touches(geom2) or geom1.intersects(geom2)

        # Compute door centroid via intersection with target street
        door_poly = box(door[0], door[1], door[0]+1, door[1]+1)
        door_line = geom.intersection(door_poly)
        if door_line.is_empty:
            raise ValueError(f"Door {door} must be adjacent to new building.")
        door_centroid = (door_line.centroid.x, door_line.centroid.y)

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

        # add building
        building_id = f"{building_type[0]}-x{door[0]}-y{door[1]}"
        self.buildings_outline = unary_union([self.buildings_outline, geom])
        # Append to buildings_gdf
        bx, by = door_centroid
        dpt = Point(bx, by)
        size_blocks = len(blocks) if blocks is not None else 0
        new_row = gpd.GeoDataFrame([
            {'id': building_id, 'type': building_type, 'door_x': bx, 'door_y': by, 'door_cell_x': door[0], 'door_cell_y': door[1], 'door_point': dpt, 'size': size_blocks, 'geometry': geom}
        ], geometry='geometry', crs=self.buildings_gdf.crs)
        new_row.set_index(['id', 'door_cell_x', 'door_cell_y'], inplace=True, drop=False)
        new_row.index.names = [None, None, None]
        self.buildings_gdf = pd.concat([self.buildings_gdf, new_row], axis=0)

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
        buffered_building_geom = geom.buffer(1)
        if not self.city_boundary.contains(buffered_building_geom):
            new_boundary = self.city_boundary.union(buffered_building_geom).envelope
            self.city_boundary = new_boundary
            self.dimensions = (int(new_boundary.bounds[2]), int(new_boundary.bounds[3]))
            # Update the streets for any new blocks within the expanded boundary
            minx, miny, maxx, maxy = map(int, new_boundary.bounds)
            for x in range(minx, maxx+1):
                for y in range(miny, maxy+1):
                    sid = f's-x{x}-y{y}'
                    exists = ((self.streets_gdf['coord_x'] == x) & (self.streets_gdf['coord_y'] == y)).any()
                    if not exists:
                        self.streets_gdf = pd.concat([
                            self.streets_gdf,
                            gpd.GeoDataFrame([{'coord_x': x, 'coord_y': y, 'id': sid, 'geometry': box(x, y, x+1, y+1)}], geometry='geometry', crs=self.streets_gdf.crs)
                        ], ignore_index=True)
                        # also add to blocks_gdf as street
                        if hasattr(self, 'blocks_gdf'):
                            new_block = gpd.GeoDataFrame([{
                                'coord_x': x,
                                'coord_y': y,
                                'kind': 'street',
                                'building_id': None,
                                'building_type': None,
                                'geometry': box(x, y, x+1, y+1)
                            }], geometry='geometry', crs=self.blocks_gdf.crs)
                            self.blocks_gdf = pd.concat([self.blocks_gdf, new_block], ignore_index=True)

    def street_adjacency_edges(self):
        """Return DataFrame of 4-neighborhood edges between street blocks (coord tuples)."""
        if not hasattr(self, 'blocks_gdf') or self.blocks_gdf.empty:
            return pd.DataFrame(columns=['u','v'])
        streets = self.blocks_gdf[self.blocks_gdf['kind'] == 'street'][['coord_x','coord_y']].copy()
        streets['u'] = list(zip(streets['coord_x'], streets['coord_y']))
        edges = []
        offsets = [(1,0),(-1,0),(0,1),(0,-1)]
        street_set = set(streets['u'])
        for (x,y) in street_set:
            for dx, dy in offsets:
                v = (x+dx, y+dy)
                if v in street_set and (x,y) < v:
                    edges.append({'u': (x,y), 'v': v})
        return pd.DataFrame(edges)

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
        """Compute graph and shortest paths using blocks/street edges."""
        edges_df = self.street_adjacency_edges()
        G = nx.Graph()
        for _, row in edges_df.iterrows():
            G.add_edge(row['u'], row['v'])
        # ensure isolated street nodes exist
        if hasattr(self, 'streets_gdf') and not self.streets_gdf.empty:
            for _, r in self.streets_gdf[['coord_x','coord_y']].iterrows():
                G.add_node((int(r['coord_x']), int(r['coord_y'])))
        sp = dict(nx.all_pairs_shortest_path(G))
        self.street_graph = {n: list(G.neighbors(n)) for n in G.nodes}
        self.shortest_paths = {node: paths for node, paths in sp.items()}
        data = []
        for origin, paths in sp.items():
            for dest, path in paths.items():
                w = (1 / (len(path) - 1) ** 2) if len(path) > 1 else 0
                data.append({'origin': origin, 'dest': dest, 'gravity': w})
        self.gravity = pd.DataFrame(data).set_index(['origin','dest'])

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
        ax.plot(np.array(x), np.array(y), linewidth=2, color='black')  # Dashed line for the boundary

        # Define colors for different building types
        if colors is None:
            colors = {
                'street': 'white',
                'home': 'skyblue',
                'work': '#C9A0DC',
                'retail': 'lightgrey',
                'park': 'lightgreen',
                'default': 'lightcoral'
            }

        # Plot streets from GeoDataFrame
        for _, row in self.streets_gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, (Polygon,)):
                x, y = geom.exterior.xy
                ax.fill(x, y, facecolor=colors['street'], linewidth=0.5, label='Street', zorder=zorder)
            elif isinstance(geom, MultiPolygon):
                for single in geom.geoms:
                    x, y = single.exterior.xy
                    ax.fill(x, y, facecolor=colors['street'], linewidth=0.5, label='Street', zorder=zorder)

        # Plot buildings
        if heatmap_agent is not None:
            weights = heatmap_agent.diary.groupby('location').duration.sum()
            norm = Normalize(vmin=0, vmax=max(weights)/2)
            base_color = np.array([1, 0, 0])  # RGB for red

            for _, brow in self.buildings_gdf.iterrows():
                btype = brow.get('type')
                geom = brow.geometry
                if isinstance(geom, Polygon):
                    x, y = geom.exterior.xy
                    weight = weights.get(brow['id'], 0)
                    alpha = norm(weight) if weight > 0 else 0

                    ax.fill(x, y, facecolor=base_color, alpha=alpha,
                            edgecolor='black', linewidth=0.5,
                            label=btype.capitalize(), zorder=zorder)
                    ax.plot(x, y, color='black', alpha=1, linewidth=0.5, zorder=zorder + 1)

                    if doors:
                        door_line = geom.intersection(self.streets_gdf[(self.streets_gdf['coord_x'] == brow['door_cell_x']) & (self.streets_gdf['coord_y'] == brow['door_cell_y'])].iloc[0].geometry)
                        scaled_door_line = scale(door_line, xfact=0.25, yfact=0.25, origin=door_line.centroid)
                        dx, dy = scaled_door_line.xy
                        ax.plot(dx, dy, linewidth=2, color='white', zorder=zorder)

                    if address:
                        door_coord = brow.get('door_x'), brow.get('door_y')
                        bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
                        axes_width_in_inches = bbox.width
                        axes_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                        fontsize = (axes_width_in_inches / axes_data_range) * 13  # Example scaling factor

                        ax.text(door_coord[0] + 0.15, door_coord[1] + 0.15,
                                f"{door_coord[0]}, {door_coord[1]}",
                                ha='left', va='bottom',
                                fontsize=fontsize, color='black')

            sm = ScalarMappable(cmap=cm.Reds, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.02)
            cbar.set_label('Minutes Spent')

        else:
            # Plot buildings from GDF
            for _, brow in self.buildings_gdf.iterrows():
                btype = brow.get('type')
                geom = brow.geometry
                if isinstance(geom, Polygon):
                    x, y = geom.exterior.xy
                    ax.fill(x, y, facecolor=colors.get(btype, colors['default']),
                            edgecolor='black', linewidth=0.5, alpha=alpha,
                            label=str(btype).capitalize() if isinstance(btype, str) else 'Building',
                            zorder=zorder)
                    for interior_ring in geom.interiors:
                        x_int, y_int = interior_ring.xy
                        ax.plot(x_int, y_int, color='black', linewidth=0.5, zorder=zorder + 1)
                        ax.fill(x_int, y_int, facecolor='white', zorder=zorder + 1)
                elif isinstance(geom, MultiPolygon):
                    for single_polygon in geom.geoms:
                        x, y = single_polygon.exterior.xy
                        ax.fill(x, y, facecolor=colors.get(btype, colors['default']),
                                edgecolor='black', linewidth=0.5, alpha=alpha,
                                label=str(btype).capitalize() if isinstance(btype, str) else 'Building',
                                zorder=zorder)
                        for interior_ring in single_polygon.interiors:
                            x_int, y_int = interior_ring.xy
                            ax.plot(x_int, y_int, color='black', linewidth=0.5, zorder=zorder + 1)
                            ax.fill(x_int, y_int, facecolor='white', zorder=zorder + 1)
                # Doors/address labels
                if doors or address:
                    dx, dy = brow.get('door_x'), brow.get('door_y')
                    if pd.notna(dx) and pd.notna(dy):
                        if doors:
                            # draw a small line marker at the door centroid
                            ax.plot([dx-0.1, dx+0.1], [dy, dy], linewidth=2, color='white', zorder=zorder + 2)
                        if address:
                            bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
                            axes_width_in_inches = bbox.width
                            axes_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                            fontsize = (axes_width_in_inches / axes_data_range) * 13
                            ax.text(dx + 0.15, dy + 0.15,
                                    f"{int(round(dx))}, {int(round(dy))}",
                                    ha='left', va='bottom', fontsize=fontsize, color='black')

        ax.set_aspect('equal')

        # Set integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def to_geodataframes(self):
        """
        Return buildings and streets as GeoDataFrames without altering existing structures.

        Returns
        -------
        (gpd.GeoDataFrame, gpd.GeoDataFrame)
            buildings_gdf with columns: id, type, door_x, door_y, size, geometry
            streets_gdf with columns: coord_x, coord_y, id, geometry
        """
        # Return current primary stores
        return self.buildings_gdf.copy(), self.streets_gdf.copy()

    def id_to_door_cell(self):
        """Return Series mapping building id -> (door_cell_x, door_cell_y) with fallbacks."""
        if self.buildings_gdf.empty:
            return pd.Series(dtype=object)
        df = self.buildings_gdf.copy()
        # prefer explicit door cell columns
        if 'door_cell_x' in df.columns and 'door_cell_y' in df.columns:
            cx = df['door_cell_x']
            cy = df['door_cell_y']
        else:
            # fallback to rounded door_x/door_y
            cx = df['door_x'].round().astype('Int64') if 'door_x' in df else pd.Series([pd.NA]*len(df), index=df.index)
            cy = df['door_y'].round().astype('Int64') if 'door_y' in df else pd.Series([pd.NA]*len(df), index=df.index)
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

    def save_geopackage(self, gpkg_path):
        """Save buildings and streets to a single GeoPackage with layers 'buildings' and 'streets'."""
        b_gdf, s_gdf = self.to_geodataframes()
        b_gdf.to_file(gpkg_path, layer='buildings', driver='GPKG')
        s_gdf.to_file(gpkg_path, layer='streets', driver='GPKG')

    @classmethod
    def from_geodataframes(cls, buildings_gdf, streets_gdf):
        """Construct a City from buildings and streets GeoDataFrames."""
        if buildings_gdf.empty:
            width, height = (0,0) if streets_gdf.empty else (int(streets_gdf['coord_x'].max()+1), int(streets_gdf['coord_y'].max()+1))
        else:
            bounds = buildings_gdf.geometry.total_bounds
            width, height = int(np.ceil(bounds[2])), int(np.ceil(bounds[3]))
        if width <= 0 or height <= 0:
            width, height = (0,0)
        city = cls(dimensions=(width, height), manual_streets=True)
        
        # Adopt input GeoDataFrames with required columns
        city.buildings_gdf = gpd.GeoDataFrame(buildings_gdf, geometry='geometry', crs=buildings_gdf.crs)
        missing_cols = set(['id','type','door_x','door_y','door_cell_x','door_cell_y','door_point','size']) - set(city.buildings_gdf.columns)
        for col in missing_cols:
            if col == 'size':
                city.buildings_gdf[col] = 0
            elif col == 'door_point':
                city.buildings_gdf[col] = city.buildings_gdf.geometry.centroid
            elif col in ['door_x', 'door_y']:
                pts = city.buildings_gdf.geometry.centroid
                city.buildings_gdf['door_x'] = pts.x if col == 'door_x' else pts.y
            elif col in ['door_cell_x', 'door_cell_y']:
                pts = city.buildings_gdf.geometry.centroid
                city.buildings_gdf['door_cell_x'] = np.floor(pts.x) if col == 'door_cell_x' else np.floor(pts.y)
            else:
                city.buildings_gdf[col] = None
        city.buildings_gdf.set_index('id', inplace=True, drop=False)
        city.buildings_gdf.index.name = None
        city.buildings_outline = unary_union(city.buildings_gdf.geometry.values) if not city.buildings_gdf.empty else Polygon()
        
        # streets_gdf may be empty if manual_streets=True
        city.streets_gdf = gpd.GeoDataFrame(streets_gdf, geometry='geometry', crs=streets_gdf.crs) if not streets_gdf.empty else gpd.GeoDataFrame(columns=['coord_x','coord_y','id','geometry'], geometry='geometry', crs=None)
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
        
        return city

    @classmethod
    def from_geopackage(cls, gpkg_path):
        b_gdf = gpd.read_file(gpkg_path, layer='buildings')
        s_gdf = gpd.read_file(gpkg_path, layer='streets')
        return cls.from_geodataframes(b_gdf, s_gdf)

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
    def __init__(self, width, height, street_spacing=5, park_ratio=0.1, home_ratio=0.4, work_ratio=0.3, retail_ratio=0.2, seed=42):
        self.seed = seed
        self.width = width
        self.height = height
        self.street_spacing = street_spacing  # Determines regular intervals for streets
        self.park_ratio = park_ratio
        self.home_ratio = home_ratio
        self.work_ratio = work_ratio
        self.retail_ratio = retail_ratio
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
        """Fills an entire block with buildings."""
        available_space = np.argwhere(~self.occupied[(block_x + 1):(block_x + self.street_spacing), 
                                                     (block_y + 1):(block_y + self.street_spacing)])
        
        if available_space.size == 0:
            return  # No available space in this block

        attempts = 0  # Termination condition to prevent infinite loops
        max_attempts = available_space.shape[0] * 2
        
        while available_space.size > 0 and attempts < max_attempts:
            size = self.building_sizes[block_type][npr.randint(len(self.building_sizes[block_type]))]
            npr.shuffle(available_space)  # Randomize placement within the block
            for x_offset, y_offset in available_space.tolist():
                x, y = block_x + 1 + x_offset, block_y + 1 + y_offset
                if x + size[0] <= block_x + self.street_spacing and y + size[1] <= block_y + self.street_spacing and \
                   np.all(self.occupied[x:(x + size[0]), y:(y + size[1])] == False):
                    door = self.get_adjacent_street((x, y))
                    if door is not None and ((self.city.streets_gdf['coord_x'] == door[0]) & (self.city.streets_gdf['coord_y'] == door[1])).any():
                        try:
                            self.city.add_building(building_type=block_type, door=door,
                                                   bbox=box(x, y, x + size[0], y + size[1]))
                        except Exception as e:
                            print(f"Skipping building placement at ({x}, {y}) due to error: {e}")
                        self.occupied[x:x + size[0], y:y + size[1]] = True  # Mark occupied
                        occupied_positions = np.array([(x_offset + dx, y_offset + dy) for dx in range(size[0]) for dy in range(size[1])])
                        mask = ~np.any(np.all(available_space[:, None, :] == occupied_positions, axis=2), axis=1)
                        available_space = available_space[mask]
                        break  # Place one building at a time and reattempt filling
            attempts += 1  # Increment attempt counter
    
    def get_adjacent_street(self, location):
        """Finds the closest predefined street to assign as the door, ensuring it’s within bounds."""
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
        # consider city generated if at least one building present in GDF
        return self.city if (hasattr(self.city, 'buildings_gdf') and len(self.city.buildings_gdf) > 0) else None

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
