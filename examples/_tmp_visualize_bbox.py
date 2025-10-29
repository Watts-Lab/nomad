import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import contextily as cx

# Philly bbox fixture
west, south, east, north = -75.16620602, 39.941158234, -75.14565573, 39.955720193

# Build polygon and project
gdf = gpd.GeoDataFrame({"name": ["philly_bbox"]}, geometry=[box(west, south, east, north)], crs="EPSG:4326")
gdf_m = gdf.to_crs("EPSG:3857")

minx, miny, maxx, maxy = gdf_m.total_bounds
width_m = maxx - minx
height_m = maxy - miny
area_m2 = float(gdf_m.area.iloc[0])
print(f"Width:  {width_m/1000:.3f} km")
print(f"Height: {height_m/1000:.3f} km")
print(f"Area:   {area_m2/1e6:.3f} km^2")

fig, ax = plt.subplots(figsize=(6, 6))
gdf_m.boundary.plot(ax=ax, color="red", linewidth=2)
ax.set_xlim(minx - 150, maxx + 150)
ax.set_ylim(miny - 150, maxy + 150)
try:
    cx.add_basemap(ax, crs=gdf_m.crs.to_string(), source=cx.providers.CartoDB.Positron)
except Exception:
    pass
ax.set_title("Philadelphia bbox fixture")
plt.tight_layout()
plt.show()
