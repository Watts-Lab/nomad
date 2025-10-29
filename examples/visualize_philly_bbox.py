import argparse
import os
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

def add_basemap(ax, crs):
    try:
        import contextily as cx
        cx.add_basemap(ax, crs=crs, source=cx.providers.CartoDB.Positron)
        return True
    except Exception as e:
        print(f"[warn] Could not add basemap: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Visualize and measure a bbox over a basemap.")
    parser.add_argument("--west", type=float, default=-75.16620602)
    parser.add_argument("--south", type=float, default=39.941158234)
    parser.add_argument("--east", type=float, default=-75.14565573)
    parser.add_argument("--north", type=float, default=39.955720193)
    parser.add_argument("--out", type=str, default="examples/plots/philly_bbox.png")
    args = parser.parse_args()

    west, south, east, north = args.west, args.south, args.east, args.north

    gdf = gpd.GeoDataFrame(
        {"name": ["philly_bbox"]},
        geometry=[box(west, south, east, north)],
        crs="EPSG:4326",
    )

    gdf_m = gdf.to_crs("EPSG:3857")
    minx, miny, maxx, maxy = gdf_m.total_bounds
    width_m = maxx - minx
    height_m = maxy - miny
    area_m2 = float(gdf_m.area.iloc[0])

    print(f"Width:  {width_m/1000:.3f} km")
    print(f"Height: {height_m/1000:.3f} km")
    print(f"Area:   {area_m2/1e6:.3f} km^2")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    gdf_m.boundary.plot(ax=ax, color="red", linewidth=2)
    pad = 150
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    added = add_basemap(ax, gdf_m.crs.to_string())
    ax.set_title("Philadelphia bbox fixture" + (" (with basemap)" if added else ""))
    plt.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()


