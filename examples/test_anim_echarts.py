"""Build an ECharts version of the stop-dashboard animation.

Run this file from the repository root:

    python examples/test_anim_echarts.py

It writes ``examples/test_anim_echarts.html`` with the Garden City trajectory,
Lachesis stop labels, and a browser-side ECharts animation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path


OUTPUT_FILE = Path(__file__).with_suffix(".html")
ECHARTS_URL = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepare_optional_imports() -> None:
    """Keep this example runnable in lightweight local environments."""

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    mpl_config = Path(tempfile.gettempdir()) / "nomad-matplotlib"
    mpl_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    try:
        import tqdm  # noqa: F401
    except ModuleNotFoundError:
        tqdm_module = types.ModuleType("tqdm")

        def tqdm(iterable=None, *args, **kwargs):
            return iterable if iterable is not None else []

        tqdm_module.tqdm = tqdm
        sys.modules["tqdm"] = tqdm_module


def _clean_float(value, digits: int = 3):
    import pandas as pd

    if pd.isna(value):
        return None
    return round(float(value), digits)


def _format_utc(timestamp: int, with_seconds: bool = True) -> str:
    import pandas as pd

    fmt = "%H:%M:%S UTC" if with_seconds else "%H:%M UTC"
    return pd.to_datetime(int(timestamp), unit="s", utc=True).strftime(fmt)


def _polygon_coords(geometry):
    if geometry.geom_type == "Polygon":
        geometries = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        geometries = list(geometry.geoms)
    else:
        return []

    polygons = []
    for polygon in geometries:
        coords = [
            [_clean_float(x), _clean_float(y)]
            for x, y in list(polygon.exterior.coords)
        ]
        polygons.append(coords)
    return polygons


def _stop_destination(stop: dict, city):
    from shapely.geometry import Point

    stop_point = Point(stop["x"], stop["y"])
    containing = city[city.geometry.contains(stop_point)]
    if len(containing) > 0:
        destination = containing.iloc[0].geometry.representative_point()
    else:
        nearest_index = city.distance(stop_point).idxmin()
        destination = city.loc[nearest_index].geometry.representative_point()
    return [_clean_float(destination.x), _clean_float(destination.y)]


def _nearest_building_id_mercator(point: tuple[float, float], buildings):
    from shapely.geometry import Point

    stop_point = Point(point)
    containing = buildings[buildings.geometry.contains(stop_point)]
    if len(containing) > 0:
        return str(containing.iloc[0]["id"])
    nearest_index = buildings.distance(stop_point).idxmin()
    return str(buildings.loc[nearest_index, "id"])


def _mercator_route_points(route_points: list[tuple[float, float]], city):
    import pandas as pd

    if not route_points:
        return []
    route = pd.DataFrame(route_points, columns=["x", "y"])
    route = city.to_mercator(route)
    return [[_clean_float(row.x), _clean_float(row.y)] for row in route.itertuples()]


def _street_route_between_buildings(origin_id: str, destination_id: str, city):
    if origin_id == destination_id:
        building = city.buildings_gdf.loc[destination_id]
        inside = building.geometry.representative_point()
        return _mercator_route_points([(inside.x, inside.y)], city)

    origin = city.buildings_gdf.loc[origin_id]
    destination = city.buildings_gdf.loc[destination_id]
    origin_inside = origin.geometry.representative_point()
    destination_inside = destination.geometry.representative_point()

    origin_door = tuple(origin["door_point"])
    destination_door = tuple(destination["door_point"])
    origin_cell = (int(origin["door_cell_x"]), int(origin["door_cell_y"]))
    destination_cell = (int(destination["door_cell_x"]), int(destination["door_cell_y"]))

    street_centers = []
    if origin_cell != destination_cell:
        try:
            street_path = city.get_shortest_path(origin_cell, destination_cell)
            street_centers = [(x + 0.5, y + 0.5) for x, y in street_path]
        except Exception:
            street_centers = []

    route = [
        (origin_inside.x, origin_inside.y),
        origin_door,
        *street_centers,
        destination_door,
        (destination_inside.x, destination_inside.y),
    ]

    deduped = []
    for point in route:
        point = (float(point[0]), float(point[1]))
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    return _mercator_route_points(deduped, city)


def build_payload() -> dict:
    _prepare_optional_imports()

    import geopandas as gpd
    import pandas as pd

    from nomad.city_gen import City
    import nomad.filters as filters
    import nomad.io.base as loader
    import nomad.stop_detection.lachesis as LACHESIS

    traj_cols = {
        "user_id": "gc_identifier",
        "x": "dev_x",
        "y": "dev_y",
        "timestamp": "unix_ts",
    }
    data_dir = REPO_ROOT / "nomad" / "data"

    traj = loader.sample_from_file(
        REPO_ROOT / "examples" / "gc_data_long",
        format="parquet",
        users=["admiring_brattain"],
        filters=("date", "==", "2024-01-01"),
        traj_cols=traj_cols,
    )
    traj = traj.sort_values(traj_cols["timestamp"]).reset_index(drop=True)
    traj["h3_cell"] = filters.to_tessellation(
        traj,
        index="h3",
        res=11,
        traj_cols=traj_cols,
        data_crs="EPSG:3857",
    )

    params = {"delta_roam": 20, "dt_max": 60, "dur_min": 5}
    stops = LACHESIS.lachesis(
        traj,
        complete_output=True,
        traj_cols=traj_cols,
        **params,
    ).sort_values("end_timestamp")
    stops = stops.reset_index(drop=True)
    labels = LACHESIS.lachesis_labels(
        traj,
        traj_cols=traj_cols,
        **params,
    ).to_numpy()

    pings = []
    for index, row in traj.iterrows():
        timestamp = int(row[traj_cols["timestamp"]])
        pings.append(
            {
                "i": int(index),
                "x": _clean_float(row[traj_cols["x"]]),
                "y": _clean_float(row[traj_cols["y"]]),
                "t": timestamp,
                "time": _format_utc(timestamp),
                "cluster": int(labels[index]),
            }
        )

    stop_records = []
    for index, row in stops.iterrows():
        start = int(row["unix_ts"])
        end = int(row["end_timestamp"])
        cluster = int(row["cluster"])
        stop_records.append(
            {
                "i": int(index),
                "cluster": cluster,
                "x": _clean_float(row["x"]),
                "y": _clean_float(row["y"]),
                "start": start,
                "end": end,
                "duration": int(row["duration"]),
                "nPings": int(row["n_pings"]),
                "diameter": _clean_float(row["diameter"], 2),
                "label": f"Stop {cluster}",
                "startTime": _format_utc(start, with_seconds=False),
                "endTime": _format_utc(end, with_seconds=False),
            }
        )

    city = gpd.read_parquet(data_dir / "garden-city-buildings-mercator.parquet")
    route_city = City.from_geopackage(data_dir / "garden-city.gpkg")
    route_city.compute_shortest_paths(callable_only=True)
    buildings = []
    for index, row in city.reset_index(drop=True).iterrows():
        for polygon in _polygon_coords(row.geometry):
            buildings.append(
                {
                    "i": int(index),
                    "type": str(row.get("building_type", "building")),
                    "coords": polygon,
                }
            )

    segments = []
    previous_building_id = _nearest_building_id_mercator(
        (pings[0]["x"], pings[0]["y"]),
        city,
    )
    for stop in stop_records:
        stop["buildingId"] = _nearest_building_id_mercator((stop["x"], stop["y"]), city)
        route_start = pings[0]["t"] if not segments else stop_records[stop["i"] - 1]["end"]
        route_end = stop["start"]
        points = _street_route_between_buildings(
            previous_building_id,
            stop["buildingId"],
            route_city,
        )
        destination = points[-1] if points else _stop_destination(stop, city)
        segments.append(
            {
                "stopIndex": stop["i"],
                "cluster": stop["cluster"],
                "detectAt": stop["end"],
                "start": route_start,
                "end": max(route_start, route_end),
                "points": points,
                "destination": destination,
                "kind": "inferred_street_route",
            }
        )
        previous_building_id = stop["buildingId"]

    xs = [ping["x"] for ping in pings] + [stop["x"] for stop in stop_records]
    ys = [ping["y"] for ping in pings] + [stop["y"] for stop in stop_records]
    for building in buildings:
        xs.extend(point[0] for point in building["coords"])
        ys.extend(point[1] for point in building["coords"])

    return {
        "meta": {
            "case": "lachesis",
            "user": "admiring_brattain",
            "date": "2024-01-01",
            "params": params,
            "route": "inferred street route between detected buildings",
        },
        "pings": pings,
        "stops": stop_records,
        "buildings": buildings,
        "segments": segments,
        "bounds": {
            "xMin": min(xs),
            "xMax": max(xs),
            "yMin": min(ys),
            "yMax": max(ys),
        },
    }


def build_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NOMAD Stop Dashboard - ECharts</title>
  <script src="{ECHARTS_URL}"></script>
  <style>
    :root {{
      color-scheme: light;
      --ink: #202124;
      --muted: #626760;
      --paper: #f6f7f2;
      --panel: #ffffff;
      --line: #d7dbd2;
      --accent: #2f6f73;
      --path: #1f6d6a;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--paper);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .shell {{
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 18px 0 22px;
    }}

    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      min-height: 56px;
      margin-bottom: 12px;
    }}

    h1 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.1;
      font-weight: 760;
      letter-spacing: 0;
    }}

    .eyebrow {{
      margin: 0 0 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    .controls {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 0 0 auto;
    }}

    button {{
      height: 34px;
      border: 1px solid #bfc6bb;
      border-radius: 6px;
      padding: 0 13px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      font-weight: 680;
      cursor: pointer;
    }}

    button:hover {{
      border-color: var(--accent);
      color: var(--accent);
    }}

    .dashboard {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 16px 45px rgba(32, 33, 36, 0.08);
    }}

    #mapChart {{
      width: 100%;
      height: min(70vh, 720px);
      min-height: 460px;
      border-bottom: 1px solid var(--line);
    }}

    #timeChart {{
      width: 100%;
      height: 126px;
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 1px;
      overflow: hidden;
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--line);
    }}

    .stat {{
      min-width: 0;
      padding: 12px 14px;
      background: #ffffff;
    }}

    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      white-space: nowrap;
    }}

    .stat strong {{
      display: block;
      margin-top: 4px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 15px;
      line-height: 1.25;
    }}

    .load-error {{
      display: none;
      margin: 16px 0;
      padding: 14px 16px;
      border: 1px solid #d08465;
      border-radius: 8px;
      background: #fff6f0;
      color: #71351f;
    }}

    body.echarts-missing .load-error {{
      display: block;
    }}

    @media (max-width: 720px) {{
      .shell {{
        width: min(100vw - 20px, 1120px);
        padding-top: 12px;
      }}

      header {{
        align-items: flex-start;
        flex-direction: column;
        gap: 10px;
      }}

      h1 {{
        font-size: 21px;
      }}

      #mapChart {{
        height: 62vh;
        min-height: 390px;
      }}

      .stats {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <p class="eyebrow">NOMAD / Lachesis</p>
        <h1>Stop Dashboard Animation</h1>
      </div>
      <div class="controls">
        <button id="toggleBtn" type="button" title="Pause or resume">Pause</button>
        <button id="replayBtn" type="button" title="Replay animation">Replay</button>
      </div>
    </header>
    <div class="load-error">ECharts did not load. Open this file with network access to the pinned CDN script.</div>
    <main class="dashboard">
      <div id="mapChart" aria-label="Animated stop detection map"></div>
      <div id="timeChart" aria-label="Animated stop detection timeline"></div>
    </main>
    <section class="stats" aria-live="polite">
      <div class="stat"><span>Current Time</span><strong id="currentTime">-</strong></div>
      <div class="stat"><span>Ping</span><strong id="pingCount">-</strong></div>
      <div class="stat"><span>Detected Stops</span><strong id="stopCount">-</strong></div>
      <div class="stat"><span>Path Event</span><strong id="pathEvent">Waiting</strong></div>
    </section>
  </div>

  <script>
    const DATA = {data_json};

    const clusterPalette = [
      "#f6d746",
      "#f89540",
      "#dc5039",
      "#a11b63",
      "#5d126e",
      "#1f4f99",
      "#149a8f"
    ];

    const buildingColors = {{
      home: "#d3d7d0",
      work: "#c8d8dd",
      park: "#c6dcc1",
      school: "#d7d0df",
      shop: "#ded3c5",
      building: "#d4d5d1"
    }};

    const mapEl = document.getElementById("mapChart");
    const timeEl = document.getElementById("timeChart");
    const toggleBtn = document.getElementById("toggleBtn");
    const replayBtn = document.getElementById("replayBtn");
    const currentTimeEl = document.getElementById("currentTime");
    const pingCountEl = document.getElementById("pingCount");
    const stopCountEl = document.getElementById("stopCount");
    const pathEventEl = document.getElementById("pathEvent");

    if (!window.echarts) {{
      document.body.classList.add("echarts-missing");
      throw new Error("ECharts failed to load");
    }}

    const mapChart = echarts.init(mapEl, null, {{ renderer: "canvas" }});
    const timeChart = echarts.init(timeEl, null, {{ renderer: "canvas" }});

    const xSpan = DATA.bounds.xMax - DATA.bounds.xMin;
    const ySpan = DATA.bounds.yMax - DATA.bounds.yMin;
    const xPad = Math.max(18, xSpan * 0.08);
    const yPad = Math.max(18, ySpan * 0.08);
    const timeMin = DATA.pings[0].t * 1000;
    const timeMax = DATA.pings[DATA.pings.length - 1].t * 1000;

    let timer = null;
    let paused = false;
    let sequenceIndex = 0;
    let currentTimestamp = DATA.pings[0].t;
    let detectedStopCount = 0;
    const TIME_STEP_SECONDS = 30;

    function clusterColor(cluster) {{
      if (cluster < 0) return "#1f2326";
      return clusterPalette[cluster % clusterPalette.length];
    }}

    function pointDatum(point) {{
      return [point.x, point.y, point.cluster, point.i, point.t];
    }}

    function stopDatum(stop) {{
      return [stop.x, stop.y, stop.cluster, stop.i, stop.nPings, stop.duration];
    }}

    function stopBarDatum(stop) {{
      return [stop.start * 1000, stop.end * 1000, stop.cluster, stop.i];
    }}

    function formatTime(timestampSeconds) {{
      return new Date(timestampSeconds * 1000).toISOString().slice(11, 19) + " UTC";
    }}

    function buildTimeSequence() {{
      const start = DATA.pings[0].t;
      const end = DATA.pings[DATA.pings.length - 1].t;
      const times = new Set([start, end]);
      for (let timestamp = start; timestamp <= end; timestamp += TIME_STEP_SECONDS) {{
        times.add(timestamp);
      }}
      DATA.pings.forEach((point) => times.add(point.t));
      DATA.stops.forEach((stop) => {{
        times.add(stop.start);
        times.add(stop.end);
      }});
      return Array.from(times).sort((a, b) => a - b);
    }}

    const timeSequence = buildTimeSequence();

    function pathLength(points) {{
      let total = 0;
      for (let index = 1; index < points.length; index += 1) {{
        total += Math.hypot(
          points[index][0] - points[index - 1][0],
          points[index][1] - points[index - 1][1]
        );
      }}
      return total;
    }}

    function partialPath(points, progress) {{
      if (points.length <= 1) return points.slice();
      const total = pathLength(points);
      if (total === 0 || progress >= 1) return points.slice();
      const target = total * Math.max(0, progress);
      let travelled = 0;
      const drawn = [points[0]];

      for (let index = 1; index < points.length; index += 1) {{
        const previous = points[index - 1];
        const next = points[index];
        const segmentLength = Math.hypot(next[0] - previous[0], next[1] - previous[1]);
        if (travelled + segmentLength >= target) {{
          const local = segmentLength === 0 ? 1 : (target - travelled) / segmentLength;
          drawn.push([
            previous[0] + (next[0] - previous[0]) * local,
            previous[1] + (next[1] - previous[1]) * local
          ]);
          return drawn;
        }}
        travelled += segmentLength;
        drawn.push(next);
      }}
      return drawn;
    }}

    function detectedCountFor(timestampSeconds) {{
      return DATA.stops.filter((stop) => stop.end <= timestampSeconds).length;
    }}

    function latestPingIndexAt(timestampSeconds) {{
      let index = 0;
      for (let candidate = 0; candidate < DATA.pings.length; candidate += 1) {{
        if (DATA.pings[candidate].t <= timestampSeconds) index = candidate;
        else break;
      }}
      return index;
    }}

    function routeProgressAt(timestampSeconds) {{
      const completed = [];
      let active = [];

      DATA.segments.forEach((segment) => {{
        if (!segment || segment.points.length < 2) return;
        if (timestampSeconds >= segment.end) {{
          completed.push(segment.points);
          return;
        }}
        if (timestampSeconds >= segment.start) {{
          const duration = Math.max(1, segment.end - segment.start);
          const progress = (timestampSeconds - segment.start) / duration;
          active = partialPath(segment.points, progress);
        }}
      }});

      return {{ completed, active }};
    }}

    function initCharts() {{
      mapChart.setOption({{
        animation: true,
        animationDurationUpdate: 120,
        grid: {{ left: 12, right: 12, top: 12, bottom: 12, containLabel: false }},
        xAxis: {{
          type: "value",
          min: DATA.bounds.xMin - xPad,
          max: DATA.bounds.xMax + xPad,
          axisLine: {{ show: false }},
          axisTick: {{ show: false }},
          axisLabel: {{ show: false }},
          splitLine: {{ show: false }}
        }},
        yAxis: {{
          type: "value",
          min: DATA.bounds.yMin - yPad,
          max: DATA.bounds.yMax + yPad,
          axisLine: {{ show: false }},
          axisTick: {{ show: false }},
          axisLabel: {{ show: false }},
          splitLine: {{ show: false }}
        }},
        tooltip: {{
          trigger: "item",
          borderWidth: 0,
          backgroundColor: "rgba(32, 33, 36, 0.92)",
          textStyle: {{ color: "#ffffff" }},
          formatter(params) {{
            if (params.seriesId === "detected-stops") {{
              const stop = DATA.stops[params.data[3]];
              return `${{stop.label}}<br>${{stop.startTime}} - ${{stop.endTime}}<br>${{stop.nPings}} pings / ${{stop.duration}} min`;
            }}
            if (params.seriesId === "cluster-pings" || params.seriesId === "noise-pings" || params.seriesId === "current-ping") {{
              const ping = DATA.pings[params.data[3]];
              const cluster = ping.cluster < 0 ? "Noise" : `Cluster ${{ping.cluster}}`;
              return `Ping ${{ping.i + 1}}<br>${{ping.time}}<br>${{cluster}}`;
            }}
            return "";
          }}
        }},
        series: [
          {{
            id: "buildings",
            type: "custom",
            coordinateSystem: "cartesian2d",
            silent: true,
            z: 0,
            data: DATA.buildings.map((_, index) => index),
            renderItem(params, api) {{
              const building = DATA.buildings[params.dataIndex];
              const points = building.coords.map((coord) => api.coord(coord));
              const fill = buildingColors[building.type] || buildingColors.building;
              return {{
                type: "polygon",
                shape: {{ points }},
                style: {{
                  fill,
                  stroke: "#ffffff",
                  lineWidth: 1
                }}
              }};
            }}
          }},
          {{
            id: "completed-paths",
            name: "Inferred street route",
            type: "lines",
            coordinateSystem: "cartesian2d",
            polyline: true,
            silent: true,
            z: 2,
            data: [],
            lineStyle: {{
              color: "#1f6d6a",
              width: 2.4,
              opacity: 0.78
            }}
          }},
          {{
            id: "active-path",
            name: "Active route",
            type: "line",
            coordinateSystem: "cartesian2d",
            smooth: 0.38,
            showSymbol: false,
            silent: true,
            z: 4,
            data: [],
            lineStyle: {{
              color: "#0f766e",
              width: 4,
              opacity: 0.95,
              shadowBlur: 10,
              shadowColor: "rgba(15, 118, 110, 0.26)"
            }}
          }},
          {{
            id: "ghost-pings",
            name: "All pings",
            type: "scatter",
            coordinateSystem: "cartesian2d",
            silent: true,
            z: 3,
            data: DATA.pings.map(pointDatum),
            symbolSize: 5,
            itemStyle: {{ color: "rgba(31, 35, 38, 0.13)" }}
          }},
          {{
            id: "noise-pings",
            name: "Noise pings",
            type: "scatter",
            coordinateSystem: "cartesian2d",
            z: 5,
            data: [],
            symbolSize: 7,
            itemStyle: {{ color: "rgba(31, 35, 38, 0.72)" }}
          }},
          {{
            id: "cluster-pings",
            name: "Cluster pings",
            type: "scatter",
            coordinateSystem: "cartesian2d",
            z: 6,
            data: [],
            symbolSize: 12,
            itemStyle: {{
              color: (params) => clusterColor(params.data[2]),
              borderColor: "#202124",
              borderWidth: 0.8,
              opacity: 0.95
            }}
          }},
          {{
            id: "detected-stops",
            name: "Detected stops",
            type: "scatter",
            coordinateSystem: "cartesian2d",
            z: 7,
            data: [],
            symbol: "diamond",
            symbolSize: 20,
            itemStyle: {{
              color: (params) => clusterColor(params.data[2]),
              borderColor: "#ffffff",
              borderWidth: 2,
              shadowBlur: 8,
              shadowColor: "rgba(32, 33, 36, 0.22)"
            }}
          }},
          {{
            id: "current-ping",
            name: "Current ping",
            type: "effectScatter",
            coordinateSystem: "cartesian2d",
            z: 8,
            data: [],
            symbolSize: 13,
            rippleEffect: {{ scale: 2.4, brushType: "stroke" }},
            itemStyle: {{
              color: (params) => clusterColor(params.data[2]),
              borderColor: "#ffffff",
              borderWidth: 1.5
            }}
          }}
        ]
      }});

      timeChart.setOption({{
        animation: true,
        animationDurationUpdate: 120,
        grid: {{ left: 34, right: 22, top: 18, bottom: 28 }},
        xAxis: {{
          type: "time",
          min: timeMin,
          max: timeMax,
          axisLine: {{ lineStyle: {{ color: "#a9aea5" }} }},
          axisTick: {{ show: false }},
          axisLabel: {{
            color: "#626760",
            formatter(value) {{
              return new Date(value).toISOString().slice(11, 16);
            }}
          }},
          splitLine: {{ show: true, lineStyle: {{ color: "#edf0ea" }} }}
        }},
        yAxis: {{
          min: 0,
          max: 1,
          axisLine: {{ show: false }},
          axisTick: {{ show: false }},
          axisLabel: {{ show: false }},
          splitLine: {{ show: false }}
        }},
        tooltip: {{ show: false }},
        series: [
          {{
            id: "ping-ticks",
            type: "custom",
            coordinateSystem: "cartesian2d",
            silent: true,
            z: 1,
            data: DATA.pings.map((_, index) => index),
            renderItem(params, api) {{
              const ping = DATA.pings[params.dataIndex];
              const top = api.coord([ping.t * 1000, 0.72]);
              const bottom = api.coord([ping.t * 1000, 0.28]);
              return {{
                type: "line",
                shape: {{ x1: top[0], y1: top[1], x2: bottom[0], y2: bottom[1] }},
                style: {{ stroke: "rgba(32, 33, 36, 0.18)", lineWidth: 1 }}
              }};
            }}
          }},
          {{
            id: "stop-bars",
            type: "custom",
            coordinateSystem: "cartesian2d",
            z: 2,
            data: [],
            renderItem(params, api) {{
              const start = api.coord([api.value(0), 0.24]);
              const end = api.coord([api.value(1), 0.76]);
              const shape = echarts.graphic.clipRectByRect(
                {{
                  x: start[0],
                  y: end[1],
                  width: Math.max(2, end[0] - start[0]),
                  height: start[1] - end[1]
                }},
                {{
                  x: params.coordSys.x,
                  y: params.coordSys.y,
                  width: params.coordSys.width,
                  height: params.coordSys.height
                }}
              );
              if (!shape) return null;
              return {{
                type: "rect",
                shape,
                style: {{
                  fill: clusterColor(api.value(2)),
                  opacity: 0.44
                }}
              }};
            }}
          }},
          {{
            id: "cursor",
            type: "line",
            coordinateSystem: "cartesian2d",
            showSymbol: false,
            z: 3,
            data: [],
            lineStyle: {{ color: "#202124", width: 2 }}
          }}
        ]
      }});
    }}

    function renderScene(timestampSeconds) {{
      currentTimestamp = timestampSeconds;
      const latestPingIndex = latestPingIndexAt(timestampSeconds);
      const current = DATA.pings[latestPingIndex];
      const visible = DATA.pings.filter((point) => point.t <= timestampSeconds);
      const noise = visible.filter((point) => point.cluster < 0).map(pointDatum);
      const clustered = visible.filter((point) => point.cluster >= 0).map(pointDatum);
      const visibleStops = DATA.stops.slice(0, detectedStopCount);
      const route = routeProgressAt(timestampSeconds);

      mapChart.setOption({{
        series: [
          {{
            id: "completed-paths",
            data: route.completed.map((points) => ({{ coords: points }}))
          }},
          {{
            id: "active-path",
            data: route.active
          }},
          {{
            id: "noise-pings",
            data: noise
          }},
          {{
            id: "cluster-pings",
            data: clustered
          }},
          {{
            id: "detected-stops",
            data: visibleStops.map(stopDatum)
          }},
          {{
            id: "current-ping",
            data: [pointDatum(current)]
          }}
        ]
      }});

      timeChart.setOption({{
        series: [
          {{
            id: "stop-bars",
            data: visibleStops.map(stopBarDatum)
          }},
          {{
            id: "cursor",
            data: [[timestampSeconds * 1000, 0.08], [timestampSeconds * 1000, 0.92]]
          }}
        ]
      }});

      currentTimeEl.textContent = formatTime(timestampSeconds);
      pingCountEl.textContent = `${{visible.length}} / ${{DATA.pings.length}}`;
      stopCountEl.textContent = `${{detectedStopCount}} / ${{DATA.stops.length}}`;
    }}

    function scheduleNext(delay = 90) {{
      clearTimeout(timer);
      if (paused) return;
      timer = setTimeout(advance, delay);
    }}

    function advance() {{
      if (paused) return;

      sequenceIndex = Math.min(sequenceIndex + 1, timeSequence.length - 1);
      const timestampSeconds = timeSequence[sequenceIndex];
      const previousStopCount = detectedStopCount;
      const nextStopCount = detectedCountFor(timestampSeconds);

      detectedStopCount = nextStopCount;
      renderScene(timestampSeconds);

      if (nextStopCount > previousStopCount) {{
        scheduleNext();
        return;
      }}

      if (sequenceIndex < timeSequence.length - 1) {{
        scheduleNext();
      }} else {{
        pathEventEl.textContent = "Complete";
        toggleBtn.textContent = "Replay";
        paused = true;
      }}
    }}

    function replay() {{
      clearTimeout(timer);
      paused = false;
      sequenceIndex = 0;
      currentTimestamp = timeSequence[0];
      detectedStopCount = 0;
      toggleBtn.textContent = "Pause";
      pathEventEl.textContent = "Route synced";
      renderScene(currentTimestamp);
      scheduleNext(900);
    }}

    toggleBtn.addEventListener("click", () => {{
      if (sequenceIndex >= timeSequence.length - 1 && paused) {{
        replay();
        return;
      }}
      paused = !paused;
      toggleBtn.textContent = paused ? "Play" : "Pause";
      if (!paused) scheduleNext(160);
    }});

    replayBtn.addEventListener("click", replay);

    window.addEventListener("resize", () => {{
      mapChart.resize();
      timeChart.resize();
    }});

    initCharts();
    replay();
  </script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    OUTPUT_FILE.write_text(build_html(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
