#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import geopandas as gpd

# Force local repo imports first (avoid picking up site-packages nomad).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nomad.io.base as loader
import nomad.data as data_folder

from nomad.stop_detection.viz import (
    animate_stop_dashboard,
)

import nomad.stop_detection.dbscan as TADBSCAN
import nomad.stop_detection.density_based as SEQSCAN
import nomad.stop_detection.grid_based as GRIDBASED
import nomad.stop_detection.lachesis as LACHESIS


def _load_input() -> dict:
    raw = sys.stdin.read().strip()
    return json.loads(raw) if raw else {}


def _to_num(params: dict, key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except Exception:
        return float(default)


def _resolve_algo(algo: str) -> str:
    norm = (algo or "").strip().lower()
    if norm in {"dbscan", "st-dbscan"}:
        return "tadbscan"
    if norm in {"grid_based", "grid-based"}:
        return "gridbased"
    return norm or "seqscan"


def _normalize_traj_columns(df):
    if "uid" in df.columns and "user_id" not in df.columns:
        df = df.rename(columns={"uid": "user_id"})
    if "identifier" in df.columns and "user_id" not in df.columns:
        df = df.rename(columns={"identifier": "user_id"})
    if "longitude" in df.columns and "x" not in df.columns:
        df = df.rename(columns={"longitude": "x"})
    if "latitude" in df.columns and "y" not in df.columns:
        df = df.rename(columns={"latitude": "y"})
    return df


def _load_notebook_data(seed: int):
    base_dir = REPO_ROOT / "examples" / "research" / "robustness-of-algorithms"
    config_path_candidates = [
        base_dir / "config_low_ha.json",
        base_dir / "config_2_stops.json",
        base_dir / "config_beta_5.json",
    ]

    sim_config = None
    for candidate in config_path_candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                sim_config = json.load(f)
            break
    if sim_config is None:
        raise RuntimeError(
            f"No config file found. Checked: {[str(p) for p in config_path_candidates]}"
        )

    data_dir = Path(data_folder.__file__).parent
    city = gpd.read_parquet(data_dir / "garden-city-buildings-mercator.parquet")

    sparse_path_value = sim_config["output_files"]["sparse_path"]
    sparse_path = Path(sparse_path_value)
    if not sparse_path.is_absolute():
        sparse_path = (base_dir / sparse_path).resolve()

    tc = {"user_id": "user_id", "x": "x", "y": "y", "timestamp": "timestamp"}
    sparse_df = loader.from_file(str(sparse_path), format="parquet", traj_cols=tc)
    sparse_df = _normalize_traj_columns(sparse_df)
    if "timestamp" not in sparse_df.columns:
        raise RuntimeError("Sparse trajectory missing 'timestamp' column.")
    sparse_df["timestamp"] = sparse_df["timestamp"].astype("int64")

    if "user_id" not in sparse_df.columns:
        raise RuntimeError("Sparse trajectory missing 'user_id' column.")

    rng = np.random.default_rng(seed)
    random_user = rng.choice(sparse_df.user_id.unique())
    return city, sparse_df, random_user, tc


def _assign_grid_locations(traj, tc: dict, cell_size: float):
    if cell_size <= 0:
        cell_size = 100.0
    xcol, ycol = tc["x"], tc["y"]
    work = traj.copy()
    x0 = float(work[xcol].min())
    y0 = float(work[ycol].min())
    gx = np.floor((work[xcol].to_numpy(dtype=float) - x0) / cell_size).astype(int)
    gy = np.floor((work[ycol].to_numpy(dtype=float) - y0) / cell_size).astype(int)
    work["location_id"] = [f"g-{ix}-{iy}" for ix, iy in zip(gx, gy)]
    return work


def _run_algo(traj, algo: str, params: dict, tc: dict):
    algo_key = _resolve_algo(algo)
    algo_registry = {
        "tadbscan": (TADBSCAN.ta_dbscan, TADBSCAN.ta_dbscan_labels),
        "seqscan": (SEQSCAN.seqscan, SEQSCAN.seqscan_labels),
        "lachesis": (LACHESIS.lachesis, LACHESIS.lachesis_labels),
        "gridbased": (GRIDBASED.grid_based, GRIDBASED.grid_based_labels),
    }
    if algo_key not in algo_registry:
        return algo_key, None, None, None
    run_fn, label_fn = algo_registry[algo_key]

    defaults = {
        "time_thresh": 60,
        "dist_thresh": 10,
        "min_pts": 3,
        "dur_min": 5,
        "dt_max": 60,
        "delta_roam": 20,
        "cell_size": 100,
        "min_time": 300,
    }
    normalized_params = {k: _to_num(params, k, v) for k, v in defaults.items()}

    if algo_key == "lachesis":
        # UI provides dt_max and dur_min in seconds; lachesis expects minutes.
        dt_max_min = max(0.01, float(normalized_params["dt_max"]) / 60.0)
        dur_min_min = max(0.01, float(normalized_params["dur_min"]) / 60.0)
        run_args = {
            "dt_max": dt_max_min,
            "delta_roam": float(normalized_params["delta_roam"]),
            "dur_min": dur_min_min,
        }
    else:
        run_args = {
            "time_thresh": int(normalized_params["time_thresh"]),
            "dist_thresh": float(normalized_params["dist_thresh"]),
            "min_pts": int(normalized_params["min_pts"]),
            "dur_min": int(normalized_params["dur_min"]),
        }
    if algo_key == "tadbscan":
        # UI exposes T_max in seconds for ST-DBSCAN; backend expects minutes.
        run_args["time_thresh"] = max(1, int(round(float(normalized_params["time_thresh"]) / 60.0)))
    if algo_key == "gridbased":
        traj = _assign_grid_locations(traj, tc=tc, cell_size=float(normalized_params["cell_size"]))
        grid_cols = {"user_id": tc["user_id"], "timestamp": tc["timestamp"], "location_id": "location_id"}
        min_time_min = max(1, int(round(float(normalized_params["min_time"]) / 60.0)))
        run_args = {
            "time_thresh": min_time_min,
            "min_cluster_size": 1,
            "dur_min": min_time_min,
        }
        stops = run_fn(
            traj,
            complete_output=True,
            traj_cols=grid_cols,
            **run_args,
        )
        labels = label_fn(
            traj,
            traj_cols=grid_cols,
            **run_args,
        )
        return algo_key, normalized_params, stops, labels

    stops = run_fn(
        traj,
        complete_output=True,
        traj_cols=tc,
        timestamp=tc["timestamp"],
        **run_args,
    )
    labels = label_fn(
        traj,
        traj_cols=tc,
        timestamp=tc["timestamp"],
        **run_args,
    )
    return algo_key, normalized_params, stops, labels


def _panel_title(algo_key: str, normalized_params: dict) -> str:
    if algo_key == "lachesis":
        dt_max_min = max(0.01, float(normalized_params["dt_max"]) / 60.0)
        dur_min_min = max(0.01, float(normalized_params["dur_min"]) / 60.0)
        return (
            f"LACHESIS | dt_max={dt_max_min:g} min, delta_roam={normalized_params['delta_roam']:g} m, dur_min={dur_min_min:g} min"
        )
    if algo_key == "gridbased":
        min_time_min = max(1, int(round(float(normalized_params["min_time"]) / 60.0)))
        return (
            f"GRID-BASED | cell_size={normalized_params['cell_size']:g} m, min_time={min_time_min} min"
        )
    if algo_key == "tadbscan":
        t_min = max(1, int(round(float(normalized_params["time_thresh"]) / 60.0)))
        return (
            f"TADBSCAN | t={t_min} min, d={normalized_params['dist_thresh']:g} m, min_pts={int(normalized_params['min_pts'])}, dur_min={int(normalized_params['dur_min'])}"
        )
    return (
        f"{algo_key.upper()} | t={int(normalized_params['time_thresh'])} min, d={normalized_params['dist_thresh']:g} m, min_pts={int(normalized_params['min_pts'])}, dur_min={int(normalized_params['dur_min'])}"
    )


def _downsample_traj(traj, tc: dict, max_frames: int = 80):
    ts_col = tc["timestamp"]
    data = traj.sort_values(ts_col).copy()
    if len(data) <= max_frames:
        return data
    idx = np.linspace(0, len(data) - 1, num=max_frames, dtype=int)
    idx = np.unique(idx)
    return data.iloc[idx].copy().reset_index(drop=True)


def _gif_data_url_from_file(path: str) -> str:
    gif_bytes = Path(path).read_bytes()
    return "data:image/gif;base64," + base64.b64encode(gif_bytes).decode("ascii")


def _render_panel_animation(
    *,
    traj,
    stops,
    title: str,
    cmap: str,
    city,
    tc: dict,
    mode: str,
) -> str:
    panel_traj = _downsample_traj(traj, tc=tc, max_frames=80)
    panel_stops = None if stops is None or getattr(stops, "empty", True) else stops
    figsize = (8.2, 5.0) if mode == "single" else (6.0, 4.0)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        gif_path = tmp.name

    try:
        animate_stop_dashboard(
            panel_traj,
            stops=panel_stops,
            interval=140,
            save_path=gif_path,
            fps=8,
            ping_color="cluster",
            ping_cmap=cmap,
            stop_cmap=cmap,
            ping_size=6,
            base_geometry=city,
            base_geom_color="#8c8c8c",
            base_geom_background="#d3d3d3",
            data_crs=None,
            traj_cols=tc,
            show_path=False,
            figsize=figsize,
        )
        return _gif_data_url_from_file(gif_path)
    finally:
        if os.path.exists(gif_path):
            os.remove(gif_path)


def _panel_output(
    *,
    sparse_df,
    random_user,
    cfg: dict,
    city,
    tc: dict,
    mode: str,
):
    traj = sparse_df.query("user_id == @random_user").copy()
    algo_key, normalized_params, stops, labels = _run_algo(
        traj, cfg.get("algo", "seqscan"), dict(cfg.get("params", {})), tc
    )

    if stops is None or labels is None or normalized_params is None:
        return {
            "algo": algo_key,
            "title": f"{algo_key.upper()} | no notebook visualization available",
            "available": False,
            "animation_data_url": None,
        }

    traj = traj.copy()
    traj["cluster"] = labels.to_numpy()
    title = _panel_title(algo_key, normalized_params)

    try:
        animation_data_url = _render_panel_animation(
            traj=traj,
            stops=stops,
            title=title,
            cmap=str(cfg.get("cmap", "inferno_r")),
            city=city,
            tc=tc,
            mode=mode,
        )
    except Exception:
        return {
            "algo": algo_key,
            "title": title,
            "available": False,
            "animation_data_url": None,
        }

    return {
        "algo": algo_key,
        "title": title,
        "available": True,
        "animation_data_url": animation_data_url,
    }


def _render_compare_payload(payload: dict) -> dict:
    seed = int(payload.get("seed", 1))
    mode = str(payload.get("mode", "compare")).strip().lower()
    left_cfg = payload.get("left", {}) or {}
    right_cfg = payload.get("right", {}) or {}

    city, sparse_df, random_user, tc = _load_notebook_data(seed=seed)
    left_panel = _panel_output(
        sparse_df=sparse_df,
        random_user=random_user,
        cfg={
            "algo": left_cfg.get("algo", "dbstop"),
            "params": left_cfg.get("params", {}),
            "cmap": left_cfg.get("cmap", "inferno_r"),
        },
        city=city,
        tc=tc,
        mode=mode,
    )

    response = {
        "seed": seed,
        "mode": mode,
        "left": left_panel,
    }
    if mode == "compare":
        response["right"] = _panel_output(
            sparse_df=sparse_df,
            random_user=random_user,
            cfg={
                "algo": right_cfg.get("algo", "seqscan"),
                "params": right_cfg.get("params", {}),
                "cmap": right_cfg.get("cmap", "inferno_r"),
            },
            city=city,
            tc=tc,
            mode=mode,
        )
    return response


def main() -> None:
    payload = _load_input()
    result = _render_compare_payload(payload)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
