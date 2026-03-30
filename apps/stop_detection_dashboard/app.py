"""
Conventional dashboard entrypoint.

Run:
    python app.py
"""

from __future__ import annotations

import colorsys
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
# Ensure local repo modules are imported before any installed site-packages.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NOMAD_IMPORT_ERROR = None
try:
    import nomad.filters as filters
    import nomad.stop_detection.dbscan as dbscan
    import nomad.stop_detection.density_based as seqscan
    import nomad.stop_detection.grid_based as grid_based
    import nomad.stop_detection.hdbscan as hdbscan
    import nomad.stop_detection.lachesis as lachesis
except Exception as exc:  # pragma: no cover - environment dependent
    NOMAD_IMPORT_ERROR = exc

DATA_PATH = REPO_ROOT / "nomad" / "data" / "gc_sample.csv"

app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)


@lru_cache(maxsize=1)
def load_gc_sample() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"uid": "user_id"}).copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    df["tz_offset"] = df["tz_offset"].astype("int64")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def add_noise_to_lon_lat(df: pd.DataFrame, noise_m: float, seed: int) -> pd.DataFrame:
    if noise_m <= 0:
        return df.copy()

    out = df.copy()
    rng = np.random.default_rng(seed)
    lat_sigma = noise_m / 111_320.0
    cos_lat = np.cos(np.radians(out["latitude"].to_numpy()))
    lon_sigma = noise_m / np.maximum(111_320.0 * np.maximum(cos_lat, 1e-3), 1e-3)

    out["latitude"] = out["latitude"] + rng.normal(0.0, lat_sigma, size=len(out))
    out["longitude"] = out["longitude"] + rng.normal(0.0, lon_sigma, size=len(out))
    return out


def prepare_traj(
    all_data: pd.DataFrame,
    user_id: str,
    beta_ping_proxy_min: float,
    noise_m: float,
    max_pings: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dense = all_data.loc[all_data["user_id"] == user_id].copy()
    dense = dense.sort_values("timestamp").reset_index(drop=True)

    window_minutes = max(1, int(round(beta_ping_proxy_min)))
    sparse = filters.downsample(
        dense,
        periods=window_minutes,
        freq="min",
        keep="first",
        traj_cols={"timestamp": "timestamp", "user_id": "user_id", "tz_offset": "tz_offset"},
    ).copy()

    sparse = add_noise_to_lon_lat(sparse, noise_m=noise_m, seed=seed)
    sparse = sparse.sort_values("timestamp").reset_index(drop=True)

    if len(sparse) > max_pings:
        step = int(np.ceil(len(sparse) / max_pings))
        sparse = sparse.iloc[::step].copy().reset_index(drop=True)

    return dense, sparse


def run_labels(algo_name: str, traj: pd.DataFrame, params: dict) -> pd.Series:
    tc = {
        "user_id": "user_id",
        "longitude": "longitude",
        "latitude": "latitude",
        "timestamp": "timestamp",
    }

    if algo_name == "SEQSCAN":
        labels = seqscan.seqscan_labels(
            traj,
            dist_thresh=params["dist_thresh"],
            time_thresh=params["time_thresh"],
            min_pts=params["min_pts"],
            dur_min=params["dur_min"],
            traj_cols=tc,
        )
    elif algo_name == "TA-DBSCAN":
        labels = dbscan.ta_dbscan_labels(
            traj,
            dist_thresh=params["dist_thresh"],
            time_thresh=params["time_thresh"],
            min_pts=params["min_pts"],
            traj_cols=tc,
        )
    elif algo_name == "Lachesis":
        labels = lachesis.lachesis_labels(
            traj,
            delta_roam=params["delta_roam"],
            dt_max=params["dt_max"],
            dur_min=params["dur_min"],
            traj_cols=tc,
        )
    elif algo_name == "HDBSCAN":
        labels = hdbscan.hdbscan_labels(
            traj,
            time_thresh=params["time_thresh"],
            min_pts=params["min_pts"],
            min_cluster_size=params["min_cluster_size"],
            dur_min=params["dur_min"],
            traj_cols=tc,
        )
    elif algo_name == "Grid-Based":
        work = traj.copy()
        work["h3_cell"] = filters.to_tessellation(
            work,
            index="h3",
            res=params["h3_res"],
            traj_cols=tc,
        )
        labels = grid_based.grid_based_labels(
            work,
            time_thresh=params["time_thresh"],
            min_cluster_size=params["min_cluster_size"],
            dur_min=params["dur_min"],
            location_id="h3_cell",
            traj_cols=tc,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    return pd.Series(labels, index=traj.index, name="cluster").fillna(-1).astype(int)


def build_stop_runs(labelled: pd.DataFrame) -> pd.DataFrame:
    work = labelled.loc[labelled["cluster"] >= 0, ["timestamp", "longitude", "latitude", "cluster"]].copy()
    if work.empty:
        return pd.DataFrame(
            columns=["cluster", "run_id", "start_timestamp", "end_timestamp", "n_pings", "duration_min", "longitude", "latitude"]
        )

    work["run_id"] = (work["cluster"] != work["cluster"].shift()).cumsum()
    runs = (
        work.groupby(["cluster", "run_id"], as_index=False)
        .agg(
            start_timestamp=("timestamp", "min"),
            end_timestamp=("timestamp", "max"),
            n_pings=("cluster", "size"),
            longitude=("longitude", "median"),
            latitude=("latitude", "median"),
        )
        .sort_values("start_timestamp")
        .reset_index(drop=True)
    )
    runs["duration_min"] = (runs["end_timestamp"] - runs["start_timestamp"]) / 60.0
    return runs


def cluster_color_map(clusters: pd.Series) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"-1": [130, 138, 150]}
    active = sorted(int(c) for c in clusters.unique() if int(c) >= 0)
    n = max(1, len(active))
    for i, cid in enumerate(active):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.55, 0.9)
        out[str(cid)] = [int(255 * r), int(255 * g), int(255 * b)]
    return out


def default_params(algo_name: str) -> dict:
    if algo_name in {"SEQSCAN", "TA-DBSCAN"}:
        return {"dist_thresh": 15, "time_thresh": 240, "min_pts": 3, "dur_min": 5}
    if algo_name == "Lachesis":
        return {"delta_roam": 20, "dt_max": 60, "dur_min": 5}
    if algo_name == "HDBSCAN":
        return {"time_thresh": 240, "min_pts": 3, "min_cluster_size": 2, "dur_min": 5}
    return {"time_thresh": 240, "min_cluster_size": 2, "dur_min": 5, "h3_res": 10}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/options")
def options():
    if NOMAD_IMPORT_ERROR is not None:
        return jsonify({"error": f"NOMAD import error: {NOMAD_IMPORT_ERROR}"}), 500

    data = load_gc_sample()
    user_counts = data["user_id"].value_counts().sort_values(ascending=False)
    users = [{"user_id": str(uid), "count": int(count)} for uid, count in user_counts.items()]
    algorithms = ["SEQSCAN", "TA-DBSCAN", "Lachesis", "HDBSCAN", "Grid-Based"]
    params = {algo: default_params(algo) for algo in algorithms}
    return jsonify({"users": users, "algorithms": algorithms, "default_params": params})


@app.post("/api/run")
def run_dashboard():
    if NOMAD_IMPORT_ERROR is not None:
        return jsonify({"error": f"NOMAD import error: {NOMAD_IMPORT_ERROR}"}), 500

    payload = request.get_json(force=True) or {}
    algo = str(payload.get("algorithm", "SEQSCAN"))
    user_id = str(payload.get("user_id", ""))
    beta_ping_proxy = float(payload.get("beta_ping_proxy", 7.0))
    noise_m = float(payload.get("noise_m", 8.0))
    max_pings = int(payload.get("max_pings", 220))
    seed = int(payload.get("seed", 2026))
    params = payload.get("params") or default_params(algo)

    data = load_gc_sample()
    available_users = set(data["user_id"].unique())
    if user_id not in available_users:
        user_id = next(iter(available_users))

    dense, sparse = prepare_traj(
        all_data=data,
        user_id=user_id,
        beta_ping_proxy_min=beta_ping_proxy,
        noise_m=noise_m,
        max_pings=max_pings,
        seed=seed,
    )

    if sparse.empty:
        return jsonify({"error": "No pings left after sparsification. Lower beta_ping proxy or increase max_pings."}), 400

    labels = run_labels(algo, sparse, params)
    labelled = sparse.copy().reset_index(drop=True)
    labelled["cluster"] = labels.to_numpy()
    stop_runs = build_stop_runs(labelled)

    dense_out = dense[["timestamp", "longitude", "latitude"]].copy()
    sparse_out = labelled[["timestamp", "longitude", "latitude", "cluster"]].copy()
    stop_out = stop_runs[
        ["cluster", "run_id", "start_timestamp", "end_timestamp", "n_pings", "duration_min", "longitude", "latitude"]
    ].copy()

    bounds = {
        "min_lat": float(dense_out["latitude"].min()),
        "max_lat": float(dense_out["latitude"].max()),
        "min_lon": float(dense_out["longitude"].min()),
        "max_lon": float(dense_out["longitude"].max()),
    }

    return jsonify(
        {
            "meta": {
                "user_id": user_id,
                "algorithm": algo,
                "input_pings": int(len(dense_out)),
                "sparse_pings": int(len(sparse_out)),
                "stop_runs": int(len(stop_out)),
                "noise_points": int((sparse_out["cluster"] == -1).sum()),
                "bounds": bounds,
            },
            "cluster_colors": cluster_color_map(sparse_out["cluster"]),
            "dense": dense_out.to_dict(orient="records"),
            "sparse": sparse_out.to_dict(orient="records"),
            "stop_runs": stop_out.to_dict(orient="records"),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
