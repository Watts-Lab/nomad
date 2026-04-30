#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

import sys

# Force local repo imports first (avoid picking up site-packages nomad).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nomad.stop_detection.lachesis as LACHESIS
import nomad.stop_detection.sequential as SEQUENTIAL
import nomad.stop_detection.density_based as DENSITY_BASED
import nomad.stop_detection.dbscan as TADBSCAN
import nomad.stop_detection.dbstop as DBSTOP
import nomad.stop_detection.utils as STOP_UTILS
import nomad.visit_attribution.visit_attribution as visits
from nomad.contact_estimation import compute_stop_detection_metrics

ROBUST_BASE_DIR = REPO_ROOT / "examples" / "research" / "robustness-of-algorithms"
ROBUST_CONFIG_PATH = ROBUST_BASE_DIR / "config_2_stops.json"
ROBUST_POI_LOCATIONS = ("w-x17-y10", "r-x19-y11")
SCRIPT_VERSION = "v3"
CACHE_DIR = Path(tempfile.gettempdir()) / "nomad_newapp_algorithm_metrics"
SUPPORTED_ALGORITHMS = ("Lachesis", "Sequential", "SeqScan", "ST-DBSCAN", "DBSTOP")
NOTEBOOK_PARAMETER_NOTES: dict[str, dict[str, Any]] = {
    "Lachesis": {
        "source": "delta_roam_vs_acc_exp.ipynb",
        "sweep_parameter": "delta_roam",
        "sweep_note": "UI sweep is scaled as 1.7 × value for Lachesis to mirror notebook setup.",
        "fixed_parameters": [
            "dt_max=60",
            "dur_min=5 (function default)",
        ],
    },
    "Sequential": {
        "source": "delta_roam_vs_acc_exp.ipynb",
        "sweep_parameter": "delta_roam",
        "fixed_parameters": [
            "dt_max=60",
            "method='centroid'",
            "dur_min=5 (function default)",
        ],
    },
    "SeqScan": {
        "source": "dist_thresh_vs_acc_exp.ipynb",
        "sweep_parameter": "dist_thresh",
        "fixed_parameters": [
            "time_thresh=60",
            "min_pts=2",
            "back_merge=False",
            "dur_min=5",
        ],
    },
    "ST-DBSCAN": {
        "source": "dist_thresh_vs_acc_exp.ipynb",
        "sweep_parameter": "dist_thresh",
        "fixed_parameters": [
            "time_thresh=60",
            "min_pts=2",
            "remove_overlaps=True",
        ],
    },
    "DBSTOP": {
        "source": "ta_seqscan_examples.ipynb (dashboard compare code path)",
        "sweep_parameter": "dist_thresh",
        "fixed_parameters": [
            "time_thresh=60",
            "min_pts=2",
            "dur_min=5",
            "back_merge=False",
        ],
    },
}


def _load_input() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    return json.loads(raw)


def _resolve_parquet_path(path_value: str) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = (ROBUST_BASE_DIR / p).resolve()
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {p}")
        return files[0]
    return p


@lru_cache(maxsize=1)
def _load_bundle() -> dict[str, Any]:
    if not ROBUST_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing robustness config: {ROBUST_CONFIG_PATH}")

    config = json.loads(ROBUST_CONFIG_PATH.read_text())
    sparse_path = _resolve_parquet_path(str(config["output_files"]["sparse_path"]))
    diaries_path = _resolve_parquet_path(str(config["output_files"]["diaries_path"]))
    buildings_path = Path(str(config["buildings_file"]))
    if not buildings_path.is_absolute():
        buildings_path = (ROBUST_BASE_DIR / buildings_path).resolve()

    sparse_df = pd.read_parquet(sparse_path).copy()
    diaries_df = pd.read_parquet(diaries_path).copy()
    if "identifier" in diaries_df.columns and "user_id" not in diaries_df.columns:
        diaries_df = diaries_df.rename(columns={"identifier": "user_id"})

    poi_table = gpd.read_parquet(buildings_path).rename(columns={"id": "location"})
    poi_subset = poi_table.loc[poi_table["location"].isin(ROBUST_POI_LOCATIONS)].copy()

    sparse_df["precomp_locations"] = visits.poi_map(
        sparse_df,
        poi_table=poi_subset,
        data_crs="EPSG:3857",
        max_distance=20,
        location_id="location",
        x="x",
        y="y",
    )

    sparse_by_user = {
        str(uid): frame.sort_values("timestamp").reset_index(drop=True).copy()
        for uid, frame in sparse_df.groupby("user_id", sort=True)
    }
    truth_by_user = {
        str(uid): frame.sort_values("timestamp").reset_index(drop=True).copy()
        for uid, frame in diaries_df.groupby("user_id", sort=True)
    }
    all_users = sorted(set(sparse_by_user).intersection(set(truth_by_user)))
    if not all_users:
        raise ValueError("No overlapping users between sparse and diary data.")

    return {
        "sparse_by_user": sparse_by_user,
        "truth_by_user": truth_by_user,
        "poi_subset": poi_subset,
        "all_users": all_users,
        "data_version": str(ROBUST_CONFIG_PATH.resolve()),
    }


def _desaturate_toward_white(color: Any, amount: float = 0.4) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    out = rgb + (white - rgb) * float(amount)
    return float(out[0]), float(out[1]), float(out[2])


def _summarize_stop_with_location(group_df: pd.DataFrame) -> pd.Series:
    return STOP_UTILS.summarize_stop(
        group_df,
        x="x",
        y="y",
        timestamp="timestamp",
        keep_col_names=True,
        passthrough_cols=["location"],
        complete_output=False,
    )


def _build_stops_with_location(labelled_pings: pd.DataFrame) -> pd.DataFrame:
    clustered = labelled_pings.loc[labelled_pings["cluster"] != -1].copy()
    if clustered.empty:
        return pd.DataFrame(columns=["timestamp", "end_timestamp", "duration", "x", "y", "location"])

    grouped = clustered.groupby("cluster", as_index=False)
    try:
        stops = grouped.apply(_summarize_stop_with_location, include_groups=False)
    except TypeError:
        stops = grouped.apply(_summarize_stop_with_location)
    return stops.reset_index(drop=True)


def _agg_labels_for_algorithm(algo_name: str, sparse_df: pd.DataFrame, sweep_value: float) -> tuple[pd.Series, float]:
    if algo_name == "Lachesis":
        actual_value = 1.7 * float(sweep_value)
        labels = LACHESIS.lachesis_labels(
            sparse_df,
            dt_max=60,
            delta_roam=actual_value,
            dur_min=5,
            x="x",
            y="y",
            timestamp="timestamp",
        )
        return labels, actual_value

    if algo_name == "Sequential":
        actual_value = float(sweep_value)
        labels = SEQUENTIAL.detect_stops_labels(
            sparse_df,
            delta_roam=actual_value,
            dt_max=60,
            dur_min=5,
            method="centroid",
            x="x",
            y="y",
            timestamp="timestamp",
        )
        return labels, actual_value

    if algo_name == "SeqScan":
        actual_value = float(sweep_value)
        labels = DENSITY_BASED.seqscan_labels(
            sparse_df,
            dist_thresh=actual_value,
            time_thresh=60,
            min_pts=2,
            dur_min=5,
            back_merge=False,
            x="x",
            y="y",
            timestamp="timestamp",
        )
        return labels, actual_value

    if algo_name == "ST-DBSCAN":
        actual_value = float(sweep_value)
        labels = TADBSCAN.ta_dbscan_labels(
            sparse_df,
            dist_thresh=actual_value,
            time_thresh=60,
            min_pts=2,
            remove_overlaps=True,
            x="x",
            y="y",
            timestamp="timestamp",
        )
        return labels, actual_value

    if algo_name == "DBSTOP":
        actual_value = float(sweep_value)
        labels = DBSTOP.dbstop_labels(
            sparse_df,
            dist_thresh=actual_value,
            time_thresh=60,
            min_pts=2,
            dur_min=5,
            back_merge=False,
            x="x",
            y="y",
            timestamp="timestamp",
        )
        return labels, actual_value

    raise ValueError(f"Unsupported algorithm: {algo_name}")


def _run_aggregated_experiment(
    *,
    algo_name: str,
    user_ids: list[str],
    sweep_values: np.ndarray,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    sparse_by_user: dict[str, pd.DataFrame] = bundle["sparse_by_user"]
    truth_by_user: dict[str, pd.DataFrame] = bundle["truth_by_user"]
    poi_subset: gpd.GeoDataFrame = bundle["poi_subset"]

    rows: list[dict[str, Any]] = []
    for user_id in user_ids:
        sparse = sparse_by_user[user_id].copy()
        truth = truth_by_user[user_id].copy()
        for sweep_value in sweep_values:
            labels, actual_value = _agg_labels_for_algorithm(algo_name, sparse, float(sweep_value))

            labelled = sparse.copy()
            labelled["cluster"] = pd.Series(labels, index=labelled.index).fillna(-1).astype(int).to_numpy()
            labelled["location"] = labelled["precomp_locations"]
            labelled["location"] = visits.point_in_polygon(
                labelled,
                poi_table=poi_subset,
                data_crs="EPSG:3857",
                max_distance=20,
                location_id="location",
                method="majority",
                x="x",
                y="y",
                recompute_location=False,
            )
            stops = _build_stops_with_location(labelled)
            metrics = compute_stop_detection_metrics(
                stops=stops,
                truth=truth,
                user_id=user_id,
                algorithm=algo_name,
                traj_cols={"location_id": "location"},
                timestamp="timestamp",
            )
            rows.append(
                {
                    "user_id": user_id,
                    "algorithm": algo_name,
                    "recall": float(metrics.get("recall", np.nan)),
                    "sweep_value": float(actual_value),
                }
            )

    return pd.DataFrame(rows)


def _algorithm_plot_meta(algo_name: str) -> tuple[str, int, str]:
    if algo_name == "Lachesis":
        return "Lachesis accuracy in 2-stop trajectory", 0, "Roaming Distance (m)"
    if algo_name == "Sequential":
        return "Sequential stop det. accuracy in 2-stop trajectory", 1, "Roaming Distance (m)"
    if algo_name == "SeqScan":
        return "SeqScan accuracy in 2-stop trajectory", 2, "Distance Threshold (m)"
    if algo_name == "ST-DBSCAN":
        return "ST-DBSCAN accuracy in 2-stop trajectory", 3, "Distance Threshold (m)"
    if algo_name == "DBSTOP":
        return "DBSTOP accuracy in 2-stop trajectory", 4, "Distance Threshold (m)"
    return f"{algo_name} accuracy", 0, "Scale Parameter"


def _figure_to_data_url(fig: plt.Figure, *, dpi: int = 180) -> str:
    out = io.BytesIO()
    fig.tight_layout()
    fig.savefig(out, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def _render_notebook_style_metric_plot(results_df: pd.DataFrame, algo_name: str) -> dict[str, Any]:
    if results_df.empty:
        raise ValueError("No aggregated metric results to plot.")

    stats = (
        results_df.groupby("sweep_value")["recall"]
        .agg(mean="mean", std="std")
        .reset_index()
        .sort_values("sweep_value")
    )
    per_user_max = (
        results_df.loc[results_df.groupby("user_id")["recall"].idxmax(), ["user_id", "sweep_value", "recall"]]
        .sort_values("sweep_value")
    )

    title, color_idx, xlabel = _algorithm_plot_meta(algo_name)
    cmap = plt.get_cmap("tab10")
    base = cmap(color_idx % cmap.N)
    scatter_color = _desaturate_toward_white(base, amount=0.4)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#e6e6e6")
    ax.set_facecolor("#e6e6e6")
    ax.scatter(
        results_df["sweep_value"],
        results_df["recall"],
        alpha=0.07,
        s=35,
        color=scatter_color,
        zorder=3,
    )
    ax.scatter(
        per_user_max["sweep_value"],
        per_user_max["recall"],
        color="red",
        s=45,
        alpha=0.9,
        zorder=6,
        label="Per-user max",
        marker="o",
    )
    ax.plot(
        stats["sweep_value"],
        stats["mean"],
        linewidth=3,
        color=base,
        label="Mean",
        zorder=4,
    )
    max_mean = float(np.nanmax(stats["mean"].to_numpy()))
    ax.axhline(
        max_mean,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Max mean ({max_mean:.3g})",
        zorder=5,
    )
    ax.set_title(title, fontsize=17)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel("Accuracy", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True, fontsize=13)

    return {
        "algorithm": algo_name,
        "title": title,
        "image_data_url": _figure_to_data_url(fig, dpi=180),
        "max_mean": max_mean,
        "points": int(len(results_df)),
    }


def _render_trajectory_preview(bundle: dict[str, Any], selected_user_id: str) -> dict[str, Any]:
    sparse_by_user: dict[str, pd.DataFrame] = bundle["sparse_by_user"]
    truth_by_user: dict[str, pd.DataFrame] = bundle["truth_by_user"]
    all_users: list[str] = bundle["all_users"]

    user_id = selected_user_id if selected_user_id in sparse_by_user else all_users[0]
    sparse = sparse_by_user[user_id].copy()
    truth = truth_by_user[user_id].copy()
    truth_stops = truth.loc[truth["location"].notna()].copy()
    if not truth_stops.empty:
        truth_stops["end_timestamp"] = truth_stops["timestamp"] + (truth_stops["duration"] * 60).astype("int64")

    unique_locs = [str(loc) for loc in truth_stops["location"].dropna().astype(str).unique().tolist()]
    cmap = plt.get_cmap("tab10")
    location_colors: dict[str, str] = {}
    for i, loc in enumerate(unique_locs):
        rgba = cmap(i % cmap.N)
        location_colors[loc] = mcolors.to_hex(rgba)

    # Map-style mini image
    fig_map, ax_map = plt.subplots(figsize=(7.0, 3.0))
    fig_map.patch.set_facecolor("#ececec")
    ax_map.set_facecolor("#8f8f8f")
    ax_map.plot(sparse["x"], sparse["y"], color=(98 / 255, 130 / 255, 166 / 255, 0.85), linewidth=2)
    for row in truth_stops.itertuples(index=False):
        if pd.isna(row.x) or pd.isna(row.y):
            continue
        color = location_colors.get(str(row.location), "#ed6f6f")
        ax_map.scatter([row.x], [row.y], s=38, color=color, edgecolor="white", linewidth=0.5, zorder=4)
    ax_map.set_title(f"Ground truth trajectory: {user_id}", fontsize=12)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    map_data_url = _figure_to_data_url(fig_map, dpi=180)

    # Timeline mini image
    fig_t, ax_t = plt.subplots(figsize=(7.0, 1.3))
    fig_t.patch.set_facecolor("#ececec")
    ax_t.set_facecolor("#ececec")
    if len(sparse) > 0:
        min_ts = int(sparse["timestamp"].iloc[0])
        max_ts = int(sparse["timestamp"].iloc[-1])
    else:
        min_ts, max_ts = 0, 1
    for row in truth_stops.itertuples(index=False):
        start_dt = pd.to_datetime(int(row.timestamp), unit="s")
        end_dt = pd.to_datetime(int(row.end_timestamp), unit="s")
        color = location_colors.get(str(row.location), "#965cb0")
        ax_t.axvspan(start_dt, end_dt, color=color, alpha=0.75, linewidth=0)
    pad = pd.Timedelta(minutes=10)
    ax_t.set_xlim(
        pd.to_datetime(min_ts, unit="s") - pad,
        pd.to_datetime(max_ts, unit="s") + pad,
    )
    ax_t.set_title("timestamps", fontsize=12)
    ax_t.yaxis.set_visible(False)
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for spine in ax_t.spines.values():
        spine.set_visible(False)
    timeline_data_url = _figure_to_data_url(fig_t, dpi=180)

    return {
        "user_id": user_id,
        "image_data_url": map_data_url,
        "timeline_data_url": timeline_data_url,
        "location_colors": location_colors,
    }


def _sample_iteration_users(all_users: list[str], iterations: int, seed: int) -> list[str]:
    n = len(all_users)
    if n == 0:
        return []
    n_pick = max(1, min(int(iterations), n))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_pick, replace=False)
    return [all_users[int(i)] for i in idx]


def _cache_file_for_payload(key_payload: dict[str, Any]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key_text = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha1(key_text.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _options_response(bundle: dict[str, Any]) -> dict[str, Any]:
    all_users: list[str] = bundle["all_users"]
    preview_count = min(5, len(all_users))
    idx = np.linspace(0, len(all_users) - 1, num=preview_count, dtype=int)
    trajectories = [{"user_id": all_users[int(i)], "label": all_users[int(i)]} for i in idx]
    return {
        "algorithms": list(SUPPORTED_ALGORITHMS),
        "trajectories": trajectories,
        "parameter_notes": NOTEBOOK_PARAMETER_NOTES,
        "defaults": {
            "mode": "single",
            "left_algorithm": "Lachesis",
            "right_algorithm": "Sequential",
            "iterations": 100,
            "sweep_min": 5.0,
            "sweep_max": 60.0,
            "sweep_points": 100,
            "seed": 2026,
        },
        "max_iterations": int(len(all_users)),
    }


def _run_response(payload: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode", "single")).strip().lower()
    if mode not in {"single", "compare"}:
        mode = "single"
    left_algorithm = str(payload.get("left_algorithm", "Lachesis"))
    right_algorithm = str(payload.get("right_algorithm", "Sequential"))
    selected_user_id = str(payload.get("trajectory_user_id", ""))
    iterations = int(payload.get("iterations", 100))
    sweep_min = float(payload.get("sweep_min", 5.0))
    sweep_max = float(payload.get("sweep_max", 60.0))
    sweep_points = int(payload.get("sweep_points", 100))
    seed = int(payload.get("seed", 2026))

    if left_algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported left algorithm: {left_algorithm}")
    if mode == "compare" and right_algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported right algorithm: {right_algorithm}")

    if sweep_points < 2:
        sweep_points = 2
    if sweep_max < sweep_min:
        sweep_min, sweep_max = sweep_max, sweep_min

    users = _sample_iteration_users(bundle["all_users"], iterations=iterations, seed=seed)
    if not users:
        raise ValueError("No users available for metric run.")

    key_payload = {
        "script_version": SCRIPT_VERSION,
        "mode": mode,
        "left_algorithm": left_algorithm,
        "right_algorithm": right_algorithm if mode == "compare" else None,
        "trajectory_user_id": selected_user_id,
        "iterations": int(len(users)),
        "seed": int(seed),
        "sweep_min": float(sweep_min),
        "sweep_max": float(sweep_max),
        "sweep_points": int(sweep_points),
        "data_version": bundle["data_version"],
    }
    cache_file = _cache_file_for_payload(key_payload)
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        cached["meta"]["cache_hit"] = True
        return cached

    started = pd.Timestamp.utcnow()
    sweep_values = np.linspace(sweep_min, sweep_max, sweep_points)
    left_results = _run_aggregated_experiment(
        algo_name=left_algorithm,
        user_ids=users,
        sweep_values=sweep_values,
        bundle=bundle,
    )
    left_panel = _render_notebook_style_metric_plot(left_results, left_algorithm)

    right_panel = None
    if mode == "compare":
        right_results = _run_aggregated_experiment(
            algo_name=right_algorithm,
            user_ids=users,
            sweep_values=sweep_values,
            bundle=bundle,
        )
        right_panel = _render_notebook_style_metric_plot(right_results, right_algorithm)

    preview = _render_trajectory_preview(bundle, selected_user_id)
    elapsed = (pd.Timestamp.utcnow() - started).total_seconds()
    output = {
        "mode": mode,
        "meta": {
            "cache_hit": False,
            "iterations": int(len(users)),
            "sweep_points": int(sweep_points),
            "elapsed_sec": float(elapsed),
        },
        "left": left_panel,
        "right": right_panel,
        "trajectory_preview": preview,
    }
    cache_file.write_text(json.dumps(output))
    return output


def main() -> None:
    payload = _load_input()
    action = str(payload.get("action", "run")).strip().lower()
    bundle = _load_bundle()

    if action == "options":
        result = _options_response(bundle)
    else:
        result = _run_response(payload, bundle)

    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
