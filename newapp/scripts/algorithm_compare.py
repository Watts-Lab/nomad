#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

# Force local repo imports first (avoid picking up site-packages nomad).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nomad.io.base as loader
import nomad.data as data_folder

from nomad.stop_detection.viz import (
    plot_circles,
    plot_stops_barcode,
    plot_time_barcode,
    plot_pings,
)

import nomad.stop_detection.dbstop as DBSTOP
import nomad.stop_detection.dbscan as TADBSCAN
import nomad.stop_detection.density_based as SEQSCAN
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


def _run_algo(traj, algo: str, params: dict, tc: dict):
    algo_key = _resolve_algo(algo)
    algo_registry = {
        "dbstop": (DBSTOP.dbstop, DBSTOP.dbstop_labels),
        "tadbscan": (TADBSCAN.ta_dbscan, TADBSCAN.ta_dbscan_labels),
        "seqscan": (SEQSCAN.seqscan, SEQSCAN.seqscan_labels),
        "lachesis": (LACHESIS.lachesis, LACHESIS.lachesis_labels),
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
    }
    normalized_params = {k: _to_num(params, k, v) for k, v in defaults.items()}

    if algo_key == "lachesis":
        run_args = {
            "dt_max": int(normalized_params["dt_max"]),
            "delta_roam": float(normalized_params["delta_roam"]),
            "dur_min": int(normalized_params["dur_min"]),
        }
    else:
        run_args = {
            "time_thresh": int(normalized_params["time_thresh"]),
            "dist_thresh": float(normalized_params["dist_thresh"]),
            "min_pts": int(normalized_params["min_pts"]),
            "dur_min": int(normalized_params["dur_min"]),
        }

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


def _plot_unavailable_panel(*, ax_map, ax_barcode, algo_key: str):
    ax_map.set_facecolor("#d3d3d3")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    ax_map.set_title(f"{algo_key.upper()} | no notebook visualization available")
    ax_map.text(
        0.5,
        0.5,
        "No plot available",
        transform=ax_map.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="#4b5563",
    )

    ax_barcode.set_facecolor("#d3d3d3")
    ax_barcode.set_xticks([])
    ax_barcode.set_yticks([])
    for spine in ax_barcode.spines.values():
        spine.set_visible(False)
    ax_barcode.set_title("timestamps")


def _plot_notebook_panel(
    *,
    ax_map,
    ax_barcode,
    traj,
    stops,
    algo_key: str,
    normalized_params: dict,
    cmap: str,
    city,
    tc: dict,
):
    plot_circles(
        traj,
        ax=ax_map,
        radius=1.5,
        color="cluster",
        cmap=cmap,
        base_geometry=city,
        base_geom_color="#8c8c8c",
        base_geom_background="#d3d3d3",
        traj_cols=tc,
    )
    plot_pings(traj, ax=ax_map, s=6, color="black", traj_cols=tc)

    if algo_key == "lachesis":
        ax_map.set_title(
            f"LACHESIS | dt_max={int(normalized_params['dt_max'])} min, delta_roam={normalized_params['delta_roam']:g} m, dur_min={int(normalized_params['dur_min'])}"
        )
    else:
        ax_map.set_title(
            f"{algo_key.upper()} | t={int(normalized_params['time_thresh'])} min, d={normalized_params['dist_thresh']:g} m, min_pts={int(normalized_params['min_pts'])}, dur_min={int(normalized_params['dur_min'])}"
        )

    plot_time_barcode(traj[tc["timestamp"]], ax=ax_barcode, set_xlim=True)
    plot_stops_barcode(
        stops,
        ax=ax_barcode,
        stop_alpha=0.4,
        cmap=cmap,
        set_xlim=False,
        timestamp=tc["timestamp"],
    )
    traj_barcode = traj[[tc["timestamp"], "cluster"]].rename(
        columns={tc["timestamp"]: "timestamp"}
    )
    plot_time_barcode(
        traj_barcode,
        color="cluster",
        ax=ax_barcode,
        cmap=cmap,
        set_xlim=False,
        lw=1,
    )
    ax_barcode.set_title("timestamps")


def _render_compare_figure(payload: dict) -> dict:
    seed = int(payload.get("seed", 1))
    mode = str(payload.get("mode", "compare")).strip().lower()
    left_cfg = payload.get("left", {}) or {}
    right_cfg = payload.get("right", {}) or {}

    city, sparse_df, random_user, tc = _load_notebook_data(seed=seed)
    resolved_algorithms = {}

    if mode == "single":
        fig, (ax_map, ax_barcode) = plt.subplots(
            2,
            1,
            figsize=(8.0, 4.8),
            gridspec_kw={"height_ratios": [10, 1]},
        )
        traj = sparse_df.query("user_id == @random_user").copy()
        algo_key, normalized_params, stops, labels = _run_algo(
            traj,
            left_cfg.get("algo", "dbstop"),
            dict(left_cfg.get("params", {})),
            tc,
        )
        resolved_algorithms["left"] = algo_key
        if stops is None or labels is None or normalized_params is None:
            _plot_unavailable_panel(ax_map=ax_map, ax_barcode=ax_barcode, algo_key=algo_key)
        else:
            traj["cluster"] = labels.to_numpy()
            _plot_notebook_panel(
                ax_map=ax_map,
                ax_barcode=ax_barcode,
                traj=traj,
                stops=stops,
                algo_key=algo_key,
                normalized_params=normalized_params,
                cmap=str(left_cfg.get("cmap", "inferno_r")),
                city=city,
                tc=tc,
            )
    else:
        panel_cfg = [
            {
                "panel": "left",
                "algo": left_cfg.get("algo", "dbstop"),
                "params": left_cfg.get("params", {}),
                "cmap": left_cfg.get("cmap", "inferno_r"),
            },
            {
                "panel": "right",
                "algo": right_cfg.get("algo", "seqscan"),
                "params": right_cfg.get("params", {}),
                "cmap": right_cfg.get("cmap", "inferno_r"),
            },
        ]

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 6.5),
            gridspec_kw={"height_ratios": [10, 1]},
        )

        panel_axes = {
            "left": (axes[0, 0], axes[1, 0]),
            "right": (axes[0, 1], axes[1, 1]),
        }

        for cfg in panel_cfg:
            ax_map, ax_barcode = panel_axes[cfg["panel"]]
            traj = sparse_df.query("user_id == @random_user").copy()
            algo_key, normalized_params, stops, labels = _run_algo(
                traj, cfg["algo"], dict(cfg["params"]), tc
            )
            resolved_algorithms[cfg["panel"]] = algo_key
            if stops is None or labels is None or normalized_params is None:
                _plot_unavailable_panel(ax_map=ax_map, ax_barcode=ax_barcode, algo_key=algo_key)
            else:
                traj["cluster"] = labels.to_numpy()
                _plot_notebook_panel(
                    ax_map=ax_map,
                    ax_barcode=ax_barcode,
                    traj=traj,
                    stops=stops,
                    algo_key=algo_key,
                    normalized_params=normalized_params,
                    cmap=str(cfg["cmap"]),
                    city=city,
                    tc=tc,
                )

    plt.tight_layout(pad=1)
    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)

    image_b64 = base64.b64encode(png_buffer.getvalue()).decode("ascii")
    return {
        "seed": seed,
        "mode": mode,
        "image_data_url": f"data:image/png;base64,{image_b64}",
        "resolved_algorithms": resolved_algorithms,
    }


def main() -> None:
    payload = _load_input()
    result = _render_compare_figure(payload)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
