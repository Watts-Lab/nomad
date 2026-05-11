from functools import partial
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nomad.contact_estimation as contact


class AlgorithmRegistry:
    """Simple registry for parameterized stop-detection runs."""

    def __init__(self):
        self._algos = []
        self._timings = []
        self._family_counts = {}

    def __len__(self):
        return len(self._algos)

    def __iter__(self):
        return iter(self._algos)

    def _next_auto_family(self, base_name):
        count = self._family_counts.get(base_name, 0) + 1
        self._family_counts[base_name] = count
        if count == 1:
            return base_name
        return f"{base_name}_{count}"

    @staticmethod
    def _normalize_algorithm_name(name):
        for suffix in ("_labels_per_user", "_per_user", "_labels"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @staticmethod
    def _expand_values(value, granularity):
        if isinstance(value, tuple):
            start, stop = value
            return np.linspace(start, stop, int(granularity)).tolist()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, list):
            return value

        return [value]

    def add_algorithm(self, fn, family=None, granularity=30, **param_specs):
        algorithm_name = self._normalize_algorithm_name(fn.__name__)
        family_name = family or self._next_auto_family(algorithm_name)

        expanded = {
            key: self._expand_values(value, granularity)
            for key, value in param_specs.items()
        }

        keys = list(expanded.keys())
        values = [expanded[k] for k in keys]

        for index, combo in enumerate(itertools.product(*values), start=1):
            params = dict(zip(keys, combo))
            self._algos.append(
                {
                    "algorithm": algorithm_name,
                    "family": family_name,
                    "fn": fn,
                    "call": partial(fn, **params),
                    "params": params,
                }
            )

        return family_name

    def iter_algorithms(self, family=None):
        if family is None:
            yield from self._algos
            return

        for algo in self._algos:
            if algo["family"] == family:
                yield algo

    def annotate_metrics(self, metrics, algo):
        row = dict(metrics)
        row["algorithm"] = algo["algorithm"]
        row["family"] = algo["family"]
        row.update(algo["params"])
        return row

    def time_call(self, algo, *args, **kwargs):
        t0 = time.perf_counter()
        output = algo["call"](*args, **kwargs)
        elapsed_s = time.perf_counter() - t0
        ping_count = len(args[0]) if args else np.nan
        self.record_timing(algo, elapsed_s, ping_count=ping_count)
        return output

    def record_timing(self, algo, elapsed_s, ping_count=np.nan):
        self._timings.append(
            {
                "algorithm": algo["algorithm"],
                "family": algo["family"],
                "elapsed_s": float(elapsed_s),
                "ping_count": float(ping_count),
                **algo["params"],
            }
        )

    def timing_frame(self):
        return pd.DataFrame(self._timings)

    def family_timing_summary(self):
        timing = self.timing_frame()
        if timing.empty:
            return pd.DataFrame(columns=["family", "n", "mean_s", "avg_pings"])

        return (
            timing.groupby("family", as_index=False)
            .agg(n=("elapsed_s", "count"), mean_s=("elapsed_s", "mean"), avg_pings=("ping_count", "mean"))
            .sort_values("mean_s")
        )


def compute_stop_detection_metrics(stops, truth, user_id=None, algorithm=None, prf_only=True, traj_cols=None, **kwargs):
    if len(stops) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "missed_fraction": 1.0,
            "merged_fraction": 0.0,
            "split_fraction": 0.0,
            "user_id": user_id,
            "algorithm": algorithm,
        }

    stops_clean = stops.fillna({"location": "Street"})
    truth_clean = truth.fillna({"location": "Street"})
    truth_buildings = truth.dropna()

    overlaps = contact.overlapping_visits(
        left=stops_clean,
        right=truth_clean,
        match_location=True,
        traj_cols=traj_cols,
        **kwargs,
    )

    total_pred = stops_clean["duration"].sum()
    total_truth = truth_clean["duration"].sum()
    tp = overlaps["duration"].sum()
    prf_metrics = contact.precision_recall_f1_from_minutes(total_pred, total_truth, tp)

    if prf_only:
        return {**prf_metrics, "user_id": user_id, "algorithm": algorithm}

    if len(truth_buildings) > 0:
        overlaps_err = contact.overlapping_visits(
            left=stops_clean,
            right=truth_buildings,
            match_location=False,
            traj_cols=traj_cols,
            **kwargs,
        )
        error_metrics = contact.compute_visitation_errors(overlaps_err, truth_buildings, traj_cols, **kwargs)
    else:
        error_metrics = {"missed_fraction": 0.0, "merged_fraction": 0.0, "split_fraction": 0.0}

    return {**prf_metrics, **error_metrics, "user_id": user_id, "algorithm": algorithm}


def _desaturate_toward_white(color, amount=0.4):
    rgb = np.array(color[:3])
    white = np.array([1.0, 1.0, 1.0])
    return tuple(rgb + (white - rgb) * amount)


def plot_metric_vs_param(
    ax,
    data,
    x,
    y,
    algo_param,
    title="",
    xlabel="",
    ylabel="",
    show_max_mean_line=False,
    show_band=False,
    color="C0",
):
    """Simple sensitivity plot used by validation notebooks."""
    if data.empty:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    stats = (
        data.groupby(algo_param)[y]
        .agg(mean="mean", std="std")
        .reset_index()
        .sort_values(algo_param)
    )

    if "user_id" in data.columns:
        per_user_max = (
            data.loc[data.groupby("user_id")[y].idxmax(), ["user_id", algo_param, y]]
            .sort_values(algo_param)
        )
    else:
        per_user_max = pd.DataFrame(columns=[algo_param, y])

    scatter_color = _desaturate_toward_white(plt.matplotlib.colors.to_rgba(color), amount=0.5)

    if show_band:
        ax.fill_between(
            stats[algo_param],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=color,
            alpha=0.2,
            label="Mean +/- 1 SD",
            zorder=1,
        )

    ax.scatter(data[x], data[y], s=20, alpha=0.07, color=scatter_color, zorder=2)
    if not per_user_max.empty:
        ax.scatter(
            per_user_max[algo_param],
            per_user_max[y],
            s=35,
            alpha=0.9,
            color="red",
            marker="x",
            label="Per-user max",
            zorder=4,
        )

    ax.plot(stats[algo_param], stats["mean"], linewidth=2.5, color=color, label="Mean", zorder=3)

    if show_max_mean_line:
        max_mean = float(np.nanmax(stats["mean"].to_numpy()))
        ax.axhline(max_mean, color="black", linestyle="--", linewidth=1.2, label=f"Max mean ({max_mean:.3g})")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True)
    return ax


def plot_family_timing(summary_df, ax=None, title="Mean Time Per Parameterization"):
    """Bar plot for family-level runtime summaries."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    if summary_df is None or summary_df.empty:
        ax.set_title(title)
        ax.set_xlabel("Family")
        ax.set_ylabel("Mean seconds")
        return ax

    ordered = summary_df.sort_values("mean_s")
    ax.bar(ordered["family"], ordered["mean_s"], alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Family")
    ax.set_ylabel("Mean seconds")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    return ax


__all__ = [
    "AlgorithmRegistry",
    "compute_stop_detection_metrics",
    "plot_metric_vs_param",
    "plot_family_timing",
]
