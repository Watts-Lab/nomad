from functools import partial
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nomad.contact_estimation as contact
import nomad.filters as filters
import nomad.io.base as loader


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


def compute_visitation_errors(overlaps, true_visits, traj_cols=None, right_traj_cols=None, **kwargs):
    if right_traj_cols is None:
        right_schema_input = traj_cols
        right_kwargs = kwargs
    else:
        right_schema_input = right_traj_cols
        right_kwargs = {}

    temp_traj_cols = loader._parse_traj_cols(true_visits.columns, right_schema_input, right_kwargs, warn=False)
    true_visits = true_visits.dropna()

    t_name, _ = loader._fallback_time_cols_dt(true_visits.columns, right_schema_input, right_kwargs)
    t_key = temp_traj_cols[t_name]

    if true_visits[t_key].duplicated().any():
        dup_ts = true_visits.loc[true_visits[t_key].duplicated(), t_key].unique()
        raise ValueError(
            "Ground-truth stops share the same start time(s), which violates the "
            "per-stop key assumption. Duplicated timestamps: " + repr(dup_ts)
        )
    n_truth = len(true_visits)

    temp_cols = loader._parse_traj_cols([], traj_cols, kwargs, warn=False)
    loc_left = f"{temp_cols['location_id']}_left"
    loc_right = f"{temp_traj_cols['location_id']}_right"

    time_keys = ["datetime", "start_datetime", "timestamp", "start_timestamp"]
    if "timestamp" in kwargs or "start_timestamp" in kwargs:
        time_keys = ["timestamp", "start_timestamp", "datetime", "start_datetime"]
    if "datetime" in kwargs or "start_datetime" in kwargs:
        time_keys = ["datetime", "start_datetime", "timestamp", "start_timestamp"]

    t_left = None
    for key in time_keys:
        col = f"{temp_cols[key]}_left"
        if col in overlaps.columns:
            t_left = col
            break

    time_keys = ["datetime", "start_datetime", "timestamp", "start_timestamp"]
    if "timestamp" in right_kwargs or "start_timestamp" in right_kwargs:
        time_keys = ["timestamp", "start_timestamp", "datetime", "start_datetime"]
    if "datetime" in right_kwargs or "start_datetime" in right_kwargs:
        time_keys = ["datetime", "start_datetime", "timestamp", "start_timestamp"]

    t_right = None
    for key in time_keys:
        col = f"{temp_traj_cols[key]}_right"
        if col in overlaps.columns:
            t_right = col
            break

    if t_left is None or t_right is None:
        raise ValueError("compute_visitation_errors: could not resolve the overlap start-time columns.")
    for col in (loc_left, loc_right):
        if col not in overlaps.columns:
            raise ValueError(f"compute_visitation_errors: expected column '{col}' in overlaps but not found.")

    overlaps = overlaps.fillna({loc_left: "Street"})

    bad_ts = set(overlaps[t_right]) - set(true_visits[t_key])
    if bad_ts:
        raise ValueError(
            "compute_visitation_errors: overlap rows reference start times that "
            "do not exist in ground truth: " + repr(sorted(bad_ts)[:10])
        )

    diff_loc = overlaps[loc_left] != overlaps[loc_right]
    same_loc = ~diff_loc

    num_overlapped = overlaps[t_right].nunique()
    missed_fraction = 1 - num_overlapped / n_truth
    merged_fraction = diff_loc.groupby(overlaps[t_right]).any().mean()
    split_fraction = overlaps[same_loc].groupby(t_right)[t_left].nunique().gt(1).mean()

    return {
        "missed_fraction": missed_fraction,
        "merged_fraction": merged_fraction,
        "split_fraction": split_fraction,
    }


def compute_stop_detection_metrics(
    stops,
    truth,
    user_id=None,
    algorithm=None,
    prf_only=True,
    traj_cols=None,
    right_traj_cols=None,
    **kwargs,
):
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

    input_traj_cols = traj_cols
    left_kwargs = dict(kwargs)
    if right_traj_cols is None:
        right_schema_input = traj_cols
        right_kwargs = left_kwargs
    else:
        right_schema_input = right_traj_cols
        right_kwargs = {}
    right_schema_hint = ""
    if right_traj_cols is None:
        right_schema_hint = (
            " The shared traj_cols/kwargs mapping appears not to fit the truth table. "
            "Pass right_traj_cols for the truth table, or make both tables use the same relevant "
            "column names for time, duration, and location."
        )

    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, left_kwargs, warn=False)
    temp_traj_cols = loader._parse_traj_cols(truth.columns, right_schema_input, right_kwargs, warn=False)

    left_loc = traj_cols["location_id"]
    right_loc = temp_traj_cols["location_id"]
    if left_loc not in stops.columns:
        raise ValueError(
            "Could not find the mapped location column in predicted stops. "
            f"Expected '{left_loc}' in columns {list(stops.columns)}."
        )
    if right_loc not in truth.columns:
        raise ValueError(
            "Could not find the mapped location column in ground-truth stops. "
            f"Expected '{right_loc}' in columns {list(truth.columns)}."
            + right_schema_hint
        )

    stops_clean = stops.copy()
    truth_clean = truth.copy()
    stops_clean[left_loc] = stops_clean[left_loc].fillna("Street")
    truth_clean[right_loc] = truth_clean[right_loc].fillna("Street")
    truth_buildings = truth[truth[right_loc].notna()].copy()

    overlaps = contact.overlapping_visits(
        left=stops_clean,
        right=truth_clean,
        match_location=True,
        traj_cols=input_traj_cols,
        right_traj_cols=right_traj_cols,
        **kwargs,
    )

    left_t_name, left_use_datetime = loader._fallback_time_cols_dt(stops_clean.columns, input_traj_cols, left_kwargs)
    left_t_key = traj_cols[left_t_name]
    left_e_t_key = traj_cols["end_datetime" if left_use_datetime else "end_timestamp"]
    left_end_col_present = loader._has_end_cols(stops_clean.columns, traj_cols)
    left_duration_col_present = loader._has_duration_cols(stops_clean.columns, traj_cols)
    if not (left_end_col_present or left_duration_col_present):
        raise ValueError("Predicted stops must provide either an end time or a duration.")
    if not left_end_col_present:
        if left_use_datetime:
            stops_clean[left_e_t_key] = stops_clean[left_t_key] + pd.to_timedelta(stops_clean[traj_cols["duration"]], unit="m")
        else:
            stops_clean[left_e_t_key] = stops_clean[left_t_key] + stops_clean[traj_cols["duration"]] * 60

    right_t_name, right_use_datetime = loader._fallback_time_cols_dt(truth_clean.columns, right_schema_input, right_kwargs)
    right_t_key = temp_traj_cols[right_t_name]
    right_e_t_key = temp_traj_cols["end_datetime" if right_use_datetime else "end_timestamp"]
    right_end_col_present = loader._has_end_cols(truth_clean.columns, temp_traj_cols)
    right_duration_col_present = loader._has_duration_cols(truth_clean.columns, temp_traj_cols)
    if not (right_end_col_present or right_duration_col_present):
        raise ValueError("Ground-truth stops must provide either an end time or a duration." + right_schema_hint)
    if not right_end_col_present:
        if right_use_datetime:
            truth_clean[right_e_t_key] = truth_clean[right_t_key] + pd.to_timedelta(truth_clean[temp_traj_cols["duration"]], unit="m")
        else:
            truth_clean[right_e_t_key] = truth_clean[right_t_key] + truth_clean[temp_traj_cols["duration"]] * 60

    left_duration = traj_cols["duration"]
    if left_duration in stops_clean.columns:
        total_pred = stops_clean[left_duration].sum()
    else:
        if left_use_datetime:
            total_pred = (filters.to_timestamp(stops_clean[left_e_t_key]) - filters.to_timestamp(stops_clean[left_t_key])).floordiv(60).sum()
        else:
            total_pred = ((stops_clean[left_e_t_key] - stops_clean[left_t_key]) // 60).sum()

    right_duration = temp_traj_cols["duration"]
    if right_duration in truth_clean.columns:
        total_truth = truth_clean[right_duration].sum()
    else:
        if right_use_datetime:
            total_truth = (filters.to_timestamp(truth_clean[right_e_t_key]) - filters.to_timestamp(truth_clean[right_t_key])).floordiv(60).sum()
        else:
            total_truth = ((truth_clean[right_e_t_key] - truth_clean[right_t_key]) // 60).sum()

    tp = overlaps[left_duration].sum()
    prf_metrics = contact.precision_recall_f1_from_minutes(total_pred, total_truth, tp)

    if prf_only:
        return {**prf_metrics, "user_id": user_id, "algorithm": algorithm}

    if len(truth_buildings) > 0:
        overlaps_err = contact.overlapping_visits(
            left=stops_clean,
            right=truth_buildings,
            match_location=False,
            traj_cols=input_traj_cols,
            right_traj_cols=right_traj_cols,
            **kwargs,
        )
        error_metrics = compute_visitation_errors(
            overlaps_err,
            truth_buildings,
            traj_cols=input_traj_cols,
            right_traj_cols=right_traj_cols,
            **kwargs,
        )
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
    "compute_visitation_errors",
    "compute_stop_detection_metrics",
    "plot_metric_vs_param",
    "plot_family_timing",
]
