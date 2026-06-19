"""
Golden test and runtime benchmark for generate_dest_diary optimizations.

Baseline uses nomad.traj_gen.Agent.generate_dest_diary (current rework flow).
Optimized uses generate_dest_diary_optimized defined in this file.

Run from repo root:
    PYTHONPATH=. python examples/research/virtual_philadelphia/golden_dest_diary_benchmark.py
    PYTHONPATH=. python examples/research/virtual_philadelphia/golden_dest_diary_benchmark.py --update-golden
    PYTHONPATH=. python examples/research/virtual_philadelphia/golden_dest_diary_benchmark.py --n-agents 5 --n-repeat 2
"""

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import numpy.random as npr
import pandas as pd

from nomad.city_gen import load as load_city_pickle
from nomad.constants import ALLOWED_BUILDINGS, DEFAULT_STAY_PROBS
from nomad.filters import to_timestamp
from nomad.traj_gen import Agent, _gravity_probs_for_unvisited

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output_rework"
CITY_CACHE = OUTPUT_DIR / "raster_city_large.pkl"
AGENT_PARAMS_PATH = OUTPUT_DIR / "agent_params_large.parquet"
GOLDEN_DIR = HERE / "golden"

EPR_PARAMS = {
    "datetime": "2025-05-23 00:00-05:00",
    "end_time": "2025-07-01 00:00-05:00",
    "epr_time_res": 15,
    "rho": 0.4,
    "gamma": 0.3,
}


def load_city_from_cache():
    """Load cached raster city, matching the rework notebook."""
    city = load_city_pickle(str(CITY_CACHE))
    if getattr(city, "mh_dist_nearby_doors", None) is not None and len(city.mh_dist_nearby_doors) > 0:
        if city.grav is None or city.grav_for_candidates is None:
            city.restore_gravity(exponent=2.0, callable_only=True)
    elif city.grav is None or city.grav_for_candidates is None:
        city.compute_gravity(exponent=2.0, callable_only=True)
    if city.shortest_paths is None:
        city.compute_shortest_paths(callable_only=True)
    return city


def make_agent(city, row):
    """Create a fresh agent matching the rework notebook setup."""
    return Agent(
        identifier=row.identifier,
        city=city,
        home=row.home,
        workplace=row.workplace,
        datetime=row.datetime,
    )


def _append_dest_segment(dest_update, entry):
    """Merge consecutive rows for the same location (optimization #1)."""
    if dest_update and dest_update[-1]["location"] == entry["location"]:
        dest_update[-1]["duration"] += entry["duration"]
    else:
        dest_update.append(entry)


def _build_epr_arrays(visit_freqs):
    """Extract NumPy arrays and per-hour allowed indices from visit_freqs."""
    building_ids = visit_freqs.index.to_numpy()
    building_types = visit_freqs["building_type"].to_numpy()
    freq = visit_freqs["freq"].to_numpy(dtype=np.int64, copy=True)
    type_labels, building_type_codes = np.unique(building_types, return_inverse=True)
    type_to_code = {label: code for code, label in enumerate(type_labels)}

    allowed_idx_by_hour = []
    for hour in range(24):
        allowed_codes = np.array([type_to_code[t] for t in ALLOWED_BUILDINGS[hour]], dtype=np.int64)
        allowed_idx_by_hour.append(np.where(np.isin(building_type_codes, allowed_codes))[0])

    return building_ids, building_types, freq, building_type_codes, allowed_idx_by_hour


def _sync_visit_freqs(building_ids, building_types, freq):
    """Rebuild visit_freqs DataFrame from NumPy arrays."""
    visit_freqs = pd.DataFrame(
        {"building_type": building_types, "freq": freq},
        index=building_ids,
    )
    visit_freqs.index.name = None
    return visit_freqs


def _choose_destination_idx(visited_idx, freq, building_ids, rng):
    """Preferential return over visited candidate indices."""
    if visited_idx.size > 0:
        weights = freq[visited_idx]
        if weights.sum() > 0:
            return int(rng.choice(visited_idx, p=weights / weights.sum()))
    return int(rng.choice(np.arange(building_ids.size)))


def _finalize_destination_diary(agent, dest_update):
    """Build destination diary without post-hoc condense_destinations."""
    if agent.destination_diary.empty:
        agent.destination_diary = pd.DataFrame(dest_update)
        return

    existing = agent.destination_diary.copy()
    if dest_update and existing.iloc[-1]["location"] == dest_update[0]["location"]:
        existing.iloc[-1, existing.columns.get_loc("duration")] += dest_update[0]["duration"]
        dest_update = dest_update[1:]

    if dest_update:
        agent.destination_diary = pd.concat(
            [existing, pd.DataFrame(dest_update)],
            ignore_index=True,
        )
    else:
        agent.destination_diary = existing


def generate_dest_diary_optimized(
    agent,
    end_time,
    epr_time_res=15,
    stay_probs=DEFAULT_STAY_PROBS,
    rho=0.4,
    gamma=0.3,
    seed=0,
):
    """
    Experimental destination diary generator.

    Current optimizations applied here:
    - #1 segment accumulation (no per-tick append + post-hoc condense)
    - #2 NumPy arrays for visit_freqs inner loop (no per-tick pandas filtering)
    - #3 candidate-only gravity via nomad.traj_gen._gravity_probs_for_unvisited
    """
    rng = npr.default_rng(seed)

    if agent.city.grav is None:
        raise RuntimeError("city.grav is not available. Call city.compute_gravity() before trajectory generation.")
    if agent.last_ping is None:
        raise RuntimeError("Agent has no last_ping.")

    if end_time.tz is None:
        tz = getattr(agent.last_ping["datetime"], "tz", None)
        if tz is not None:
            end_time = end_time.tz_localize(tz)
    if isinstance(end_time, pd.Timestamp):
        end_time = to_timestamp(end_time)

    visit_freqs = agent.visit_freqs
    if (visit_freqs is None) or (not (visit_freqs["freq"] > 0).any()):
        visit_freqs = agent._initialize_visits_unif(
            rng=rng,
            home_work_freq=20,
            initial_k={"retail": 4, "workplace": 2, "home": 2, "park": 2},
            other_locs_freq=2,
        )

    if agent.destination_diary.empty:
        start_time_local = agent.last_ping["datetime"]
        start_time = agent.last_ping["timestamp"]
        curr_info = agent.city.get_block(
            (int(np.floor(agent.last_ping["x"])), int(np.floor(agent.last_ping["y"])))
        )
        curr = (
            curr_info["building_id"]
            if curr_info["building_type"] is not None
            and curr_info["building_type"] != "street"
            and curr_info["building_id"] is not None
            else agent.home
        )
    else:
        last_entry = agent.destination_diary.iloc[-1]
        last_datetime = (
            pd.to_datetime(last_entry.datetime)
            if not isinstance(last_entry.datetime, pd.Timestamp)
            else last_entry.datetime
        )
        start_time_local = last_datetime + timedelta(minutes=int(last_entry.duration))
        if "timestamp" in agent.destination_diary.columns:
            start_time = int(last_entry.timestamp + last_entry.duration * 60)
        else:
            start_time = to_timestamp(last_entry.datetime) + int(last_entry.duration * 60)
        curr = last_entry.location

    if start_time > end_time:
        raise ValueError(
            f"Agent {agent.identifier}: last_ping timestamp ({start_time}) is at or beyond end_time ({end_time})."
        )

    building_ids, building_types, freq, _, allowed_idx_by_hour = _build_epr_arrays(
        visit_freqs
    )
    bid_to_idx = {bid: i for i, bid in enumerate(building_ids)}
    curr_idx = bid_to_idx[curr]

    dest_update = []
    while start_time < end_time:
        allowed_idx = allowed_idx_by_hour[start_time_local.hour]
        allowed_freq = freq[allowed_idx]
        visited_idx = allowed_idx[allowed_freq > 0]
        unvisited_idx = allowed_idx[allowed_freq == 0]

        S = visited_idx.size if visited_idx.size > 0 else 1
        p_exp = rho * (S ** (-gamma))

        curr_type = building_types[curr_idx] if curr_idx >= 0 else "home"
        allowed_types = ALLOWED_BUILDINGS[start_time_local.hour]

        u_stay = rng.uniform()
        if (curr_type in allowed_types) & (u_stay < stay_probs.get(curr_type, 0.5)):
            pass
        elif rng.uniform() < p_exp:
            if unvisited_idx.size > 0:
                probs = _gravity_probs_for_unvisited(
                    agent.city, curr_idx, unvisited_idx, building_ids
                )
                probs = probs / probs.sum()
                curr_idx = int(rng.choice(unvisited_idx, p=probs))
            else:
                curr_idx = _choose_destination_idx(visited_idx, freq, building_ids, rng)
            freq[curr_idx] += 1
        else:
            curr_idx = _choose_destination_idx(visited_idx, freq, building_ids, rng)
            freq[curr_idx] += 1

        _append_dest_segment(
            dest_update,
            {
                "datetime": start_time_local,
                "timestamp": start_time,
                "duration": epr_time_res,
                "location": building_ids[curr_idx],
            },
        )

        start_time_local = start_time_local + timedelta(minutes=int(epr_time_res))
        start_time = start_time + epr_time_res * 60

    _finalize_destination_diary(agent, dest_update)
    agent.visit_freqs = _sync_visit_freqs(building_ids, building_types, freq)


def run_baseline(agent, end_time, dest_seed):
    """Run current nomad generate_dest_diary and return outputs + elapsed seconds."""
    t0 = time.perf_counter()
    agent.generate_dest_diary(
        end_time=end_time,
        epr_time_res=EPR_PARAMS["epr_time_res"],
        rho=EPR_PARAMS["rho"],
        gamma=EPR_PARAMS["gamma"],
        seed=int(dest_seed),
    )
    elapsed = time.perf_counter() - t0
    return agent.destination_diary.copy(), agent.visit_freqs.copy(), elapsed


def run_optimized(agent, end_time, dest_seed):
    """Run experimental optimized generator and return outputs + elapsed seconds."""
    t0 = time.perf_counter()
    generate_dest_diary_optimized(
        agent,
        end_time=end_time,
        epr_time_res=EPR_PARAMS["epr_time_res"],
        stay_probs=DEFAULT_STAY_PROBS,
        rho=EPR_PARAMS["rho"],
        gamma=EPR_PARAMS["gamma"],
        seed=int(dest_seed),
    )
    elapsed = time.perf_counter() - t0
    return agent.destination_diary.copy(), agent.visit_freqs.copy(), elapsed


def assert_outputs_equal(actual_dest, actual_freqs, expected_dest, expected_freqs):
    """Assert destination diary and visit_freqs match golden reference."""
    pd.testing.assert_frame_equal(
        actual_dest.reset_index(drop=True),
        expected_dest.reset_index(drop=True),
        check_dtype=False,
    )
    actual_freqs = actual_freqs.copy()
    expected_freqs = expected_freqs.copy()
    actual_freqs.index = actual_freqs.index.astype(str)
    expected_freqs.index = expected_freqs.index.astype(str)
    pd.testing.assert_frame_equal(actual_freqs, expected_freqs, check_dtype=False)


def save_golden(dest_diary, visit_freqs, agent_id):
    """Write golden reference outputs."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    dest_diary.to_parquet(GOLDEN_DIR / f"dest_diary_{agent_id}.parquet", index=False)
    visit_freqs.reset_index(names="id").to_parquet(GOLDEN_DIR / f"visit_freqs_{agent_id}.parquet", index=False)


def load_golden(agent_id):
    """Load golden reference outputs."""
    dest_diary = pd.read_parquet(GOLDEN_DIR / f"dest_diary_{agent_id}.parquet")
    visit_freqs = pd.read_parquet(GOLDEN_DIR / f"visit_freqs_{agent_id}.parquet").set_index("id")
    visit_freqs.index.name = None
    return dest_diary, visit_freqs


def benchmark_agents(city, agent_params, end_time, n_agents, n_repeat):
    """Time baseline vs optimized across multiple agents."""
    rows = agent_params.iloc[:n_agents]
    baseline_times = []
    optimized_times = []

    for _ in range(n_repeat):
        for _, row in rows.iterrows():
            agent = make_agent(city, row)
            _, _, baseline_elapsed = run_baseline(agent, end_time, row.dest_seed)
            baseline_times.append(baseline_elapsed)

            agent = make_agent(city, row)
            _, _, optimized_elapsed = run_optimized(agent, end_time, row.dest_seed)
            optimized_times.append(optimized_elapsed)

    baseline_arr = np.array(baseline_times)
    optimized_arr = np.array(optimized_times)
    speedup = baseline_arr.sum() / optimized_arr.sum()

    print(f"\nBenchmark ({n_agents} agents x {n_repeat} repeats = {len(baseline_times)} runs each)")
    print(f"  baseline : {baseline_arr.mean():.3f}s mean  ({baseline_arr.sum():.2f}s total)")
    print(f"  optimized: {optimized_arr.mean():.3f}s mean  ({optimized_arr.sum():.2f}s total)")
    print(f"  speedup  : {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Golden test and dest_diary benchmark")
    parser.add_argument("--update-golden", action="store_true", help="Regenerate golden files from baseline")
    parser.add_argument("--golden-agent", default="clever_colden", help="Agent identifier for golden test")
    parser.add_argument("--n-agents", type=int, default=3, help="Agents to include in runtime benchmark")
    parser.add_argument("--n-repeat", type=int, default=1, help="Repeat count per agent for runtime benchmark")
    parser.add_argument("--skip-benchmark", action="store_true", help="Only run golden correctness check")
    args = parser.parse_args()

    if not CITY_CACHE.exists():
        print(f"Missing city cache: {CITY_CACHE}", file=sys.stderr)
        sys.exit(1)
    if not AGENT_PARAMS_PATH.exists():
        print(f"Missing agent params: {AGENT_PARAMS_PATH}", file=sys.stderr)
        sys.exit(1)

    city = load_city_from_cache()
    agent_params = pd.read_parquet(AGENT_PARAMS_PATH)
    end_time = pd.Timestamp(EPR_PARAMS["end_time"])

    golden_row = agent_params.loc[agent_params["identifier"] == args.golden_agent]
    if golden_row.empty:
        print(f"Golden agent not found: {args.golden_agent}", file=sys.stderr)
        sys.exit(1)
    golden_row = golden_row.iloc[0]

    if args.update_golden:
        agent = make_agent(city, golden_row)
        dest_diary, visit_freqs, elapsed = run_baseline(agent, end_time, golden_row.dest_seed)
        save_golden(dest_diary, visit_freqs, args.golden_agent)
        print(f"Golden updated for {args.golden_agent} ({len(dest_diary)} dest entries, {elapsed:.2f}s)")
        return

    golden_dest, golden_freqs = load_golden(args.golden_agent)

    agent = make_agent(city, golden_row)
    opt_dest, opt_freqs, _ = run_optimized(agent, end_time, golden_row.dest_seed)
    assert_outputs_equal(opt_dest, opt_freqs, golden_dest, golden_freqs)
    print(f"Golden match: optimized output equals baseline for {args.golden_agent}")
    print(f"  dest entries: {len(opt_dest)}")
    print(f"  visited buildings (freq > 0): {(opt_freqs['freq'] > 0).sum()}")

    if not args.skip_benchmark:
        benchmark_agents(city, agent_params, end_time, args.n_agents, args.n_repeat)


if __name__ == "__main__":
    main()
