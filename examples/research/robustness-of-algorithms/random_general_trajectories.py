"""
Generate random general trajectories for robustness-of-algorithms experiments.

Differences from four_stop_trajectories_exp1:
  - Each agent gets a randomly-assigned home and workplace (no fixed diary).
  - Destinations are produced by the EPR gravity model, giving each agent a
    unique multi-day itinerary across all building types.
  - beta_ping is swept linearly across agents to produce a spread of sparsity
    levels (q values) within the same population.

Reads:   config_random_general.json  (produced by generate_configs.py)
Writes:  sparse_traj_2/   — sampled sparse trajectories (Parquet)
         diaries_2/       — ground-truth visit diaries   (Parquet)
         homes_2/         — per-agent home/workplace table (Parquet)

Usage:
    python generate_configs.py random_general   # write config
    python random_general_trajectories.py       # generate data
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from nomad.city_gen import City
from nomad.traj_gen import Population


def main():
    config_file = Path(__file__).resolve().parent / "config_random_general.json"
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    par = config["agent_params"]
    out = config["output_files"]

    # ── City setup ─────────────────────────────────────────────────────────
    # EPR destination generation requires city.grav; _build_hub_network and
    # compute_gravity are called explicitly in case the saved geopackage does
    # not already contain pre-computed gravity data.
    city = City.from_geopackage(config["city_file"])
    city._build_hub_network(hub_size=16)
    if city.grav is None:
        city.compute_gravity(exponent=2.0, use_proxy_hub_distance=True)
    city.compute_shortest_paths(callable_only=True)

    # POI coordinates used when reprojecting diaries to Mercator.
    poi_data = pd.DataFrame({
        "building_id": city.buildings_gdf["id"].values,
        "x": city.buildings_gdf["door_point"].apply(lambda p: p[0]).values,
        "y": city.buildings_gdf["door_point"].apply(lambda p: p[1]).values,
    })

    start_time = pd.Timestamp(config["start_time"])
    end_time   = pd.Timestamp(config["end_time"])

    # ── Population ─────────────────────────────────────────────────────────
    # home=None and workplace=None (defaults) → each Agent randomly samples
    # its own home and workplace from the city's building catalogue, seeded
    # by the agent's individual seed so results are reproducible.
    population = Population(city)
    population.generate_agents(
        N=config["N"],
        seed=config["name_seed"],
        name_count=config["name_count"],
    )

    beta_ping = par["beta_ping"]  # list of N values (linspace 4→9)

    for i, agent in enumerate(tqdm(population.roster.values(), desc="Generating trajectories")):
        # Full trajectory via EPR model.  Passing datetime as a kwarg
        # sets the agent's initial position timestamp to start_time.
        agent.generate_trajectory(
            datetime=start_time,
            end_time=end_time,
            epr_time_res=config["epr_time_res"],
            dt=config["dt"],
            seed=par["seed_trajectory"][i],
        )

        agent.sample_trajectory(
            beta_ping=beta_ping[i] if isinstance(beta_ping, list) else beta_ping,
            ha=par["ha"],
            seed=par["seed_sparsity"][i],
            replace_sparse_traj=True,
        )

    # ── Reproject & save ───────────────────────────────────────────────────
    population.reproject_to_mercator(
        sparse_traj=True,
        full_traj=False,
        diaries=True,
        poi_data=poi_data,
    )
    population.save_pop(
        sparse_path=out["sparse_path"],
        diaries_path=out["diaries_path"],
        homes_path=out["homes_path"],
        beta_ping=beta_ping,
        ha=par["ha"],
    )
    print(f"Saved {config['N']} agents")
    print(f"  sparse → {out['sparse_path']}")
    print(f"  diaries → {out['diaries_path']}")


if __name__ == "__main__":
    main()
