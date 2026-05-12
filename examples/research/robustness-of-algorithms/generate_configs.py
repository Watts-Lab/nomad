import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nomad.data as data_folder


def main():
    key = None
    if len(sys.argv) > 1:
        key = sys.argv[1]

    output_dir = Path(__file__).resolve().parent
    data_dir = Path(data_folder.__file__).parent

    registry = {}

    ######################################################################
    # Destination diary: four-stop balanced
    # Used for fig_1 in the paper and less basic trajectory experiments.
    ######################################################################
    start = "2024-06-01 00:00-04:00"
    start_time = pd.date_range(start=start, periods=4, freq="60min")
    unix_timestamp = [int(t.timestamp()) for t in start_time]
    duration = [60] * 4
    location = ["h-x14-y11", "h-x14-y9", "w-x17-y10", "w-x17-y8"]
    destinations = pd.DataFrame(
        {
            "datetime": start_time,
            "timestamp": unix_timestamp,
            "duration": duration,
            "location": location,
        }
    )
    destinations.to_csv(output_dir / "exp_1_destinations_balanced.csv", index=False)

    ######################################################################
    # Destination diary: four-stop unbalanced
    # Used for fig_1 in the paper and less basic trajectory experiments.
    ######################################################################
    start = "2024-06-01 00:00-04:00"
    start_time = pd.date_range(start=start, periods=8, freq="30min")
    unix_timestamp = [int(t.timestamp()) for t in start_time]
    duration = [30] * 8
    location = [
        "h-x14-y11",
        "h-x14-y11",
        "h-x14-y9",
        "w-x17-y10",
        "w-x17-y10",
        "w-x17-y8",
        "w-x17-y8",
        "w-x17-y8",
    ]
    destinations = pd.DataFrame(
        {
            "datetime": start_time,
            "timestamp": unix_timestamp,
            "duration": duration,
            "location": location,
        }
    )
    destinations.to_csv(output_dir / "exp_1_destinations_unbalanced.csv", index=False)

    ######################################################################
    # Destination diary: two-stop
    ######################################################################
    start = "2024-06-01 00:00-04:00"
    start_time = pd.date_range(start=start, periods=2, freq="90min")
    unix_timestamp = [int(t.timestamp()) for t in start_time]
    duration = [90] * 2
    location = ["w-x17-y10", "r-x19-y11"]
    destinations = pd.DataFrame(
        {
            "datetime": start_time,
            "timestamp": unix_timestamp,
            "duration": duration,
            "location": location,
        }
    )
    destinations.to_csv(output_dir / "exp_2_stops.csv", index=False)

    ######################################################################
    # Destination diary: two-stop aws
    ######################################################################
    start = "2024-06-01 00:00-04:00"
    start_time = pd.date_range(start=start, periods=2, freq="180min")
    unix_timestamp = [int(t.timestamp()) for t in start_time]
    duration = [180] * 2
    location = ["w-x17-y10", "r-x19-y11"]
    destinations = pd.DataFrame(
        {
            "datetime": start_time,
            "timestamp": unix_timestamp,
            "duration": duration,
            "location": location,
        }
    )
    destinations.to_csv(output_dir / "exp_2_stops_aws.csv", index=False)

    ######################################################################
    # Config family: four-stop
    ######################################################################
    N_reps = 5
    sparsity_samples = 10
    N = N_reps * sparsity_samples
    beta_ping = np.repeat(np.linspace(4, 9, sparsity_samples), N_reps).tolist()

    registry["low_ha"] = {
        "dt": 0.25,
        "N": N,
        "name_count": 2,
        "name_seed": 2025,
        "city_file": str(data_dir / "garden-city.gpkg"),
        "buildings_file": str(data_dir / "garden-city-buildings-mercator.parquet"),
        "destination_diary_file": str(output_dir / "exp_1_destinations_balanced.csv"),
        "output_files": {
            "sparse_path": str(output_dir / "sparse_traj_1"),
            "diaries_path": str(output_dir / "diaries_1"),
            "homes_path": str(output_dir / "homes_1"),
        },
        "agent_params": {
            "agent_homes": "h-x14-y11",
            "agent_workplaces": "w-x17-y8",
            "seed_trajectory": list(range(N)),
            "seed_sparsity": list(range(N)),
            "beta_ping": beta_ping,
            "beta_durations": None,
            "beta_start": None,
            "ha": 11.5 / 15,
        },
    }

    registry["high_ha"] = {
        "dt": 0.5,
        "N": N,
        "name_count": 2,
        "name_seed": 2025,
        "city_file": str(data_dir / "garden-city.gpkg"),
        "buildings_file": str(data_dir / "garden-city-buildings-mercator.parquet"),
        "destination_diary_file": str(output_dir / "exp_1_destinations_unbalanced.csv"),
        "output_files": {
            "sparse_path": str(output_dir / "sparse_traj_2"),
            "diaries_path": str(output_dir / "diaries_2"),
            "homes_path": str(output_dir / "homes_2"),
        },
        "agent_params": {
            "agent_homes": "h-x14-y11",
            "agent_workplaces": "w-x17-y8",
            "seed_trajectory": list(range(N)),
            "seed_sparsity": list(range(N)),
            "beta_ping": beta_ping,
            "beta_durations": None,
            "beta_start": None,
            "ha": 15 / 15,
        },
    }

    ######################################################################
    # Config family: random general trajectories
    # Agents receive random home/workplace; destinations are generated by
    # the EPR gravity model (no fixed diary).  Intended as the main
    # multi-stop, multi-user dataset for validation experiments.
    ######################################################################
    N_random = 250
    beta_ping_random = np.linspace(4, 9, N_random).tolist()

    registry["random_general"] = {
        "N":            N_random,
        "name_count":   3,
        "name_seed":    2025,
        "city_file":    str(data_dir / "garden-city.gpkg"),
        "start_time":   "2024-06-01T00:00:00-04:00",
        "end_time":     "2024-06-08T00:00:00-04:00",
        "epr_time_res": 15,
        "dt":           1,
        "output_files": {
            "sparse_path":  str(output_dir / "sparse_traj_2"),
            "diaries_path": str(output_dir / "diaries_2"),
            "homes_path":   str(output_dir / "homes_2"),
        },
        "agent_params": {
            "seed_trajectory": list(range(N_random)),
            "seed_sparsity":   list(range(N_random)),
            "beta_ping":       beta_ping_random,
            "ha":              3 / 4,
        },
    }

    ######################################################################
    # Config family: two-stop
    # Small comparisons and prototypes.
    ######################################################################
    registry["2_stops"] = {
        "dt": 0.2,
        "N": 250,
        "name_count": 2,
        "name_seed": 2025,
        "city_file": str(data_dir / "garden-city.gpkg"),
        "buildings_file": str(data_dir / "garden-city-buildings-mercator.parquet"),
        "destination_diary_file": str(output_dir / "exp_2_stops.csv"),
        "output_files": {
            "sparse_path": str(output_dir / "sparse_default"),
            "diaries_path": str(output_dir / "diaries_default"),
            "homes_path": str(output_dir / "homes_default"),
        },
        "agent_params": {
            "agent_homes": "h-x14-y11",
            "agent_workplaces": "w-x17-y8",
            "seed_trajectory": list(range(250)),
            "seed_sparsity": list(range(250)),
            "beta_ping": 7,
            "beta_durations": None,
            "beta_start": None,
            "ha": 15 / 15,
        },
    }

    registry["beta_5"] = {
        "dt": 0.2,
        "N": 250,
        "name_count": 2,
        "name_seed": 2025,
        "city_file": str(data_dir / "garden-city.gpkg"),
        "buildings_file": str(data_dir / "garden-city-buildings-mercator.parquet"),
        "destination_diary_file": str(output_dir / "exp_2_stops.csv"),
        "output_files": {
            "sparse_path": str(output_dir / "sparse_beta_5"),
            "diaries_path": str(output_dir / "diaries_beta_5"),
            "homes_path": str(output_dir / "homes_beta_5"),
        },
        "agent_params": {
            "agent_homes": "h-x14-y11",
            "agent_workplaces": "w-x17-y8",
            "seed_trajectory": list(range(250)),
            "seed_sparsity": list(range(250)),
            "beta_ping": 5,
            "beta_durations": None,
            "beta_start": None,
            "ha": 15 / 15,
        },
    }

    registry["beta_8_5"] = {
        "dt": 0.2,
        "N": 250,
        "name_count": 2,
        "name_seed": 2025,
        "city_file": str(data_dir / "garden-city.gpkg"),
        "buildings_file": str(data_dir / "garden-city-buildings-mercator.parquet"),
        "destination_diary_file": str(output_dir / "exp_2_stops.csv"),
        "output_files": {
            "sparse_path": str(output_dir / "sparse_beta_8_5"),
            "diaries_path": str(output_dir / "diaries_beta_8_5"),
            "homes_path": str(output_dir / "homes_beta_8_5"),
        },
        "agent_params": {
            "agent_homes": "h-x14-y11",
            "agent_workplaces": "w-x17-y8",
            "seed_trajectory": list(range(250)),
            "seed_sparsity": list(range(250)),
            "beta_ping": 8.5,
            "beta_durations": None,
            "beta_start": None,
            "ha": 15 / 15,
        },
    }

    ######################################################################
    # Config family: two-stop aws sweep
    # Large-scale comparisons (2,000 users per parameter).
    ######################################################################
    ha_values = [13 / 15, 15 / 15, 17 / 15]
    beta_values = [5, 6, 7, 8]

    for ha in ha_values:
        for beta in beta_values:
            k = f"ha{round(ha * 15)}_beta{beta}"
            registry[k] = {
                "dt": 0.2,
                "N": 2000,
                "name_count": 3,
                "name_seed": 2025,
                "city_file": str(data_dir / "garden-city.gpkg"),
                "buildings_file": str(data_dir / "garden-city-buildings-mercator.parquet"),
                "destination_diary_file": str(output_dir / "exp_2_stops_aws.csv"),
                "output_files": {
                    "sparse_path": str(output_dir / f"sparse_{k}"),
                    "diaries_path": str(output_dir / f"diaries_{k}"),
                    "homes_path": str(output_dir / f"homes_{k}"),
                },
                "agent_params": {
                    "agent_homes": "h-x14-y11",
                    "agent_workplaces": "w-x17-y8",
                    "seed_trajectory": list(range(2000)),
                    "seed_sparsity": list(range(2000)),
                    "beta_ping": beta,
                    "beta_durations": None,
                    "beta_start": None,
                    "ha": ha,
                },
            }

    if key is not None:
        with open(output_dir / f"config_{key}.json", "w", encoding="utf-8") as f:
            json.dump(registry[key], f, ensure_ascii=False, indent=4)
        print(f"Wrote config_{key}.json")
        return

    for k in sorted(registry.keys()):
        with open(output_dir / f"config_{k}.json", "w", encoding="utf-8") as f:
            json.dump(registry[k], f, ensure_ascii=False, indent=4)
        print(f"Wrote config_{k}.json")


if __name__ == "__main__":
    main()
