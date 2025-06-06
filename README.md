# Nomad: Network for Open Mobility Analysis and Data
This repository hosts the open‑source software library of the NSF-funded NOMAD research‑infrastructure project (nomad.seas.upenn.edu). The code base supports end‑to‑end processing of large‑scale GPS mobility data: ingestion, quality control, spatial‑temporal transformation, derivation of mobility metrics, and synthetic trajectory generation. All functions are implemented in Python with parallel equivalents in PySpark, enabling the same analysis notebook to run on a workstation or a Spark cluster without API changes.

NOMAD builds on previous software resources—like *scikit‑mobility*, *mobilkit*, and *trackintel*—with the goal of providing a single, production‑ready library that covers the entire processing pipeline in a form suitable for analysis of massive datasets and to aid in the replicability of existing research.

## Modules 

| Module | Description |
|-------|---------|
| Data ingestion | Read CSV, Parquet, GeoJSON, and partitioned datasets; validation of data types; return Pandas or Spark DataFrames |
| Filtering & completeness | Quantify hourly/daily/weekly coverage, filtering according to completeness, geography, and time window; handles spatial projections and timezones|
| Tessellation | Map pings to H3, S2, or custom grids for grid-based algorithms |
| Stop / trip detection | Density-based algorithms and sequential algorithm (_Project Lachesis_) |
| Home / workplace inference | Frequency‑ and time‑window heuristics to assign residential and workplace locations |
| Mobility metrics | Radius of gyration, travel distance, time at home, entropy, and related indicators |
| Co‑location & contact networks | Build proximity graphs from POI visits or spatial–temporal proximity |
| POI attribution & generation | Match stops to existing POIs or cluster frequent stops to obtain unsupervised POI datasets|
| Aggregation & debiasing | differential‑privacy preserving aggregated metrics, and weights for debiasing and post‑stratification |
| Trajectory simulation | Exploration and Preferential Return models; Point process samplers to generate sparse signals |

Unit tests currently cover data ingestion, filtering, stop detection, and trajectory simulation.

## Installation
```bash
# install directly from GitHub
pip install git+https://github.com/Watts-Lab/nomad.git

# Spark extras (optional)
pip install git+https://github.com/Watts-Lab/nomad.git#egg=nomad[spark]
```

## Examples
The examples/ folder contains notebooks and small sample datasets that demonstrate loading partitioned data, measuring completeness, detecting stops, computing mobility metrics, and generating synthetic trajectories. Notebooks run unchanged in local Python or on Spark.

## Contribute 
For development clone the repository and ensure unit tests, located in `nomad\tests\' are passed before submitting a pull request.
## License 
MIT © University of Pennsylvania 2025.

Further information on the NOMAD Trusted Research Environment, data catalog, and community resources will be available at [https://nomad.seas.upenn.edu](https://nomad.seas.upenn.edu).
