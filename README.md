# Nomad: Network for Open Mobility Analysis and Data
This repository hosts the open‑source software library of the NSF-funded NOMAD research‑infrastructure project (nomad.seas.upenn.edu). The code base supports end‑to‑end processing of large‑scale GPS mobility data: ingestion, quality control, spatial‑temporal transformation, derivation of mobility metrics, and synthetic trajectory generation. All functions are implemented in Python with parallel equivalents in PySpark, enabling the same analysis notebook to run on a workstation or a Spark cluster without API changes.

NOMAD builds on previous software resources—like *scikit‑mobility*, *mobilkit*, and *trackintel*—with the goal of providing a single, production‑ready library that covers the entire processing pipeline in a form suitable for analysis of massive datasets and to aid in the replicability of existing research.

## Modules 

| Module | Description |
|-------|---------|
| Data ingestion | Ingestion and persistence of data in multiple spatio-temporal formats, including partitioned datasets and S3, through Pandas, PyArrow, and PySpark APIs. |
| Filtering & completeness | Completeness metrics, filtering, and format standardization for GPS data in Pandas, PyArrow, and PySpark, with shared schemas for in-memory and distributed workflows. |
| Tessellation | Map pings to H3, S2, or custom grids for grid-based algorithms |
| Stop detection | Methods that group pings into stops with 4 sequential and 4 density-based algorithms, support multiple spatio-temporal input formats, and return a unified stop-table schema using shared summarization helpers. |
| Visit, home, and workplace attribution | Attribution of stops to polygon layers using shapely spatial indices, plus daytime/nighttime multi-week filters for home and workplace inference. |
| Mobility metrics | Periodic mobility indicators from stop tables, including weighted radius of gyration, time at home, and self-containment metrics. |
| Co‑location estimates | Temporal joins across visit tables to estimate user co-location, also used by evaluation tooling for stop-detection methods against ground truth. |
| Mapping utilities | OpenStreetMap-based utilities for downloading and processing real city layers (boundaries, buildings, streets), plus synthetic city generation utilities. |
| Aggregation & debiasing | differential‑privacy preserving aggregated metrics, and weights for debiasing and post‑stratification |
| Mobility models and trajectory generation | Agent-based mobility models (including Exploration and Preferential Return) for city-scale trajectory simulation, with optional sparse burst/gap temporal sampling and horizontal-noise spatial models. |

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
