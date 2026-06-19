Getting Started
===============

What is NOMAD?
--------------

**NOMAD** (Network for Open Mobility Analysis and Data) is an open-source Python library for end-to-end processing of large-scale GPS mobility data. The library supports:

* Data ingestion from various formats (CSV, Parquet, GeoJSON)
* Quality control and filtering
* Spatial-temporal transformation
* Derivation of mobility metrics
* Synthetic trajectory generation

NOMAD is part of the NSF-funded NOMAD research infrastructure project and builds on previous software resources like *scikit-mobility*, *mobilkit*, and *trackintel*. The goal is to provide a single, production-ready library that covers the entire processing pipeline and is suitable for analysis of massive datasets.

Key Features
------------

* **Dual API**: All functions implemented in Python with parallel equivalents in PySpark
* **Scalable**: Same analysis notebook runs on a workstation or Spark cluster without API changes
* **Production-ready**: Designed for large-scale mobility data processing
* **Research-focused**: Aids in the replicability of existing research

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

Install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/Watts-Lab/nomad.git

With Spark Support
~~~~~~~~~~~~~~~~~~

For large-scale data processing with PySpark:

.. code-block:: bash

   pip install git+https://github.com/Watts-Lab/nomad.git#egg=nomad[spark]

Requirements
~~~~~~~~~~~~

* Python 3.9 or higher
* Core dependencies: pandas, geopandas, numpy, shapely, networkx
* Optional: PySpark 3.4.4+ for distributed processing

Quick Start
-----------

Here's a simple example to get started with NOMAD:

.. code-block:: python

   import nomad.io.base as loader
   
   # Load mobility data
   df = loader.read_data('data/sample.csv')
   
   # Perform stop detection
   from nomad.stop_detection import dbscan
   stops = dbscan.detect_stops(df, eps=100, min_samples=3)
   
   # Compute mobility metrics
   from nomad.metrics import metrics
   metrics_df = metrics.compute_radius_of_gyration(df)

Next Steps
----------

* :doc:`data_ingestion` - Learn how to load and validate your mobility data
* :doc:`filtering` - Filter and assess data completeness
* :doc:`stop_detection` - Detect stops and trips from GPS traces
* :doc:`visit_attribution` - Infer home and workplace locations
* :doc:`metrics` - Compute mobility metrics
* :doc:`synthetic_data_generation` - Generate synthetic trajectories

Examples
--------

The ``examples/`` folder contains Jupyter notebooks demonstrating various NOMAD features:

* **[1] Reading Data** - Loading different data formats
* **[2] Filtering** - Data quality control and filtering
* **[3] Stop Detection** - Different stop detection algorithms
* **[4] Home Attribution** - Inferring residential locations

License
-------

MIT © University of Pennsylvania 2025

For more information, visit `nomad.seas.upenn.edu <https://nomad.seas.upenn.edu>`_.

