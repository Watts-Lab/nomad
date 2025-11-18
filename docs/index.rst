.. NOMAD documentation master file

NOMAD: Network for Open Mobility Analysis and Data
===================================================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

**NOMAD** is an open-source Python library for end-to-end processing of large-scale GPS mobility data. 
Part of the NSF-funded NOMAD research infrastructure project, it provides production-ready tools for 
mobility data analysis with seamless scaling from local workstations to Spark clusters.

NOMAD builds on previous software resources—like scikit-mobility, mobilkit, and trackintel—with the 
goal of providing a single, production-ready library that covers the entire processing pipeline in a 
form suitable for analysis of massive datasets and to aid in the replicability of existing research.

All functions are implemented in Python with parallel equivalents in PySpark, enabling the same 
analysis notebook to run on a workstation or a Spark cluster without API changes.

Quick Links
-----------

* **GitHub**: `github.com/Watts-Lab/nomad <https://github.com/Watts-Lab/nomad>`_
* **Website**: `nomad.seas.upenn.edu <https://nomad.seas.upenn.edu>`_
* **Installation**: ``pip install git+https://github.com/Watts-Lab/nomad.git``

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/Watts-Lab/nomad.git

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   data_ingestion
   filtering
   tessellation
   stop_detection
   visit_attribution
   metrics
   colocation
   aggregation
   synthetic_data_generation

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   source/benchmarking_of_stop_detection_algorithms
   source/lachesis_demo
   source/tadbscan_demo
   source/hdbscan_demo
   source/grid_based_demo
   source/poi_synthetic
   source/poi_osm

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api_reference

Modules Overview
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Description
   * - :doc:`data_ingestion`
     - Read CSV, Parquet, GeoJSON; validate schemas; return Pandas or Spark DataFrames
   * - :doc:`filtering`
     - Assess coverage; filter by completeness, geography, time; handle projections
   * - :doc:`tessellation`
     - Map pings to H3, S2, or custom grids for grid-based algorithms
   * - :doc:`stop_detection`
     - DBSCAN, HDBSCAN, grid-based, and sequential (Lachesis) algorithms
   * - :doc:`visit_attribution`
     - Frequency and time-window heuristics for home/workplace inference
   * - :doc:`metrics`
     - Radius of gyration, travel distance, entropy, and related indicators
   * - :doc:`colocation`
     - Build proximity graphs from POI visits or spatial-temporal proximity
   * - :doc:`aggregation`
     - Differential privacy, k-anonymity, debiasing, and post-stratification
   * - :doc:`synthetic_data_generation`
     - EPR models and point process samplers for trajectory generation

Community & Support
-------------------

* **Issues**: Report bugs on `GitHub Issues <https://github.com/Watts-Lab/nomad/issues>`_
* **Contribute**: See our contribution guidelines
* **License**: MIT © University of Pennsylvania 2025

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
