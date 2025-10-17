.. _stop_detection:

==============
Stop Detection
==============

Stop detection is the process of identifying stationary periods in GPS mobility 
trajectories. It distinguishes between **stops** (meaningful stationary locations) 
and **trips** (movement between stops), which is fundamental for understanding 
human mobility patterns.

Overview of stop-detection methods
===================================

A comparison of the stop-detection algorithms in NOMAD is shown below.

.. figure:: _images/source_benchmarking_of_stop_detection_algorithms_10_0.png
   :target: source/benchmarking_of_stop_detection_algorithms.html
   :align: center
   :width: 80%

   Runtime comparison of stop-detection algorithms.


.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Method name
     - Parameters
     - Use case
     - Characteristics

   * - :ref:`DBSCAN <dbscan_stop>`
     - eps (radius), min_samples
     - General-purpose, uniform density, fast computation
     - Density-based, finds stops as spatial clusters

   * - :ref:`HDBSCAN <hdbscan_stop>`
     - min_cluster_size, min_samples
     - Variable density stops, automatic parameter selection
     - Hierarchical density-based, adapts to density variations

   * - :ref:`Grid-based <grid_based_stop>`
     - grid_resolution, min_duration
     - Privacy-preserving, sparse data, real-time processing
     - Tessellation-based, privacy-aware, deterministic

   * - :ref:`Lachesis <lachesis_stop>`
     - distance_threshold, time_threshold
     - Trajectory segmentation, sparse temporal sampling
     - Sequential algorithm, time-aware, good for low-frequency GPS


.. _dbscan_stop:

DBSCAN
======

To be implemented.


.. _hdbscan_stop:

HDBSCAN
=======

To be implemented.


.. _grid_based_stop:

Grid-based
==========

To be implemented.


.. _lachesis_stop:

Lachesis
========

To be implemented.
