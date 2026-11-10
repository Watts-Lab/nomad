=================
Visit Attribution
=================

Recurring locations
===================

``detect_locations`` assigns recurring-location IDs to coordinate rows. DBSCAN
remains the default. Infomap is an opt-in alternative for stop tables containing
a start time and either duration or end time. The optional integration uses
Infomap 2.16 or newer and requires Python 3.11 or newer. Install it from a NOMAD
checkout:

.. code-block:: bash

   python -m pip install -e '.[infomap]'

.. code-block:: python

   from nomad.visit_attribution.visit_attribution import detect_locations

   location_ids = detect_locations(
       stops, method='infomap', epsilon=50, sigma=15,
       seed=123, num_trials=10,
   )

Assign the returned Series to ``stops['location_id']`` to retain locations
alongside the stop table. Custom column names use ``traj_cols``.

Infomap graph construction
==========================

Each stop row is one node at its representative coordinates. Two distinct nodes
are connected when their distance is at most ``epsilon``. Connections can be
followed in either direction with the same weight (an undirected graph), and no
stop is connected to itself. An edge at distance ``d`` receives Gaussian similarity
weight ``exp(-d**2 / (2 * sigma**2))``; zero-distance pairs receive weight 1.
Numerically zero weights are omitted. Isolated stops remain separate locations.

``epsilon`` determines which pairs can be connected. ``sigma`` determines how
quickly their connection strength decreases with distance and must be supplied
explicitly. With ``sigma=15``, distances of 15, 30 and 45 units receive weights
of approximately 0.61, 0.14 and 0.01, provided they are within ``epsilon``.
Smaller values emphasize close neighbors; larger values make connection weights
more similar within the radius. Both parameters use the same distance units.
The values in the example are illustrative and can be adjusted for the data.

Distances use Euclidean distance in projected coordinate units, or haversine
distance in meters for longitude/latitude, including across the antimeridian.
Use a suitable projection: projected coordinates are not necessarily in meters.
Radius-neighbor searches avoid allocating a dense pairwise distance matrix,
although a large radius can still produce a dense graph.

Infomap models a walker moving between connected stops, favoring stronger
connections. It looks for groups where that walker tends to remain. Each stop
receives one group assignment, with no nested subgroups (``two_level=True``).
The ``markov_time`` setting controls the grouping resolution. We use its default
value of 1; it does not represent minutes or any other unit of trajectory time.

The default seed is 123 and the default number of search trials is 10. Repeating
an identical ordered input with the same parameters and dependency versions is
reproducible. Record the parameters and dependency versions to reproduce a run.
See the `Infomap documentation <https://mapequation.org/infomap-python-docs/>`_
for the map equation and flow model.

This random walk represents spatial similarity, not observed trips. Timestamps
and durations establish that the input contains stops; they do not weight edges.
``min_pts`` applies only to DBSCAN. Infomap IDs are numbered by first occurrence
and are arbitrary identifiers, not comparable labels across independent runs.

Users and summaries
===================

With a user column, Infomap constructs a separate graph per user. Their location
IDs are distinct within the returned Series; another user's stops cannot alter
a user's partition. Input ordering and indices, including duplicate indices,
are preserved, and the input is not modified. Single-user data need not include
a user column. DBSCAN and custom methods retain their single-user restriction.

``return_locations=True`` also returns centers, convex-hull extents and stop
counts. Multi-user summaries include user identity. Geometry summaries currently
require projected coordinates.

Lachesis post-processing
========================

.. code-block:: python

   from nomad.stop_detection.sequential_algs import lachesis_per_user

   result = lachesis_per_user(
       trajectory, delta_roam=20, dt_max=30, dur_min=5,
       postprocessing='infomap', eps=50,
       infomap_kwargs={'sigma': 15, 'seed': 123, 'num_trials': 10},
   )

Lachesis first detects stops, summarizes their representative coordinates, and
assigns locations using the chosen method. It then uses the existing merge rule:
consecutive stops must share a location and their end-to-start gap must be at
most ``dt_max`` minutes. An intervening stop at another location prevents merging.
Repeated visits separated by a longer gap remain separate stops.

The same options are available on ``lachesis``, ``lachesis_labels`` and
``lachesis_labels_per_user``. Location assignments are intermediate: existing
return schemas, noise labels, user identity and anchor values are retained.
``postprocessing=None`` remains the default. ``postprocessing='dbscan'`` retains
its existing ``eps`` radius and minimum of two stops per location cluster.

Parameter effects
=================

The radius limits direct edges, not community diameter. Chains of nearby stops
can connect distant stops, while a small ``sigma`` can separate nearby groups.
The number and spatial distribution of stop rows also affect graph connectivity
and the resulting partition.

The :doc:`Lachesis demonstration <source/lachesis_demo>` includes a controlled
comparison with DBSCAN and equal-weight Infomap across several synthetic layouts.
