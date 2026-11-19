=================
Visit Attribution
=================

Recurring locations
===================

``detect_locations`` clusters a single user's coordinate rows with DBSCAN
(or a custom callable). It returns an aligned Series named ``location_cluster``;
``location_id`` is reserved for polygon attribution. Customize the cluster name
through ``traj_cols={'location_cluster': 'place'}``.

.. code-block:: python

   from nomad.visit_attribution.visit_attribution import detect_locations

   stops['location_cluster'] = detect_locations(stops, epsilon=50)

``epsilon`` limits neighbor distance, rather than the diameter of a whole cluster.
Distances use projected coordinate units, or meters for longitude/latitude.
Noise rows receive individual location clusters. ``return_locations=True`` also
returns centers, convex-hull extents, and row counts for projected coordinates.

For trajectory stop detection using Infomap, see :ref:`infostop`.
