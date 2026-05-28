Quick Start
===========

.. code-block:: python

   from nemo import WaveletDetector, FlowTracker, load_cube, active_channels

   # Load cube → float32 (n_ch, H, W)
   cube = load_cube("data/cube.fits")

   # Select channels with significant positive flux
   channels = active_channels(cube, threshold_frac=0.05)

   # Configure the wavelet detector
   detector = WaveletDetector(
       scales=6,       # total starlet scales
       k_sigma=5.0,    # detection threshold (user-controlled)
       use_scale=4,    # detail band used for component extraction
       min_area=10,    # minimum component area in pixels
   )

   # Configure the flow tracker
   tracker = FlowTracker(
       detector=detector,
       min_match_overlap=5,    # min advected∩component overlap for continuation
       min_split_overlap=3,    # min advected∩component overlap to attribute a split
       max_gap_dist=15.0,      # max centroid distance (px) for gap bridging
       min_displacement=3.0,   # min cumulative travel (px) for kinematic flag
       wav_abrupt_thresh=0.5,  # wavelet abruptness threshold for false-detection filter
       flow_iou_thresh=0.25,   # flow-IoU threshold for false-detection filter
   )

   # Run the full pipeline (verbose prints a splash + per-stage progress)
   result = tracker.run(cube, channels, verbose=True)

   print(f"{len(result.sources)} real sources")
   print(f"{len(result.false_detections)} false detections removed")
   print(f"{len(result.tracks)} individual tracks")

   # Inspect a source
   src = result.sources[0]
   print(src["channels"])       # spectral channels spanned
   print(src["split_events"])   # channels where the footprint split

Result object
-------------

:class:`nemo.TrackingResult` exposes:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``sources``
     - Real sources after false-detection removal, each a dict with
       ``id``, ``track_ids``, ``channels``, ``split_events``, ``merge_events``.
   * - ``false_detections``
     - Sources flagged as false positives, kept for inspection.
   * - ``tracks``
     - All tracks annotated with ``source_id``, ``kinematic``,
       ``displacement``, ``has_split``.
   * - ``flow_seq``
     - TV-L1 flow for every consecutive channel pair as
       ``(ch_ref, ch_tgt, flow (2,H,W), joint_mask)`` tuples.
   * - ``src_data``
     - Per-source classification metrics: ``flow_iou``, ``wav_abrupt``, ``n_det``.
   * - ``src_colors``
     - tab10 RGBA colour assigned to each source.
