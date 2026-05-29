nemo.track
==========

.. py:module:: nemo.track

.. autoapi-nested-parse::

   Flow-guided source tracker for per-channel wavelet detections.

   Takes the list of per-channel detections produced by
   ``wavelet_detections.detect_cube_per_channel`` and runs a four-stage pipeline:

   Stage 1 — Masked optical flow
       TV-L1 flow is computed between every consecutive channel pair, but only
       inside the intersection of the two channels' union footprint masks.
       Zeroing the images outside detected sources prevents artefact-level flow
       vectors from leaking into the tracking step.

   Stage 2 — Track linking with split/merge detection
       Two-pass approach for symmetric split and merge detection.

       Forward pass (ch_0 → ch_N): advected masks are propagated through the flow and
       matched to component detections via Hungarian assignment.  Unmatched predictions
       that overlap an already-claimed component are flagged as **merges**.  Unmatched
       detections start new independent tracks.

       Backward pass (ch_N → ch_0): the same algorithm runs on reversed detections
       and negated flow.  A forward split (one parent → two children) appears as a
       backward merge (two tracks → one component) and is captured symmetrically by the
       same merge-detection step.

       Reconciliation maps backward merge events onto forward tracks to annotate
       ``split_from`` and ``split_at`` fields.  No Euclidean distance is used
       anywhere in split or merge detection.

   Stage 3 — Kinematic classification
       A track is **kinematically active** if its cumulative centroid displacement
       across channels exceeds MIN_DISPLACEMENT pixels, or if it was involved in
       a split or merge event.

   Stage 4 — Source grouping
       Tracks connected by split_from / merge_into relationships are grouped into
       **sources** via union-find.  A source represents one physical object whose
       emission footprint may split into several components across channels (due to
       kinematics / Doppler shear) and later rejoin.

   Output
   ------
   ``run_flow_tracker`` returns a :class:`TrackingResult` dataclass with:

   ``detections``
       list[ChannelDetection], one per processed channel.
   ``flow_seq``
       list of (ch_ref, ch_tgt, flow (2,H,W), joint_mask) tuples.
   ``tracks``
       list of track dicts.  Each dict contains ``id``, ``source_id``,
       ``trajectory``, ``masks``, ``split_at``, ``split_from``,
       ``merge_into``, ``displacement``, ``has_split``, ``kinematic``.
   ``sources``
       list of source dicts.  Each dict contains ``id``, ``track_ids``,
       ``channels``, ``n_channels``, ``split_events``, ``merge_events``.

   Usage (standalone)::

       python flow_tracker.py \
           --cube  data/clean_cube.npy \
           --out   /tmp/tracks \
           --channels 70,74 \
           --min-match-overlap 5 --min-displacement 3



Classes
-------

.. autoapisummary::

   nemo.track.TrackingResult
   nemo.track.FlowTracker


Functions
---------

.. autoapisummary::

   nemo.track.masked_flow_tvl1
   nemo.track.compute_flow_sequence
   nemo.track.link_tracks
   nemo.track.classify_kinematic
   nemo.track.group_into_sources
   nemo.track.classify_sources
   nemo.track.run_flow_tracker
   nemo.track.main


Module Contents
---------------

.. py:function:: masked_flow_tvl1(img_ref: numpy.ndarray, img_tgt: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray

   TV-L1 optical flow restricted to *mask* pixels.

   Both images are zeroed outside *mask* before the solver runs, so emission
   structure outside detected source footprints never influences the flow
   estimate inside them.

   :param img_ref: 2-D float32 channel images, shape (H, W).
   :param img_tgt: 2-D float32 channel images, shape (H, W).
   :param mask: Boolean (H, W) — True where flow should be estimated.

   :returns: Shape (2, H, W) float32.  ``flow[0]`` = v (row displacement),
             ``flow[1]`` = u (col displacement).  Zero everywhere outside *mask*.
   :rtype: np.ndarray


.. py:function:: compute_flow_sequence(detections: list[nemo.detect.ChannelDetection], verbose: bool = False) -> list[tuple[int, int, numpy.ndarray, numpy.ndarray]]

   Compute masked TV-L1 flow for every consecutive detection pair.

   The joint mask is the *union* of the source footprints from both channels.
   Using the union (rather than the intersection) is critical for split
   detection: when a source splits into a new spatial location between
   channels, the two components may not overlap at all.  With an intersection
   mask the flow would be zero everywhere and the predicted centroid would
   not move — causing the split-off component to be mis-classified as a new
   independent source.  With the union mask the TV-L1 solver sees the
   source signal on both sides and produces flow vectors that point from
   the pre-split footprint toward the post-split footprint, allowing
   :func:`link_tracks` to attribute the new component to the correct parent.

   :param detections: Ordered list of :class:`~wavelet_detections.ChannelDetection` objects.

   :rtype: list of (ch_ref, ch_tgt, flow, joint_mask) tuples.


.. py:function:: link_tracks(detections: list[nemo.detect.ChannelDetection], flow_seq: list[tuple[int, int, numpy.ndarray, numpy.ndarray]], min_match_overlap: int = 5, max_gap_channels: int = 5, verbose: bool = False) -> list[dict]

   Link per-channel component detections into multi-channel tracks.

   Uses advected masks and pixel-overlap matching — no Euclidean distance.

   Algorithm
   ---------
   Each track maintains an *advected mask*: its most recently known wavelet
   footprint, advected channel-by-channel through the flow via Catmull-Rom
   cubic interpolation.  All matching uses pixel-overlap only.

   For each consecutive channel pair (ref → tgt):

   A. Advect every active track's footprint mask through the flow → adv_maps.
   B. Hungarian matching on negative-overlap cost matrix.  Pairs with overlap
      ≥ min_match_overlap are matched; advected mask reset to the matched
      component footprint.
   C. Unmatched predictions: check for merge via overlap, then freeze advected
      mask and extrapolate centroid; deactivate if gap exceeds max_gap_channels.
   D. Unmatched detections: always start new independent tracks.  Split
      attribution is performed separately by :func:`_reconcile_splits` using
      the backward pass, where splits look like merges and are detected
      symmetrically.

   :param detections: Per-channel detection results in channel order.
   :param flow_seq: Output of :func:`compute_flow_sequence`.
   :param min_match_overlap: Minimum pixel overlap (advected mask ∩ component footprint) to accept a
                             continuation match.
   :param max_gap_channels: Maximum number of consecutive unmatched channels before a track is
                            deactivated.

   :returns: One dict per track with keys: ``id``, ``trajectory``, ``masks``,
             ``split_at``, ``split_from``, ``merge_into``, ``active``.
   :rtype: list[dict]


.. py:function:: classify_kinematic(tracks: list[dict], min_displacement: float = 3.0, verbose: bool = False) -> list[dict]

   Add kinematic classification fields to each track dict (in-place).

   A track is **kinematically active** if:
   - Its cumulative centroid displacement across channels ≥ *min_displacement*, or
   - It was involved in a split event (either as parent or as split-off child).

   Adds keys ``displacement`` (float, px), ``has_split`` (bool),
   and ``kinematic`` (bool) to each track dict.


.. py:function:: group_into_sources(tracks: list[dict]) -> list[dict]

   Group related tracks into sources via union-find over split/merge edges.

   Two tracks belong to the same source if they are connected by any chain of
   ``split_from`` (child→parent) or ``merge_into`` (merging track → target)
   relationships.  The result is one source per connected component.

   Annotates each track dict in-place with a ``source_id`` key.

   :param tracks: Output of :func:`classify_kinematic` (or :func:`link_tracks`).

   :returns: One dict per source, sorted by ascending ``id``, with keys:
             ``id``, ``track_ids``, ``channels``, ``n_channels``,
             ``split_events``, ``merge_events``.
   :rtype: list[dict]


.. py:function:: classify_sources(sources: list[dict], tracks: list[dict], detections: list[nemo.detect.ChannelDetection], flow_seq: list[tuple], wav_scale_idx: int = 3, wav_abrupt_thresh: float = 0.5, flow_iou_thresh: float = 0.25, short_det_max: int = 8, verbose: bool = True, plot: bool = False, vel_array: numpy.ndarray | None = None, results_dir: str | pathlib.Path | None = None) -> tuple[list[dict], list[dict], dict, dict]

   Classify sources as real detections or false positives.

   Uses two complementary metrics computed from wavelet coefficients and
   optical flow:

   - **flow_iou**: advect each source's footprint through the flow field
     (backward warp) and measure IoU with the next channel's footprint.
     Real sources follow the flow and score high; artefacts don't move
     coherently and score low.
   - **wav_abrupt**: ratio of the edge wavelet flux (first or last channel
     of the detection) to the peak flux.  Step-function artefacts that
     appear or disappear abruptly score ≈ 1; real sources fade in/out
     and score lower.

   A source is classified as a false detection if:
       wav_abrupt > wav_abrupt_thresh
       OR (flow_iou < flow_iou_thresh AND n_detected_channels < short_det_max)

   :param sources: Direct outputs of :func:`run_flow_tracker`.
   :param tracks: Direct outputs of :func:`run_flow_tracker`.
   :param detections: Direct outputs of :func:`run_flow_tracker`.
   :param flow_seq: Direct outputs of :func:`run_flow_tracker`.
   :param wav_scale_idx: 0-based wavelet scale index to use for spectral profile extraction.
                         Should match ``use_scale - 1`` used in :func:`run_flow_tracker`.
   :param wav_abrupt_thresh: Classification thresholds (see above).
   :param flow_iou_thresh: Classification thresholds (see above).
   :param short_det_max: Classification thresholds (see above).
   :param verbose: Print a formatted table of sources and false detections.
   :param plot: Render and save the two-panel separation figure
                (IoU scatter + normalised wavelet profiles).  Requires
                ``vel_array`` and ``results_dir``.
   :param vel_array: 1-D velocity array aligned with cube channels (km/s).  Required
                     when ``plot=True``.
   :param results_dir: Directory for saved figures.  Created if absent.  Required when
                       ``plot=True``.

   :returns: * **good_sources** (*list[dict]*)
             * **false_dets** (*list[dict]*)
             * **src_data** (*dict  {source_id → metric dict}*)
             * **src_colors** (*dict  {source_id → rgba tuple}  (tab10, cycled mod 10)*)


.. py:function:: run_flow_tracker(cube: numpy.ndarray, channel_list: list[int] | None = None, scales: int = 6, k_sigma: float = 5.0, use_scale: int = 5, min_area: int = 20, thresh: float | None = None, use_mean_map_sigma: bool = True, min_match_overlap: int = 5, max_gap_channels: int = 5, min_displacement: float = 3.0, wav_scale_idx: int = 3, wav_abrupt_thresh: float = 0.5, flow_iou_thresh: float = 0.25, short_det_max: int = 8, vel_array: numpy.ndarray | None = None, results_dir: str | pathlib.Path | None = None, plot: bool = False, verbose: bool = False) -> tuple[list[nemo.detect.ChannelDetection], list[tuple], list[dict], list[dict], list[dict], list[dict], dict, dict]

   Detect → flow → track → classify → group → classify sources.

   Runs all five pipeline stages and returns their combined outputs.

   :param use_mean_map_sigma: Passed to :func:`~wavelet_detections.detect_cube_per_channel`.
                              When ``True`` (default) the wavelet threshold is anchored to the
                              per-scale noise from the mean map, preventing spurious detections
                              on near-empty channels.
   :param wav_scale_idx: 0-based wavelet scale index for source classification (should equal
                         ``use_scale - 1``).
   :param wav_abrupt_thresh: Thresholds for :func:`classify_sources`.
   :param flow_iou_thresh: Thresholds for :func:`classify_sources`.
   :param short_det_max: Thresholds for :func:`classify_sources`.
   :param vel_array: 1-D velocity array (km/s, length = cube.shape[0]).  Passed to
                     :func:`classify_sources` for axis labelling when ``plot=True``.
   :param results_dir: Output directory for saved figures.  Passed to
                       :func:`classify_sources` when ``plot=True``.
   :param plot: Render and save the false-detection separation figure.
   :param verbose: Print per-step progress and summary tables.

   :returns: * **detections** (*list[ChannelDetection]*)
             * **flow_seq** (*list of (ch_ref, ch_tgt, flow, joint_mask)*)
             * **tracks** (*list of classified track dicts (each annotated with source_id)*)
             * **sources** (*list of all source dicts*)
             * **good_sources** (*list of source dicts that passed the false-detection filter*)
             * **false_dets** (*list of source dicts flagged as false detections*)
             * **src_data** (*dict {source_id → metric dict (flow_iou, wav_abrupt, …)}*)
             * **src_colors** (*dict {source_id → rgba tuple}  (tab10, cycled mod 10)*)


.. py:class:: TrackingResult

   Output of :meth:`FlowTracker.run`.

   Wraps the full pipeline output — detections, flow, tracks, sources,
   false-detection list, and per-source metrics — in a single object.


   .. py:attribute:: detections
      :type:  list


   .. py:attribute:: flow_seq
      :type:  list


   .. py:attribute:: tracks
      :type:  list


   .. py:attribute:: sources
      :type:  list


   .. py:attribute:: false_detections
      :type:  list


   .. py:attribute:: src_data
      :type:  dict


   .. py:attribute:: src_colors
      :type:  dict


.. py:class:: FlowTracker(detector=None, min_match_overlap: int = 5, max_gap_channels: int = 5, min_displacement: float = 3.0, wav_scale_idx: int = 3, wav_abrupt_thresh: float = 0.5, flow_iou_thresh: float = 0.25, short_det_max: int = 8)

   Full STORM pipeline: wavelet detection → optical flow → track linking
   → kinematic classification → source grouping → false-detection removal.

   :param detector: Wavelet detector instance.  ``None`` uses default settings.
   :type detector: WaveletDetector or None
   :param min_match_overlap: Minimum pixel overlap to accept an advected→component continuation match.
   :type min_match_overlap: int
   :param max_gap_channels: Maximum consecutive unmatched channels before a track is deactivated.
   :type max_gap_channels: int
   :param min_displacement: Minimum cumulative centroid travel (px) to call a track kinematic.
   :type min_displacement: float
   :param wav_scale_idx: 0-based wavelet scale index for source classification metrics.
   :type wav_scale_idx: int
   :param wav_abrupt_thresh: Abruptness threshold above which a source is flagged as a false
                             detection.
   :type wav_abrupt_thresh: float
   :param flow_iou_thresh: Flow-IoU threshold below which a short source is flagged as a false
                           detection.
   :type flow_iou_thresh: float
   :param short_det_max: Maximum channel span for the flow-IoU false-detection criterion.
   :type short_det_max: int

   .. rubric:: Examples

   >>> from storm.detect import WaveletDetector
   >>> from storm.track import FlowTracker
   >>>
   >>> detector = WaveletDetector(scales=6, k_sigma=5.0, use_scale=5)
   >>> tracker  = FlowTracker(detector, min_match_overlap=5)
   >>> result   = tracker.run(cube, channel_list, verbose=True)
   >>>
   >>> result.sources          # real sources
   >>> result.false_detections # flagged false positives
   >>> result.tracks           # all individual tracks


   .. py:attribute:: detector


   .. py:attribute:: min_match_overlap
      :value: 5



   .. py:attribute:: max_gap_channels
      :value: 5



   .. py:attribute:: min_displacement
      :value: 3.0



   .. py:attribute:: wav_scale_idx
      :value: 3



   .. py:attribute:: wav_abrupt_thresh
      :value: 0.5



   .. py:attribute:: flow_iou_thresh
      :value: 0.25



   .. py:attribute:: short_det_max
      :value: 8



   .. py:method:: run(cube: numpy.ndarray, channel_list: list[int] | None = None, vel_array: numpy.ndarray | None = None, results_dir=None, plot: bool = False, verbose: bool = False) -> TrackingResult

      Detect sources in *cube* and run the full tracking pipeline.

      :param cube:
      :type cube: (n_ch, H, W) float32
      :param channel_list: Channels to process.  ``None`` processes all channels.
      :type channel_list: list of int or None
      :param vel_array: Velocity axis (km/s) for plot axis labelling.
      :type vel_array: 1-D array or None
      :param results_dir: Directory for saved figures when ``plot=True``.
      :type results_dir: path-like or None
      :param plot: Render and save the false-detection separation figure.
      :type plot: bool
      :param verbose: Print per-step progress and summary tables.
      :type verbose: bool

      :rtype: TrackingResult



   .. py:method:: run_from_detections(detections: list, vel_array: numpy.ndarray | None = None, results_dir=None, plot: bool = False, verbose: bool = False) -> TrackingResult

      Run the tracking pipeline on pre-computed *detections*.

      Useful when you want to inspect or filter detections before tracking.

      :param detections: Output of :meth:`WaveletDetector.detect`.
      :type detections: list[ChannelDetection]



.. py:function:: main() -> None

