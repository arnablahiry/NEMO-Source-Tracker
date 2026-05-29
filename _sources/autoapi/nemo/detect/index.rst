nemo.detect
===========

.. py:module:: nemo.detect

.. autoapi-nested-parse::

   Per-channel starlet (à trous IUWT) source detection for 3-D spectral cubes.

   Strategy
   --------
   The starlet transform decomposes each 2-D spectral slice into detail bands at
   increasing angular scales plus a coarse residual.  Compact sources produce
   strong coefficients in the fine-scale bands; diffuse emission sits in the
   coarser bands and is naturally suppressed.

   For each channel slice, ``wavelet_footprints``:

   1. Computes the starlet transform via a manual PyTorch à trous convolution with
      the B3-spline scaling function.
   2. Thresholds each detail scale independently using a per-scale MAD noise
      estimate, so the threshold adapts to the actual signal level at that scale
      rather than collapsing on noise-free or low-signal data.
   3. Applies an absolute floor of 10 % of the detection-plane peak to suppress
      float32 rounding artefacts in noise-free cubes.
   4. Extracts connected emission components on the chosen detection scale and returns peak
      coordinates, binary footprint masks, and bounding boxes.

   Input formats
   -------------
   ``load_cube`` accepts .h5/.hdf5, .fits/.fit, .npy, .npz.

   Usage (standalone)::

       python wavelet_detections.py \
           --cube  data/clean_cube.npy \
           --out   /tmp/detections \
           --channels 70,74 \
           --k-sigma 5 --scales 6 --use-scale 5 --min-area 20



Classes
-------

.. autoapisummary::

   nemo.detect.ChannelDetection
   nemo.detect.WaveletDetector


Functions
---------

.. autoapisummary::

   nemo.detect.starlet_transform
   nemo.detect.load_cube
   nemo.detect.active_channels
   nemo.detect.reference_sigmas_from_mean_map
   nemo.detect.wavelet_footprints
   nemo.detect.detect_cube_per_channel
   nemo.detect.main


Module Contents
---------------

.. py:function:: starlet_transform(image: numpy.ndarray, scales: int) -> numpy.ndarray

   Starlet (à trous IUWT) transform of a 2-D image.

   :param image:
   :type image: (H, W) float32
   :param scales:
   :type scales: total planes = (scales-1) detail bands + 1 coarse residual

   :returns: Planes 0 … scales-2 are detail bands (finest → coarsest).
             Plane scales-1 is the coarse residual.
   :rtype: np.ndarray, shape (scales, H, W)


.. py:function:: load_cube(path: str | pathlib.Path) -> numpy.ndarray

   Load a spectral cube from HDF5, FITS, .npy, or .npz.

   Always returns float32 (n_ch, H, W).  NaNs are replaced with 0.


.. py:function:: active_channels(cube: numpy.ndarray, threshold_frac: float = 0.05) -> list[int]

   Return indices of channels whose positive flux exceeds *threshold_frac* × max.

   Uses only positive flux so noise-dominated channels (where positive and
   negative values roughly cancel) do not inflate the total.


.. py:class:: ChannelDetection

   Bases: :py:obj:`NamedTuple`


   All detection results for a single spectral channel.


   .. py:attribute:: channel
      :type:  int


   .. py:attribute:: image
      :type:  numpy.ndarray


   .. py:attribute:: footprint_masks
      :type:  list


   .. py:attribute:: peaks
      :type:  list


   .. py:attribute:: boxes
      :type:  list


   .. py:attribute:: detect_coeffs
      :type:  numpy.ndarray


.. py:function:: reference_sigmas_from_mean_map(cube: numpy.ndarray, channel_list: list[int] | None, scales: int) -> numpy.ndarray

   Per-scale noise reference derived from the mean map across *channel_list*.

   The mean map has noise σ/√N relative to a single channel.  Per-scale MAD
   of its starlet coefficients is therefore multiplied by √N to recover an
   estimate of the single-channel noise at each wavelet scale.  This gives a
   stable, channel-independent threshold that is immune to the per-channel MAD
   collapse that occurs on near-empty channels (where residuals are tiny and
   deterministic, making per-channel σ → 0 and the threshold meaninglessly low).

   :param cube:
   :type cube: (n_ch, H, W) float32
   :param channel_list:
   :type channel_list: list of channel indices to include; ``None`` uses all.
   :param scales:
   :type scales: total number of starlet scales (including coarse residual).

   :returns: Single-channel noise estimate for each detail scale.
   :rtype: np.ndarray, shape (scales - 1,)


.. py:function:: wavelet_footprints(image: numpy.ndarray, scales: int = 4, k_sigma: float = 2.3, use_scale: int = 2, min_area: int = 10, thresh: float | None = None, sigma_per_scale: numpy.ndarray | None = None) -> ChannelDetection

   Detect compact-source footprints in a single 2-D image via starlet thresholding.

   :param image: 2-D spectral slice (H, W).
   :param scales: Total number of starlet scales (including the coarse residual plane).
   :param k_sigma: Detection threshold in units of per-scale noise.
   :param use_scale: 1-based index of the detail band used for component detection.
                     Scale 1 is the finest (sub-pixel structure); higher scales capture
                     progressively larger compact sources.
   :param min_area: Minimum component area in pixels; smaller components are discarded as artefacts.
   :param thresh: Absolute lower bound on the detection-plane value.  ``None`` (default)
                  sets it to 10 % of the detection-plane maximum.
   :param sigma_per_scale: Pre-computed per-scale noise array, shape (scales-1,).  When provided,
                           these values replace the per-channel MAD estimate so that the threshold
                           is anchored to a stable global reference (typically derived from the
                           mean map via :func:`reference_sigmas_from_mean_map`).  Passing ``None``
                           falls back to the original per-channel MAD behaviour.

   :returns: channel is set to -1 here; callers should replace it via ``._replace``.
   :rtype: ChannelDetection

   .. rubric:: Notes

   When ``sigma_per_scale`` is ``None`` each scale is thresholded with its own
   MAD estimate.  This is adaptive but can collapse to near-zero on nearly
   empty channels (where denoised residuals are tiny), causing spurious
   detections.  Providing ``sigma_per_scale`` from the mean-map decomposition
   pins the threshold to the cube-wide noise floor and eliminates this failure.


.. py:function:: detect_cube_per_channel(cube: numpy.ndarray, channel_list: list[int] | None = None, scales: int = 4, k_sigma: float = 2.3, use_scale: int = 2, min_area: int = 10, thresh: float | None = None, use_mean_map_sigma: bool = True, verbose: bool = False) -> list[ChannelDetection]

   Run ``wavelet_footprints`` on every channel in *channel_list*.

   :param use_mean_map_sigma: When ``True`` (default), compute per-scale noise reference from the
                              mean map across *channel_list* via :func:`reference_sigmas_from_mean_map`
                              and pass it to every per-channel call.  This prevents the per-channel
                              MAD from collapsing on near-empty channels and eliminates spurious
                              detections in signal-free parts of the cube.  Set to ``False`` to
                              revert to the original per-channel MAD behaviour.
   :param verbose: Print progress and summary statistics to stdout.

   :returns: One entry per channel, with the ``channel`` field set to the cube channel index.
   :rtype: list[ChannelDetection]


.. py:class:: WaveletDetector(scales: int = 6, k_sigma: float = 5.0, use_scale: int = 5, min_area: int = 20, thresh: float | None = None, use_mean_map_sigma: bool = True)

   Starlet-wavelet per-channel source detector for 3-D spectral cubes.

   :param scales: Total number of starlet scales (including coarse residual).
   :type scales: int
   :param k_sigma: Detection threshold in units of per-scale noise.
   :type k_sigma: float
   :param use_scale: 1-based detail band used for component detection.
   :type use_scale: int
   :param min_area: Minimum component area in pixels.
   :type min_area: int
   :param thresh: Absolute lower bound on detection-plane value.  ``None`` uses 10 % of
                  the channel peak.
   :type thresh: float or None
   :param use_mean_map_sigma: Anchor the noise estimate to the mean map across all channels rather
                              than computing it per-channel.  Prevents spurious detections on nearly
                              empty channels.
   :type use_mean_map_sigma: bool

   .. rubric:: Examples

   >>> detector = WaveletDetector(scales=6, k_sigma=5.0, use_scale=5)
   >>> detections = detector.detect(cube, channel_list)


   .. py:attribute:: scales
      :value: 6



   .. py:attribute:: k_sigma
      :value: 5.0



   .. py:attribute:: use_scale
      :value: 5



   .. py:attribute:: min_area
      :value: 20



   .. py:attribute:: thresh
      :value: None



   .. py:attribute:: use_mean_map_sigma
      :value: True



   .. py:method:: detect(cube: numpy.ndarray, channel_list: list[int] | None = None, verbose: bool = False) -> list[ChannelDetection]

      Run per-channel wavelet detection on *cube*.

      :param cube:
      :type cube: (n_ch, H, W) float32
      :param channel_list: Channel indices to process.  ``None`` processes all channels.
      :type channel_list: list of int or None
      :param verbose: Print per-channel progress and summary statistics.
      :type verbose: bool

      :returns: One entry per channel in *channel_list*, in order.
      :rtype: list[ChannelDetection]



.. py:function:: main() -> None

