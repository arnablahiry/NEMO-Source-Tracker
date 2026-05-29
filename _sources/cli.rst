CLI Reference
=============

NEMO installs three command-line entry points.

nemo-detect
-----------

Run per-channel starlet wavelet detection only.

.. code-block:: bash

   nemo-detect --cube data/cube.fits --out results/ \
               --scales 6 --k-sigma 5.0 --use-scale 5 --min-area 20

.. option:: --cube

   Input cube file: ``.fits``, ``.h5``, ``.npy``, or ``.npz``.

.. option:: --out

   Output directory for ``detections_summary.json``.

.. option:: --channels

   Comma-separated channel indices. Default: auto-selected active channels.

.. option:: --scales

   Total starlet scales including coarse residual.

.. option:: --k-sigma

   Detection threshold in units of per-scale noise.

.. option:: --use-scale

   1-based detail band used for component extraction.

.. option:: --min-area

   Minimum component area in pixels.

nemo-track
----------

Run the full detection + tracking pipeline.

.. code-block:: bash

   nemo-track --cube data/cube.fits --out results/ \
              --scales 6 --k-sigma 5.0 --use-scale 5 --min-area 20 \
              --min-match-overlap 5 --min-split-overlap 3 --min-displacement 3.0

Produces ``tracks.csv``, ``sources.csv``, and ``summary.json`` in the output
directory.

.. option:: --min-match-overlap

   Minimum pixel overlap (advected mask ∩ component) to accept a continuation match.

.. option:: --min-split-overlap

   Minimum pixel overlap to attribute an unmatched component as a split.

.. option:: --min-displacement

   Minimum cumulative centroid travel in pixels to classify a track as kinematic.

nemo-denoise
------------

Denoise a FITS cube with the IST algorithm (requires ``cosmostat``).

.. code-block:: bash

   nemo-denoise cube.fits --threshold 5.0 --thresh-increm 2.0 --num-iter 20
