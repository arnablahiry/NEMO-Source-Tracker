nemo
====

.. py:module:: nemo

.. autoapi-nested-parse::

   NEMO — Non-stationary Extraction via Multiscale Optical-flow.

   Typical usage::

       from nemo import WaveletDetector, FlowTracker

       detector = WaveletDetector(scales=6, k_sigma=5.0, use_scale=5)
       tracker  = FlowTracker(detector, min_match_overlap=5)
       result   = tracker.run(cube, channel_list, verbose=True)

       result.sources          # real sources (false detections removed)
       result.tracks           # all individual tracks
       result.false_detections # flagged false positives



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/nemo/denoise/index
   /autoapi/nemo/detect/index
   /autoapi/nemo/gui/index
   /autoapi/nemo/track/index
   /autoapi/nemo/utils/index


