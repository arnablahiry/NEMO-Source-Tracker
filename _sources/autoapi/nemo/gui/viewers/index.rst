nemo.gui.viewers
================

.. py:module:: nemo.gui.viewers


Classes
-------

.. autoapisummary::

   nemo.gui.viewers.SliceViewer
   nemo.gui.viewers.ScaleViewer


Module Contents
---------------

.. py:class:: SliceViewer(master, cube: numpy.ndarray, detections: list | None = None, flow_seq: list | None = None, tracks: list | None = None, sources: list | None = None, mode: str = 'raw', initial_norm: str = 'linear', initial_gamma: float = 0.5)

   Bases: :py:obj:`tkinter.Toplevel`


   Channel-by-channel viewer with normalization controls and optional overlays.

   mode : "raw"        — plain channel images
          "detections" — white contour overlays from detection footprints
          "flow"       — quiver overlay from flow_seq


.. py:class:: ScaleViewer(master, cube: numpy.ndarray, wav_params: dict, detections=None, on_scale_chosen=None, on_params_saved=None)

   Bases: :py:obj:`tkinter.Toplevel`


   Browse per-channel 2D wavelet coefficient maps with embedded parameters.

   Number of scales is chosen via radio buttons (2…max), where max depends on
   the image dimensions.  The figure is always laid out as 2 rows × ceil(n/2)
   cols.  Changing n_scales re-runs the starlet transform and redraws in real
   time.


   .. py:attribute:: N_ROWS
      :value: 2



