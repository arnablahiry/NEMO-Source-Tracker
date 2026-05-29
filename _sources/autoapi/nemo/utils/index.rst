nemo.utils
==========

.. py:module:: nemo.utils

.. autoapi-nested-parse::

   Shared plotting utilities for IFU source identification notebooks.



Functions
---------

.. autoapisummary::

   nemo.utils.add_beam


Module Contents
---------------

.. py:function:: add_beam(ax, bmin_pix: float, bmaj_pix: float, bpa_deg: float, xy_offset: tuple[float, float] = (10, 10), color: str = 'white', crosshair: bool = True) -> None

   Draw a synthesized beam ellipse on *ax*.

   :param ax:
   :type ax: matplotlib.axes.Axes
   :param bmin_pix: Minor-axis FWHM in pixels.
   :type bmin_pix: float
   :param bmaj_pix: Major-axis FWHM in pixels (≥ bmin_pix).
   :type bmaj_pix: float
   :param bpa_deg: Beam position angle in degrees, CCW from the x-axis (East).
   :type bpa_deg: float
   :param xy_offset: Offset from the bottom-left corner of the current axes limits, in
                     data units (pixels).  Default (10, 10).
   :type xy_offset: (float, float)
   :param color: Colour of the ellipse outline and crosshairs.
   :type color: str
   :param crosshair: Draw major/minor axis crosshairs inside the ellipse.
   :type crosshair: bool


