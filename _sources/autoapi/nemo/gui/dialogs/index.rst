nemo.gui.dialogs
================

.. py:module:: nemo.gui.dialogs


Classes
-------

.. autoapisummary::

   nemo.gui.dialogs.WaveletParamsDialog
   nemo.gui.dialogs.FlowParamsDialog
   nemo.gui.dialogs.ScalingDialog


Module Contents
---------------

.. py:class:: WaveletParamsDialog(master, on_save, current: dict | None = None)

   Bases: :py:obj:`tkinter.Toplevel`


   Modal dialog: edit and save WaveletDetector parameters.


.. py:class:: FlowParamsDialog(master, on_save, current: dict | None = None)

   Bases: :py:obj:`tkinter.Toplevel`


   Modal dialog: edit and save FlowTracker parameters.


.. py:class:: ScalingDialog(master, on_save, current: dict | None = None)

   Bases: :py:obj:`tkinter.Toplevel`


   Choose Linear / Log / Power scaling for the cube.


