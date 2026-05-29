nemo.denoise
============

.. py:module:: nemo.denoise

.. autoapi-nested-parse::

   Denoise any FITS spectral cube using pycs Denoiser2D1D IST.

   The denoised data replaces the primary HDU array and is saved as a new FITS
   file alongside the input (suffix _denoised_ist.fits).

   Requires the cosmostat package (not on PyPI).  Install it manually and ensure
   it is importable, or set the COSMOSTAT_PATH environment variable to its root::

       export COSMOSTAT_PATH=/path/to/cosmostat
       storm-denoise cube.fits

   Run with the cosmostat conda env:
       /path/to/envs/cosmostat/bin/python -m storm.denoise <cube.fits> [options]



Functions
---------

.. autoapisummary::

   nemo.denoise.main


Module Contents
---------------

.. py:function:: main()

