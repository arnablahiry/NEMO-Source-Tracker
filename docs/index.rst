N.E.M.O. [Non-stationary Extraction via Multiscale Optical-flow]
================================================================

.. image:: ../assets/nemo_logo.png
   :width: 100%
   :alt: NEMO logo

**NEMO** is a Python pipeline for detecting and tracking compact emission sources
across the spectral axis of 3-D radio interferometric data cubes (FITS, HDF5, NumPy).
It combines a multiscale starlet wavelet detector with TV-L1 optical flow tracking,
kinematic classification, and a dual-metric false-detection filter.

.. grid:: 2

   .. grid-item-card:: Getting Started
      :link: installation
      :link-type: doc

      Install NEMO and run the pipeline on your first cube.

   .. grid-item-card:: Methodology
      :link: methodology
      :link-type: doc

      Starlet wavelet detection, masked optical flow, track linking,
      and false-detection removal — explained with equations.

.. grid:: 2

   .. grid-item-card:: Graphical Interface
      :link: gui
      :link-type: doc

      Load cubes, tune parameters, and run the full pipeline without writing
      any code.  Covers the four-card workspace, all viewer windows, and
      analysis tools.

   .. grid-item-card:: Results: W2246-0526
      :link: results
      :link-type: doc

      Application to ALMA [C II] observations of a hyper-luminous quasar at *z* = 4.6.

.. grid:: 2

   .. grid-item-card:: API Reference
      :link: autoapi/index
      :link-type: doc

      Auto-generated reference for all public classes and functions.

   .. grid-item-card:: CLI Reference
      :link: cli
      :link-type: doc

      Command-line tools for detection, tracking, and denoising.

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   quickstart
   methodology
   gui
   results
   cli
   autoapi/index
