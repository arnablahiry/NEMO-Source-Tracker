Installation
============

Requirements
------------

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy, SciPy, scikit-image, Astropy, h5py, Matplotlib

From source
-----------

.. code-block:: bash

   git clone https://github.com/arnablahiry/nemo.git
   cd nemo
   pip install -e .

Supported input formats
-----------------------

:func:`nemo.load_cube` accepts ``*.fits``, ``*.fit``, ``*.h5``, ``*.hdf5``,
``*.npy``, and ``*.npz`` files. All data are cast to ``float32 (n_ch, H, W)``
with NaNs replaced by zero.
