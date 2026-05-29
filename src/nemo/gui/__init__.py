"""NEMO GUI — spectral cube browser with moment-0 preview and slice viewer.

Launch with::

    python -m nemo.gui

or from Python::

    from nemo.gui import launch
    launch()
"""
import matplotlib
matplotlib.use("TkAgg")

from .app import launch, NemoGUI
from .loaders import load_cube_file

__all__ = ["launch", "NemoGUI", "load_cube_file"]
