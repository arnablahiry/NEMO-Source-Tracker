"""NEMO — Non-stationary Extraction via Multiscale Optical-flow.

Typical usage::

    from nemo import WaveletDetector, FlowTracker

    detector = WaveletDetector(scales=6, k_sigma=5.0, use_scale=5)
    tracker  = FlowTracker(detector, min_match_overlap=5)
    result   = tracker.run(cube, channel_list, verbose=True)

    result.sources          # real sources (false detections removed)
    result.tracks           # all individual tracks
    result.false_detections # flagged false positives
"""

from .detect import (
    ChannelDetection,
    WaveletDetector,
    load_cube,
    active_channels,
    detect_cube_per_channel,
    detect_all_scales,
    wavelet_footprints,
    reference_sigmas_from_mean_map,
)
from .gui import NemoGUI, launch
from .hierarchy import (
    HierarchicalSourceGroup,
    ScaleDetection,
    PerChannelScaleDetections,
    build_hierarchical_sources,
)
from .track import (
    FlowTracker,
    TrackingResult,
    run_flow_tracker,
    compute_flow_sequence,
    link_tracks,
    link_tracks_per_scale,
    classify_kinematic,
    group_into_sources,
    source_per_scale,
    classify_sources,
    masked_flow_tvl1,
)

__all__ = [
    # GUI
    "NemoGUI",
    "launch",
    # Primary API
    "WaveletDetector",
    "FlowTracker",
    "TrackingResult",
    # Data containers
    "ChannelDetection",
    "HierarchicalSourceGroup",
    "ScaleDetection",
    "PerChannelScaleDetections",
    # I/O helpers
    "load_cube",
    "active_channels",
    # Lower-level functions (for advanced use)
    "detect_cube_per_channel",
    "detect_all_scales",
    "wavelet_footprints",
    "reference_sigmas_from_mean_map",
    "run_flow_tracker",
    "compute_flow_sequence",
    "link_tracks",
    "link_tracks_per_scale",
    "classify_kinematic",
    "group_into_sources",
    "source_per_scale",
    "classify_sources",
    "build_hierarchical_sources",
    "masked_flow_tvl1",
]
