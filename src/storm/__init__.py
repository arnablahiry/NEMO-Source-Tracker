"""STORM — Source Tracking via Optical-flow and Resolved Multiscale-wavelets.

Main entry point::

    from storm.track import run_flow_tracker
    detections, flow_seq, tracks, sources, good_sources, false_dets, src_data, src_colors = (
        run_flow_tracker(cube, channel_list=channel_list, verbose=True)
    )
"""

from .detect import (
    ChannelDetection,
    load_cube,
    active_channels,
    detect_cube_per_channel,
    wavelet_footprints_scarlet2,
    reference_sigmas_from_mean_map,
)
from .track import (
    run_flow_tracker,
    compute_flow_sequence,
    link_tracks,
    classify_kinematic,
    group_into_sources,
    classify_sources,
    masked_flow_tvl1,
)

__all__ = [
    # detect
    "ChannelDetection",
    "load_cube",
    "active_channels",
    "detect_cube_per_channel",
    "wavelet_footprints_scarlet2",
    "reference_sigmas_from_mean_map",
    # track
    "run_flow_tracker",
    "compute_flow_sequence",
    "link_tracks",
    "classify_kinematic",
    "group_into_sources",
    "classify_sources",
    "masked_flow_tvl1",
]
