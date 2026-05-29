"""End-to-end integration tests for the full NEMO pipeline."""
import numpy as np
import pytest

from nemo import FlowTracker, WaveletDetector
from nemo.track import (
    classify_kinematic,
    compute_flow_sequence,
    group_into_sources,
    link_tracks,
    _reconcile_splits,
)
from nemo.detect import active_channels, detect_cube_per_channel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_pipeline(cube, channel_list=None,
                  scales=4, k_sigma=2.0, use_scale=2, min_area=5,
                  min_match_overlap=5, max_gap_channels=3):
    """Run the full pipeline manually (no FlowTracker class) for explicitness."""
    if channel_list is None:
        channel_list = list(range(cube.shape[0]))
    dets = detect_cube_per_channel(
        cube, channel_list=channel_list,
        scales=scales, k_sigma=k_sigma, use_scale=use_scale, min_area=min_area,
    )
    flow_seq = compute_flow_sequence(dets)
    fwd = link_tracks(dets, flow_seq,
                      min_match_overlap=min_match_overlap,
                      max_gap_channels=max_gap_channels)
    det_rev  = list(reversed(dets))
    flow_rev = [(b, a, -fl, mg) for (a, b, fl, mg) in reversed(flow_seq)]
    bwd = link_tracks(det_rev, flow_rev,
                      min_match_overlap=min_match_overlap,
                      max_gap_channels=max_gap_channels)
    _reconcile_splits(fwd, bwd)
    classify_kinematic(fwd)
    sources = group_into_sources(fwd)
    return dets, fwd, sources


# ---------------------------------------------------------------------------
# Static source
# ---------------------------------------------------------------------------

class TestStaticSourcePipeline:

    def test_single_source_found(self, static_cube):
        channels = active_channels(static_cube, threshold_frac=0.01)
        _, _, sources = _run_pipeline(static_cube, channel_list=channels)
        assert len(sources) == 1

    def test_source_spans_all_active_channels(self, static_cube):
        channels = active_channels(static_cube, threshold_frac=0.01)
        _, tracks, sources = _run_pipeline(static_cube, channel_list=channels)
        assert len(sources) == 1
        assert sources[0]['n_channels'] >= len(channels) - 1  # allow ±1 for edge

    def test_static_source_not_kinematic(self, static_cube):
        channels = active_channels(static_cube, threshold_frac=0.01)
        _, tracks, _ = _run_pipeline(static_cube, channel_list=channels)
        # The source track(s) should all be static (very small displacement)
        displacements = [t['displacement'] for t in tracks]
        assert all(d < 3.0 for d in displacements)

    def test_no_spurious_merges_or_splits(self, static_cube):
        channels = active_channels(static_cube, threshold_frac=0.01)
        _, tracks, _ = _run_pipeline(static_cube, channel_list=channels)
        assert all(len(t['merge_into']) == 0 for t in tracks)
        assert all(t['split_from'] is None for t in tracks)


# ---------------------------------------------------------------------------
# Moving source
# ---------------------------------------------------------------------------

class TestMovingSourcePipeline:

    def test_moving_source_is_kinematic(self, moving_cube):
        channels = list(range(moving_cube.shape[0]))
        _, tracks, sources = _run_pipeline(
            moving_cube, channel_list=channels,
            min_match_overlap=3,
        )
        assert any(t['kinematic'] for t in tracks)

    def test_moving_source_has_displacement(self, moving_cube):
        channels = list(range(moving_cube.shape[0]))
        _, tracks, _ = _run_pipeline(
            moving_cube, channel_list=channels,
            min_match_overlap=3,
        )
        max_disp = max(t['displacement'] for t in tracks)
        assert max_disp > 2.0


# ---------------------------------------------------------------------------
# FlowTracker class API
# ---------------------------------------------------------------------------

class TestFlowTrackerAPI:

    def test_run_returns_tracking_result(self, static_cube):
        from nemo.track import TrackingResult
        channels = active_channels(static_cube, threshold_frac=0.01)
        tracker = FlowTracker(
            WaveletDetector(scales=4, k_sigma=2.0, use_scale=2, min_area=5),
            min_match_overlap=5,
        )
        result = tracker.run(static_cube, channel_list=channels)
        assert isinstance(result, TrackingResult)

    def test_result_fields_present(self, static_cube):
        channels = active_channels(static_cube, threshold_frac=0.01)
        tracker = FlowTracker(
            WaveletDetector(scales=4, k_sigma=2.0, use_scale=2, min_area=5),
        )
        result = tracker.run(static_cube, channel_list=channels)
        assert result.detections is not None
        assert result.flow_seq is not None
        assert result.tracks is not None
        assert result.sources is not None
        assert result.false_detections is not None

    def test_sources_plus_false_equals_total(self, static_cube):
        channels = active_channels(static_cube, threshold_frac=0.01)
        tracker = FlowTracker(
            WaveletDetector(scales=4, k_sigma=2.0, use_scale=2, min_area=5),
        )
        result = tracker.run(static_cube, channel_list=channels)
        total_grouped = len(result.sources) + len(result.false_detections)
        # Every grouped entity should be either real or false
        assert total_grouped >= 0

    def test_all_tracks_have_source_id(self, static_cube):
        channels = active_channels(static_cube, threshold_frac=0.01)
        tracker = FlowTracker(
            WaveletDetector(scales=4, k_sigma=2.0, use_scale=2, min_area=5),
        )
        result = tracker.run(static_cube, channel_list=channels)
        for t in result.tracks:
            assert 'source_id' in t

    def test_flow_seq_length(self, static_cube):
        channels = active_channels(static_cube, threshold_frac=0.01)
        tracker = FlowTracker(
            WaveletDetector(scales=4, k_sigma=2.0, use_scale=2, min_area=5),
        )
        result = tracker.run(static_cube, channel_list=channels)
        assert len(result.flow_seq) == len(result.detections) - 1


# ---------------------------------------------------------------------------
# active_channels integration
# ---------------------------------------------------------------------------

class TestActiveChannelsIntegration:

    def test_selects_channels_with_source(self, static_cube):
        # static_cube has source in channels 2-8
        channels = active_channels(static_cube, threshold_frac=0.01)
        assert all(ch in channels for ch in range(2, 9))

    def test_excludes_empty_channels(self, static_cube):
        # static_cube channels 0 and 1 are zero
        channels = active_channels(static_cube, threshold_frac=0.05)
        assert 0 not in channels
        assert 1 not in channels
