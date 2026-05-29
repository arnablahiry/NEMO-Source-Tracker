"""Tests for nemo.track — flow, link_tracks, reconcile, classify, group_into_sources."""
import numpy as np
import pytest

from nemo.track import (
    _advect_mask,
    _reconcile_splits,
    classify_kinematic,
    compute_flow_sequence,
    group_into_sources,
    link_tracks,
    masked_flow_tvl1,
)
from tests.conftest import make_detection, zero_flow_seq


# ---------------------------------------------------------------------------
# masked_flow_tvl1
# ---------------------------------------------------------------------------

class TestMaskedFlowTvl1:
    def test_output_shape(self):
        H, W = 20, 20
        img = np.random.rand(H, W).astype(np.float32)
        mask = np.ones((H, W), dtype=bool)
        flow = masked_flow_tvl1(img, img, mask)
        assert flow.shape == (2, H, W)

    def test_output_dtype(self):
        H, W = 16, 16
        img = np.random.rand(H, W).astype(np.float32)
        mask = np.ones((H, W), dtype=bool)
        flow = masked_flow_tvl1(img, img, mask)
        assert flow.dtype == np.float32

    def test_zero_outside_mask(self):
        H, W = 20, 20
        img = np.random.rand(H, W).astype(np.float32)
        mask = np.zeros((H, W), dtype=bool)
        mask[5:15, 5:15] = True
        flow = masked_flow_tvl1(img, img, mask)
        assert (flow[:, ~mask] == 0.0).all()

    def test_identical_images_near_zero_flow(self):
        """Identical images should produce flow close to zero."""
        H, W = 20, 20
        img = np.zeros((H, W), dtype=np.float32)
        img[8:12, 8:12] = 1.0
        mask = np.ones((H, W), dtype=bool)
        flow = masked_flow_tvl1(img, img, mask)
        assert float(np.abs(flow).mean()) < 0.5


# ---------------------------------------------------------------------------
# compute_flow_sequence
# ---------------------------------------------------------------------------

class TestComputeFlowSequence:
    def _make_detections(self, n_ch, H=16, W=16):
        return [make_detection(ch, H, W, [(8, 8)]) for ch in range(n_ch)]

    def test_length(self):
        dets = self._make_detections(5)
        seq = compute_flow_sequence(dets)
        assert len(seq) == 4  # n_ch - 1

    def test_tuple_structure(self):
        dets = self._make_detections(3)
        seq = compute_flow_sequence(dets)
        for item in seq:
            ch_ref, ch_tgt, flow, mask = item
            assert isinstance(ch_ref, int)
            assert isinstance(ch_tgt, int)
            assert ch_tgt == ch_ref + 1
            assert flow.shape == (2, 16, 16)
            assert mask.shape == (16, 16)

    def test_consecutive_channels(self):
        dets = self._make_detections(4)
        seq = compute_flow_sequence(dets)
        for i, (ch_ref, ch_tgt, _, _) in enumerate(seq):
            assert ch_ref == i
            assert ch_tgt == i + 1

    def test_empty_mask_gives_zero_flow(self):
        """Channels with no detections produce zero flow."""
        H, W = 16, 16
        d0 = make_detection(0, H, W, [])  # no detections
        d1 = make_detection(1, H, W, [])
        seq = compute_flow_sequence([d0, d1])
        _, _, flow, _ = seq[0]
        np.testing.assert_array_equal(flow, 0.0)


# ---------------------------------------------------------------------------
# _advect_mask
# ---------------------------------------------------------------------------

class TestAdvectMask:
    def test_zero_flow_unchanged(self):
        """Zero flow must not move the mask."""
        H, W = 16, 16
        mask = np.zeros((H, W), dtype=bool)
        mask[6:10, 6:10] = True
        flow = np.zeros((2, H, W), dtype=np.float32)
        result = _advect_mask(mask, flow)
        assert result.shape == (H, W)
        # Every pixel in the mask should map back to itself
        assert (result[6:10, 6:10] > 0).all()

    def test_output_shape(self):
        H, W = 20, 20
        mask = np.zeros((H, W), dtype=bool)
        mask[8:12, 8:12] = True
        flow = np.zeros((2, H, W), dtype=np.float32)
        out = _advect_mask(mask, flow)
        assert out.shape == (H, W)

    def test_empty_mask_stays_empty(self):
        H, W = 16, 16
        mask = np.zeros((H, W), dtype=bool)
        flow = np.zeros((2, H, W), dtype=np.float32)
        out = _advect_mask(mask, flow)
        assert out.sum() == 0.0

    def test_shift_by_integer_flow(self):
        """Uniform +2 row flow should shift all mask pixels 2 rows down."""
        H, W = 20, 20
        mask = np.zeros((H, W), dtype=bool)
        mask[5:8, 8:11] = True
        flow = np.zeros((2, H, W), dtype=np.float32)
        flow[0] = 2.0  # shift 2 rows
        out = _advect_mask(mask, flow)
        # Weight map should be concentrated around row 7-9
        rows_with_hits = np.where(out.sum(axis=1) > 0)[0]
        assert rows_with_hits.min() >= 6
        assert rows_with_hits.max() <= 10


# ---------------------------------------------------------------------------
# link_tracks
# ---------------------------------------------------------------------------

class TestLinkTracks:

    # --- single persistent source ---

    def test_single_static_source_one_track(self):
        H, W = 20, 20
        dets = [make_detection(ch, H, W, [(10, 10)]) for ch in range(5)]
        seq = zero_flow_seq(dets)
        tracks = link_tracks(dets, seq, min_match_overlap=5)
        assert len(tracks) == 1

    def test_single_source_trajectory_length(self):
        H, W = 20, 20
        dets = [make_detection(ch, H, W, [(10, 10)]) for ch in range(5)]
        seq = zero_flow_seq(dets)
        tracks = link_tracks(dets, seq, min_match_overlap=5)
        assert len(tracks[0]['trajectory']) == 5

    def test_single_source_correct_channels(self):
        H, W = 20, 20
        dets = [make_detection(ch, H, W, [(10, 10)]) for ch in range(5)]
        seq = zero_flow_seq(dets)
        tracks = link_tracks(dets, seq, min_match_overlap=5)
        channels_in_traj = [pt[0] for pt in tracks[0]['trajectory']]
        assert channels_in_traj == list(range(5))

    # --- two independent sources ---

    def test_two_independent_sources_two_tracks(self):
        H, W = 32, 32
        dets = [make_detection(ch, H, W, [(8, 8), (24, 24)]) for ch in range(4)]
        seq = zero_flow_seq(dets)
        tracks = link_tracks(dets, seq, min_match_overlap=5)
        assert len(tracks) == 2

    def test_two_sources_no_merge(self):
        H, W = 32, 32
        dets = [make_detection(ch, H, W, [(8, 8), (24, 24)]) for ch in range(4)]
        seq = zero_flow_seq(dets)
        tracks = link_tracks(dets, seq, min_match_overlap=5)
        assert all(len(t['merge_into']) == 0 for t in tracks)

    # --- gap bridging ---

    def test_source_survives_short_gap(self):
        """Track should survive when source disappears for fewer than max_gap_channels."""
        H, W = 20, 20
        dets = [
            make_detection(0, H, W, [(10, 10)]),
            make_detection(1, H, W, []),           # gap: no detection
            make_detection(2, H, W, [(10, 10)]),
        ]
        seq = zero_flow_seq(dets)
        tracks = link_tracks(dets, seq, min_match_overlap=5, max_gap_channels=2)
        active_track = [t for t in tracks if t['active']]
        assert len(active_track) == 1

    def test_source_deactivates_after_long_gap(self):
        """Track should deactivate when source is gone longer than max_gap_channels."""
        H, W = 20, 20
        dets = (
            [make_detection(ch, H, W, [(10, 10)]) for ch in range(2)]
            + [make_detection(ch, H, W, []) for ch in range(2, 7)]   # 5-channel gap
        )
        seq = zero_flow_seq(dets)
        tracks = link_tracks(dets, seq, min_match_overlap=5, max_gap_channels=2)
        # Original source track should be deactivated
        original_track = next(t for t in tracks if t['trajectory'][0][0] == 0)
        assert not original_track['active']

    # --- merge detection ---

    def test_merge_recorded_in_merge_into(self):
        """Two nearby sources converging to one detection should record a merge."""
        H, W = 32, 32
        # Ch0: two close sources (overlapping masks with radius=5)
        # Ch1+: single source at midpoint — both tracks advect into it
        dets = [
            make_detection(0, H, W, [(10, 16), (14, 16)], radius=5),
            make_detection(1, H, W, [(12, 16)], radius=5),
            make_detection(2, H, W, [(12, 16)], radius=5),
        ]
        seq = zero_flow_seq(dets)
        tracks = link_tracks(dets, seq, min_match_overlap=5)
        merging = [t for t in tracks if t['merge_into']]
        assert len(merging) >= 1

    # --- new track for unmatched detection ---

    def test_new_detection_starts_new_track(self):
        """A component that appears mid-sequence starts a new track."""
        H, W = 32, 32
        dets = [
            make_detection(0, H, W, [(8, 8)]),
            make_detection(1, H, W, [(8, 8), (24, 24)]),  # second source appears
            make_detection(2, H, W, [(8, 8), (24, 24)]),
        ]
        seq = zero_flow_seq(dets)
        tracks = link_tracks(dets, seq, min_match_overlap=5)
        assert len(tracks) == 2

    # --- track dict fields ---

    def test_track_dict_has_required_keys(self):
        H, W = 20, 20
        dets = [make_detection(ch, H, W, [(10, 10)]) for ch in range(3)]
        seq = zero_flow_seq(dets)
        tracks = link_tracks(dets, seq)
        required = {'id', 'trajectory', 'masks', 'split_at', 'split_from',
                    'merge_into', 'active', 'gap_age'}
        for t in tracks:
            assert required.issubset(set(t.keys()))


# ---------------------------------------------------------------------------
# _reconcile_splits
# ---------------------------------------------------------------------------

class TestReconcileSplits:

    def _build_track(self, tid, trajectory, split_from=None, merge_into=None):
        return {
            'id': tid,
            'trajectory': trajectory,
            'split_at': [],
            'split_from': split_from,
            'merge_into': merge_into or [],
            'active': True,
            'gap_age': 0,
            'masks': {},
        }

    def test_backward_merge_becomes_forward_split(self):
        """Backward track merging into another → forward child gets split_from.

        Positions are chosen 15 px apart so trajectory-vote matching doesn't
        cross-assign tracks (tol_px default is 5 px).
        """
        # Forward tracks: track 0 (parent at y=5) and track 1 (child at y=20)
        fwd_0 = self._build_track(0, [(0, 5.0, 5.0), (1, 5.0, 5.0), (2, 5.0, 5.0)])
        fwd_1 = self._build_track(1, [(2, 20.0, 5.0), (3, 20.0, 5.0)])
        fwd_tracks = [fwd_0, fwd_1]

        # Backward tracks mirror the forward trajectories (reversed channel order)
        # bwd_1 corresponds to fwd_1; it merges into bwd_0 (fwd_0) at ch 2
        bwd_0 = self._build_track(0, [(3, 5.0, 5.0), (2, 5.0, 5.0), (1, 5.0, 5.0)])
        bwd_1 = self._build_track(1, [(3, 20.0, 5.0), (2, 20.0, 5.0)],
                                  merge_into=[(2, 0)])  # merges into bwd_0 at ch 2
        bwd_tracks = [bwd_0, bwd_1]

        _reconcile_splits(fwd_tracks, bwd_tracks, tol_px=3.0)

        # fwd_1 (child) should have split_from pointing to fwd_0 (parent)
        assert fwd_1['split_from'] == fwd_0['id']

    def test_no_merge_no_split(self):
        """No merge events in backward pass → no splits annotated."""
        fwd_0 = self._build_track(0, [(0, 8.0, 8.0), (1, 8.0, 8.0)])
        fwd_1 = self._build_track(1, [(0, 20.0, 20.0), (1, 20.0, 20.0)])
        bwd_0 = self._build_track(0, [(1, 8.0, 8.0), (0, 8.0, 8.0)])
        bwd_1 = self._build_track(1, [(1, 20.0, 20.0), (0, 20.0, 20.0)])

        _reconcile_splits([fwd_0, fwd_1], [bwd_0, bwd_1])

        assert fwd_0['split_from'] is None
        assert fwd_1['split_from'] is None

    def test_returns_fwd_tracks(self):
        fwd = [self._build_track(0, [(0, 5.0, 5.0)])]
        bwd = [self._build_track(0, [(0, 5.0, 5.0)])]
        result = _reconcile_splits(fwd, bwd)
        assert result is fwd


# ---------------------------------------------------------------------------
# classify_kinematic
# ---------------------------------------------------------------------------

class TestClassifyKinematic:

    def _make_track(self, trajectory, split_at=None, split_from=None, merge_into=None):
        return {
            'id': 0,
            'trajectory': trajectory,
            'split_at': split_at or [],
            'split_from': split_from,
            'merge_into': merge_into or [],
            'active': True,
            'gap_age': 0,
            'masks': {},
        }

    def test_static_track_not_kinematic(self):
        t = self._make_track([(i, 10.0, 10.0) for i in range(5)])
        classify_kinematic([t], min_displacement=3.0)
        assert not t['kinematic']
        assert t['displacement'] == pytest.approx(0.0)

    def test_moving_track_is_kinematic(self):
        # Moves 2 px per channel → total displacement = 8 px over 5 steps
        t = self._make_track([(i, 10.0, 10.0 + 2*i) for i in range(5)])
        classify_kinematic([t], min_displacement=3.0)
        assert t['kinematic']
        assert t['displacement'] == pytest.approx(8.0, abs=0.1)

    def test_split_track_is_kinematic_regardless_of_displacement(self):
        t = self._make_track([(i, 10.0, 10.0) for i in range(3)],
                             split_at=[1])  # involved in split
        classify_kinematic([t], min_displacement=100.0)  # high threshold
        assert t['kinematic']
        assert t['has_split']

    def test_merge_track_is_kinematic(self):
        t = self._make_track([(i, 10.0, 10.0) for i in range(3)],
                             merge_into=[(2, 1)])
        classify_kinematic([t], min_displacement=100.0)
        assert t['kinematic']

    def test_displacement_field_set(self):
        t = self._make_track([(0, 0.0, 0.0), (1, 3.0, 4.0)])  # 5 px step
        classify_kinematic([t])
        assert t['displacement'] == pytest.approx(5.0, abs=0.1)

    def test_has_split_false_for_clean_track(self):
        t = self._make_track([(i, 5.0, 5.0) for i in range(4)])
        classify_kinematic([t])
        assert not t['has_split']


# ---------------------------------------------------------------------------
# group_into_sources
# ---------------------------------------------------------------------------

class TestGroupIntoSources:

    def _track(self, tid, trajectory, split_from=None, merge_into=None,
               split_at=None):
        return {
            'id': tid,
            'trajectory': trajectory,
            'split_from': split_from,
            'merge_into': merge_into or [],
            'split_at': split_at or [],
            'active': True,
            'gap_age': 0,
            'masks': {},
            'kinematic': False,
            'has_split': False,
            'displacement': 0.0,
        }

    def test_independent_tracks_separate_sources(self):
        tracks = [
            self._track(0, [(ch, 5.0, 5.0) for ch in range(3)]),
            self._track(1, [(ch, 15.0, 15.0) for ch in range(3)]),
            self._track(2, [(ch, 25.0, 5.0) for ch in range(3)]),
        ]
        sources = group_into_sources(tracks)
        assert len(sources) == 3

    def test_split_tracks_same_source(self):
        parent = self._track(0, [(ch, 10.0, 10.0) for ch in range(4)],
                             split_at=[2])
        child  = self._track(1, [(ch, 10.0, 14.0) for ch in range(2, 4)],
                             split_from=0)
        sources = group_into_sources([parent, child])
        assert len(sources) == 1

    def test_merge_tracks_same_source(self):
        t0 = self._track(0, [(ch, 8.0, 8.0) for ch in range(4)],
                         merge_into=[(3, 1)])
        t1 = self._track(1, [(ch, 12.0, 8.0) for ch in range(4)])
        sources = group_into_sources([t0, t1])
        assert len(sources) == 1

    def test_chain_a_splits_from_b_merges_into_c(self):
        """Indirect chain: A→B (split), B→C (merge) → all in same source."""
        a = self._track(0, [(ch, 5.0, 5.0) for ch in range(3)], split_from=1)
        b = self._track(1, [(ch, 5.0, 5.0) for ch in range(5)],
                        split_at=[1], merge_into=[(4, 2)])
        c = self._track(2, [(ch, 5.0, 5.0) for ch in range(5)])
        sources = group_into_sources([a, b, c])
        assert len(sources) == 1

    def test_source_id_annotated_on_tracks(self):
        tracks = [
            self._track(0, [(0, 5.0, 5.0)]),
            self._track(1, [(0, 15.0, 5.0)]),
        ]
        group_into_sources(tracks)
        assert 'source_id' in tracks[0]
        assert 'source_id' in tracks[1]

    def test_source_track_ids_field(self):
        parent = self._track(0, [(0, 5.0, 5.0), (1, 5.0, 5.0)], split_at=[1])
        child  = self._track(1, [(1, 5.0, 8.0)], split_from=0)
        sources = group_into_sources([parent, child])
        assert set(sources[0]['track_ids']) == {0, 1}

    def test_source_channels_field(self):
        t = self._track(0, [(0, 5.0, 5.0), (1, 5.0, 5.0), (2, 5.0, 5.0)])
        t['masks'] = {0: None, 1: None, 2: None}
        sources = group_into_sources([t])
        assert set(sources[0]['channels']) == {0, 1, 2}

    def test_split_events_recorded(self):
        parent = self._track(0, [(ch, 5.0, 5.0) for ch in range(4)],
                             split_at=[2])
        child  = self._track(1, [(2, 5.0, 8.0), (3, 5.0, 8.0)], split_from=0)
        sources = group_into_sources([parent, child])
        assert 2 in sources[0]['split_events']

    def test_returns_sorted_by_id(self):
        tracks = [self._track(i, [(0, float(i)*5, 0.0)]) for i in range(4)]
        sources = group_into_sources(tracks)
        ids = [s['id'] for s in sources]
        assert ids == sorted(ids)
