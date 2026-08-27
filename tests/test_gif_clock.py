"""Tests for the shared GIF clock in the main window.

The clock steps every card through the same channel each tick.  Cards cover
different channel ranges, so the choice of union vs intersection decides
whether some cards run out of frames mid-loop and appear to freeze.
"""
import pytest

from nemo.gui.app import NemoGUI


master = NemoGUI._master_channels


class TestMasterChannels:
    def test_overlapping_lists_intersect(self):
        out = master([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4]])
        assert out == [3, 4]

    def test_no_card_is_ever_missing_a_frame(self):
        """The property the whole fix exists for."""
        lists = [[10, 11, 12, 13, 14], [10, 11, 12, 13], [11, 12, 13, 14]]
        out = master(lists)
        for chs in lists:
            assert set(out) <= set(chs)

    def test_flow_card_short_by_its_last_channel(self):
        """Flow is computed between consecutive pairs, so it lacks the last."""
        detections = [0, 1, 2, 3, 4]
        flow = [0, 1, 2, 3]                 # one fewer, by construction
        out = master([detections, flow])
        assert 4 not in out, "the channel flow cannot render must be dropped"
        assert out == [0, 1, 2, 3]

    def test_narrower_sources_card_trims_both_ends(self):
        out = master([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4], [2, 3, 4]])
        assert out == [2, 3, 4]

    def test_card_with_no_frames_is_ignored(self):
        """An empty card must not collapse the intersection to nothing."""
        out = master([[1, 2, 3], [], [2, 3, 4]])
        assert out == [2, 3]

    def test_all_cards_empty_gives_nothing(self):
        assert master([[], [], []]) == []
        assert master([]) == []

    def test_disjoint_lists_fall_back_to_the_union(self):
        """Rather than freezing the animation on a single image."""
        out = master([[0, 1, 2], [7, 8, 9]])
        assert out == [0, 1, 2, 7, 8, 9]

    def test_single_common_frame_falls_back_to_the_union(self):
        """One shared channel is not an animation; prefer the old behaviour."""
        out = master([[1, 2, 3], [3, 4, 5]])
        assert out == [1, 2, 3, 4, 5]

    def test_single_card_keeps_all_its_channels(self):
        assert master([[4, 5, 6]]) == [4, 5, 6]

    def test_result_is_sorted_and_deduplicated(self):
        out = master([[3, 1, 2, 2], [1, 2, 3]])
        assert out == [1, 2, 3]
