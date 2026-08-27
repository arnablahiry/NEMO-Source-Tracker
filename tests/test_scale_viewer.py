"""Tests for the Configure Detection window (ScaleViewer).

The coarse residual is a selectable detection band like any other, so it has to
toggle, highlight, and reach the detector.
"""
import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from nemo.gui.viewers import ScaleViewer, _accent   # noqa: E402


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError:                       # pragma: no cover - no display
        pytest.skip("no display available")
    r.geometry("1200x900")
    yield r
    r.destroy()


class _Card0:
    beam = (2.0, 1.0, 30.0)
    pixscale = 0.25


@pytest.fixture
def viewer(root):
    rng = np.random.default_rng(0)
    H = W = 128
    ys, xs = np.mgrid[0:H, 0:W]
    blob = np.exp(-((ys - 64) ** 2 + (xs - 64) ** 2) / (2 * 4.0 ** 2))
    cube = (rng.normal(0, 0.02, (6, H, W)) + blob).astype(np.float32)
    v = ScaleViewer(root, cube, wav_params={}, card_0=_Card0())
    v._cleanup_canvas = lambda: None
    root.update_idletasks()
    yield v


class TestCoarseScale:
    def test_coarse_pill_exists_and_is_labelled_C(self, viewer):
        labels = [w._label for w in viewer._rf.winfo_children()[-1].winfo_children()
                  if hasattr(w, "_label")]
        assert labels[-1] == "C"

    def test_coarse_starts_unselected(self, viewer):
        n = viewer._n_scales_var.get()
        assert viewer._scale_selections[n].get() is False

    def test_coarse_can_be_selected(self, viewer):
        n = viewer._n_scales_var.get()
        viewer._scale_selections[n].set(True)
        assert viewer._scale_selections[n].get() is True

    def test_selecting_coarse_highlights_its_panel(self, viewer):
        """Regression: the highlight set stopped at n_detail, so C never lit."""
        n = viewer._n_scales_var.get()
        ax = viewer._panel_figs[n]._nemo_ax

        viewer._scale_selections[n].set(False)
        viewer._draw()
        off_colour = ax.title.get_color()
        off_lw = list(ax.spines.values())[0].get_linewidth()

        viewer._scale_selections[n].set(True)
        viewer._draw()
        on_colour = ax.title.get_color()
        on_lw = list(ax.spines.values())[0].get_linewidth()

        assert on_colour == _accent(), "selected coarse title must use the accent"
        assert on_colour != off_colour
        assert on_lw > off_lw, "selected panel border must thicken"

    def test_coarse_panel_is_titled_coarse_scale(self, viewer):
        n = viewer._n_scales_var.get()
        viewer._draw()
        assert viewer._panel_figs[n]._nemo_ax.get_title() == "Coarse Scale"

    def test_selected_coarse_reaches_detect_scales(self, viewer):
        n = viewer._n_scales_var.get()
        viewer._scale_selections[n].set(True)
        saved = {}
        viewer._on_params_saved = saved.update
        viewer.destroy = lambda: None
        viewer._save_params()
        assert n in saved["detect_scales"]

    def test_detail_scales_still_highlight(self, viewer):
        """The fix must not disturb the ordinary bands."""
        viewer._scale_selections[3].set(True)
        viewer._draw()
        assert viewer._panel_figs[3]._nemo_ax.title.get_color() == _accent()


class TestPanelLayout:
    def test_one_panel_per_scale(self, viewer):
        n = viewer._n_scales_var.get()
        assert sorted(viewer._panel_figs) == list(range(1, n + 1))

    def test_panels_are_square(self, viewer):
        for fig in viewer._panel_figs.values():
            w, h = fig.get_size_inches()
            assert abs(w - h) < 1e-6
