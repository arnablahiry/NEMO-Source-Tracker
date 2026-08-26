"""Bounding boxes must never extend past the image frame.

A box drawn outside the data extent makes matplotlib grow the axes, which pads
the rendered figure with background colour and shrinks the image inside it —
visible as white/black bands around the preview.  These tests pin both the
geometry helper and the rendering behaviour it exists to protect.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402

from nemo.utils import clamped_bbox                  # noqa: E402

H = W = 186


class TestClampedBbox:
    def test_interior_box_is_padded_normally(self):
        x, y, w, h, lx, ly = clamped_bbox(40, 50, 30, 40, 5, (H, W))
        assert (x, y) == (25, 35)
        assert (w, h) == (20, 20)

    @pytest.mark.parametrize("r0,r1,c0,c1", [
        (0, 10, 0, 10),           # bottom-left corner
        (H - 11, H - 1, 0, 10),   # top-left
        (0, 10, W - 11, W - 1),   # bottom-right
        (H - 11, H - 1, W - 11, W - 1),  # top-right
    ])
    def test_box_never_escapes_frame(self, r0, r1, c0, c1):
        x, y, w, h, lx, ly = clamped_bbox(r0, r1, c0, c1, 8, (H, W))
        assert x >= 0 and y >= 0
        assert x + w <= W - 1
        assert y + h <= H - 1

    @pytest.mark.parametrize("r0,r1,c0,c1", [
        (0, 10, 0, 10),
        (H - 11, H - 1, W - 11, W - 1),
    ])
    def test_label_stays_inside_frame(self, r0, r1, c0, c1):
        """An off-image label is clipped away and the source goes unnamed."""
        *_, lx, ly = clamped_bbox(r0, r1, c0, c1, 8, (H, W))
        assert 0 <= lx <= W - 1
        assert 0 <= ly <= H - 1

    def test_degenerate_single_pixel_source(self):
        x, y, w, h, _, _ = clamped_bbox(5, 5, 5, 5, 0, (H, W))
        assert w >= 1 and h >= 1, "zero-size rectangles render as nothing"

    def test_source_larger_than_frame(self):
        x, y, w, h, _, _ = clamped_bbox(-20, H + 20, -20, W + 20, 10, (H, W))
        assert (x, y) == (0, 0)
        assert x + w <= W - 1 and y + h <= H - 1


class TestNoFramePadding:
    """Renders with the same axes setup the GIF renderers use."""

    @staticmethod
    def _render(box):
        img = np.random.default_rng(0).random((H, W))
        mask = np.zeros((H, W), dtype=bool)
        mask[H - 16:H, 0:20] = True          # source flush against the edge
        fig = plt.Figure(figsize=(4, 4), dpi=96, facecolor="#0a0a14")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.imshow(img, origin="lower")
        ax.contour(mask.astype(float), [0.5], colors=["r"], linewidths=0.7)
        if box is not None:
            x, y, w, h, lx, ly = box
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="r"))
            ax.text(lx, ly, "S1")
        fig.canvas.draw()
        xl, yl = ax.get_xlim(), ax.get_ylim()
        plt.close(fig)
        return (xl[1] - xl[0]) - W, (yl[1] - yl[0]) - H

    def test_contours_alone_add_no_padding(self):
        assert self._render(None) == (0.0, 0.0)

    def test_unclamped_box_pads_the_frame(self):
        """Regression guard: this is the bug, and it must stay reproducible."""
        r0, r1, c0, c1, pad = H - 16, H - 1, 0, 19, 4
        raw = (c0 - pad, r0 - pad, c1 - c0 + 2 * pad, r1 - r0 + 2 * pad,
               c1 + pad, r1 + pad)
        dx, dy = self._render(raw)
        assert dx > 0 or dy > 0, "unclamped box should have expanded the axes"

    def test_clamped_box_adds_no_padding(self):
        box = clamped_bbox(H - 16, H - 1, 0, 19, 4, (H, W))
        assert self._render(box) == (0.0, 0.0)
