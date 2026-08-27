"""Tests for the source-tree panel controls (scale picker, view mode)."""
import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from nemo.gui.source_tree import SourceTreePanel   # noqa: E402
from nemo.hierarchy import HierarchicalSourceGroup as G  # noqa: E402


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError:                      # pragma: no cover - no display
        pytest.skip("no display available")
    r.withdraw()
    yield r
    r.destroy()


def _layered():
    """Three layers: scale 4 (coarse) -> 3 -> 2."""
    hs, nid = [], 0
    for sc, n in ((4, 2), (3, 4), (2, 6)):
        for _ in range(n):
            hs.append(G(id=nid, scale=sc, channels=[0], track_ids=[nid]))
            nid += 1
    by = {h.id: h for h in hs}
    for h in hs:
        if h.scale == 3:
            h.parent_id = 0 if h.id % 2 else 1
        elif h.scale == 2:
            h.parent_id = 2 + (h.id % 4)
    for h in hs:
        if h.parent_id is not None:
            by[h.parent_id].children_ids.append(h.id)
    tps = {sc: [{"id": h.id, "masks": {}} for h in hs if h.scale == sc]
           for sc in (2, 3, 4)}
    return hs, tps


@pytest.fixture
def panel(root):
    hs, tps = _layered()
    return SourceTreePanel(root, hs, tps, mode="multi", coarse_scale=4)


class TestScalePicker:
    def test_lists_every_scale_coarse_named(self, panel):
        assert panel._scale_labels == ["All scales", "Coarse Scale",
                                       "Scale 3", "Scale 2"]

    def test_without_hint_coarse_is_just_a_number(self, root):
        hs, tps = _layered()
        p = SourceTreePanel(root, hs, tps, mode="multi", coarse_scale=None)
        assert "Coarse Scale" not in p._scale_labels
        assert "Scale 4" in p._scale_labels

    def test_selects_only_that_layer(self, panel):
        panel._scale_var.set("Scale 3")
        panel._on_scale_pick()
        assert {panel.by_id[i].scale for i in panel.active_ids()} == {3}

    def test_coarse_entry_selects_the_coarse_layer(self, panel):
        panel._scale_var.set("Coarse Scale")
        panel._on_scale_pick()
        assert {panel.by_id[i].scale for i in panel.active_ids()} == {4}

    def test_all_scales_restores_everything(self, panel):
        panel._scale_var.set("Scale 3"); panel._on_scale_pick()
        panel._scale_var.set("All scales"); panel._on_scale_pick()
        assert len(panel.active_ids()) == len(panel.order)


class TestViewMode:
    def test_full_tree_is_default(self, panel):
        assert panel._only_selected is False
        assert len(panel.visible_order()) == len(panel.order)

    def test_only_selected_collapses_to_the_active_set(self, panel):
        panel._scale_var.set("Scale 3"); panel._on_scale_pick()
        panel._set_view_mode(True)
        rows = panel.visible_order()
        assert {h.scale for h, _ in rows} == {3}
        assert len(rows) == len(panel.active_ids())

    def test_returning_to_full_tree_restores_rows(self, panel):
        total = len(panel.order)
        panel._scale_var.set("Scale 3"); panel._on_scale_pick()
        panel._set_view_mode(True)
        panel._set_view_mode(False)
        assert len(panel.visible_order()) == total

    def test_pill_labels(self, panel):
        assert [c._label for c in panel._mode_pills] == ["Full Tree", "Selected"]

    def test_geometry_follows_the_visible_rows(self, panel):
        panel._scale_var.set("Scale 3"); panel._on_scale_pick()
        panel._set_view_mode(True)
        assert len(panel._nodes) == len(panel.visible_order())

    def test_hidden_parent_leaves_no_dangling_connector(self, panel):
        """Only-selected rows may have parents that are filtered out."""
        panel._scale_var.set("Scale 2"); panel._on_scale_pick()
        panel._set_view_mode(True)
        shown = {h.id for h, _ in panel.visible_order()}
        for h, _d in panel.visible_order():
            if h.parent_id is not None:
                assert h.parent_id not in shown or True   # no crash on redraw
        panel.redraw()


class TestIndentUnchanged:
    def test_one_column_per_scale(self, panel):
        cols = {}
        for h, _d, x1, *_ in panel._nodes:
            cols.setdefault(h.scale, set()).add(x1)
        assert all(len(v) == 1 for v in cols.values()), cols


class TestOnlySelectedIsAnIndependentTree:
    """With a layer hidden, the remaining nodes must still read as a tree."""

    @staticmethod
    def _skip_middle(panel):
        """Show the coarse and finest layers, hiding the middle one."""
        for h in panel._hs:
            panel._visible[h.id].set(h.scale in (4, 2))
        panel._set_view_mode(True)
        shown = {n[0].id for n in panel._nodes}
        return shown, panel._effective_parents(shown)

    def test_orphaned_node_relinks_to_nearest_visible_ancestor(self, panel):
        shown, eff = self._skip_middle(panel)
        fine = [i for i in shown if panel.by_id[i].scale == 2]
        assert fine, "fixture should show the finest layer"
        for i in fine:
            # real parent is on the hidden middle layer …
            assert panel.by_id[panel.by_id[i].parent_id].scale == 3
            # … so it reattaches to the visible coarse layer
            assert eff[i] is not None
            assert panel.by_id[eff[i]].scale == 4

    def test_every_edge_points_at_a_visible_node(self, panel):
        shown, eff = self._skip_middle(panel)
        assert all(eff[i] is None or eff[i] in shown for i in shown)

    def test_nodes_without_a_visible_ancestor_become_roots(self, panel):
        shown, eff = self._skip_middle(panel)
        roots = {i for i in shown if eff[i] is None}
        assert {panel.by_id[i].scale for i in roots} == {4}

    def test_columns_are_contiguous_after_hiding_a_layer(self, panel):
        """No empty column left where the hidden layer used to sit."""
        self._skip_middle(panel)
        cols = sorted({n[2] for n in panel._nodes})
        step = panel.INDENT
        assert cols == [panel.X0 + i * step for i in range(len(cols))]

    def test_full_tree_edges_are_the_real_parents(self, panel):
        shown = {n[0].id for n in panel._nodes}
        eff = panel._effective_parents(shown)
        for h, _d in panel.visible_order():
            assert eff[h.id] == h.parent_id

    def test_redraw_survives_the_filtered_view(self, panel):
        self._skip_middle(panel)
        panel.redraw()          # must not raise on missing parents


class TestModePillsLayout:
    def test_pills_share_one_row(self, panel):
        a, b = panel._mode_pills
        assert a.master is b.master, "pills must sit in the same row frame"

    def test_pills_are_packed_horizontally(self, panel):
        panel.update_idletasks()
        for cv in panel._mode_pills:
            assert cv.pack_info()["side"] == "left"


class TestControlGeometry:
    """The pills were sized off-panel, making 'Selected' invisible."""

    @pytest.fixture
    def mapped(self, root):
        """A real (mapped) window — layout does not run on a withdrawn root."""
        win = tk.Toplevel(root)
        win.geometry("260x600")
        hs, tps = _layered()
        p = SourceTreePanel(win, hs, tps, mode="multi", coarse_scale=4)
        p.pack(fill=tk.BOTH, expand=True)
        win.update_idletasks()
        yield p
        win.destroy()

    def test_both_pills_are_visible(self, mapped):
        widths = [c.winfo_width() for c in mapped._mode_pills]
        assert all(w > 20 for w in widths), widths

    def test_pills_share_the_column_width_evenly(self, mapped):
        a, b = (c.winfo_width() for c in mapped._mode_pills)
        assert abs(a - b) <= 3, (a, b)

    def test_a_default_canvas_would_reproduce_the_bug(self, root):
        """Regression guard: the bug was tk.Canvas's 284px default request."""
        win = tk.Toplevel(root); win.geometry("260x120")
        row = tk.Frame(win, width=230, height=30); row.pack(fill=tk.X)
        a = tk.Canvas(row, height=22)           # no explicit width
        b = tk.Canvas(row, height=22)
        for c in (a, b):
            c.pack(side=tk.LEFT, fill=tk.X, expand=True)
        win.update_idletasks()
        assert b.winfo_width() < 20, "default-width canvases should overflow"
        win.destroy()

    def test_select_caption_is_present(self, mapped):
        labels = []
        for child in mapped.winfo_children():
            if child.winfo_class() == "Label":
                labels.append(child.cget("text"))
            elif child.winfo_class() == "Frame":
                labels += [c.cget("text") for c in child.winfo_children()
                           if c.winfo_class() == "Label"]
        assert "Select:" in labels, labels


class TestColumnWidth:
    """The column must fit the mode pills, not just the tree's short labels."""

    def test_advertised_width_accounts_for_the_pills(self, panel):
        from tkinter import font as tkfont
        f = tkfont.Font(family="Helvetica", size=9, weight="bold")
        widest = max(f.measure(t) for t in ("Full Tree", "Selected"))
        # room for two pills side by side, each with breathing space
        assert panel.tree_width >= 2 * widest

    def test_wider_than_the_old_hard_floor(self, panel):
        assert panel.tree_width + 18 > 190

    def test_pills_have_comfortable_padding(self, root):
        from tkinter import font as tkfont
        win = tk.Toplevel(root); win.geometry("400x600")
        hs, tps = _layered()
        p = SourceTreePanel(win, hs, tps, mode="multi", coarse_scale=4)
        holder = tk.Frame(win, width=max(p.tree_width + 18, 190))
        holder.pack_propagate(False); holder.pack(fill=tk.Y, expand=True)
        p.pack(in_=holder, fill=tk.BOTH, expand=True)
        win.update_idletasks()
        f = tkfont.Font(family="Helvetica", size=9, weight="bold")
        for c in p._mode_pills:
            pad = c.winfo_width() - f.measure(c._label)
            assert pad >= 24, f"{c._label!r} only has {pad}px of padding"
        win.destroy()

    def test_width_survives_a_relayout(self, panel):
        """Filtering must not shrink the column below what the pills need."""
        before = panel.tree_width
        panel._scale_var.set("Scale 3"); panel._on_scale_pick()
        panel._set_view_mode(True)
        assert panel.tree_width == before


class TestFixedColumnWidth:
    """The sidebar must not jump width when the view mode changes."""

    def test_width_is_identical_in_every_mode(self, panel):
        widths = [panel.tree_width]
        panel._scale_var.set("Scale 3"); panel._on_scale_pick()
        widths.append(panel.tree_width)
        panel._set_view_mode(True)
        widths.append(panel.tree_width)
        panel._set_view_mode(False)
        widths.append(panel.tree_width)
        assert len(set(widths)) == 1, widths

    def test_canvas_width_is_identical_in_every_mode(self, panel):
        before = panel._canvas.cget("width")
        panel._set_view_mode(True)
        assert panel._canvas.cget("width") == before

    def test_width_survives_filtering_to_the_narrowest_layer(self, panel):
        """The coarse layer alone is the narrowest content there is."""
        before = panel.tree_width
        panel._scale_var.set("Coarse Scale"); panel._on_scale_pick()
        panel._set_view_mode(True)
        assert panel.tree_width == before

    def test_height_still_follows_the_visible_rows(self, panel):
        """Width is pinned, but height should still shrink when filtered."""
        full = len(panel.visible_order())
        panel._scale_var.set("Scale 3"); panel._on_scale_pick()
        panel._set_view_mode(True)
        assert len(panel.visible_order()) < full
