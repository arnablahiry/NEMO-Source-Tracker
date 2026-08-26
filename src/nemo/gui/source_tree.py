"""Reusable hierarchical source-tree panel.

A canvas-drawn tree of source "pills" with connector lines, used by the
detections/sources viewer and the combined/individual analysis windows.

* ``mode="multi"``  — every source has its own visibility toggle (checkbox-like).
* ``mode="single"`` — exactly one source is selected at a time (radio-like).

Pills are filled with each source's *current* contour colour, shaded by its
per-scale opacity (coarse faint → leaves bright).  In multi mode, hiding a
source's parent promotes that source to its own independently-coloured root.
"""
import tkinter as tk

from . import _constants as C
from ..hierarchy import (assign_tree_colors, assign_tree_names, TREE_PALETTE)


def _accent():
    return C.LOG_TXT if C._current_theme == "light" else C.ACCENT


def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))



def hsrc_channel_masks(hierarchical_sources, tracks_per_scale):
    """Return {h_id: {channel: [mask, ...]}} for every hierarchical source."""
    tracks_by_scale_id = {}
    for scale, tracks in (tracks_per_scale or {}).items():
        for t in tracks:
            tracks_by_scale_id[(scale, t["id"])] = t
    out = {}
    for h in hierarchical_sources:
        ch_dict = {}
        for tid in h.track_ids:
            t = tracks_by_scale_id.get((h.scale, tid))
            if not t:
                continue
            for ch, m in t["masks"].items():
                ch_dict.setdefault(ch, []).append(m)
        out[h.id] = ch_dict
    return out


class SourceTreePanel(tk.Frame):
    INDENT = 22
    ROW_H  = 30
    PILL_H = 24
    SPINE_X = 8
    X0 = 24

    def __init__(self, parent, hierarchical_sources, tracks_per_scale,
                 on_change=None, mode="multi", title="Source tree",
                 default_active_letter=None):
        super().__init__(parent, bg=C.BG)
        self._hs = list(hierarchical_sources or [])
        self._tracks_per_scale = tracks_per_scale or {}
        self._on_change = on_change
        self._mode = mode

        self.by_id  = {h.id: h for h in self._hs}
        self.colors = assign_tree_colors(self._hs)          # {id: (rgb, alpha)}
        self.alpha  = {hid: a for hid, (_c, a) in self.colors.items()}
        self.name   = assign_tree_names(self._hs)
        self.masks_by_ch = hsrc_channel_masks(self._hs, self._tracks_per_scale)

        # hierarchical DFS order (coarse → fine) + root lookup
        self.order = []   # [(h, depth)]
        self.root_of = {}

        def _root_of(h):
            seen = set()
            while (h.parent_id is not None and h.parent_id in self.by_id
                   and h.id not in seen):
                seen.add(h.id)
                h = self.by_id[h.parent_id]
            return h
        for h in self._hs:
            self.root_of[h.id] = _root_of(h).id

        roots = [h for h in self._hs if h.is_root()]
        roots.sort(key=lambda h: (-h.scale, h.id))

        def _walk(h, depth):
            self.order.append((h, depth))
            kids = [self.by_id[c] for c in h.children_ids if c in self.by_id]
            kids.sort(key=lambda k: (-k.scale, k.id))
            for k in kids:
                _walk(k, depth + 1)
        for r in roots:
            _walk(r, 0)

        # stable per-source colour (never shifts when others toggle)
        root_ids    = [h.id for h, d in self.order if d == 0]
        nonroot_ids = [h.id for h, d in self.order if d > 0]
        self.stable_color = {}
        for i, rid in enumerate(root_ids):
            self.stable_color[rid] = TREE_PALETTE[i % len(TREE_PALETTE)]
        for j, nid in enumerate(nonroot_ids):
            self.stable_color[nid] = TREE_PALETTE[
                (len(root_ids) + j) % len(TREE_PALETTE)]

        # activation state
        if mode == "single":
            init = self.order[0][0].id if self.order else 0
            self._selected = tk.IntVar(value=init)
        else:
            # Default visibility: all on, unless a letter (scale level) is given
            # and at least one source carries it (e.g. only the "B" sources).
            letter = default_active_letter
            has_letter = letter is not None and any(
                self.name.get(h.id, "").startswith(letter) for h in self._hs)
            self._visible = {
                h.id: tk.BooleanVar(
                    value=(not has_letter) or self.name.get(h.id, "").startswith(letter))
                for h in self._hs
            }

        # ---- widgets ----
        tk.Label(self, text=title, bg=C.BG, fg=_accent(),
                 font=("Helvetica", 8, "bold")).pack(pady=(2, 4), anchor="w")

        from tkinter import font as tkfont
        self._font = tkfont.Font(family="Helvetica", size=10, weight="bold")

        # Indent by *scale*, not by tree depth.  Names come from
        # ``assign_tree_names`` which letters nodes by scale (A = coarsest), so
        # indenting by depth put a D attached straight to a B at the same
        # column as a C — the letters said one thing and the layout another.
        #
        # A node skips a level whenever no source at the intervening scale
        # contained it: the linking loop walks nearest-coarser-first and breaks
        # on the first match (hierarchy.py:334), so with no C-scale parent
        # available a D attaches directly to B.  Indenting by scale makes every
        # C share a column and every D share a column, and renders that skip as
        # a visible gap instead of hiding it.
        scales_desc = sorted({h.scale for h, _d in self.order}, reverse=True)
        self._scale_rank = {s: i for i, s in enumerate(scales_desc)}

        # pill geometry
        self._nodes = []   # (h, depth, x1, y1, x2, y2, cy, label)
        max_x = self.X0
        for i, (h, depth) in enumerate(self.order):
            label = self.name.get(h.id, f"#{h.id}")
            tw = self._font.measure(label)
            x1 = self.X0 + self._scale_rank.get(h.scale, depth) * self.INDENT
            cy = self.ROW_H // 2 + i * self.ROW_H
            x2 = x1 + tw + 20
            self._nodes.append((h, depth, x1, cy - self.PILL_H // 2,
                                x2, cy + self.PILL_H // 2, cy, label))
            max_x = max(max_x, x2)
        cv_w = int(max_x + 8)
        cv_h = max(self.ROW_H, self.ROW_H * len(self.order))


        self._canvas = tk.Canvas(self, bg=C.BG, width=cv_w,
                                 highlightthickness=0, bd=0)
        vsb = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set,
                               scrollregion=(0, 0, cv_w, cv_h))
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, anchor="nw", fill=tk.BOTH, expand=True)
        self._canvas.bind("<Button-1>", self._on_click)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._canvas.bind(seq, self._on_scroll)
        self.tree_width = cv_w
        self.redraw()

    # ------------------------------------------------------------------ API
    def selected_id(self):
        return int(self._selected.get()) if self._mode == "single" else None

    def descendants(self, hid):
        """Set of *hid* and all of its descendants."""
        out, stack = set(), [hid]
        while stack:
            n = stack.pop()
            if n in out or n not in self.by_id:
                continue
            out.add(n)
            for c in self.by_id[n].children_ids:
                if c in self.by_id:
                    stack.append(c)
        return out

    def active_ids(self):
        if self._mode == "single":
            # selecting a source activates it together with all its children
            return self.descendants(int(self._selected.get()))
        return {hid for hid, v in self._visible.items() if v.get()}

    def coloring(self):
        """(root_of, base_rgb_by_id) for the currently active sources."""
        active = self.active_ids()

        def vis_root(hid):
            h = self.by_id[hid]
            seen = set()
            while h.parent_id in active and h.id not in seen:
                seen.add(h.id)
                h = self.by_id[h.parent_id]
            return h.id
        root_of = {hid: vis_root(hid) for hid in active}
        return root_of, {hid: self.stable_color[root_of[hid]] for hid in active}

    # --------------------------------------------------------------- drawing
    def redraw(self):
        cv = self._canvas
        cv.delete("all")
        line_col = C.DIM_TXT
        node_by_id = {n[0].id: n for n in self._nodes}
        _root_of, base_of = self.coloring()
        bg = _hex_to_rgb01(C.BG)

        # left root spine
        root_nodes = [n for n in self._nodes if n[1] == 0]
        if len(root_nodes) >= 2:
            cv.create_line(self.SPINE_X, root_nodes[0][6],
                           self.SPINE_X, root_nodes[-1][6],
                           fill=line_col, width=1)
        for h, depth, x1, y1, x2, y2, cy, label in root_nodes:
            cv.create_line(self.SPINE_X, cy, x1, cy, fill=line_col, width=1)

        # parent → child connectors
        for h, depth, x1, y1, x2, y2, cy, label in self._nodes:
            if depth == 0:
                continue
            parent = self.by_id.get(h.parent_id)
            if parent is None or parent.id not in node_by_id:
                continue
            p_cy = node_by_id[parent.id][6]
            # Elbow sits just left of the child's own scale column, so a
            # skipped scale level shows as a longer horizontal run.
            rank = self._scale_rank.get(h.scale, depth)
            vx = self.X0 + rank * self.INDENT - self.INDENT // 2
            cv.create_line(vx, p_cy, vx, cy, fill=line_col, width=1)
            cv.create_line(vx, cy, x1, cy, fill=line_col, width=1)

        # pills (sharp rectangles), shaded like the contour
        for h, depth, x1, y1, x2, y2, cy, label in self._nodes:
            base = base_of.get(h.id)
            if base is not None:   # active
                a = self.alpha.get(h.id, 1.0)
                r, g, b = (c * a + bb * (1 - a) for c, bb in zip(base, bg))
                fill = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
                outline = fill
                lum = 0.299*r + 0.587*g + 0.114*b
                txt_col = "#000000" if lum > 0.55 else "#ffffff"
            else:                  # inactive
                fill = C.CARD_BG
                outline = C.DIM_TXT
                txt_col = C.DIM_TXT
            cv.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=1)
            cv.create_text((x1 + x2) // 2, cy, text=label,
                           font=self._font, fill=txt_col)


    # ---------------------------------------------------------------- events
    def _on_click(self, event):
        cx, cy_e = self._canvas.canvasx(event.x), self._canvas.canvasy(event.y)
        for h, depth, x1, y1, x2, y2, cy, label in self._nodes:
            if x1 <= cx <= x2 and y1 <= cy_e <= y2:
                if self._mode == "single":
                    self._selected.set(h.id)
                else:
                    v = self._visible[h.id]
                    v.set(not v.get())
                self.redraw()
                if self._on_change:
                    self._on_change()
                return

    def _on_scroll(self, event):
        if getattr(event, "num", None) == 4:
            self._canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
