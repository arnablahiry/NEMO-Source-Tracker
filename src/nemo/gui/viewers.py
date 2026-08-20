import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import PowerNorm, LogNorm, Normalize

from . import _constants as C
from ._constants import _CMAPS
from ._transport import TransportControls
from .widgets import _FlatBtn
from .dialogs import ScalingDialog


def _subsample_for_stats(cube: np.ndarray, max_elems: int = 4_000_000):
    """Return a strided view of *cube* with at most ~max_elems voxels.

    Whole-cube reductions (min/max/percentile) stall the UI on large cubes.
    A strided subsample gives display statistics that are visually identical
    for free — it's a view, so no copy or extra memory.
    """
    n = cube.size
    if n <= max_elems or cube.ndim != 3:
        return cube
    # Stride only the spatial axes so every channel is still represented.
    step = int(np.ceil(np.sqrt(n / max_elems)))
    return cube[:, ::step, ::step]


def _contour_color():
    return "black" if C._current_theme == "light" else "white"

def _accent():
    return C.LOG_TXT if C._current_theme == "light" else C.ACCENT


def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def _round_rect(canvas, x1, y1, x2, y2, r, **kw):
    """Draw a rounded rectangle (pill) on *canvas* and return its item id."""
    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    pts = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)


def _rich_label(parent, segments, bg=None, fg=None):
    """Render a math-style symbol with subscripts on a tk.Canvas (SONGS style).

    segments: list of (text, style) where style is:
      'n'  — normal baseline (Georgia 11 bold italic)
      's'  — subscript (Georgia 8 italic, lowered)
      'p'  — superscript (Georgia 8 italic, raised)
    """
    from tkinter import font as tkfont
    bg  = bg  or parent.cget("bg")
    fg  = fg  or _accent()

    base_f  = tkfont.Font(family="Georgia", size=11, weight="bold",  slant="italic")
    small_f = tkfont.Font(family="Georgia", size=8,                  slant="italic")

    BASELINE = 13
    SUB_DROP =  3
    SUP_LIFT =  5
    CANVAS_H = BASELINE + SUB_DROP + small_f.metrics("linespace") // 2 + 2

    total_w = 4
    for text, style in segments:
        f = base_f if style == "n" else small_f
        total_w += f.measure(text)
    total_w += 4

    cv = tk.Canvas(parent, width=total_w, height=CANVAS_H,
                   bg=bg, highlightthickness=0, bd=0)
    x = 2
    for text, style in segments:
        f = base_f if style == "n" else small_f
        if style == "n":
            cv.create_text(x, BASELINE, text=text, font=f, fill=fg, anchor="sw")
        elif style == "s":
            cv.create_text(x, BASELINE + SUB_DROP, text=text, font=f, fill=fg, anchor="sw")
        elif style == "p":
            cv.create_text(x, BASELINE - SUP_LIFT, text=text, font=f, fill=fg, anchor="sw")
        x += f.measure(text)
    return cv


class SliceViewer(TransportControls, tk.Toplevel):
    """Channel-by-channel viewer with normalization controls and optional overlays.

    mode : "raw"        — plain channel images
           "detections" — contour overlays from detection footprints
           "flow"       — quiver overlay from flow_seq
           "sources"    — per-source mask overlays
    """

    def __init__(self, master, cube: np.ndarray,
                 detections: list | None = None,
                 multi_scale_dets: list | None = None,
                 hierarchical_sources: list | None = None,
                 flow_seq: list | None = None,
                 flow_seq_per_scale: dict | None = None,
                 detections_per_scale: dict | None = None,
                 tracks_per_scale: dict | None = None,
                 tracks: list | None = None,
                 sources: list | None = None,
                 mode: str = "raw",
                 initial_norm: str = "linear",
                 initial_gamma: float = 0.5,
                 beam=None, pixscale=None, kpc_per_pix=None,
                 flux_unit: str | None = None,
                 vel_array=None,
                 card_0=None):
        super().__init__(master)
        self._flow_seq_per_scale   = flow_seq_per_scale or {}
        self._detections_per_scale = detections_per_scale or {}
        self._tracks_per_scale     = tracks_per_scale or {}
        self._initial_norm  = initial_norm
        self._initial_gamma = float(initial_gamma)
        self._beam        = beam
        self._pixscale    = pixscale
        self._kpc_per_pix = kpc_per_pix
        self._vel_array   = vel_array
        self._flux_unit   = flux_unit or "Jy beam⁻¹"
        self.title({
            "raw":        "Channel Viewer",
            "detections": "Wavelet Detections — Channel Viewer",
            "flow":       "Optical Flow — Channel Viewer",
            "sources":    "Sources — Channel Viewer",
        }.get(mode, "Channel Viewer"))
        self.configure(bg=C.BG)
        self.resizable(True, True)

        self._cube      = cube
        self._dets      = detections or []
        self._multi_scale_dets = multi_scale_dets or []
        self._hierarchical_sources = hierarchical_sources or []
        self._mode      = mode
        self._tracks    = tracks or []
        self._sources   = sources or []
        self._det_by_ch = {d.channel: d for d in self._dets}
        self._card_0    = card_0

        # Multi-scale detection support
        self._multi_scale_mode = len(self._multi_scale_dets) > 0
        # Use card_0's flag if it's a BooleanVar, otherwise create new one
        if card_0 and hasattr(card_0, '_multi_scale_enabled') and isinstance(card_0._multi_scale_enabled, tk.BooleanVar):
            self._multi_scale_enabled = card_0._multi_scale_enabled
        else:
            self._multi_scale_enabled = tk.BooleanVar(value=True)
        if self._multi_scale_mode:
            self._scales_available = sorted(
                set().union(*[sd.scales.keys() for sd in self._multi_scale_dets])
            )
            self._selected_scale = tk.IntVar(value=self._scales_available[-1])
        else:
            self._scales_available = []
            self._selected_scale = tk.IntVar(value=1)
        self._flow_by_ch = {}
        if flow_seq:
            for ch_ref, _ch_tgt, flow, _mask in flow_seq:
                self._flow_by_ch[ch_ref] = flow

        self._src_masks_by_ch: dict[int, dict[int, list]] = {}
        if mode == "sources" and self._tracks and self._sources:
            tracks_by_id = {t["id"]: t for t in self._tracks}
            for src in self._sources:
                ch_dict: dict[int, list] = {}
                for tid in src["track_ids"]:
                    t = tracks_by_id.get(tid)
                    if not t:
                        continue
                    for ch, mask in t["masks"].items():
                        ch_dict.setdefault(ch, []).append(mask)
                self._src_masks_by_ch[src["id"]] = ch_dict

        from .analysis import _source_colors
        self._src_color = _source_colors(self._sources)
        self._src_visible: dict[int, tk.BooleanVar] = {}

        # Multi-scale detection lookup (used in "detections" mode)
        self._ms_det_by_ch: dict = {}
        self._scale_visible: dict[int, tk.BooleanVar] = {}
        if self._multi_scale_dets and mode == "detections":
            self._ms_det_by_ch = {sd.channel: sd for sd in self._multi_scale_dets}
            self._scales_available = sorted(
                {s for sd in self._multi_scale_dets for s in sd.scales.keys()}
            )
            self._scale_visible = {s: tk.BooleanVar(value=True)
                                   for s in self._scales_available}

        # Multi-scale flow (used in "flow" mode): one scale shown at a time
        self._flow_scale_mode = mode == "flow" and bool(self._flow_seq_per_scale)
        self._flow_by_scale: dict = {}
        self._flowdet_by_scale_ch: dict = {}
        if self._flow_scale_mode:
            for s, fseq in self._flow_seq_per_scale.items():
                self._flow_by_scale[s] = {cr: fl for cr, _ct, fl, _m in fseq}
            for s, dets in self._detections_per_scale.items():
                self._flowdet_by_scale_ch[s] = {d.channel: d for d in dets}
            self._flow_scales_available = sorted(self._flow_seq_per_scale.keys())
            self._flow_selected_scale = tk.IntVar(
                value=self._flow_scales_available[-1])  # coarsest by default

        # Hierarchical sources (used in "sources" mode)
        self._hsrc_mode = (mode == "sources" and bool(self._hierarchical_sources)
                           and bool(self._tracks_per_scale))
        self._hsrc_masks_by_ch: dict = {}
        self._hsrc_colors: dict = {}
        self._hsrc_alpha: dict = {}
        self._hsrc_name: dict = {}
        self._hsrc_by_id: dict = {}
        self._hsrc_visible: dict = {}
        self._hsrc_root: dict = {}
        self._hsrc_order: list = []  # [(h, depth)] in hierarchical DFS order
        if self._hsrc_mode:
            from ..hierarchy import assign_tree_colors, assign_tree_names
            self._hsrc_by_id = {h.id: h for h in self._hierarchical_sources}
            self._hsrc_colors = assign_tree_colors(self._hierarchical_sources)
            self._hsrc_alpha = {hid: a for hid, (_rgb, a) in self._hsrc_colors.items()}
            self._hsrc_name = assign_tree_names(self._hierarchical_sources)
            tracks_by_scale_id = {}
            for scale, tracks in self._tracks_per_scale.items():
                for t in tracks:
                    tracks_by_scale_id[(scale, t['id'])] = t
            for h in self._hierarchical_sources:
                ch_dict: dict = {}
                for tid in h.track_ids:
                    t = tracks_by_scale_id.get((h.scale, tid))
                    if not t:
                        continue
                    for ch, m in t['masks'].items():
                        ch_dict.setdefault(ch, []).append(m)
                self._hsrc_masks_by_ch[h.id] = ch_dict
            # Default visibility: only the "B" (second-coarsest level) sources,
            # matching the Combined analysis window.  Fall back to all-on if no
            # source carries a "B" name.
            _has_B = any(self._hsrc_name.get(h.id, "").startswith("B")
                         for h in self._hierarchical_sources)
            self._hsrc_visible = {
                h.id: tk.BooleanVar(
                    value=(not _has_B)
                    or self._hsrc_name.get(h.id, "").startswith("B"))
                for h in self._hierarchical_sources}

            # Root lookup + hierarchical DFS order (coarse → fine) for the panel
            byid = self._hsrc_by_id

            def _root_of(h):
                seen = set()
                while (h.parent_id is not None and h.parent_id in byid
                       and h.id not in seen):
                    seen.add(h.id)
                    h = byid[h.parent_id]
                return h
            for h in self._hierarchical_sources:
                self._hsrc_root[h.id] = _root_of(h).id

            roots = [h for h in self._hierarchical_sources if h.is_root()]
            roots.sort(key=lambda h: (-h.scale, h.id))

            def _walk(h, depth):
                self._hsrc_order.append((h, depth))
                kids = [byid[c] for c in h.children_ids if c in byid]
                kids.sort(key=lambda k: (-k.scale, k.id))
                for k in kids:
                    _walk(k, depth + 1)
            for r in roots:
                _walk(r, 0)

            # Stable per-source colour: every source gets a fixed colour that
            # never changes when other sources are toggled.  Roots take their
            # tree colour (matching the GIF); any node that *could* be promoted
            # to a root gets its own fixed leftover-palette colour.  A source's
            # displayed colour is the stable colour of its current visible root.
            from ..hierarchy import TREE_PALETTE
            root_ids    = [h.id for h, d in self._hsrc_order if d == 0]
            nonroot_ids = [h.id for h, d in self._hsrc_order if d > 0]
            self._hsrc_stable_color = {}
            for i, rid in enumerate(root_ids):
                self._hsrc_stable_color[rid] = TREE_PALETTE[i % len(TREE_PALETTE)]
            for j, nid in enumerate(nonroot_ids):
                self._hsrc_stable_color[nid] = TREE_PALETTE[
                    (len(root_ids) + j) % len(TREE_PALETTE)]

        if self._dets and mode != "sources":
            self._channels = [d.channel for d in self._dets]
        elif self._ms_det_by_ch:
            self._channels = sorted(self._ms_det_by_ch.keys())
        else:
            self._channels = list(range(cube.shape[0]))

        VW = 500
        FW = 700

        # Subsample for the vmin/vmax slider bounds — a strided view keeps this
        # fast on large cubes (full-cube reductions stall the UI on load).
        _sample = _subsample_for_stats(cube)
        self._data_min = float(np.nanmin(_sample))
        self._data_max = float(np.nanmax(_sample))

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        self._fig   = plt.Figure(figsize=(FW/96, VW/96), dpi=96, facecolor=C.LOG_BG)
        self._ax    = self._fig.add_axes([0.02, 0.02, 0.94, 0.96])
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_edgecolor(C.DIM_TXT)
            spine.set_linewidth(0.8)
        _div        = make_axes_locatable(self._ax)
        self._ax_cb = _div.append_axes("right", size="4%", pad=0.1)
        self._ax_cb.set_facecolor(C.LOG_BG)

        needs_panel = (
            self._hsrc_mode or
            (self._mode == "sources" and self._sources) or
            (self._mode == "detections" and self._scale_visible) or
            self._flow_scale_mode
        )
        if needs_panel:
            top = tk.Frame(self, bg=C.BG)
            top.pack(fill=tk.BOTH, expand=True)

            if self._hsrc_mode:
                panel_w = self._source_tree_width() + 18  # + scrollbar
                pad = (4, 0)
            else:
                panel_w = 80
                pad = (6, 2)
            cb_panel = tk.Frame(top, bg=C.BG, width=panel_w)
            cb_panel.pack(side=tk.LEFT, fill=tk.Y, padx=pad, pady=4)
            cb_panel.pack_propagate(False)

            def _scale_swatch(scale):
                n_sc = len(self._scales_available)
                rank = self._scales_available.index(scale)
                alpha = 0.95 if n_sc <= 1 else 0.95 - rank * 0.70 / (n_sc - 1)
                g = int(alpha * 255)
                return "#{:02x}{:02x}{:02x}".format(g, g, g)

            if self._hsrc_mode:
                tk.Label(cb_panel, text="Source tree", bg=C.BG, fg=_accent(),
                         font=("Helvetica", 8, "bold")).pack(pady=(2, 4), anchor="w")
                self._build_source_tree(cb_panel)

            elif self._mode == "sources":
                tk.Label(cb_panel, text="Sources", bg=C.BG, fg=_accent(),
                         font=("Helvetica", 8, "bold")).pack(pady=(2, 4), anchor="w")
                for s in self._sources:
                    sid = s["id"]
                    col = self._src_color[sid]
                    hex_col = "#{:02x}{:02x}{:02x}".format(
                        int(col[0]*255), int(col[1]*255), int(col[2]*255))
                    var = tk.BooleanVar(value=True)
                    self._src_visible[sid] = var
                    row = tk.Frame(cb_panel, bg=C.BG)
                    row.pack(fill=tk.X, pady=1, anchor="w")
                    tk.Label(row, text="■", bg=C.BG, fg=hex_col,
                             font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 2))
                    tk.Checkbutton(row, text=f"S{sid}", variable=var,
                                   command=self._draw,
                                   bg=C.BG, fg=C.STEP_LABEL_TXT, selectcolor=C.CARD_BG,
                                   activebackground=C.BG, activeforeground=_accent(),
                                   font=("Helvetica", 8), relief=tk.FLAT,
                                   anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

            elif self._mode == "detections":
                tk.Label(cb_panel, text="Scales", bg=C.BG, fg=_accent(),
                         font=("Helvetica", 8, "bold")).pack(pady=(2, 4), anchor="w")
                self._build_scale_pills(cb_panel)

            elif self._flow_scale_mode:
                tk.Label(cb_panel, text="Flow scale", bg=C.BG, fg=_accent(),
                         font=("Helvetica", 8, "bold")).pack(pady=(2, 4), anchor="w")
                for s in self._flow_scales_available:
                    is_coarse = (s == self._flow_scales_available[-1])
                    tag = " (coarse)" if is_coarse else ""
                    tk.Radiobutton(cb_panel, text=f"j={s}{tag}", value=s,
                                   variable=self._flow_selected_scale,
                                   command=self._draw,
                                   bg=C.BG, fg=C.STEP_LABEL_TXT, selectcolor=C.CARD_BG,
                                   activebackground=C.BG, activeforeground=_accent(),
                                   font=("Helvetica", 8), relief=tk.FLAT,
                                   anchor="w").pack(fill=tk.X, pady=1, anchor="w")

            self._canvas = FigureCanvasTkAgg(self._fig, master=top)
            self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        else:
            self._canvas = FigureCanvasTkAgg(self._fig, master=self)
            self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ctrl = tk.Frame(self, bg=C.BG)
        ctrl.pack(fill=tk.X, padx=10, pady=(6, 0))

        light = C._current_theme == "light"
        default_cmap = "cubehelix" if light else "inferno"
        self._cmap     = tk.StringVar(value=default_cmap)
        self._inverted = tk.BooleanVar(value=light)

        tk.Label(ctrl, text="Colormap:", bg=C.BG, fg=C.STEP_LABEL_TXT,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 4))
        # Native tk.OptionMenu — fully styleable unlike ttk.Combobox
        om = tk.OptionMenu(ctrl, self._cmap, *_CMAPS, command=lambda _v: self._draw())
        om.configure(
            bg=C.CARD_BG, fg=C.STEP_LABEL_TXT,
            activebackground=C.DIM, activeforeground=C.STEP_LABEL_TXT,
            highlightthickness=0, relief=tk.FLAT,
            font=("Helvetica", 9), width=9,
        )
        om["menu"].configure(
            bg=C.CARD_BG, fg=C.STEP_LABEL_TXT,
            activebackground=C.DIM, activeforeground=C.STEP_LABEL_TXT,
            font=("Helvetica", 9),
        )
        om.pack(side=tk.LEFT, padx=(0, 6))

        tk.Checkbutton(ctrl, text="Inverted", variable=self._inverted,
                       command=self._draw,
                       bg=C.BG, fg=C.STEP_LABEL_TXT, selectcolor=C.CARD_BG,
                       activebackground=C.BG, activeforeground=_accent(),
                       font=("Helvetica", 9), relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 14))

        self._norm_mode = tk.StringVar(value=self._initial_norm)
        tk.Label(ctrl, text="Scale:", bg=C.BG, fg=C.STEP_LABEL_TXT,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self._build_norm_pills(ctrl)
        self._build_inline_gamma(ctrl)
        self._set_gamma_enabled(self._norm_mode.get() == "power")

        # Multi-scale scale selector
        if self._multi_scale_mode:
            tk.Label(ctrl, text="Scale:", bg=C.BG, fg=C.STEP_LABEL_TXT,
                     font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(14, 4))
            for scale in self._scales_available:
                tk.Radiobutton(ctrl, text=str(scale), variable=self._selected_scale, value=scale,
                               command=self._draw,
                               bg=C.BG, fg=C.STEP_LABEL_TXT, selectcolor=C.CARD_BG,
                               activebackground=C.BG, activeforeground=_accent(),
                               font=("Helvetica", 9), relief=tk.FLAT).pack(
                                   side=tk.LEFT, padx=2)

        # Below the image: sliders on the left half, integrated spectrum on the
        # right half (two equal-width columns).
        body = tk.Frame(self, bg=C.BG)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1, uniform="half")
        body.columnconfigure(1, weight=1, uniform="half")
        body.rowconfigure(0, weight=1)
        left  = tk.Frame(body, bg=C.BG)
        left.grid(row=0, column=0, sticky="nsew")
        right = tk.Frame(body, bg=C.BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=(4, 2))

        self._ch_lbl = tk.Label(left, text="", bg=C.BG, fg=_accent(),
                                font=("Helvetica", 9, "italic"), anchor="w")
        self._ch_lbl.pack(fill=tk.X, padx=12, pady=(4, 2))

        def _bordered_slider(label, from_, to_, default, cmd):
            """Card (DIM border) with label above; slider+entry wrapped in accent border."""
            # Outer card — DIM border
            card = tk.Frame(left, bg=C.DIM, padx=1, pady=1)
            card.pack(fill=tk.X, padx=10, pady=(2, 0))
            card_inner = tk.Frame(card, bg=C.CARD_BG, padx=6, pady=4)
            card_inner.pack(fill=tk.BOTH, expand=True)

            txt = tk.Label(card_inner, text=label, bg=C.CARD_BG, fg=C.STEP_LABEL_TXT,
                           font=("Helvetica", 8), anchor="w")
            txt.pack(anchor="w", pady=(0, 3))

            # Inner slider+entry box — ACCENT border
            slider_box = tk.Frame(card_inner, bg=_accent(), padx=1, pady=1)
            slider_box.pack(fill=tk.X)
            row = tk.Frame(slider_box, bg=C.CARD_BG, padx=4, pady=3)
            row.pack(fill=tk.BOTH, expand=True)

            s = tk.Scale(row, from_=from_, to=to_, resolution=(to_ - from_) / 500,
                         orient=tk.HORIZONTAL, command=cmd,
                         bg=_accent(), fg=_accent(), troughcolor=C.LOG_BG,
                         activebackground=C.ACCENT_HOVER, highlightthickness=0,
                         sliderrelief=tk.FLAT, bd=0, showvalue=False, width=12)
            s.set(default)
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)

            tk.Frame(row, bg=_accent(), width=1).pack(side=tk.LEFT, fill=tk.Y, padx=(3, 0))

            val_var = tk.StringVar()
            val_entry = tk.Entry(row, textvariable=val_var, width=9,
                                 justify="right", bg=C.CARD_BG,
                                 readonlybackground=C.CARD_BG,
                                 fg=_accent(), insertbackground=_accent(),
                                 relief=tk.FLAT, highlightthickness=0,
                                 font=("Courier", 7), bd=0, state="readonly")
            val_entry.pack(side=tk.RIGHT, padx=(3, 0))
            val_entry._var = val_var
            return s, val_entry, txt, slider_box

        self._vmin_sl, self._vmin_lbl, self._vmin_txt, _ = _bordered_slider(
            "vmin", self._data_min, self._data_max, self._data_min,
            lambda _v: self._draw())
        self._vmax_sl, self._vmax_lbl, self._vmax_txt, _ = _bordered_slider(
            "vmax", self._data_min, self._data_max, self._data_max,
            lambda _v: self._draw())

        # Channel slider — DIM card, ACCENT slider box
        ch_card = tk.Frame(left, bg=C.DIM, padx=1, pady=1)
        ch_card.pack(fill=tk.X, padx=10, pady=(2, 8))
        ch_card_inner = tk.Frame(ch_card, bg=C.CARD_BG, padx=6, pady=4)
        ch_card_inner.pack(fill=tk.BOTH, expand=True)
        tk.Label(ch_card_inner, text="Channel", bg=C.CARD_BG, fg=C.STEP_LABEL_TXT,
                 font=("Helvetica", 8), anchor="w").pack(anchor="w", pady=(0, 3))
        ctrl_row = tk.Frame(ch_card_inner, bg=C.CARD_BG)
        ctrl_row.pack(fill=tk.X)
        ch_box = tk.Frame(ctrl_row, bg=_accent(), padx=1, pady=1)
        ch_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._slider_box = ch_box
        ch_row = tk.Frame(ch_box, bg=C.CARD_BG, padx=4, pady=3)
        ch_row.pack(fill=tk.BOTH, expand=True)
        N_ch = len(self._channels)
        self._slider = tk.Scale(
            ch_row, from_=0, to=N_ch - 1,
            orient=tk.HORIZONTAL, command=lambda _v: self._draw(),
            bg=_accent(), fg=_accent(), troughcolor=C.LOG_BG,
            activebackground=C.ACCENT_HOVER, highlightthickness=0,
            sliderrelief=tk.FLAT, bd=0, width=11, showvalue=False,
        )
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Frame(ch_row, bg=_accent(), width=1).pack(side=tk.LEFT, fill=tk.Y, padx=(3, 0))
        self._ch_disp = tk.Canvas(ch_row, bg=C.LOG_BG, highlightthickness=0,
                                  width=60, height=20)
        self._ch_disp.pack(side=tk.RIGHT, padx=(3, 0))
        def _update_ch_disp(idx):
            self._ch_disp.delete("all")
            w, h = 60, 20
            ch = self._channels[int(idx)]
            n  = str(ch + 1)
            self._ch_disp.create_text(w//2 - 8, h//2, text=n,
                                      fill=_accent(), font=("Courier", 8, "bold"), anchor="e")
            self._ch_disp.create_text(w//2 - 6, h//2, text=f"/{N_ch}",
                                      fill=_accent(), font=("Courier", 8), anchor="w")
        self._update_ch_disp = _update_ch_disp
        _update_ch_disp(N_ch // 2)
        self._slider.set(N_ch // 2)

        # play / loop pills + FPS box (shared transport), right of the slider
        self._init_transport_state()
        self._build_transport(ctrl_row)

        self._build_spectrum(right)

        self._draw()
        self.update_idletasks()
        from ._theme import set_titlebar_appearance
        set_titlebar_appearance(self)
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry(f"{w}x{h}")
        self.minsize(w, h)
        if self._hsrc_mode:
            # Lock the source viewer to a fixed size; the tree scrolls instead.
            self.resizable(False, False)
            self.maxsize(w, h)
        else:
            self.maxsize(w, 9999)

    def _build_source_tree(self, parent):
        """Canvas-drawn hierarchy: connector lines + clickable name boxes.

        An active (visible) source's box is filled with that source's current
        contour colour; a hidden source's box is outline-only.  Clicking a box
        toggles the source — colours update dynamically with the visualisation.
        """
        from tkinter import font as tkfont
        self._tree_font = tkfont.Font(family="Helvetica", size=10, weight="bold")

        self.INDENT   = 22
        self.ROW_H    = 30
        self.PILL_H   = 24
        self.SPINE_X  = 8
        self.X0       = 24   # leaves room for the left root spine + connectors

        # Pre-compute box geometry per node
        self._tree_nodes = []  # (h, depth, x1, y1, x2, y2, cy, label)
        max_x = self.X0
        for i, (h, depth) in enumerate(self._hsrc_order):
            label = self._hsrc_name.get(h.id, f"#{h.id}")
            tw = self._tree_font.measure(label)
            x1 = self.X0 + depth * self.INDENT
            cy = self.ROW_H // 2 + i * self.ROW_H
            y1 = cy - self.PILL_H // 2
            x2 = x1 + tw + 20
            y2 = cy + self.PILL_H // 2
            self._tree_nodes.append((h, depth, x1, y1, x2, y2, cy, label))
            max_x = max(max_x, x2)

        cv_h = max(self.ROW_H, self.ROW_H * len(self._hsrc_order))
        cv_w = int(max_x + 8)
        self._tree_canvas = tk.Canvas(parent, bg=C.BG, width=cv_w,
                                      highlightthickness=0, bd=0)
        vsb = tk.Scrollbar(parent, orient="vertical",
                           command=self._tree_canvas.yview)
        self._tree_canvas.configure(yscrollcommand=vsb.set,
                                    scrollregion=(0, 0, cv_w, cv_h))
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree_canvas.pack(side=tk.LEFT, anchor="nw", fill=tk.BOTH, expand=True)
        self._tree_canvas.bind("<Button-1>", self._on_tree_click)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._tree_canvas.bind(seq, self._on_tree_scroll)
        self._draw_source_tree()

    def _source_tree_width(self) -> int:
        """Pixel width the source-tree canvas needs for its widest pill."""
        from tkinter import font as tkfont
        f = tkfont.Font(family="Helvetica", size=10, weight="bold")
        max_depth = max((d for _, d in self._hsrc_order), default=0)
        max_lbl = max((f.measure(self._hsrc_name.get(h.id, f"#{h.id}"))
                       for h, _ in self._hsrc_order), default=20)
        return 24 + max_depth * 22 + max_lbl + 28

    def _on_tree_scroll(self, event):
        if getattr(event, "num", None) == 4:
            self._tree_canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self._tree_canvas.yview_scroll(1, "units")
        else:
            self._tree_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _draw_source_tree(self):
        cv = self._tree_canvas
        cv.delete("all")
        line_col = C.DIM_TXT

        node_by_id = {n[0].id: n for n in self._tree_nodes}
        # current contour colours of the visible forest (dynamic)
        _root_of, base_of = self._visible_hsrc_coloring()

        # left root spine: one vertical line tying all base (depth-0) sources
        # together, with a horizontal connector into each root pill.
        root_nodes = [n for n in self._tree_nodes if n[1] == 0]
        if len(root_nodes) >= 2:
            cv.create_line(self.SPINE_X, root_nodes[0][6],
                           self.SPINE_X, root_nodes[-1][6],
                           fill=line_col, width=1)
        for h, depth, x1, y1, x2, y2, cy, label in root_nodes:
            cv.create_line(self.SPINE_X, cy, x1, cy, fill=line_col, width=1)

        # connector lines: vertical spine per parent + horizontal stub per child
        for h, depth, x1, y1, x2, y2, cy, label in self._tree_nodes:
            if depth == 0:
                continue
            parent = self._hsrc_by_id.get(h.parent_id)
            if parent is None or parent.id not in node_by_id:
                continue
            p_cy = node_by_id[parent.id][6]
            vx = self.X0 + depth * self.INDENT - self.INDENT // 2
            cv.create_line(vx, p_cy, vx, cy, fill=line_col, width=1)
            cv.create_line(vx, cy, x1, cy, fill=line_col, width=1)

        # boxes (proper rectangles), filled with the source's contour colour
        # shaded by its per-scale opacity (coarse = faint, leaves = full),
        # so the pill shade matches the contour shade.
        bg = _hex_to_rgb01(C.BG)
        for h, depth, x1, y1, x2, y2, cy, label in self._tree_nodes:
            base = base_of.get(h.id)
            if base is not None:  # active / visible
                alpha = self._hsrc_alpha.get(h.id, 1.0)
                r, g, b = (c * alpha + bb * (1 - alpha) for c, bb in zip(base, bg))
                fill = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
                outline = fill
                lum = 0.299*r + 0.587*g + 0.114*b
                txt_col = "#000000" if lum > 0.55 else "#ffffff"
            else:  # hidden
                fill = C.CARD_BG
                outline = C.DIM_TXT
                txt_col = C.DIM_TXT
            cv.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=1)
            cv.create_text((x1 + x2) // 2, cy, text=label,
                           font=self._tree_font, fill=txt_col)

    def _on_tree_click(self, event):
        cv = self._tree_canvas
        cx, cy_e = cv.canvasx(event.x), cv.canvasy(event.y)
        for h, depth, x1, y1, x2, y2, cy, label in self._tree_nodes:
            if x1 <= cx <= x2 and y1 <= cy_e <= y2:
                var = self._hsrc_visible[h.id]
                var.set(not var.get())
                self._draw_source_tree()
                self._draw()
                return

    def _scale_shade(self, s):
        """Grey-level (0–1) matching the white contour opacity for scale *s*."""
        n = len(self._scales_available)
        rank = self._scales_available.index(s)
        return 0.95 if n <= 1 else 0.95 - rank * 0.70 / (n - 1)

    def _build_norm_pills(self, parent):
        """Horizontal single-select pills choosing the display scaling."""
        from tkinter import font as tkfont
        self._npill_font = tkfont.Font(family="Helvetica", size=9, weight="bold")
        PILL_H, PAD_Y, GAP, X0 = 22, 2, 6, 2
        items = [("linear", "Linear Scale"),
                 ("log",    "Log Scale"),
                 ("power",  "Power Scale")]
        self._npill_nodes = []
        x  = X0
        cy = PAD_Y + PILL_H // 2
        for mode, label in items:
            tw = self._npill_font.measure(label)
            x1, x2 = x, x + tw + 18
            self._npill_nodes.append((mode, x1, cy - PILL_H // 2,
                                      x2, cy + PILL_H // 2, cy, label))
            x = x2 + GAP
        cv = tk.Canvas(parent, bg=C.BG, width=x, height=PILL_H + 2 * PAD_Y,
                       highlightthickness=0, bd=0)
        cv.pack(side=tk.LEFT, padx=(0, 4))
        cv.bind("<Button-1>", self._on_norm_pill_click)
        self._npill_canvas = cv
        self._draw_norm_pills()

    def _draw_norm_pills(self):
        cv = self._npill_canvas
        cv.delete("all")
        sel   = self._norm_mode.get()
        light = C._current_theme == "light"
        for mode, x1, y1, x2, y2, cy, label in self._npill_nodes:
            if mode == sel:
                fill = outline = _accent()
                txt_col = "#ffffff" if light else "#000000"
            else:
                fill, outline, txt_col = C.CARD_BG, C.DIM_TXT, C.DIM_TXT
            cv.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=1)
            cv.create_text((x1 + x2) // 2, cy, text=label,
                           font=self._npill_font, fill=txt_col)

    def _on_norm_pill_click(self, event):
        for mode, x1, y1, x2, y2, cy, label in self._npill_nodes:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self._norm_mode.set(mode)
                self._draw_norm_pills()
                self._set_gamma_enabled(mode == "power")
                self._draw()
                return

    def _build_inline_gamma(self, parent):
        """Compact γ slider (display only) shown to the right of the norm pills.

        Initialised from ``initial_gamma`` (the card's scaling) but otherwise
        independent — changing it here only affects this viewer's display.
        """
        H = 26   # match the norm-pill height
        box = tk.Frame(parent, bg=_accent(), padx=1, pady=1, width=150, height=H)
        box.pack_propagate(False)
        box.pack(side=tk.LEFT, padx=(2, 0))
        self._gamma_box = box
        row = tk.Frame(box, bg=C.CARD_BG)
        row.pack(fill=tk.BOTH, expand=True)
        self._gamma_txt = tk.Label(row, text="γ", bg=C.CARD_BG, fg=_accent(),
                                   font=("Georgia", 11, "italic"))
        self._gamma_txt.pack(side=tk.LEFT, padx=(3, 3))
        self._gamma_sl = tk.Scale(row, from_=0.1, to=2.0, resolution=0.05,
                                  orient=tk.HORIZONTAL,
                                  bg=_accent(), fg=_accent(), troughcolor=C.LOG_BG,
                                  activebackground=C.ACCENT_HOVER,
                                  highlightthickness=0, sliderrelief=tk.FLAT,
                                  bd=0, showvalue=False, width=8)
        self._gamma_sl.set(self._initial_gamma)
        self._gamma_sl.configure(command=self._on_gamma)   # attach after .set()
        self._gamma_sl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Frame(row, bg=_accent(), width=1).pack(side=tk.LEFT, fill=tk.Y,
                                                  padx=(3, 0))   # divider
        var = tk.StringVar(value=f"{self._initial_gamma:.2f}")
        self._gamma_lbl = tk.Entry(row, textvariable=var, width=4, justify="right",
                                   bg=C.CARD_BG, readonlybackground=C.CARD_BG,
                                   fg=_accent(), relief=tk.FLAT, highlightthickness=0,
                                   font=("Courier", 7), bd=0, state="readonly")
        self._gamma_lbl._var = var
        self._gamma_lbl.pack(side=tk.RIGHT, padx=(2, 2))

    def _set_gamma_enabled(self, on):
        """Enable/disable the power-exponent slider with a dimmed look when off."""
        s, box   = self._gamma_sl, self._gamma_box
        lbl, val = self._gamma_txt, self._gamma_lbl
        if on:
            s.configure(state="normal", fg=_accent(), bg=_accent(),
                        troughcolor=C.LOG_BG, activebackground=C.ACCENT_HOVER)
            box.configure(bg=_accent())
            lbl.configure(fg=_accent())
            val.configure(fg=_accent())
        else:
            s.configure(state="disabled", fg=C.DIM, bg=C.DIM,
                        troughcolor=C.CARD_BG, activebackground=C.DIM)
            box.configure(bg=C.DIM)
            lbl.configure(fg=C.DIM_TXT)
            val.configure(fg=C.DIM_TXT)

    def _on_gamma(self, _v=None):
        self._gamma_lbl._var.set(f"{float(self._gamma_sl.get()):.2f}")
        self._draw()

    def _build_scale_pills(self, parent):
        """Sharp-cornered scale toggle pills for the detections viewer."""
        from tkinter import font as tkfont
        self._spill_font = tkfont.Font(family="Helvetica", size=10, weight="bold")
        PILL_H, ROW_H, X0 = 22, 28, 8
        self._spill_nodes = []  # (scale, x1, y1, x2, y2, cy, label)
        max_x = X0
        for i, s in enumerate(self._scales_available):
            label = f"j={s}"
            tw = self._spill_font.measure(label)
            cy = ROW_H // 2 + i * ROW_H
            x1, x2 = X0, X0 + tw + 20
            y1, y2 = cy - PILL_H // 2, cy + PILL_H // 2
            self._spill_nodes.append((s, x1, y1, x2, y2, cy, label))
            max_x = max(max_x, x2)
        cv_h = max(ROW_H, ROW_H * len(self._scales_available))
        self._spill_canvas = tk.Canvas(parent, bg=C.BG, width=int(max_x + 8),
                                       height=cv_h, highlightthickness=0, bd=0)
        self._spill_canvas.pack(anchor="nw", fill=tk.BOTH, expand=True)
        self._spill_canvas.bind("<Button-1>", self._on_scale_pill_click)
        self._draw_scale_pills()

    def _draw_scale_pills(self):
        cv = self._spill_canvas
        cv.delete("all")
        # Contours are white (dark theme) / black (light theme) at graded
        # opacity; fill each pill with that exact colour over the panel bg.
        cbase = 1.0 if C._current_theme != "light" else 0.0
        bg = _hex_to_rgb01(C.BG)
        for s, x1, y1, x2, y2, cy, label in self._spill_nodes:
            if self._scale_visible[s].get():  # active — same shade as its contour
                a = self._scale_shade(s)
                r, g, b = (cbase * a + bb * (1 - a) for bb in bg)
                fill = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
                outline = fill
                lum = 0.299*r + 0.587*g + 0.114*b
                txt_col = "#000000" if lum > 0.55 else "#ffffff"
            else:  # hidden — outline only
                fill = C.CARD_BG
                outline = C.DIM_TXT
                txt_col = C.DIM_TXT
            cv.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=1)
            cv.create_text((x1 + x2) // 2, cy, text=label,
                           font=self._spill_font, fill=txt_col)

    def _on_scale_pill_click(self, event):
        for s, x1, y1, x2, y2, cy, label in self._spill_nodes:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                var = self._scale_visible[s]
                var.set(not var.get())
                self._draw_scale_pills()
                self._draw()
                return

    def _visible_hsrc_coloring(self):
        """Colour the currently-visible hierarchical sources.

        A source's colour is the *stable* colour of its current visible root (a
        visible source whose parent is hidden/absent).  Because every source's
        stable colour is fixed up front, toggling one source never changes the
        colour of any source you didn't touch: hiding a node only re-roots its
        own descendants onto their fixed colour.  Returns (root_of, base_by_id).
        """
        byid = self._hsrc_by_id
        visible = {hid for hid, v in self._hsrc_visible.items() if v.get()}

        def vis_root(hid):
            h = byid[hid];  seen = set()
            while h.parent_id in visible and h.id not in seen:
                seen.add(h.id);  h = byid[h.parent_id]
            return h.id

        root_of = {hid: vis_root(hid) for hid in visible}
        return root_of, {hid: self._hsrc_stable_color[root_of[hid]]
                         for hid in visible}

    def _norm(self):
        vmin = float(self._vmin_sl.get())
        vmax = float(self._vmax_sl.get())
        if vmin >= vmax:
            vmax = vmin + 1e-9
        mode = self._norm_mode.get()
        if mode == "log":
            vmin = max(vmin, 1e-12)
            vmax = max(vmax, vmin + 1e-12)
            return LogNorm(vmin=vmin, vmax=vmax)
        elif mode == "power":
            vmin = max(vmin, 0)
            return PowerNorm(gamma=float(self._gamma_sl.get()), vmin=vmin, vmax=vmax)
        else:
            return Normalize(vmin=vmin, vmax=vmax)

    def _fmt_val(self, v: float) -> str:
        if self._norm_mode.get() == "log":
            exp = np.log10(max(abs(v), 1e-30))
            return f"10^{exp:.2f}"
        return f"{v:.2e}"

    def _update_value_labels(self):
        for w in (self._vmin_txt, self._vmax_txt):
            w.configure(fg=C.STEP_LABEL_TXT)
        self._vmin_lbl._var.set(self._fmt_val(float(self._vmin_sl.get())))
        self._vmax_lbl._var.set(self._fmt_val(float(self._vmax_sl.get())))

    def _build_spectrum(self, parent):
        """Spectrum panel with a dashed marker tracking the current channel.

        In the sources viewer (hierarchical or flat) the panel is *dynamic*
        and mirrors the Combined-analysis spectrum: a dashed whole-field
        "Total" curve plus one curve per currently-selected source, each in
        that source's contour colour, redrawn as sources are toggled.  In all
        other modes it is the static whole-field integrated spectrum.
        """
        cube  = self._cube
        nchan = cube.shape[0]
        self._spec_H, self._spec_W = cube.shape[1], cube.shape[2]
        if self._vel_array is not None and len(self._vel_array) == nchan:
            self._spec_x = np.asarray(self._vel_array, dtype=float)
            self._spec_xlabel = "Velocity"
        else:
            self._spec_x = np.arange(nchan, dtype=float)
            self._spec_xlabel = "Channel"

        # Per-source spectra in the sources viewer (hierarchical or flat);
        # static whole-field integrated spectrum everywhere else.
        masks_by_ch = (self._hsrc_masks_by_ch if self._hsrc_mode
                       else self._src_masks_by_ch if self._mode == "sources"
                       else None)
        self._spec_source_mode = bool(masks_by_ch)
        self._spec_total = np.nansum(cube, axis=(1, 2))
        if self._spec_source_mode:
            self._spec_curve: dict = {}
            for sid, ch_dict in masks_by_ch.items():
                fp = np.zeros((self._spec_H, self._spec_W), dtype=bool)
                for masks in ch_dict.values():
                    for m in masks:
                        fp |= m
                self._spec_curve[sid] = (cube[:, fp].sum(axis=1) if fp.any()
                                         else np.zeros(nchan, dtype=float))

        fig = plt.Figure(figsize=(3.0, 2.6), dpi=96, facecolor=C.LOG_BG)
        ax  = fig.add_subplot(111)
        self._spec_fig    = fig
        self._spec_ax     = ax
        self._spec_canvas = FigureCanvasTkAgg(fig, master=parent)
        self._spec_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._last_vis_key = None
        self._redraw_spectrum()

    def _spec_visible_ids(self):
        vis = self._hsrc_visible if self._hsrc_mode else self._src_visible
        return frozenset(sid for sid, v in vis.items() if v.get())

    def _spec_visible_items(self):
        """[(label, sid, rgb)] for visible sources, coloured to match the image."""
        from .analysis import _shade
        items = []
        if self._hsrc_mode:
            _root_of, base_of = self._visible_hsrc_coloring()
            for h, _depth in self._hsrc_order:
                hid = h.id
                if hid not in base_of:
                    continue
                rgb = _shade(base_of[hid], self._hsrc_alpha.get(hid, 1.0))
                items.append((self._hsrc_name.get(hid, str(hid)), hid, rgb))
        else:
            for s in self._sources:
                sid = s["id"]
                v = self._src_visible.get(sid)
                if v is not None and not v.get():
                    continue
                items.append((f"S{sid}", sid, self._src_color[sid][:3]))
        return items

    def _redraw_spectrum(self):
        """Clear and replot the spectrum (Total + one curve per visible source)."""
        ax = self._spec_ax
        ax.clear()
        txt_col = C.STEP_LABEL_TXT
        ax.set_facecolor(C.LOG_BG)
        ax.grid(True, color=C.DIM_TXT, alpha=0.22, linewidth=0.4)
        ax.set_axisbelow(True)
        xs = self._spec_x

        if self._spec_source_mode:
            ax.plot(xs, self._spec_total, color=C.DIM_TXT, lw=1.0, ls="--",
                    label="Total")
            items = self._spec_visible_items()
            for label, sid, rgb in items:
                ax.plot(xs, self._spec_curve[sid], color=rgb, lw=1.2, label=label)
            if items:
                leg = ax.legend(fontsize=6, ncol=2, framealpha=0.6,
                                facecolor=C.LOG_BG, edgecolor=C.DIM_TXT,
                                labelcolor=txt_col, handlelength=1.2,
                                columnspacing=1.0, handletextpad=0.4,
                                borderpad=0.3, loc="best")
                leg.get_frame().set_linewidth(0.6)
            self._last_vis_key = self._spec_visible_ids()
        else:
            ax.plot(xs, self._spec_total, color=_accent(), lw=1.1)

        ax.set_xlabel(self._spec_xlabel, color=txt_col, fontsize=9)
        ax.set_ylabel(self._flux_unit, color=txt_col, fontsize=9)
        ax.tick_params(colors=txt_col, labelsize=8, length=3, direction="in",
                       top=True, bottom=True, left=True, right=True)
        ax.margins(x=0.02)
        for sp in ax.spines.values():
            sp.set_edgecolor(C.DIM_TXT)
            sp.set_linewidth(0.8)
        ch = self._channels[int(self._slider.get())]
        self._spec_vline = ax.axvline(self._spec_x[ch],
                                      color=_contour_color(), ls="--", lw=1.1)
        self._spec_fig.tight_layout(pad=1.0)
        self._spec_canvas.draw_idle()

    def _update_spectrum_data(self):
        """Replot the per-source spectra when the source selection changes."""
        if not getattr(self, "_spec_source_mode", False) \
                or not hasattr(self, "_spec_ax"):
            return
        if self._spec_visible_ids() == self._last_vis_key:
            return
        self._redraw_spectrum()

    def _update_spectrum_marker(self, ch):
        if not hasattr(self, "_spec_vline"):
            return
        x = float(self._spec_x[ch])
        self._spec_vline.set_xdata([x, x])
        self._spec_canvas.draw_idle()

    def _draw(self):
        idx = int(self._slider.get())
        ch  = self._channels[idx]
        img = self._cube[ch]
        if hasattr(self, '_update_ch_disp'):
            self._update_ch_disp(idx)
        self._update_spectrum_data()
        self._update_spectrum_marker(ch)

        norm = self._norm()
        cmap = self._cmap.get() + ("_r" if self._inverted.get() else "")

        self._ax.clear()
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_edgecolor(C.DIM_TXT)
            spine.set_linewidth(0.8)
        self._ax.imshow(img, cmap=cmap, norm=norm, origin="lower")

        self._fig.set_facecolor(C.LOG_BG)
        self._ax_cb.set_facecolor(C.LOG_BG)
        self._ax_cb.clear()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = self._fig.colorbar(sm, cax=self._ax_cb)
        cb.ax.tick_params(colors=C.STEP_LABEL_TXT, labelsize=9, length=3)
        cb.outline.set_edgecolor(C.DIM)
        plt.setp(plt.getp(cb.ax, "yticklabels"), color=C.STEP_LABEL_TXT, fontsize=9)
        cb.set_label(self._flux_unit, color=C.STEP_LABEL_TXT, fontsize=9, labelpad=6)

        contour_col = _contour_color()

        if self._mode == "detections":
            if self._ms_det_by_ch:
                sd = self._ms_det_by_ch.get(ch)
                if sd:
                    n_sc = len(self._scales_available)
                    cv   = 0.0 if C._current_theme == "light" else 1.0
                    for scale_idx in self._scales_available:
                        if not self._scale_visible[scale_idx].get():
                            continue
                        if scale_idx not in sd.scales:
                            continue
                        rank  = self._scales_available.index(scale_idx)
                        alpha = 0.95 if n_sc <= 1 else 0.95 - rank * 0.70 / (n_sc - 1)
                        masks, _, _ = sd.scales[scale_idx]
                        for mask in masks:
                            self._ax.contour(mask.astype(float), [0.5],
                                             colors=[[cv, cv, cv, alpha]],
                                             linewidths=0.8)
            else:
                d = self._det_by_ch.get(ch)
                if d:
                    for mask in d.footprint_masks:
                        self._ax.contour(mask.astype(float), [0.5],
                                         colors=[contour_col], linewidths=0.8)

        elif self._hsrc_mode:
            from matplotlib.patches import Rectangle as _Rect
            H_img, W_img = img.shape
            PAD_BB = 4

            # Dynamic colouring of the visible forest: a deselected root
            # promotes its branches to independent (separately-coloured) roots.
            root_of, base_of = self._visible_hsrc_coloring()

            # bbox owner per visible subtree = its coarsest visible source at
            # this channel.  Deselecting the coarsest hands the box to the next.
            bbox_owner: dict = {}
            for h_id, ch_dict in self._hsrc_masks_by_ch.items():
                if h_id not in base_of or not ch_dict.get(ch):
                    continue
                vroot = root_of[h_id]
                scale = self._hsrc_by_id[h_id].scale
                cur = bbox_owner.get(vroot)
                if cur is None or scale > cur[1]:
                    bbox_owner[vroot] = (h_id, scale)
            bbox_ids = {v[0] for v in bbox_owner.values()}

            for h_id, ch_dict in self._hsrc_masks_by_ch.items():
                if h_id not in base_of:
                    continue
                masks = ch_dict.get(ch)
                if not masks:
                    continue
                r, g, b = base_of[h_id]
                alpha = self._hsrc_alpha[h_id]
                for mask in masks:
                    self._ax.contour(mask.astype(float), [0.5],
                                     colors=[[r, g, b, alpha]], linewidths=0.8)
                if h_id in bbox_ids:  # bbox around the largest visible contour
                    union = np.zeros((H_img, W_img), dtype=bool)
                    for m in masks:
                        union |= m
                    rows, cols = np.where(union)
                    if not len(rows):
                        continue
                    r0, r1 = int(rows.min()), int(rows.max())
                    c0, c1 = int(cols.min()), int(cols.max())
                    self._ax.add_patch(_Rect(
                        (c0 - PAD_BB, r0 - PAD_BB),
                        c1 - c0 + 2*PAD_BB, r1 - r0 + 2*PAD_BB,
                        linewidth=0.9, edgecolor=(r, g, b, 0.9),
                        facecolor="none", zorder=4,
                    ))
                    self._ax.text(
                        c1 + PAD_BB, r1 + PAD_BB, self._hsrc_name.get(h_id, str(h_id)),
                        ha="center", va="center", fontsize=7,
                        color="black", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.22", fc=(r, g, b), ec=(r, g, b), lw=1.2),
                        zorder=6,
                    )

        elif self._mode == "sources":
            from matplotlib.patches import Rectangle as _Rect
            H_img, W_img = img.shape
            PAD_BB = 4

            for sid, ch_dict in self._src_masks_by_ch.items():
                if not self._src_visible[sid].get():
                    continue
                masks = ch_dict.get(ch)
                if not masks:
                    continue
                col  = self._src_color[sid]
                lcol = (min(col[0]+0.3, 1.0), min(col[1]+0.3, 1.0), min(col[2]+0.3, 1.0))
                for mask in masks:
                    self._ax.contour(mask.astype(float), [0.5],
                                     colors=[col], linewidths=0.7)
                    rows, cols = np.where(mask)
                    if not len(rows):
                        continue
                    r0, r1 = int(rows.min()), int(rows.max())
                    c0, c1 = int(cols.min()), int(cols.max())
                    self._ax.add_patch(_Rect(
                        (c0 - PAD_BB, r0 - PAD_BB),
                        c1 - c0 + 2*PAD_BB, r1 - r0 + 2*PAD_BB,
                        linewidth=0.8, edgecolor=lcol, facecolor="none", zorder=4,
                    ))
                    self._ax.text(
                        c1 + PAD_BB, r1 + PAD_BB, str(sid),
                        ha="center", va="center", fontsize=7,
                        color="black", fontweight="bold",
                        bbox=dict(boxstyle="circle,pad=0.22", fc=lcol, ec=lcol, lw=1.2),
                        zorder=6,
                    )

        elif self._mode == "flow":
            if self._flow_scale_mode:
                # One scale at a time: that scale's flow arrows + its contours
                sel = int(self._flow_selected_scale.get())
                flow = self._flow_by_scale.get(sel, {}).get(ch)
                d = self._flowdet_by_scale_ch.get(sel, {}).get(ch)
            else:
                flow = self._flow_by_ch.get(ch)
                d = self._det_by_ch.get(ch)
            if flow is not None:
                H, W = img.shape
                qs = max(min(H, W) // 28, 2)
                ys = np.arange(0, H, qs);  xs = np.arange(0, W, qs)
                Xq, Yq = np.meshgrid(xs, ys)
                u = flow[1][ys[:, None], xs[None, :]].ravel()
                v = flow[0][ys[:, None], xs[None, :]].ravel()
                mag = np.hypot(u, v)
                pk  = float(mag.max())
                if pk > 1e-6:
                    scale = qs * 0.8 / pk
                    qcmap = "cool" if C._current_theme == "dark" else "viridis"
                    self._ax.quiver(
                        Xq.ravel(), Yq.ravel(), u * scale, v * scale,
                        mag, cmap=qcmap, angles="xy", scale_units="xy", scale=1,
                        width=0.003, headwidth=4, headlength=5,
                        alpha=0.85, clim=(0, pk),
                    )
            if d:
                for mask in d.footprint_masks:
                    self._ax.contour(mask.astype(float), [0.5],
                                     colors=[contour_col], linewidths=0.6, alpha=0.7)

        if self._beam is not None or self._pixscale is not None or self._kpc_per_pix is not None:
            from .card import _draw_annotations
            H_img, W_img = img.shape
            _draw_annotations(self._ax, H_img, W_img,
                              self._beam, self._pixscale, self._kpc_per_pix,
                              light=C._current_theme == "light")

        self._canvas.draw()
        self._update_value_labels()
        total_det = sum(len(d.peaks) for d in self._dets) if self._dets else 0
        ch_det = len(self._det_by_ch[ch].peaks) if ch in self._det_by_ch else 0
        parts = [f"Channel {ch}  ({idx+1}/{len(self._channels)})"]
        if self._mode == "detections":
            parts.append(f"{ch_det} det. · {total_det} total")
        elif self._flow_scale_mode:
            sel = int(self._flow_selected_scale.get())
            shown = ch in self._flow_by_scale.get(sel, {})
            parts.append(f"scale j={sel}" + ("  · flow shown" if shown else ""))
        elif self._mode == "flow" and ch in self._flow_by_ch:
            parts.append("flow shown")
        elif self._hsrc_mode:
            n_active = sum(1 for h_id, cd in self._hsrc_masks_by_ch.items()
                           if cd.get(ch) and self._hsrc_visible[h_id].get())
            parts.append(f"{n_active} source(s) here")
        elif self._mode == "sources":
            n_active = sum(1 for sid, v in self._src_visible.items()
                           if v.get() and self._src_masks_by_ch.get(sid, {}).get(ch))
            parts.append(f"{n_active} source(s) here")
        self._ch_lbl.configure(text="  ·  ".join(parts))


class ScaleViewer(tk.Toplevel):
    """Browse per-channel 2D wavelet coefficient maps with embedded parameters."""

    N_ROWS = 2

    def __init__(self, master, cube: np.ndarray, wav_params: dict,
                 detections=None, on_scale_chosen=None, on_params_saved=None,
                 card_0=None):
        super().__init__(master)
        self.title("Wavelet Scale Viewer — Configure")
        self.configure(bg=C.BG)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._cube            = cube
        self._on_scale_chosen = on_scale_chosen
        self._on_params_saved = on_params_saved
        self._wav_params      = dict(wav_params or {})

        from ..detect import max_2d_scales, default_detect_scales
        H, W = cube.shape[1], cube.shape[2]
        self._max_scales = max_2d_scales(H, W)
        n_scales = int(self._wav_params.get("scales", self._max_scales))
        n_scales = max(2, min(self._max_scales, n_scales))
        self._wav_params["scales"] = n_scales

        self._n_scales_var   = tk.IntVar(value=n_scales)
        current_scale        = int(self._wav_params.get("use_scale", max(1, n_scales - 2)))
        self._selected_scale = tk.IntVar(value=current_scale)

        self._channels = [d.channel for d in detections] if detections \
                         else list(range(cube.shape[0]))

        VW = 700

        ctrl = tk.Frame(self, bg=C.BG)
        ctrl.pack(fill=tk.X, padx=10, pady=(10, 14))

        # Multi-scale detection control
        if card_0 and hasattr(card_0, '_multi_scale_enabled') and isinstance(card_0._multi_scale_enabled, tk.BooleanVar):
            self._multi_scale_enabled = card_0._multi_scale_enabled
        else:
            self._multi_scale_enabled = tk.BooleanVar(value=True)

        # Default multi-scale selection: the configured detect_scales, else the
        # three bands below the coarsest (Nmax-1, Nmax-2, Nmax-3).
        _cfg = self._wav_params.get("detect_scales")
        self._default_scales = set(_cfg) if _cfg else set(default_detect_scales(n_scales))
        self._scale_selections = {i: tk.BooleanVar(value=(i in self._default_scales))
                                  for i in range(1, 11)}

        # Detection approach pills in the top bar (replaces channel count)
        _PWL = 160
        _PH  = 20
        _approach_cvs = []

        def _draw_ap(cv, selected, hover=False):
            cv.delete("all")
            if selected:
                fill, fg = _accent(), C.BG
            elif hover:
                fill, fg = C.PLACEHOLDER_BG_EN, _accent()
            else:
                fill, fg = C.CARD_BG, C.STEP_LABEL_TXT
            w, h = int(cv.cget("width")), int(cv.cget("height"))
            cv.create_rectangle(0, 0, w, h, fill=fill, outline=fill)
            cv.create_text(w // 2, h // 2, text=cv._label,
                           fill=fg, font=("Helvetica", 9, "bold"))

        def _refresh_ap():
            for c in _approach_cvs:
                _draw_ap(c, c._is_sel())

        # Container for centered approach pills
        pills_container = tk.Frame(ctrl, bg=C.BG)
        pills_container.pack(fill=tk.BOTH, expand=True)

        pills_inner = tk.Frame(pills_container, bg=C.BG)
        pills_inner.pack(side=tk.TOP, expand=True, anchor="center")

        for ap_txt, ap_val in [("Multi-scale [hierarchical]", True),
                                ("Single-scale [non-hierarchical]", False)]:
            _v = ap_val
            cv = tk.Canvas(pills_inner, width=_PWL, height=_PH,
                           bg=C.CARD_BG, highlightthickness=0, bd=0, cursor="pointinghand")
            cv._label  = ap_txt
            cv._is_sel = lambda v=_v: bool(self._multi_scale_enabled.get()) == v
            _approach_cvs.append(cv)
            cv.pack(side=tk.LEFT, padx=(0, 4))
            cv.bind("<ButtonRelease-1>",
                    lambda e, v=_v: (self._multi_scale_enabled.set(v),
                                     self._rebuild_scale_selector(), _refresh_ap()))
            cv.bind("<Enter>",  lambda e, c=cv: _draw_ap(c, c._is_sel(), hover=True))
            cv.bind("<Leave>",  lambda e, c=cv: _draw_ap(c, c._is_sel()))
            _draw_ap(cv, cv._is_sel())

        self._approach_pills_refresh = _refresh_ap

        self._fig_frame = tk.Frame(self, bg=C.BG)
        self._fig_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 16))
        self._fig = None
        self._img_label = None;  self._img_photo = None
        self._n_scales_last = 0

        _PW, _PH = 26, 20   # pill width / height for numeric pills
        _PWL     = 160      # wider pill for text labels

        def _pill_row(parent, label):
            """Return (row_frame, make_pill_fn, refresh_fn) for a pill button row."""
            f = tk.Frame(parent, bg=C.BG)
            f.pack(fill=tk.X, padx=10, pady=(4, 0))
            tk.Label(f, text=label, bg=C.BG, fg=_accent(),
                     font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
            canvases = []

            def _draw(cv, selected, hover=False):
                cv.delete("all")
                if selected:
                    fill, fg = _accent(), C.BG
                elif hover:
                    fill, fg = C.PLACEHOLDER_BG_EN, _accent()
                else:
                    fill, fg = C.CARD_BG, C.STEP_LABEL_TXT
                w, h = int(cv.cget("width")), int(cv.cget("height"))
                cv.create_rectangle(0, 0, w, h, fill=fill, outline=fill)
                cv.create_text(w // 2, h // 2, text=cv._label,
                               fill=fg, font=cv._font)

            def make_pill(label_txt, w, on_click, is_selected_fn):
                cv = tk.Canvas(f, width=w, height=_PH,
                               bg=C.CARD_BG, highlightthickness=0, bd=0,
                               cursor="pointinghand")
                cv._label = label_txt
                cv._font  = ("Helvetica", 9, "bold")
                canvases.append((cv, is_selected_fn))
                cv.pack(side=tk.LEFT, padx=2)

                def _refresh_all(*_):
                    for c, selfn in canvases:
                        _draw(c, selfn())

                cv.bind("<ButtonRelease-1>", lambda e: (on_click(), _refresh_all()))
                cv.bind("<Enter>",  lambda e: _draw(cv, is_selected_fn(), hover=True))
                cv.bind("<Leave>",  lambda e: _draw(cv, is_selected_fn()))
                _draw(cv, is_selected_fn())
                return cv, _refresh_all

            def refresh():
                for c, selfn in canvases:
                    _draw(c, selfn())

            return f, make_pill, refresh

        # ── Starlet decomposition equation (between subplots and params) ──────
        self._build_equation(self)

        # ── Number of scales ──────────────────────────────────────────────────
        _, _mk_ns, _ref_ns = _pill_row(self, "Number of scales:")
        self._n_scales_pills_refresh = _ref_ns
        for s in range(2, self._max_scales + 1):
            _val = s
            _mk_ns(str(s), _PW,
                   on_click=lambda v=_val: (self._n_scales_var.set(v), self._on_nscales_changed()),
                   is_selected_fn=lambda v=_val: int(self._n_scales_var.get()) == v)

        self._VW = VW
        self._rf = tk.Frame(self, bg=C.BG)
        self._rf.pack(fill=tk.X, padx=10, pady=(4, 2))
        self._rebuild_scale_selector()

        # ── Detection Parameters — SONGS-style slider cards (horizontal) ─────
        tk.Label(self, text="Detection Parameters", bg=C.BG, fg=_accent(),
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 0))
        params_row = tk.Frame(self, bg=C.BG)
        params_row.pack(fill=tk.X, padx=7, pady=(0, 0))

        # `thresh` is the detection threshold as a fraction of the per-scale
        # peak wavelet coefficient (mathematical, noise-model-free).
        _stored = self._wav_params.get("thresh")
        _thresh_init = 0.1 if _stored is None else float(_stored)

        self._pvars: dict[str, tk.Variable] = {}
        SLIDER_H = 11    # uniform height for all slider tracks

        def _symbol_fg():
            return "#000000" if C._current_theme == "light" else "#ffffff"

        def _param_slider(segs, desc, key, from_, to_, init, resolution, fmt, integer=False):
            """DIM card; plain desc above; symbol left of ACCENT-bordered slider+entry."""
            var = tk.DoubleVar(value=init) if not integer else tk.IntVar(value=int(init))
            self._pvars[key] = var

            entry_var = tk.StringVar(value=fmt.format(int(init) if integer else init))
            busy = {'v': False}

            def _fmt_v(v):
                try:    return fmt.format(int(round(v)) if integer else v)
                except: return str(v)

            card = tk.Frame(params_row, bg=C.DIM, padx=1, pady=1)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3, pady=(3, 0))
            fr = tk.Frame(card, bg=C.CARD_BG, padx=6, pady=4)
            fr.pack(fill=tk.BOTH, expand=True)

            # Plain text description above (wraps within the narrower column)
            tk.Label(fr, text=desc, bg=C.CARD_BG, fg=C.STEP_LABEL_TXT,
                     font=("Helvetica", 8), anchor="w", justify="left",
                     wraplength=185).pack(anchor="w", pady=(0, 3))

            # Symbol left + slider box right
            body = tk.Frame(fr, bg=C.CARD_BG)
            body.pack(fill=tk.X)

            sym = _rich_label(body, segs, bg=C.CARD_BG, fg=_symbol_fg())
            sym.pack(side=tk.LEFT, anchor="center", padx=(0, 6))

            slider_box = tk.Frame(body, bg=_accent(), padx=1, pady=1)
            slider_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
            row = tk.Frame(slider_box, bg=C.CARD_BG, padx=4, pady=3)
            row.pack(fill=tk.BOTH, expand=True)

            def _on_scale(val):
                if busy['v']: return
                busy['v'] = True
                v = int(round(float(val))) if integer else float(val)
                var.set(v);  entry_var.set(_fmt_v(v))
                busy['v'] = False
                # live-update the detail-scale overlays as thresholds change
                if key in ("thresh", "k_sigma", "min_area") and \
                        getattr(self, "_img_label", None) is not None:
                    self._draw()

            scale = tk.Scale(row, from_=from_, to=to_, resolution=resolution,
                             orient=tk.HORIZONTAL, command=_on_scale,
                             bg=_accent(), fg=_accent(), troughcolor=C.LOG_BG,
                             activebackground=C.ACCENT_HOVER, highlightthickness=0,
                             sliderrelief=tk.FLAT, bd=0, showvalue=False, width=SLIDER_H)
            scale.set(init)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Frame(row, bg=_accent(), width=1).pack(side=tk.LEFT, fill=tk.Y, padx=(3, 0))

            entry = tk.Entry(row, textvariable=entry_var, width=7,
                             justify="right", bg=C.LOG_BG, fg=_accent(),
                             insertbackground=_accent(), relief=tk.FLAT,
                             highlightthickness=0, font=("Courier", 8), bd=0)
            entry.pack(side=tk.RIGHT, padx=(4, 0))

            def _commit(*_):
                if busy['v']: return
                try:
                    raw = max(from_, min(to_, float(entry_var.get().strip())))
                    busy['v'] = True
                    v = int(round(raw)) if integer else raw
                    var.set(v);  scale.set(v);  entry_var.set(_fmt_v(v))
                    busy['v'] = False
                    if key in ("thresh", "k_sigma", "min_area") and \
                            getattr(self, "_img_label", None) is not None:
                        self._draw()
                except (ValueError, tk.TclError):
                    pass

            entry.bind("<Return>",   _commit)
            entry.bind("<FocusOut>", _commit)

        # w / w_max  — fraction of the per-scale peak coefficient (noise-free)
        _param_slider([("w","n"), ("/w","n"), ("max","s")],
                      "Peak-fraction threshold (per scale).",
                      "thresh", 0.0, 0.9, _thresh_init, 0.01, "{:.2f}")
        # λ_α (σ_α) per-scale noise gate (for noisy cubes; 0 disables)
        _param_slider([("λ","n"), ("α","s"), (" (σ","n"), ("α","s"), (")","n")],
                      "Noise threshold (per scale); 0 = off.",
                      "k_sigma", 0.0, 20.0,
                      float(self._wav_params.get("k_sigma", 0.0)),
                      0.1, "{:.1f}")
        # A_min (px)
        _param_slider([("A","n"), ("min","s"), (" (px)","n")],
                      "Minimum source area.",
                      "min_area", 1, 200,
                      int(self._wav_params.get("min_area", 20)),
                      1, "{:d}", integer=True)

        _FlatBtn(self, "Save Parameters", self._save_params,
                 bg_on=C.ACCENT, active=True).pack(pady=(8, 4))

        sw_card = tk.Frame(self, bg=C.DIM, padx=1, pady=1)
        sw_card.pack(fill=tk.X, padx=10, pady=(4, 10))
        sw_inner = tk.Frame(sw_card, bg=C.CARD_BG, padx=6, pady=4)
        sw_inner.pack(fill=tk.BOTH, expand=True)
        tk.Label(sw_inner, text="Channel", bg=C.CARD_BG, fg=C.STEP_LABEL_TXT,
                 font=("Helvetica", 8), anchor="w").pack(anchor="w", pady=(0, 3))
        sw_box = tk.Frame(sw_inner, bg=_accent(), padx=1, pady=1)
        sw_box.pack(fill=tk.X)
        sw_row = tk.Frame(sw_box, bg=C.CARD_BG, padx=4, pady=3)
        sw_row.pack(fill=tk.BOTH, expand=True)
        N_sw = len(self._channels)
        self._slider = tk.Scale(
            sw_row, from_=0, to=N_sw - 1,
            orient=tk.HORIZONTAL, command=lambda _v: self._draw(),
            bg=_accent(), fg=_accent(), troughcolor=C.LOG_BG,
            activebackground=C.ACCENT_HOVER, highlightthickness=0,
            sliderrelief=tk.FLAT, bd=0, width=11, showvalue=False,
        )
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Frame(sw_row, bg=_accent(), width=1).pack(side=tk.LEFT, fill=tk.Y, padx=(3, 0))
        self._ch_disp = tk.Canvas(sw_row, bg=C.LOG_BG, highlightthickness=0,
                                  width=60, height=20)
        self._ch_disp.pack(side=tk.RIGHT, padx=(3, 0))
        def _upd_sw(idx, N=N_sw):
            self._ch_disp.delete("all")
            w, h = 60, 20
            n = str(self._channels[int(idx)] + 1)
            self._ch_disp.create_text(w//2 - 8, h//2, text=n,
                                      fill=_accent(), font=("Courier", 8, "bold"), anchor="e")
            self._ch_disp.create_text(w//2 - 6, h//2, text=f"/{N}",
                                      fill=_accent(), font=("Courier", 8), anchor="w")
        self._upd_sw = _upd_sw
        _upd_sw(N_sw // 2)
        self._slider.set(N_sw // 2)

        self._rebuild_figure()
        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())
        self.maxsize(self.winfo_width(), 9999)

    def _on_nscales_changed(self):
        from ..detect import default_detect_scales
        n_new = int(self._n_scales_var.get())
        if int(self._selected_scale.get()) > n_new - 1:
            self._selected_scale.set(max(1, n_new - 1))
        self._wav_params["scales"]    = n_new
        self._wav_params["use_scale"] = int(self._selected_scale.get())
        # re-apply the default selection (Nmax-1, Nmax-2, Nmax-3) for the new count
        self._default_scales = set(default_detect_scales(n_new))
        for i in range(1, 11):
            if i not in self._scale_selections:
                self._scale_selections[i] = tk.BooleanVar()
            self._scale_selections[i].set(i in self._default_scales)
        self._rebuild_scale_selector()

    def _rebuild_scale_selector(self):
        """Rebuild scale selector as pills (multi-scale toggle or single-scale pick)."""
        for w in self._rf.winfo_children():
            w.destroy()

        n_detail  = int(self._n_scales_var.get()) - 1
        _PW, _PH  = 26, 20
        is_multi  = self._multi_scale_enabled.get()
        lbl_text  = "Select scales:" if is_multi else "Choose scale:"

        tk.Label(self._rf, text=lbl_text, bg=C.BG, fg=_accent(),
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))

        pills = []

        def _draw(cv, selected, hover=False):
            cv.delete("all")
            if selected:
                fill, fg = _accent(), C.BG
            elif hover:
                fill, fg = C.PLACEHOLDER_BG_EN, _accent()
            else:
                fill, fg = C.CARD_BG, C.STEP_LABEL_TXT
            cv.create_rectangle(0, 0, _PW, _PH, fill=fill, outline=fill)
            cv.create_text(_PW // 2, _PH // 2, text=cv._label,
                           fill=fg, font=("Helvetica", 9, "bold"))

        def _refresh_pills():
            for cv in pills:
                _draw(cv, cv._is_selected())

        for s in range(1, n_detail + 1):
            if s not in self._scale_selections:
                self._scale_selections[s] = tk.BooleanVar(
                    value=(s in getattr(self, "_default_scales", set())))

            cv = tk.Canvas(self._rf, width=_PW, height=_PH,
                           bg=C.CARD_BG, highlightthickness=0, bd=0,
                           cursor="pointinghand")
            cv._label = str(s)

            if is_multi:
                _s = s
                cv._is_selected = lambda sc=_s: self._scale_selections[sc].get()
                def _on_click(sc=_s):
                    self._scale_selections[sc].set(not self._scale_selections[sc].get())
                    _refresh_pills()
                    self._on_scale_selection_changed()
            else:
                _s = s
                cv._is_selected = lambda sc=_s: int(self._selected_scale.get()) == sc
                def _on_click(sc=_s):
                    self._selected_scale.set(sc)
                    _refresh_pills()
                    self._on_radio()

            cv.bind("<ButtonRelease-1>", lambda e, fn=_on_click: fn())
            cv.bind("<Enter>",  lambda e, c=cv: _draw(c, c._is_selected(), hover=True))
            cv.bind("<Leave>",  lambda e, c=cv: _draw(c, c._is_selected()))
            _draw(cv, cv._is_selected())
            pills.append(cv)
            cv.pack(side=tk.LEFT, padx=2)

        # refresh approach + n_scales pills too so they stay in sync
        if hasattr(self, '_approach_pills_refresh'):
            self._approach_pills_refresh()
        if hasattr(self, '_n_scales_pills_refresh'):
            self._n_scales_pills_refresh()

        self._rebuild_figure()

    def _rebuild_choose_scale_radios(self):
        """Legacy method for rebuilding radio buttons (called by number-of-scales change)."""
        for w in self._rf.winfo_children():
            w.destroy()
        tk.Label(self._rf, text="Choose scale:", bg=C.BG, fg=_accent(),
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        n_detail = int(self._n_scales_var.get()) - 1
        for s in range(1, n_detail + 1):
            tk.Radiobutton(self._rf, text=str(s),
                           variable=self._selected_scale, value=s,
                           command=self._on_radio,
                           bg=C.BG, fg=_accent(), selectcolor=C.CARD_BG,
                           activebackground=C.BG, activeforeground=_accent(),
                           font=("Helvetica", 9), relief=tk.FLAT).pack(
                               side=tk.LEFT, padx=2)

    def _on_scale_selection_changed(self):
        """Called when a checkbox in multi-scale mode is toggled."""
        self._rebuild_figure()

    def _build_equation(self, parent):
        """Starlet synthesis equation as serif text with sub/superscripts."""
        cap = tk.Label(parent, text="Starlet (à trous IUWT) decomposition",
                       bg=C.BG, fg=_accent(), font=("Helvetica", 8, "bold"))
        cap.pack(fill=tk.X, padx=10, pady=(2, 0))

        txt = tk.Text(parent, bg=C.BG, fg=C.STEP_LABEL_TXT, relief=tk.FLAT,
                      bd=0, highlightthickness=0, height=2, width=1, wrap="none",
                      cursor="arrow", font=("Georgia", 17))
        txt.tag_configure("center", justify="center")
        txt.tag_configure("sup", font=("Georgia", 12), offset=8)
        txt.tag_configure("sub", font=("Georgia", 12), offset=-4)
        txt.tag_configure("big", font=("Georgia", 22))
        runs = [
            ("I(x, y) = ", ""),
            ("Σ", "big"), ("J", "sup"), ("j=1", "sub"),
            ("  w", ""), ("j", "sub"), ("(x, y)  +  c", ""),
            ("J", "sub"), ("(x, y)", ""),
        ]
        for s, tag in runs:
            txt.insert("end", s, (tag,) if tag else ())
        txt.tag_add("center", "1.0", "end")
        txt.configure(state="disabled")
        txt.pack(fill=tk.X, padx=10, pady=(2, 8))

    def _on_close(self):
        self._cleanup_canvas()
        self.destroy()

    def _save_params(self):
        def _f(k): return self._pvars[k].get()
        thresh_s = str(_f("thresh")).strip()
        try:
            # thresh = fraction of the per-scale peak wavelet coefficient
            thresh_frac = float(thresh_s) if thresh_s else None
            params = dict(
                scales=int(self._n_scales_var.get()),
                k_sigma=float(_f("k_sigma")),
                use_scale=int(self._selected_scale.get()),
                min_area=int(_f("min_area")),
                thresh=thresh_frac,
                use_mean_map_sigma=True,
            )
            # In multi-scale mode, honour the checked detail scales.
            if self._multi_scale_enabled.get():
                n_detail = int(self._n_scales_var.get()) - 1
                chosen = [s for s in range(1, n_detail + 1)
                          if self._scale_selections.get(s)
                          and self._scale_selections[s].get()]
                params["detect_scales"] = chosen or list(range(1, n_detail + 1))
        except ValueError as exc:
            messagebox.showerror("Bad parameter", str(exc), parent=self)
            return
        if self._on_params_saved:
            self._on_params_saved(params)
        self._cleanup_canvas()
        self.destroy()

    def _cleanup_canvas(self):
        """Close matplotlib figure and drop the label image before destroy()."""
        self._img_photo = None
        if self._img_label is not None:
            try:
                self._img_label.destroy()
            except Exception:
                pass
            self._img_label = None
        if self._fig is not None:
            try:
                plt.close(self._fig)
            except Exception:
                pass
            self._fig = None

    def _rebuild_figure(self):
        n_scales = int(self._n_scales_var.get())
        import math
        n_rows = math.ceil(n_scales / 4)
        n_cols = math.ceil(n_scales / n_rows)

        if self._fig:
            plt.close(self._fig)
            self._fig = None
        if self._img_label:
            self._img_label.destroy()
            self._img_label = None
        self._img_photo = None

        cell_px = self._VW // n_cols
        dpi     = 96
        n_slots = n_rows * n_cols
        self._fig = plt.Figure(
            figsize=(self._VW / dpi, (n_rows * cell_px) / dpi),
            dpi=dpi, facecolor=C.LOG_BG,
        )
        self._axes = []
        for i in range(n_slots):
            ax = self._fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_xticks([]);  ax.set_yticks([])
            ax.set_facecolor(C.LOG_BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(C.DIM);  sp.set_linewidth(0.5)
            self._axes.append(ax)
        self._fig.subplots_adjust(left=0.06, right=0.94,
                                  top=0.86, bottom=0.06,
                                  hspace=0.42, wspace=0.10)
        self._img_label = tk.Label(self._fig_frame, bg=C.LOG_BG, bd=0)
        self._img_label.pack(fill=tk.BOTH, expand=True)
        self._n_scales_last = n_scales
        if hasattr(self, '_slider'):
            self._draw()

    @staticmethod
    def _detection_mask(coeff_plane, band, alpha, ksig, min_area):
        """Detected-region mask for one detail scale at the current thresholds.

        Mirrors `detect.detect_all_scales`: peak-fraction gate (alpha · max),
        optional per-scale noise gate (ksig · MAD σ), then a min-area filter.
        """
        peak = float(band.max())
        if peak <= 0:
            return None
        binary = band > alpha * peak
        if ksig and ksig > 0:
            sig = 1.4826 * np.median(np.abs(coeff_plane - np.median(coeff_plane))) + 1e-12
            binary &= band > (ksig * sig)
        if not binary.any():
            return None
        from scipy.ndimage import label as _label
        from skimage.measure import regionprops as _rprops
        lab, _ = _label(binary)
        keep = np.zeros_like(binary)
        for r in _rprops(lab):
            if r.area >= min_area:
                keep |= (lab == r.label)
        return keep

    def _on_radio(self):
        scale = int(self._selected_scale.get())
        if self._on_scale_chosen:
            self._on_scale_chosen(scale)
        self._draw()

    def _draw(self):
        n_scales = int(self._n_scales_var.get())
        if n_scales != self._n_scales_last:
            self._rebuild_figure();  return

        idx = int(self._slider.get())
        ch  = self._channels[idx]
        if hasattr(self, '_upd_sw'):
            self._upd_sw(idx)
        img = self._cube[ch].astype(np.float32)

        from ..detect import starlet_transform
        coeffs   = starlet_transform(img, scales=n_scales)
        n_detail = n_scales - 1

        # Get selected scales for highlighting
        is_multi_scale = self._multi_scale_enabled.get()
        if is_multi_scale:
            selected_scales = {s for s in range(1, n_detail + 1) if self._scale_selections[s].get()}
        else:
            chosen = int(self._selected_scale.get())
            selected_scales = {chosen}

        # Current threshold parameters (live overlay on the detail scales only)
        def _pget(k, default):
            v = self._pvars.get(k) if hasattr(self, "_pvars") else None
            try:
                return float(v.get()) if v is not None else default
            except Exception:
                return default
        alpha    = _pget("thresh", 0.1)
        ksig     = _pget("k_sigma", 0.0)
        min_area = int(_pget("min_area", 20))

        for i, ax in enumerate(self._axes):
            ax.clear()
            ax.set_xticks([]);  ax.set_yticks([])
            ax.set_facecolor(C.LOG_BG)

            if i < n_scales:
                is_coarse = (i == n_detail)
                band      = np.clip(coeffs[i], 0, None)
                vmax      = float(np.nanpercentile(band, 99.5)) if band.max() > 0 else 1e-9
                ax.imshow(band, cmap="seismic", origin="lower", vmin=-vmax, vmax=vmax)
                scale_num = i + 1
                is_chosen = (not is_coarse) and (scale_num in selected_scales)
                label     = "Coarse Scale" if is_coarse else f"Scale {scale_num}"
                ax.set_title(label,
                             color=_accent() if is_chosen else C.STEP_LABEL_TXT,
                             fontsize=8, pad=8,
                             fontweight="bold" if is_chosen else "normal")
                for sp in ax.spines.values():
                    sp.set_edgecolor(_accent() if is_chosen else C.DIM)
                    sp.set_linewidth(1.5 if is_chosen else 0.5)

                # Overlay the detected regions at the current thresholds: paint
                # everything *outside* the mask white and outline it in white.
                # The coarse residual is never thresholded — it stays untouched.
                if not is_coarse:
                    mask = self._detection_mask(coeffs[i], band, alpha, ksig, min_area)
                    if mask is None:
                        mask = np.zeros(band.shape, dtype=bool)
                    ov = np.ones(mask.shape + (4,), dtype=np.float32)  # opaque white
                    ov[mask, 3] = 0.0                                  # clear inside mask
                    ax.imshow(ov, origin="lower", interpolation="nearest")
                    if mask.any():
                        ax.contour(mask.astype(float), [0.5],
                                   colors=["white"], linewidths=0.8)
            else:
                ax.set_visible(False)

        if self._img_label and self._img_label.winfo_exists():
            import io as _io
            from PIL import Image as _PilImg, ImageTk as _ImageTk
            buf = _io.BytesIO()
            self._fig.savefig(buf, format="png", dpi=96,
                              bbox_inches="tight", pad_inches=0.1,
                              facecolor=self._fig.get_facecolor())
            buf.seek(0)
            pil = _PilImg.open(buf).convert("RGB")
            # `bbox_inches="tight"` crops to content; pad extra whitespace on
            # the left/right (figure background colour) so the subplots are
            # not flush against the window edges.
            side_pad = 48
            fc = self._fig.get_facecolor()
            bg = tuple(int(round(c * 255)) for c in fc[:3])
            padded = _PilImg.new("RGB", (pil.width + 2 * side_pad, pil.height), bg)
            padded.paste(pil, (side_pad, 0))
            self._img_photo = _ImageTk.PhotoImage(padded)
            self._img_label.configure(image=self._img_photo)
