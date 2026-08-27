import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..utils import clamped_bbox
from . import _constants as C
from ._theme import set_titlebar_appearance
from ._transport import TransportControls
from .source_tree import SourceTreePanel


def _accent():
    return C.LOG_TXT if C._current_theme == "light" else C.ACCENT


def _mpl_theme():
    """Matplotlib colours for the current GUI theme."""
    light = C._current_theme == "light"
    return dict(
        fig_bg="#ffffff" if light else "#0a0a14",
        # Pure black in light mode: every piece of figure text — colourbar
        # labels and tick labels, axis labels, panel tags, legend entries —
        # routes through this one key, and the old near-navy read as grey
        # against a white figure background.
        fg="#000000" if light else "white",
        spine="#b0b0c8" if light else "#555577",
        m1_bg="#dfe2ec" if light else "#222236",
        legend_bg="#eef0f6" if light else "#16213e",
        bbox_bg="white" if light else "black",
        total="0.35" if light else "0.7",
        cmap="cubehelix_r" if light else "inferno",
        # RGBA used to dim the region outside the source mask (lighter in light)
        dim=(0.0, 0.0, 0.0, 0.28) if light else (0.0, 0.0, 0.0, 0.55),
    )


def _build_source_unions(tracks, sources, H, W):
    """Return (src_union, all_union, det_chs_all) — boolean masks and channel list."""
    src_union = {}
    tracks_by_id = {t["id"]: t for t in tracks}
    for src in sources:
        m = np.zeros((H, W), dtype=bool)
        for tid in src["track_ids"]:
            t = tracks_by_id.get(tid)
            if not t:
                continue
            for mask in t["masks"].values():
                m |= mask
        src_union[src["id"]] = m
    all_union = np.zeros((H, W), dtype=bool)
    for m in src_union.values():
        all_union |= m
    det_chs_all = sorted({
        ch for src in sources
        for tid in src["track_ids"] if tid in tracks_by_id
        for ch in tracks_by_id[tid]["masks"]
    })
    return src_union, all_union, det_chs_all


def _source_colors(sources):
    from . import _constants as C
    cmap = plt.get_cmap("tab10")
    colors = {s["id"]: cmap(i % 10) for i, s in enumerate(sources)}
    if C._current_theme == "light":
        # Darker saturated versions of the dark-mode tab10 colors
        colors = {sid: tuple(min(c * 0.6, 1.0) for c in rgba[:3]) + (rgba[3],)
                  for sid, rgba in colors.items()}
    return colors


def _mom1_valid(total_flux, cube, n_chan, k=5.0):
    """Pixels where the moment-1 denominator is trustworthy.

    Moment 1 is ``Σ(flux·v) / Σ(flux)``.  Guarding only on ``total_flux > 0``
    is not enough for interferometric data: positive and negative noise very
    nearly cancel, and a denominator a hair above zero sends the ratio to
    absurd velocities.  Measured on an IC5179 CO(2-1) cube, that guard produced
    a moment-1 map spanning −296,002 … +397,240 km/s against a real velocity
    axis of 2780 … 3979 — 5.7% of pixels unphysical, and because they set the
    colour scale the whole map rendered as one flat tone.

    Requiring the summed flux to exceed ``k·σ·√n_chan`` (σ from a MAD estimate
    of the cube) keeps only pixels with a genuinely significant denominator.
    On the same cube this returns 2859 … 3693 km/s.
    """
    finite = cube[np.isfinite(cube)]
    if finite.size == 0:
        return total_flux > 0
    sigma = 1.4826 * np.median(np.abs(finite - np.median(finite)))
    if not np.isfinite(sigma) or sigma <= 0:
        return total_flux > 0
    return total_flux > (k * sigma * np.sqrt(max(int(n_chan), 1)))


def _mom1_limits(m1):
    """Diverging colour limits for a moment-1 map, centred on the map itself.

    A moment-1 map holds *absolute* velocities (e.g. 2780–3979 km/s for a
    galaxy at cz ≈ 3400), not offsets from zero.  Scaling it symmetrically
    about 0 therefore pushes every pixel into the top colour and the rotation
    signature disappears — the map renders as one flat block.

    Centre on the median (the systemic velocity, near enough) and take the
    half-range from a high percentile rather than the extremum, so a handful of
    edge pixels cannot flatten the whole map.
    """
    if m1 is None or not np.any(~np.isnan(m1)):
        return 0.0, 1.0
    c = float(np.nanmedian(m1))
    half = float(np.nanpercentile(np.abs(m1 - c), 99.0))
    if not np.isfinite(half) or half <= 0:
        half = 1.0
    return c - half, c + half


def _clip_to_axis(mom1, ch_axis):
    """Blank any moment-1 pixel outside the actual velocity axis.

    A backstop: an intensity-weighted mean of velocities can only legitimately
    land inside the range of those velocities, so anything outside is numerical
    fallout, not a measurement.
    """
    if ch_axis is None or len(ch_axis) == 0:
        return mom1
    lo, hi = float(np.min(ch_axis)), float(np.max(ch_axis))
    return np.where((mom1 >= lo) & (mom1 <= hi), mom1, np.nan)


def _shade(base, alpha):
    """Blend an RGB colour toward the current figure background by *alpha*."""
    bg = (1.0, 1.0, 1.0) if C._current_theme == "light" else (0.04, 0.04, 0.08)
    return tuple(c * alpha + b * (1 - alpha) for c, b in zip(base, bg))


class CombinedAnalysisWindow(TransportControls, tk.Toplevel):
    """Channel viewer + full-field Moment 0/1 + integrated spectra in one window."""

    SYMMETRIC_CMAPS = ["RdBu_r", "seismic", "coolwarm", "bwr",
                       "PuOr_r", "PiYG_r", "Spectral_r"]

    def __init__(self, master, cube, tracks, sources, vel_array=None,
                 hierarchical_sources=None, tracks_per_scale=None,
                 beam=None, pixscale=None, kpc_per_pix=None,
                 coarse_scale=None):
        super().__init__(master)
        self.title("Source Analysis — Channels, Moments & Spectra")
        self.configure(bg=C.BG)
        self.geometry("1320x840")
        self.minsize(1120, 740)
        self.resizable(True, True)

        self._cube      = cube
        self._tracks    = tracks
        self._sources   = sources
        self._beam      = beam
        self._pixscale  = pixscale
        self._kpc_per_pix = kpc_per_pix
        self._vel_array = (np.asarray(vel_array, dtype=np.float64)
                           if vel_array is not None else None)
        self._has_vel  = self._vel_array is not None
        self._sp_axis  = self._vel_array if self._has_vel \
                         else np.arange(cube.shape[0], dtype=np.float64)
        self._sp_label = r"Velocity (km s$^{-1}$)" if self._has_vel else "Channel"
        self._m1_unit  = r"km s$^{-1}$" if self._has_vel else "channel"

        H, W = cube.shape[1], cube.shape[2]
        self._H, self._W = H, W
        self._hier = bool(hierarchical_sources and tracks_per_scale)

        # plot-parameter state
        light = C._current_theme == "light"
        self._img_cmap   = tk.StringVar(value="cubehelix" if light else "inferno")
        self._img_invert = tk.BooleanVar(value=light)
        self._m1_cmap    = tk.StringVar(value="RdBu_r")
        self._m1_invert  = tk.BooleanVar(value=False)
        self._norm_mode  = tk.StringVar(value="Per-source cube")
        _vmin, _vmax = float(np.nanmin(cube)), float(np.nanmax(cube))
        if not (np.isfinite(_vmin) and np.isfinite(_vmax)) or _vmax <= _vmin:
            _vmin, _vmax = 0.0, 1.0
        self._cube_vmin, self._cube_vmax = _vmin, _vmax
        # vmin/vmax sliders (used in per-cube mode) + scale/gamma
        self._vmin_var  = tk.DoubleVar(value=_vmin)
        self._vmax_var  = tk.DoubleVar(value=_vmax)
        self._scale_mode = tk.StringVar(value="Linear")   # Linear / Log / Power
        self._gamma_var = tk.DoubleVar(value=0.5)
        self._init_transport_state()

        body = tk.Frame(self, bg=C.BG)
        body.pack(fill=tk.BOTH, expand=True)
        left_col = tk.Frame(body, bg=C.BG)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fig_frame = tk.Frame(left_col, bg=C.BG)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._fig = plt.Figure(figsize=(11.5, 7.0), dpi=96,
                               facecolor=_mpl_theme()["fig_bg"])
        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._build_controls_row(left_col)

        # per-source union masks + per-channel masks for the channel overlays
        self._masks_by_ch: dict = {}
        if self._hier:
            self._tree = SourceTreePanel(body, hierarchical_sources,
                                         tracks_per_scale, on_change=self._draw,
                                         mode="multi", title="Source tree",
                                         default_active_letter="B",
                                         coarse_scale=coarse_scale)
            self._tree.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=8)
            self._units = {}
            for hid, ch_dict in self._tree.masks_by_ch.items():
                m = np.zeros((H, W), dtype=bool)
                mbc = {}
                for ch, masks in ch_dict.items():
                    cm = np.zeros((H, W), dtype=bool)
                    for mm in masks:
                        cm |= mm
                    mbc[ch] = cm
                    m |= cm
                self._units[hid] = m
                self._masks_by_ch[hid] = mbc
            det_chs = sorted({ch for cd in self._tree.masks_by_ch.values() for ch in cd})
        else:
            self._src_union, _all, self._det_chs_all = \
                _build_source_unions(tracks, sources, H, W)
            self._src_color = _source_colors(sources)
            self._units = self._src_union
            det_chs = self._det_chs_all
            tracks_by_id = {t["id"]: t for t in tracks}
            for s in sources:
                mbc: dict = {}
                for tid in s["track_ids"]:
                    t = tracks_by_id.get(tid)
                    if not t:
                        continue
                    for ch, mask in t["masks"].items():
                        mbc[ch] = mbc.get(ch, np.zeros((H, W), dtype=bool)) | mask
                self._masks_by_ch[s["id"]] = mbc
            self._show_total = tk.BooleanVar(value=True)
            self._show_diff  = tk.BooleanVar(value=True)
            self._show_src: dict[int, tk.BooleanVar] = {}
            self._build_checkbox_panel(body, sources)

        self._all_union = np.zeros((H, W), dtype=bool)
        for m in self._units.values():
            self._all_union |= m

        self._total_spec = cube.sum(axis=(1, 2))
        self._spec = {
            sid: (cube[:, m].sum(axis=1) if m.any() else np.zeros(cube.shape[0]))
            for sid, m in self._units.items()
        }
        src_flux = (cube[:, self._all_union].sum(axis=1)
                    if self._all_union.any() else np.zeros(cube.shape[0]))
        self._diffuse_spec = self._total_spec - src_flux
        self._src_spec = self._spec

        det_idx = np.array(det_chs) if det_chs else np.arange(cube.shape[0])
        self._mom0 = cube[det_idx].sum(axis=0)
        flux_stack = cube[det_idx]
        total_flux = flux_stack.sum(axis=0)
        ch_axis = self._sp_axis[det_idx].astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            self._mom1 = _clip_to_axis(np.where(
                _mom1_valid(total_flux, cube, len(det_idx)) & self._all_union,
                (flux_stack * ch_axis[:, None, None]).sum(axis=0) / total_flux,
                np.nan,
            ), ch_axis)

        # channel slider scrubs the whole cube
        self._chan_list = list(range(cube.shape[0]))
        mid = len(self._chan_list) // 2
        self._slider.configure(to=max(len(self._chan_list) - 1, 0))
        self._slider.set(mid)
        self._draw()
        set_titlebar_appearance(self)

    # ─────────────────────────── plot-param widgets ─────────────────────────
    @staticmethod
    def _inv_cmap(name, invert):
        if not invert:
            return name
        return name[:-2] if name.endswith("_r") else name + "_r"

    @staticmethod
    def _mask_edge(ax, mask, color, lw, k=10):
        """Pixel-accurate outline of a binary mask (no contour corner-cutting)."""
        if mask is None or not mask.any():
            return
        H, W = mask.shape
        up = np.repeat(np.repeat(mask.astype(float), k, axis=0), k, axis=1)
        xs = (np.arange(W * k) + 0.5) / k - 0.5
        ys = (np.arange(H * k) + 0.5) / k - 0.5
        ax.contour(xs, ys, up, [0.5], colors=[color], linewidths=lw)

    def _card(self, parent, title, title_fg=None,
              title_font=("Helvetica", 7), pad=(5, 3)):
        outer = tk.Frame(parent, bg=C.DIM, padx=1, pady=1)
        inner = tk.Frame(outer, bg=C.CARD_BG, padx=pad[0], pady=pad[1])
        inner.pack(fill=tk.BOTH, expand=True)
        tk.Label(inner, text=title, bg=C.CARD_BG, fg=title_fg or C.STEP_LABEL_TXT,
                 font=title_font).pack(anchor="w", pady=(0, 3))
        return outer, inner

    def _styled_optionmenu(self, parent, var, options, width=10):
        om = tk.OptionMenu(parent, var, *options, command=lambda _v: self._draw())
        om.configure(bg=C.CARD_BG, fg=C.STEP_LABEL_TXT,
                     activebackground=C.DIM, activeforeground=C.STEP_LABEL_TXT,
                     highlightthickness=0, relief=tk.FLAT,
                     font=("Helvetica", 8), width=width, anchor="w")
        om["menu"].configure(bg=C.CARD_BG, fg=C.STEP_LABEL_TXT,
                             activebackground=C.DIM, activeforeground=C.STEP_LABEL_TXT,
                             font=("Helvetica", 8))
        return om

    def _value_slider(self, parent, label, var, frm, to, res, fmt, label_fg=None,
                      label_widget=None, expand=False):
        """Label + accent box [slider | divider | value textbox]; returns (frame, scale).

        *label_widget* is an optional callable(parent)->widget for a custom label
        (e.g. an italic V₍ₘᵢₙ₎ canvas).  *expand* makes the slider fill its parent.
        """
        f = tk.Frame(parent, bg=C.CARD_BG)
        if label_widget is not None:
            label_widget(f).pack(side=tk.LEFT, padx=(0, 3))
        else:
            tk.Label(f, text=label, bg=C.CARD_BG, fg=label_fg or C.STEP_LABEL_TXT,
                     font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 3))
        box = tk.Frame(f, bg=_accent(), padx=1, pady=1)
        box.pack(side=tk.LEFT, fill=tk.X if expand else None, expand=expand)
        rowf = tk.Frame(box, bg=C.CARD_BG, padx=3, pady=2); rowf.pack(fill=tk.BOTH, expand=True)
        s = tk.Scale(rowf, from_=frm, to=to, resolution=res, orient=tk.HORIZONTAL,
                     variable=var, bg=_accent(), fg=_accent(), troughcolor=C.LOG_BG,
                     activebackground=C.ACCENT_HOVER, highlightthickness=0,
                     sliderrelief=tk.FLAT, bd=0, width=9, showvalue=False, length=60)
        s.pack(side=tk.LEFT, fill=tk.X if expand else None, expand=expand)
        tk.Frame(rowf, bg=_accent(), width=1).pack(side=tk.LEFT, fill=tk.Y, padx=(3, 0))
        sv = tk.StringVar(value=fmt(float(var.get())))
        ent = tk.Entry(rowf, textvariable=sv, width=8, justify="right", bg=C.CARD_BG,
                       readonlybackground=C.CARD_BG, fg=_accent(), relief=tk.FLAT,
                       highlightthickness=0, font=("Courier", 7), bd=0, state="readonly")
        ent.pack(side=tk.LEFT, padx=(3, 0))

        def _cmd(_v=None):
            sv.set(fmt(float(var.get())))
            self._draw()
        s.configure(command=_cmd)
        s._box = box
        return f, s

    def _build_scale_pills(self, parent):
        from tkinter import font as tkfont
        self._spill_font = tkfont.Font(family="Helvetica", size=8, weight="bold")
        PILL_H, GAP, X0 = 20, 4, 1
        self._spill_nodes = []
        x = X0; cy = PILL_H // 2 + 1
        for mode in ("Linear", "Log", "Power"):
            tw = self._spill_font.measure(mode)
            self._spill_nodes.append((mode, x, cy - PILL_H // 2, x + tw + 12,
                                      cy + PILL_H // 2, cy, mode))
            x += tw + 12 + GAP
        cv = tk.Canvas(parent, bg=C.CARD_BG, width=x, height=PILL_H + 2,
                       highlightthickness=0, bd=0)
        cv.bind("<Button-1>", self._on_scale_pill_click)
        self._spill_canvas = cv
        self._draw_scale_pills()
        return cv

    def _draw_scale_pills(self):
        cv = self._spill_canvas
        cv.delete("all")
        sel = self._scale_mode.get()
        light = C._current_theme == "light"
        for mode, x1, y1, x2, y2, cy, label in self._spill_nodes:
            if mode == sel:
                fill = outline = _accent()
                txt = "#ffffff" if light else "#000000"
            else:
                fill, outline, txt = C.CARD_BG, C.DIM_TXT, C.DIM_TXT
            cv.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=1)
            cv.create_text((x1 + x2) // 2, cy, text=label, font=self._spill_font, fill=txt)

    def _on_scale_pill_click(self, event):
        for mode, x1, y1, x2, y2, cy, label in self._spill_nodes:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self._scale_mode.set(mode)
                self._draw_scale_pills()
                self._on_scale_change()
                return

    def _set_slider_enabled(self, s, on):
        if on:
            s.configure(state="normal", fg=_accent(), bg=_accent(),
                        troughcolor=C.LOG_BG, activebackground=C.ACCENT_HOVER)
            s._box.configure(bg=_accent())
        else:
            s.configure(state="disabled", fg=C.DIM, bg=C.DIM,
                        troughcolor=C.CARD_BG, activebackground=C.DIM)
            s._box.configure(bg=C.DIM)

    def _update_norm_enabled(self):
        on = self._norm_mode.get().startswith("Per-source")   # vmin/vmax usable per-cube
        self._set_slider_enabled(self._vmin_scale, on)
        self._set_slider_enabled(self._vmax_scale, on)

    def _on_norm_change(self):
        self._update_norm_enabled()
        self._draw()

    def _on_scale_change(self, redraw=True):
        self._set_slider_enabled(self._gamma_scale, self._scale_mode.get() == "Power")
        if redraw:
            self._draw()

    def _build_norm_scale_card(self, parent):
        from .viewers import _rich_label
        outer = tk.Frame(parent, bg=C.DIM, padx=1, pady=1)
        inner = tk.Frame(outer, bg=C.CARD_BG, padx=6, pady=8)   # slightly taller
        inner.pack(fill=tk.BOTH, expand=True)
        row = tk.Frame(inner, bg=C.CARD_BG); row.pack(fill=tk.X)
        light = C._current_theme == "light"
        vfg = "#000000" if light else "#ffffff"   # V_min / V_max symbols

        # 1 — per-source / per-channel normalisation
        om = tk.OptionMenu(row, self._norm_mode, "Per-source cube", "Per-channel",
                           command=lambda _v: self._on_norm_change())
        om.configure(bg=C.CARD_BG, fg=C.STEP_LABEL_TXT, activebackground=C.DIM,
                     activeforeground=C.STEP_LABEL_TXT, highlightthickness=0,
                     relief=tk.FLAT, font=("Helvetica", 8), width=13, anchor="w")
        om["menu"].configure(bg=C.CARD_BG, fg=C.STEP_LABEL_TXT, activebackground=C.DIM,
                             activeforeground=C.STEP_LABEL_TXT, font=("Helvetica", 8))
        om.pack(side=tk.LEFT, padx=(0, 8))

        # 2 — scaling pills
        self._build_scale_pills(row).pack(side=tk.LEFT, padx=(0, 8))

        # 3 — gamma slider + divider + value
        res = (self._cube_vmax - self._cube_vmin) / 500.0 or 1e-6
        gf, self._gamma_scale = self._value_slider(
            row, "γ", self._gamma_var, 0.1, 2.0, 0.05, lambda v: f"{v:.2f}")
        gf.pack(side=tk.LEFT, padx=(0, 8))

        # 4 + 5 — vmin / vmax: italic V with full "min"/"max" subscript, wide sliders
        def _vlabel(sub):
            return lambda p: _rich_label(p, [("V", "n"), (sub, "s")],
                                         bg=C.CARD_BG, fg=vfg)
        vminf, self._vmin_scale = self._value_slider(
            row, "vmin", self._vmin_var, self._cube_vmin, self._cube_vmax, res,
            lambda v: f"{v:.2e}", label_widget=_vlabel("min"), expand=True)
        vminf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        vmaxf, self._vmax_scale = self._value_slider(
            row, "vmax", self._vmax_var, self._cube_vmin, self._cube_vmax, res,
            lambda v: f"{v:.2e}", label_widget=_vlabel("max"), expand=True)
        vmaxf.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._update_norm_enabled()
        self._on_scale_change(redraw=False)
        return outer

    def _build_controls_row(self, parent):
        row = tk.Frame(parent, bg=C.BG)
        row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 6))
        grey = C.STEP_LABEL_TXT

        # left column (wide): normalisation/scaling card above the channel card
        leftcol = tk.Frame(row, bg=C.BG)
        leftcol.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_norm_scale_card(leftcol).pack(side=tk.TOP, fill=tk.X)
        self._build_channel_card(leftcol).pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        # right column: flux + velocity colour-map cards stacked, matching height
        rightcol = tk.Frame(row, bg=C.BG)
        rightcol.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0))

        def _chk(p, var):
            return tk.Checkbutton(
                p, text="Invert", variable=var, command=self._draw,
                bg=C.CARD_BG, fg=C.STEP_LABEL_TXT, selectcolor=C.LOG_BG,
                activebackground=C.CARD_BG, activeforeground=_accent(),
                font=("Helvetica", 8), relief=tk.FLAT)

        f_out, f_in = self._card(rightcol, "Flux colour map", grey)
        f_out.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        fr = tk.Frame(f_in, bg=C.CARD_BG); fr.pack(fill=tk.X)
        self._styled_optionmenu(fr, self._img_cmap, list(C._CMAPS), width=9).pack(side=tk.LEFT)
        _chk(fr, self._img_invert).pack(side=tk.LEFT, padx=(1, 0))

        v_out, v_in = self._card(rightcol, "Velocity colour map", grey)
        v_out.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(4, 0))
        vr = tk.Frame(v_in, bg=C.CARD_BG); vr.pack(fill=tk.X)
        self._styled_optionmenu(vr, self._m1_cmap, self.SYMMETRIC_CMAPS, width=9).pack(side=tk.LEFT)
        _chk(vr, self._m1_invert).pack(side=tk.LEFT, padx=(1, 0))

    @staticmethod
    def _contrast_color(rgb):
        """White on a dark colour, black on a light one."""
        r, g, b = rgb[:3]
        return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"

    def _draw_beam(self, ax, Hc, Wc, color):
        if self._beam is None:
            return
        from ..utils import add_beam
        ps = self._pixscale
        bmaj = self._beam[0] / ps if (ps and ps != 1.0) else self._beam[0]
        bmin = self._beam[1] / ps if (ps and ps != 1.0) else self._beam[1]
        off = max(bmaj * 0.75, min(Hc, Wc) * 0.06)
        add_beam(ax, bmin_pix=bmin, bmaj_pix=bmaj, bpa_deg=self._beam[2],
                 xy_offset=(off, off), color=color)

    def _draw_scalebar(self, ax, Hc, Wc, col):
        if self._kpc_per_pix:
            steps, unit, per = (0.5, 1, 2, 5, 10, 20, 50, 100, 200), "kpc", self._kpc_per_pix
        elif self._pixscale:
            steps, unit, per = (1, 2, 5, 10, 20, 30, 60, 120), '"', self._pixscale
        else:
            return
        bar = steps[0] / per
        for v in steps:
            bar = v / per
            if Wc * 0.12 <= bar <= Wc * 0.35:
                break
        x0, y0 = Wc * 0.64, Hc * 0.06
        ax.plot([x0, x0 + bar], [y0, y0], color=col, lw=1.8)
        txt = f"{v} {unit}" if unit == "kpc" else f'{v}"'
        ax.text(x0 + bar / 2, y0 + Hc * 0.04, txt, color=col, ha="center",
                va="bottom", fontsize=7)

    def _build_channel_card(self, parent):
        outer = tk.Frame(parent, bg=C.DIM, padx=1, pady=1)
        inner = tk.Frame(outer, bg=C.CARD_BG, padx=6, pady=5)
        inner.pack(fill=tk.BOTH, expand=True)

        trow = tk.Frame(inner, bg=C.CARD_BG); trow.pack(anchor="w", pady=(0, 3))
        pre = "Channel Velocity " if self._has_vel else "Channel "
        suf = " km s⁻¹" if self._has_vel else ""
        tk.Label(trow, text=pre, bg=C.CARD_BG, fg=_accent(),
                 font=("Helvetica", 8, "bold")).pack(side=tk.LEFT)
        self._chan_vel_lbl = tk.Label(trow, text="", bg=C.CARD_BG, fg=_accent(),
                                      font=("Helvetica", 8, "italic"))
        self._chan_vel_lbl.pack(side=tk.LEFT)
        tk.Label(trow, text=suf, bg=C.CARD_BG, fg=_accent(),
                 font=("Helvetica", 8, "bold")).pack(side=tk.LEFT)

        crow = tk.Frame(inner, bg=C.CARD_BG); crow.pack(fill=tk.X)
        box = tk.Frame(crow, bg=_accent(), padx=1, pady=1)
        box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._slider_box = box
        srow = tk.Frame(box, bg=C.CARD_BG, padx=4, pady=3); srow.pack(fill=tk.BOTH, expand=True)
        self._slider = tk.Scale(
            srow, from_=0, to=1, orient=tk.HORIZONTAL,
            command=lambda _v: self._draw(),
            bg=_accent(), fg=_accent(), troughcolor=C.LOG_BG,
            activebackground=C.ACCENT_HOVER, highlightthickness=0,
            sliderrelief=tk.FLAT, bd=0, width=12, showvalue=False)
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Frame(srow, bg=_accent(), width=1).pack(side=tk.LEFT, fill=tk.Y, padx=(3, 0))
        self._ch_disp = tk.Canvas(srow, bg=C.LOG_BG, highlightthickness=0,
                                  width=58, height=20)
        self._ch_disp.pack(side=tk.RIGHT, padx=(3, 0))

        self._build_transport(crow)
        return outer

    def _update_ch_disp(self, ch):
        cv = self._ch_disp
        cv.delete("all")
        w, h = 58, 20
        cv.create_text(w // 2 - 6, h // 2, text=str(ch + 1),
                       fill=_accent(), font=("Courier", 8, "bold"), anchor="e")
        cv.create_text(w // 2 - 4, h // 2, text=f"/{self._cube.shape[0]}",
                       fill=_accent(), font=("Courier", 8), anchor="w")
        if hasattr(self, "_chan_vel_lbl"):
            self._chan_vel_lbl.configure(
                text=(f"{self._sp_axis[ch]:.1f}" if self._has_vel else str(ch + 1)))

    def _build_checkbox_panel(self, body, sources):
        cb_panel = tk.Frame(body, bg=C.BG, width=170)
        cb_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=8)
        cb_panel.pack_propagate(False)
        tk.Label(cb_panel, text="Spectra", bg=C.BG, fg=C.ACCENT,
                 font=("Helvetica", 10, "bold")).pack(pady=(2, 6), anchor="w")

        row = tk.Frame(cb_panel, bg=C.BG); row.pack(fill=tk.X, pady=1, anchor="w")
        tk.Label(row, text="—", bg=C.BG, fg=C.STEP_LABEL_TXT,
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(row, text="Total (always on)",
                       variable=self._show_total, state=tk.DISABLED,
                       bg=C.BG, fg=C.STEP_LABEL_TXT, selectcolor=C.CARD_BG,
                       disabledforeground=C.STEP_LABEL_TXT,
                       activebackground=C.BG, font=("Helvetica", 8),
                       relief=tk.FLAT, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        row = tk.Frame(cb_panel, bg=C.BG); row.pack(fill=tk.X, pady=1, anchor="w")
        tk.Label(row, text="■", bg=C.BG, fg="#8b4513",
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(row, text="Diffuse",
                       variable=self._show_diff, command=self._draw,
                       bg=C.BG, fg=C.STEP_LABEL_TXT, selectcolor=C.CARD_BG,
                       activebackground=C.BG, activeforeground=C.ACCENT,
                       font=("Helvetica", 8), relief=tk.FLAT,
                       anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        for s in sources:
            sid = s["id"]
            col = self._src_color[sid]
            hexc = "#{:02x}{:02x}{:02x}".format(
                int(col[0]*255), int(col[1]*255), int(col[2]*255))
            var = tk.BooleanVar(value=True)
            self._show_src[sid] = var
            row = tk.Frame(cb_panel, bg=C.BG); row.pack(fill=tk.X, pady=1, anchor="w")
            tk.Label(row, text="■", bg=C.BG, fg=hexc,
                     font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
            tk.Checkbutton(row, text=f"Source {sid}",
                           variable=var, command=self._draw,
                           bg=C.BG, fg=C.STEP_LABEL_TXT, selectcolor=C.CARD_BG,
                           activebackground=C.BG, activeforeground=C.ACCENT,
                           font=("Helvetica", 8), relief=tk.FLAT,
                           anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _iter_visible(self):
        """Yield (label, sid, mask, rgb) for each source currently shown."""
        if self._hier:
            _root_of, base_of = self._tree.coloring()
            order_ids = [h.id for h, _d in self._tree.order]
            for hid in order_ids:
                if hid not in base_of:
                    continue
                rgb = _shade(base_of[hid], self._tree.alpha.get(hid, 1.0))
                yield self._tree.name.get(hid, str(hid)), hid, self._units[hid], rgb
        else:
            for s in self._sources:
                sid = s["id"]
                if not self._show_src[sid].get():
                    continue
                yield f"Source {sid}", sid, self._units[sid], self._src_color[sid][:3]

    def _draw(self):
        t = _mpl_theme()
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from mpl_toolkits.axes_grid1.axes_size import Fixed
        self._fig.clear()
        # Big channel viewer (left, full height) | Mom0 Mom1 (top) / spectrum (bottom).
        gs = self._fig.add_gridspec(2, 3, width_ratios=[2.8, 1, 1],
                                    height_ratios=[1, 1.18],   # taller spectrum
                                    hspace=0.13, wspace=0.14,
                                    left=0.03, right=0.95, top=0.92, bottom=0.09)
        ax_ch = self._fig.add_subplot(gs[:, 0])
        ax_m0 = self._fig.add_subplot(gs[0, 1])
        ax_m1 = self._fig.add_subplot(gs[0, 2])
        ax_sp = self._fig.add_subplot(gs[1, 1:])
        for ax in (ax_ch, ax_m0, ax_m1):
            ax.set_facecolor(t["fig_bg"])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_anchor("N")          # anchor images to the top → top colorbars align
            for sp in ax.spines.values():
                sp.set_edgecolor(t["spine"]); sp.set_linewidth(0.5)
        ax_sp.set_facecolor(t["fig_bg"])
        for sp in ax_sp.spines.values():
            sp.set_edgecolor(t["spine"])
        ax_sp.grid(True, color=t["spine"], alpha=0.22, linewidth=0.4)
        ax_sp.set_axisbelow(True)

        CBAR = Fixed(0.10)                 # identical colorbar thickness everywhere
        PAD  = Fixed(0.05)

        def _top_cbar(ax, im, label):
            cax = make_axes_locatable(ax).append_axes(
                "top", size=CBAR, pad=PAD, axes_class=plt.matplotlib.axes.Axes)
            cb = self._fig.colorbar(im, cax=cax, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
            cb.ax.tick_params(colors=t["fg"], labelsize=7, direction="out")
            cb.outline.set_edgecolor(t["spine"])
            cb.set_label(label, fontsize=8, color=t["fg"], labelpad=4)

        def _panel_label(ax, text):
            ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left",
                    fontsize=9, color=t["fg"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=t["bbox_bg"],
                              alpha=0.5, ec="none"))

        vis_items = list(self._iter_visible())
        vis_union = np.zeros((self._H, self._W), dtype=bool)
        for _label, _sid, m, _rgb in vis_items:
            vis_union |= m
        if not vis_union.any():
            vis_union = self._all_union

        img_cmap = self._inv_cmap(self._img_cmap.get(), self._img_invert.get())
        m1_cmap  = self._inv_cmap(self._m1_cmap.get(), self._m1_invert.get())
        # beam/scalebar colour contrasts the colourmap's low end (white on dark)
        flux_oc = self._contrast_color(plt.get_cmap(img_cmap)(0.12))
        _hx = t["m1_bg"].lstrip("#")
        m1_oc = self._contrast_color(tuple(int(_hx[i:i+2], 16) / 255 for i in (0, 2, 4)))

        # ── big channel viewer (current channel + per-channel source contours) ─
        idx = min(int(self._slider.get()), len(self._chan_list) - 1)
        ch  = self._chan_list[idx]
        self._update_ch_disp(ch)
        chan_img = self._cube[ch]
        if self._norm_mode.get().startswith("Per-source"):
            cv0, cv1 = float(self._vmin_var.get()), float(self._vmax_var.get())
        else:
            finite = chan_img[np.isfinite(chan_img)]
            if finite.size:
                cv0, cv1 = float(np.nanmin(finite)), float(np.nanmax(finite))
            else:
                cv0, cv1 = 0.0, 1.0
        if cv1 <= cv0:
            cv1 = cv0 + 1e-9
        from matplotlib.colors import Normalize, LogNorm, PowerNorm
        sm = self._scale_mode.get()
        if sm == "Log":
            norm = LogNorm(vmin=max(cv0, 1e-12), vmax=max(cv1, max(cv0, 1e-12) + 1e-12))
        elif sm == "Power":
            norm = PowerNorm(gamma=float(self._gamma_var.get()),
                             vmin=max(cv0, 0.0), vmax=cv1)
        else:
            norm = Normalize(vmin=cv0, vmax=cv1)
        im_ch = ax_ch.imshow(chan_img, cmap=img_cmap, norm=norm, origin="lower")
        from matplotlib.patches import Rectangle as _Rect

        # Bounding boxes: one per source *tree*, on the coarsest (largest-scale)
        # currently-visible source that has a footprint at this channel.
        bbox_owner: dict = {}   # root → (sid, scale, rgb, label)
        for label, sid, _m, rgb in vis_items:
            mch = self._masks_by_ch.get(sid, {}).get(ch)
            if mch is None or not mch.any():
                continue
            if self._hier:
                root = self._tree.root_of.get(sid, sid)
                scale = self._tree.by_id[sid].scale
            else:
                root, scale = sid, 0
            cur = bbox_owner.get(root)
            if cur is None or scale > cur[1]:
                bbox_owner[root] = (sid, scale, rgb, label)
        bbox_sids = {v[0] for v in bbox_owner.values()}

        for label, sid, _m, rgb in vis_items:
            mch = self._masks_by_ch.get(sid, {}).get(ch)
            self._mask_edge(ax_ch, mch, rgb, 1.0)
            if sid in bbox_sids and mch is not None and mch.any():
                rows, cols = np.where(mch)
                r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
                PAD_BB = 4                          # match the GIF / square preview
                rgb3 = (rgb[0], rgb[1], rgb[2])
                _bx, _by, _bw, _bh, _lx, _ly = clamped_bbox(
                    r0, r1, c0, c1, PAD_BB, chan_img.shape)
                ax_ch.add_patch(_Rect(
                    (_bx, _by), _bw, _bh, linewidth=0.9,
                    edgecolor=(rgb3[0], rgb3[1], rgb3[2], 0.9),
                    facecolor="none", zorder=4))
                ax_ch.text(_lx, _ly,
                           label, ha="center",
                           va="center", fontsize=6, color="black",
                           fontweight="bold", zorder=6,
                           bbox=dict(boxstyle="round,pad=0.2",
                                     fc=rgb3, ec=rgb3, lw=1.0))

        self._draw_beam(ax_ch, self._H, self._W, flux_oc)
        self._draw_scalebar(ax_ch, self._H, self._W, flux_oc)
        _panel_label(ax_ch, f"Channel {ch + 1}")
        _top_cbar(ax_ch, im_ch, r"Jy beam$^{-1}$")

        # ── Moment 0 ──────────────────────────────────────────────────────────
        v0, v1 = np.nanpercentile(self._mom0, [1, 99])
        im0 = ax_m0.imshow(self._mom0, cmap=img_cmap, origin="lower",
                           vmin=v0, vmax=v1)
        rgba = np.zeros((self._H, self._W, 4), dtype=np.float32)
        rgba[~vis_union] = list(t["dim"])
        for _label, _sid, m, rgb in vis_items:
            if m.any():
                rgba[m] = [rgb[0], rgb[1], rgb[2], 0.45]
        ax_m0.imshow(rgba, origin="lower", interpolation="nearest")
        self._draw_beam(ax_m0, self._H, self._W, flux_oc)
        _panel_label(ax_m0, "Moment 0")
        _top_cbar(ax_m0, im0, r"Jy beam$^{-1}$")

        # ── Moment 1 ──────────────────────────────────────────────────────────
        ax_m1.set_facecolor(t["m1_bg"])
        m1_show = np.where(vis_union, self._mom1, np.nan)
        _v0, _v1 = _mom1_limits(m1_show)
        im1 = ax_m1.imshow(m1_show, cmap=m1_cmap, origin="lower",
                           vmin=_v0, vmax=_v1)
        self._draw_beam(ax_m1, self._H, self._W, m1_oc)
        _panel_label(ax_m1, "Moment 1")
        _top_cbar(ax_m1, im1, self._m1_unit)

        # ── spectrum (with the dynamic channel marker) ────────────────────────
        xs = self._sp_axis
        ax_sp.plot(xs, self._total_spec, color=t["total"], lw=1, ls="--", label="Total")
        if getattr(self, "_show_diff", None) is None or self._show_diff.get():
            ax_sp.plot(xs, self._diffuse_spec, color="#cd853f", lw=1, ls=":",
                       label="Diffuse")
        for label, sid, m, rgb in vis_items:
            ax_sp.plot(xs, self._spec[sid], color=rgb, lw=1.2, label=label)
        ax_sp.axvline(float(self._sp_axis[ch]), color=t["fg"], ls="--",
                      lw=1.0, alpha=0.8)
        ax_sp.set_xlabel(self._sp_label, color=t["fg"], fontsize=9)
        # y axis (ticks + label) on the right
        ax_sp.yaxis.set_label_position("right")
        ax_sp.yaxis.tick_right()
        ax_sp.set_ylabel(r"Integrated flux (Jy beam$^{-1}$)", color=t["fg"], fontsize=9)
        # tick marks on all four sides; labels only bottom (x) and right (y)
        ax_sp.tick_params(colors=t["fg"], labelsize=8, which="both", direction="in",
                          top=True, bottom=True, left=True, right=True,
                          labeltop=False, labelleft=False,
                          labelbottom=True, labelright=True)
        ax_sp.margins(x=0.02)
        leg = ax_sp.legend(fontsize=7, facecolor=t["legend_bg"],
                           edgecolor=t["spine"], labelcolor=t["fg"])
        for txt in leg.get_texts():
            txt.set_color(t["fg"])

        self._canvas.draw()


def _component_palette(n):
    """*n* visually distinct RGB colours for per-track curves/contours."""
    cmap = plt.get_cmap("tab10")
    light = C._current_theme == "light"
    cols = []
    for i in range(max(n, 1)):
        rgb = cmap(i % 10)[:3]
        if light:
            rgb = tuple(min(c * 0.6, 1.0) for c in rgb)
        cols.append(rgb)
    return cols


class IndividualAnalysisWindow(TransportControls, tk.Toplevel):
    PAD = 8

    def __init__(self, master, cube, tracks, sources, vel_array=None,
                 hierarchical_sources=None, tracks_per_scale=None,
                 beam=None, pixscale=None, coarse_scale=None):
        super().__init__(master)
        self.title("Individual Source Analysis")
        self.configure(bg=C.BG)
        self.geometry("900x820")
        self.minsize(1200, 760)
        self.resizable(True, True)

        self._cube      = cube
        self._tracks    = tracks
        self._sources   = sources
        self._beam      = beam
        self._pixscale  = pixscale
        self._tracks_per_scale = tracks_per_scale or {}
        self._tracks_by_id = {t["id"]: t for t in tracks}
        self._src_color = _source_colors(sources)
        self._vel_array = (np.asarray(vel_array, dtype=np.float64)
                           if vel_array is not None else None)
        self._has_vel   = self._vel_array is not None
        self._sp_axis   = self._vel_array if self._has_vel \
                          else np.arange(cube.shape[0], dtype=np.float64)
        self._sp_label  = r"Velocity (km s$^{-1}$)" if self._has_vel else "Channel"
        self._m1_unit   = r"km s$^{-1}$" if self._has_vel else "channel"
        self._hier = bool(hierarchical_sources and tracks_per_scale)

        # per-source render cache (filled by _on_source_change)
        self._components: list[dict] = []
        self._track_active: dict[int, tk.BooleanVar] = {}
        self._chan_list: list[int] = []
        self._show_tracks = tk.BooleanVar(value=False)

        # plot-parameter state
        light = C._current_theme == "light"
        self._img_cmap   = tk.StringVar(value="cubehelix" if light else "inferno")
        self._img_invert = tk.BooleanVar(value=light)   # light → reversed (dark-on-light)
        self._m1_cmap    = tk.StringVar(value="RdBu_r")
        self._m1_invert  = tk.BooleanVar(value=False)
        self._norm_mode  = tk.StringVar(value="Per-source cube")
        # whole-cube nanmin/nanmax for per-source channel normalisation
        _vmin, _vmax = float(np.nanmin(cube)), float(np.nanmax(cube))
        if not (np.isfinite(_vmin) and np.isfinite(_vmax)) or _vmax <= _vmin:
            _vmin, _vmax = 0.0, 1.0
        self._cube_vmin, self._cube_vmax = _vmin, _vmax

        # playback transport state (play/loop/FPS)
        self._init_transport_state()

        body = tk.Frame(self, bg=C.BG)
        body.pack(fill=tk.BOTH, expand=True)

        # ── LEFT column: figure + channel slider ─────────────────────────────
        left_col = tk.Frame(body, bg=C.BG)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fig_frame = tk.Frame(left_col, bg=C.BG)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._fig = plt.Figure(figsize=(7.0, 7.0), dpi=96,
                               facecolor=_mpl_theme()["fig_bg"])
        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._build_controls_row(left_col)

        # ── RIGHT column: source chooser + track controls ────────────────────
        if self._hier:
            self._full_mom0 = cube.sum(axis=0)
            right = tk.Frame(body, bg=C.BG)
            right.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=8)
            self._tree = SourceTreePanel(right, hierarchical_sources,
                                         tracks_per_scale,
                                         on_change=self._on_source_change,
                                         mode="single", title="Choose source",
                                         coarse_scale=coarse_scale)
            self._tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            right.configure(width=max(self._tree.tree_width + 18, 190))
            right.pack_propagate(False)

            # accent-bordered moment-0 thumbnail showing the source's position
            border = tk.Frame(right, bg=C.ACCENT)
            border.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
            inner = tk.Frame(border, bg=C.BG)
            inner.pack(fill=tk.BOTH, padx=2, pady=2)
            tk.Label(inner, text="Position", bg=C.BG, fg=C.STEP_LABEL_TXT,
                     font=("Helvetica", 7)).pack(anchor="w")
            self._thumb_fig = plt.Figure(figsize=(1.6, 1.6), dpi=96,
                                         facecolor=_mpl_theme()["fig_bg"])
            self._thumb_canvas = FigureCanvasTkAgg(self._thumb_fig, master=inner)
            thumb_w = self._thumb_canvas.get_tk_widget()
            thumb_w.pack()                    # fixed size (not fill) so it stays square

            # Keep the position plot a perfect square that tracks the sidebar
            # width: on resize, set the canvas height equal to its width.
            def _square_thumb(event):
                s = max(int(event.width) - 4, 1)
                if thumb_w.winfo_reqwidth() != s:
                    thumb_w.configure(width=s, height=s)
            inner.bind("<Configure>", _square_thumb)

            self._build_track_panel(right)
            self._on_source_change()
        else:
            self._selected = tk.IntVar(value=sources[0]["id"] if sources else 0)
            right = tk.Frame(body, bg=C.BG, width=190)
            right.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=8)
            right.pack_propagate(False)
            tk.Label(right, text="Choose source", bg=C.BG, fg=C.ACCENT,
                     font=("Helvetica", 10, "bold")).pack(pady=(2, 6), anchor="w")
            for s in sources:
                sid = s["id"]
                col = self._src_color[sid]
                hexc = "#{:02x}{:02x}{:02x}".format(
                    int(col[0]*255), int(col[1]*255), int(col[2]*255))
                row = tk.Frame(right, bg=C.BG); row.pack(fill=tk.X, pady=1, anchor="w")
                tk.Label(row, text="■", bg=C.BG, fg=hexc,
                         font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
                tk.Radiobutton(row, text=f"Source {sid}",
                               variable=self._selected, value=sid,
                               command=self._on_source_change,
                               bg=C.BG, fg=C.STEP_LABEL_TXT, selectcolor=C.CARD_BG,
                               activebackground=C.BG, activeforeground=C.ACCENT,
                               font=("Helvetica", 9), relief=tk.FLAT,
                               anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._build_track_panel(right)
            if sources:
                self._on_source_change()

        self.protocol("WM_DELETE_WINDOW",
                      lambda: (self._anim_stop(), self.destroy()))
        set_titlebar_appearance(self)

    def _build_track_panel(self, parent):
        """Tracks header + select/deselect + vertical pill list + checkbox."""
        tk.Frame(parent, bg=C.DIM, height=1).pack(fill=tk.X, pady=(10, 4))
        hdr = tk.Frame(parent, bg=C.BG); hdr.pack(fill=tk.X, pady=(0, 4))
        tk.Label(hdr, text="Tracks", bg=C.BG, fg=_accent(),
                 font=("Helvetica", 8, "bold")).pack(side=tk.LEFT)

        def _mini_btn(text, cmd):
            b = tk.Label(hdr, text=text, bg=C.CARD_BG, fg=_accent(),
                         font=("Helvetica", 7, "bold"), padx=4, pady=1,
                         cursor="pointinghand")
            b.bind("<Button-1>", lambda _e: cmd())
            return b
        _mini_btn("Deselect all", lambda: self._set_all_tracks(False)).pack(
            side=tk.RIGHT, padx=(3, 0))
        _mini_btn("Select all", lambda: self._set_all_tracks(True)).pack(
            side=tk.RIGHT)

        self._build_track_pills(parent)
        tk.Checkbutton(parent, text="Show track footprint / spectra",
                       variable=self._show_tracks, command=self._render,
                       bg=C.BG, fg=C.STEP_LABEL_TXT, selectcolor=C.CARD_BG,
                       activebackground=C.BG, activeforeground=_accent(),
                       font=("Helvetica", 8), relief=tk.FLAT, anchor="w",
                       justify="left", wraplength=170).pack(anchor="w", pady=(4, 2))

    # ─────────────────────────── widgets ────────────────────────────────────
    SYMMETRIC_CMAPS = ["RdBu_r", "seismic", "coolwarm", "bwr",
                       "PuOr_r", "PiYG_r", "Spectral_r"]

    @staticmethod
    def _inv_cmap(name, invert):
        if not invert:
            return name
        return name[:-2] if name.endswith("_r") else name + "_r"

    def _card(self, parent, title, title_fg=None,
              title_font=("Helvetica", 8, "bold"), pad=(6, 5)):
        """DIM-bordered card with a title; returns (outer, inner, title_label)."""
        outer = tk.Frame(parent, bg=C.DIM, padx=1, pady=1)
        inner = tk.Frame(outer, bg=C.CARD_BG, padx=pad[0], pady=pad[1])
        inner.pack(fill=tk.BOTH, expand=True)
        lbl = tk.Label(inner, text=title, bg=C.CARD_BG, fg=title_fg or _accent(),
                       font=title_font)
        lbl.pack(anchor="w", pady=(0, 3))
        return outer, inner, lbl

    def _build_controls_row(self, parent):
        """Channel card (slider + transport) + three tight plot-parameter cards."""
        row = tk.Frame(parent, bg=C.BG)
        row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 6))

        grey, tfont, tpad = C.STEP_LABEL_TXT, ("Helvetica", 7), (5, 3)

        ch_out = self._build_channel_card(row)
        ch_out.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        f_out, f_in, _ = self._card(row, "Flux colour map", grey, tfont, tpad)
        v_out, v_in, _ = self._card(row, "Velocity colour map", grey, tfont, tpad)
        n_out, n_in, _ = self._card(row, "Channel normalisation", grey, tfont, tpad)
        # fill=Y → all cards stretch to the (tallest) channel card's height
        f_out.pack(side=tk.LEFT, fill=tk.Y, padx=4)
        v_out.pack(side=tk.LEFT, fill=tk.Y, padx=4)
        n_out.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0))

        def _chk(p, var):
            return tk.Checkbutton(
                p, text="Inv", variable=var, command=self._render,
                bg=C.CARD_BG, fg=C.STEP_LABEL_TXT, selectcolor=C.LOG_BG,
                activebackground=C.CARD_BG, activeforeground=_accent(),
                font=("Helvetica", 8), relief=tk.FLAT)

        fr = tk.Frame(f_in, bg=C.CARD_BG); fr.pack(fill=tk.X)
        self._styled_optionmenu(fr, self._img_cmap, list(C._CMAPS), width=8).pack(side=tk.LEFT)
        _chk(fr, self._img_invert).pack(side=tk.LEFT, padx=(1, 0))

        vr = tk.Frame(v_in, bg=C.CARD_BG); vr.pack(fill=tk.X)
        self._styled_optionmenu(vr, self._m1_cmap, self.SYMMETRIC_CMAPS, width=8).pack(side=tk.LEFT)
        _chk(vr, self._m1_invert).pack(side=tk.LEFT, padx=(1, 0))

        self._styled_optionmenu(n_in, self._norm_mode,
                                ["Per-source cube", "Per-channel"], width=14).pack(anchor="w")

    def _build_channel_card(self, parent):
        """Channel card: dynamic velocity title + (slider, play, loop, FPS) row."""
        outer = tk.Frame(parent, bg=C.DIM, padx=1, pady=1)
        inner = tk.Frame(outer, bg=C.CARD_BG, padx=6, pady=5)
        inner.pack(fill=tk.BOTH, expand=True)

        # dynamic title with the current channel velocity italicised
        trow = tk.Frame(inner, bg=C.CARD_BG); trow.pack(anchor="w", pady=(0, 3))
        pre = "Channel Velocity " if self._has_vel else "Channel "
        suf = " km s⁻¹" if self._has_vel else ""
        tk.Label(trow, text=pre, bg=C.CARD_BG, fg=_accent(),
                 font=("Helvetica", 8, "bold")).pack(side=tk.LEFT)
        self._chan_vel_lbl = tk.Label(trow, text="", bg=C.CARD_BG, fg=_accent(),
                                      font=("Helvetica", 8, "italic"))
        self._chan_vel_lbl.pack(side=tk.LEFT)
        tk.Label(trow, text=suf, bg=C.CARD_BG, fg=_accent(),
                 font=("Helvetica", 8, "bold")).pack(side=tk.LEFT)

        crow = tk.Frame(inner, bg=C.CARD_BG); crow.pack(fill=tk.X)

        box = tk.Frame(crow, bg=_accent(), padx=1, pady=1)
        box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._slider_box = box
        srow = tk.Frame(box, bg=C.CARD_BG, padx=4, pady=3); srow.pack(fill=tk.BOTH, expand=True)
        self._slider = tk.Scale(
            srow, from_=0, to=1, orient=tk.HORIZONTAL,
            command=lambda _v: self._render(),
            bg=_accent(), fg=_accent(), troughcolor=C.LOG_BG,
            activebackground=C.ACCENT_HOVER, highlightthickness=0,
            sliderrelief=tk.FLAT, bd=0, width=11, showvalue=False)
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Frame(srow, bg=_accent(), width=1).pack(side=tk.LEFT, fill=tk.Y, padx=(3, 0))
        self._ch_disp = tk.Canvas(srow, bg=C.LOG_BG, highlightthickness=0,
                                  width=58, height=20)
        self._ch_disp.pack(side=tk.RIGHT, padx=(3, 0))

        # order: slider, play, loop, FPS  (shared transport mixin)
        self._build_transport(crow)
        return outer

    def _styled_optionmenu(self, parent, var, options, width=10):
        om = tk.OptionMenu(parent, var, *options, command=lambda _v: self._render())
        om.configure(bg=C.CARD_BG, fg=C.STEP_LABEL_TXT,
                     activebackground=C.DIM, activeforeground=C.STEP_LABEL_TXT,
                     highlightthickness=0, relief=tk.FLAT,
                     font=("Helvetica", 8), width=width, anchor="w")
        om["menu"].configure(bg=C.CARD_BG, fg=C.STEP_LABEL_TXT,
                             activebackground=C.DIM, activeforeground=C.STEP_LABEL_TXT,
                             font=("Helvetica", 8))
        return om

    def _update_ch_disp(self, ch):
        cv = self._ch_disp
        cv.delete("all")
        w, h = 58, 20
        cv.create_text(w // 2 - 6, h // 2, text=str(ch + 1),
                       fill=_accent(), font=("Courier", 8, "bold"), anchor="e")
        cv.create_text(w // 2 - 4, h // 2, text=f"/{self._cube.shape[0]}",
                       fill=_accent(), font=("Courier", 8), anchor="w")
        # dynamic card title: current channel velocity (or channel number)
        if hasattr(self, "_chan_vel_lbl"):
            val = self._sp_axis[ch]
            self._chan_vel_lbl.configure(
                text=(f"{val:.1f}" if self._has_vel else str(ch + 1)))

    def _build_track_pills(self, parent):
        from tkinter import font as tkfont
        self._tpill_font = tkfont.Font(family="Helvetica", size=9, weight="bold")
        self._tdesc_font = tkfont.Font(family="Helvetica", size=8)
        container = tk.Frame(parent, bg=C.BG)
        container.pack(fill=tk.X, pady=(2, 2))
        self._tpill_canvas = tk.Canvas(container, bg=C.BG, height=30,
                                       highlightthickness=0, bd=0)
        self._tpill_vsb = tk.Scrollbar(container, orient="vertical",
                                       command=self._tpill_canvas.yview)
        self._tpill_canvas.configure(yscrollcommand=self._tpill_vsb.set)
        self._tpill_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tpill_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._tpill_canvas.bind("<Button-1>", self._on_track_pill_click)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._tpill_canvas.bind(seq, self._on_tpill_scroll)
        self._tpill_nodes = []

    def _on_tpill_scroll(self, event):
        cv = self._tpill_canvas
        if getattr(event, "num", None) == 4:
            cv.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            cv.yview_scroll(1, "units")
        else:
            cv.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _track_description(self, comp) -> str:
        """Caption describing a track's velocity span and split/merge event."""
        chans = sorted(comp["masks_by_ch"].keys())
        if not chans:
            return ""
        v0, v1 = self._sp_axis[chans[0]], self._sp_axis[chans[-1]]
        if self._has_vel:
            fmt = lambda v: f"{v:.1f} km s⁻¹"
        else:
            fmt = lambda v: f"channel {int(round(v))}"
        rel = comp.get("rel")
        plabel = comp.get("parent_label", "the source")
        if rel == "split_merge":
            return (f"Split from {plabel} at {fmt(v0)} and merged again "
                    f"with {plabel} at {fmt(v1)}")
        if rel == "split":
            return f"Split from {plabel} at {fmt(v0)} and observed to {fmt(v1)}"
        if rel == "merge":
            return f"Observed from {fmt(v0)} until merging into {plabel} at {fmt(v1)}"
        return f"Observed from {fmt(v0)} to {fmt(v1)}"

    def _refresh_track_pills(self):
        """Hierarchical pill tree: each track as an indented pill tied to its
        parent by connectors, with a caption (velocity span / split / merge)
        wrapped underneath."""
        cv = self._tpill_canvas
        cv.delete("all")
        self._tpill_nodes = []
        PILL_H, INDENT, SPINE_X, X0, GAP, DESC_PAD, V_OFF = 22, 18, 6, 16, 8, 2, 5
        cv_w = cv.winfo_width()
        if cv_w <= 1:
            cv_w = 184                       # window not mapped yet at first draw

        # Top-down layout: place each pill, then its wrapped caption below it.
        line_col = C.DIM_TXT                  # faint connectors
        desc_col = C.STEP_LABEL_TXT           # darker caption text
        cy_by_tid, x1_by_tid = {}, {}
        desc_ids = []
        y = 4
        for i, comp in enumerate(self._components):
            depth = comp.get("depth", 0)
            x1 = X0 + depth * INDENT
            tw = self._tpill_font.measure(comp["label"])
            x2 = x1 + tw + 18
            cy = y + PILL_H // 2
            self._tpill_nodes.append((i, x1, cy - PILL_H // 2, x2,
                                      cy + PILL_H // 2, cy, comp["label"]))
            cy_by_tid[comp["tid"]] = cy
            x1_by_tid[comp["tid"]] = x1

            desc = self._track_description(comp)
            tid_txt = cv.create_text(
                x1, cy + PILL_H // 2 + DESC_PAD, text=desc, anchor="nw",
                font=self._tdesc_font, fill=desc_col,
                width=max(cv_w - x1 - 6, 70))
            desc_ids.append(tid_txt)
            bb = cv.bbox(tid_txt)
            desc_h = (bb[3] - bb[1]) if bb else 12
            y = cy + PILL_H // 2 + DESC_PAD + desc_h + GAP
        # Bound the visible height; scroll through the rest so the checkbox
        # below the pill list always stays on screen.
        MAX_H = 260
        content_h = max(int(y), 30)
        cv.configure(height=min(content_h, MAX_H),
                     scrollregion=(0, 0, cv_w, content_h))

        # left spine linking the root (depth-0) pills
        roots = [n for n in self._tpill_nodes
                 if self._components[n[0]].get("depth", 0) == 0]
        if len(roots) >= 2:
            cv.create_line(SPINE_X, roots[0][5], SPINE_X, roots[-1][5],
                           fill=line_col, width=1)
        for i, x1, y1, x2, y2, cy, label in roots:
            cv.create_line(SPINE_X, cy, x1, cy, fill=line_col, width=1)

        # parent → child connectors: the vertical spine sits just left of the
        # PARENT pill (never through it), with a horizontal stub into the child.
        for i, x1, y1, x2, y2, cy, label in self._tpill_nodes:
            comp = self._components[i]
            ptid = comp.get("parent_tid")
            if comp.get("depth", 0) == 0 or ptid not in cy_by_tid:
                continue
            p_cy = cy_by_tid[ptid]
            vx = x1_by_tid[ptid] - V_OFF
            cv.create_line(vx, p_cy, vx, cy, fill=line_col, width=1)
            cv.create_line(vx, cy, x1, cy, fill=line_col, width=1)

        # pills
        for i, x1, y1, x2, y2, cyy, label in self._tpill_nodes:
            if self._track_active[i].get():
                r, g, b = self._components[i]["color"]
                fill = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
                outline = fill
                lum = 0.299*r + 0.587*g + 0.114*b
                txt_col = "#000000" if lum > 0.55 else "#ffffff"
            else:
                fill, outline, txt_col = C.CARD_BG, C.DIM_TXT, C.DIM_TXT
            cv.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=1)
            cv.create_text((x1 + x2) // 2, cyy, text=label,
                           font=self._tpill_font, fill=txt_col)

        for tid_txt in desc_ids:             # keep captions above the connectors
            cv.tag_raise(tid_txt)

    def _set_all_tracks(self, on):
        """Activate/deactivate every track (the source pill is always active)."""
        for i, comp in enumerate(self._components):
            if comp.get("is_source"):
                continue
            var = self._track_active.get(i)
            if var is not None:
                var.set(on)
        self._refresh_track_pills()
        self._render()

    def _on_track_pill_click(self, event):
        cx = self._tpill_canvas.canvasx(event.x)   # account for scroll offset
        cy_e = self._tpill_canvas.canvasy(event.y)
        for i, x1, y1, x2, y2, cy, label in self._tpill_nodes:
            if x1 <= cx <= x2 and y1 <= cy_e <= y2:
                if self._components[i].get("is_source"):
                    return            # the source is always shown
                var = self._track_active[i]
                var.set(not var.get())
                self._refresh_track_pills()
                self._render()
                return

    # ───────────────────────── data assembly ────────────────────────────────
    def _collect_components(self):
        """Per-track components of the current source: label, colour, masks, spectrum.

        Components are the individual tracks of the selected source *at its own
        scale* — never the sub-sources of finer scales nested below it.
        """
        cube = self._cube
        nchan = cube.shape[0]
        comps = []

        # Resolve the tracks belonging to the selected source, all at one scale.
        if self._hier:
            sid = self._tree.selected_id()
            h = self._tree.by_id.get(sid)
            if h is None:
                return []
            tps = self._tracks_per_scale.get(h.scale, [])
            tracks_by_id = {t["id"]: t for t in tps}
            track_ids = list(h.track_ids)
        else:
            sid = int(self._selected.get())
            src = next((s for s in self._sources if s["id"] == sid), None)
            if src is None:
                return []
            tracks_by_id = self._tracks_by_id
            track_ids = list(src["track_ids"])

        present = {tid: tracks_by_id[tid] for tid in track_ids if tracks_by_id.get(tid)}
        if not present:
            return []
        ids_in = set(present)
        src_rgb = (self._tree.stable_color.get(sid, (0.7, 0.7, 0.7)) if self._hier
                   else self._src_color.get(sid, (0.7, 0.7, 0.7))[:3])

        # Per-track spectrum + total flux (flux picks the dominant "source" track).
        spec_of, flux_of = {}, {}
        for tid, trk in present.items():
            spec = np.zeros(nchan, dtype=float)
            for ch, m in trk["masks"].items():
                spec[ch] = float(cube[ch][m].sum())
            spec_of[tid] = spec
            flux_of[tid] = float(spec.sum())

        # Parent of a track within the source: the track it split *from*, else
        # the track it merges *into* (both make it a sub-track, not a source).
        # ``rel`` records which relationship linked it ("split"/"merge"/None).
        def _parent(tid):
            trk = present[tid]
            sf = trk.get("split_from")
            sf = sf if (sf is not None and sf in ids_in and sf != tid) else None
            merge_owners = [owner for (_ch, owner) in (trk.get("merge_into") or [])
                            if owner in ids_in and owner != tid]
            if sf is not None and sf in merge_owners:   # split off then merged back
                return sf, "split_merge"
            if sf is not None:
                return sf, "split"
            if merge_owners:
                return merge_owners[0], "merge"
            return None, None

        parent_of, rel_of = {}, {}
        for tid in present:
            parent_of[tid], rel_of[tid] = _parent(tid)
        # The source is the single dominant root; any other root is a track that
        # belongs to the source, so re-parent it onto the primary.
        roots = [tid for tid, p in parent_of.items() if p is None]
        primary = max(roots or list(present), key=lambda t: flux_of[t])
        parent_of[primary] = None
        rel_of[primary] = None
        for tid in present:
            if tid != primary and parent_of[tid] is None:
                parent_of[tid] = primary
                rel_of[tid] = None   # grouped into source, no split/merge link

        children_of: dict = {tid: [] for tid in present}
        for tid, p in parent_of.items():
            if p is not None:
                children_of[p].append(tid)

        ordered: list = []   # (tid, depth, parent_tid)
        seen: set = set()

        def _walk(tid, depth, parent_tid):
            if tid in seen:
                return
            seen.add(tid)
            ordered.append((tid, depth, parent_tid))
            for c in sorted(children_of[tid], key=lambda t: -flux_of[t]):
                _walk(c, depth + 1, tid)
        _walk(primary, 0, None)
        for tid in present:          # any cycle leftovers → directly under source
            if tid not in seen:
                seen.add(tid)
                ordered.append((tid, 1, primary))

        # Labels: every generation is a Greek letter, numbered sequentially
        # within it — α = the dominant (source) track, β = its tracks, γ = …
        GREEK = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ"]
        gen_k: dict = {}
        for tid, depth, parent_tid in ordered:
            g = GREEK[depth] if depth < len(GREEK) else f"L{depth}"
            k = gen_k.get(depth, 0)
            gen_k[depth] = k + 1
            lbl = f"Track {g}{k + 1}"
            comps.append({
                "label": lbl,
                "tid": tid,
                "depth": depth,
                "parent_tid": parent_tid,
                "rel": rel_of.get(tid),
                "is_source": depth == 0,
                "masks_by_ch": dict(present[tid]["masks"]),
                "spec": spec_of[tid],
            })

        # Colours: the source keeps its original colour; tracks get new colours.
        pal = _component_palette(max(len(comps) - 1, 1))
        pi = 0
        for comp in comps:
            if comp["is_source"]:
                comp["color"] = tuple(src_rgb[:3])
            else:
                comp["color"] = pal[pi]
                pi += 1
        # parent label (e.g. "Track α1") for each track's caption
        label_by_tid = {c["tid"]: c["label"] for c in comps}
        for comp in comps:
            comp["parent_label"] = label_by_tid.get(comp["parent_tid"], "")
        return comps

    def _on_source_change(self):
        """Recompute everything that depends on the selected source."""
        if getattr(self, "_anim_playing", False):
            self._anim_stop()
        self._components = self._collect_components()
        # (re)build per-track activation state, all on by default
        self._track_active = {i: tk.BooleanVar(value=True)
                              for i in range(len(self._components))}
        self._refresh_track_pills()

        H, W = self._cube.shape[1], self._cube.shape[2]
        union = np.zeros((H, W), dtype=bool)
        ch_to_mask: dict[int, np.ndarray] = {}
        for comp in self._components:
            for ch, m in comp["masks_by_ch"].items():
                union |= m
                ch_to_mask[ch] = ch_to_mask.get(ch, np.zeros((H, W), dtype=bool)) | m
        self._union = union
        self._ch_to_mask = ch_to_mask

        # source-level colour + name/title
        if self._hier:
            sid = self._tree.selected_id()
            self._src_level_color = self._tree.stable_color.get(sid, (0.7, 0.7, 0.7))
            self._source_name = self._tree.name.get(sid, str(sid))
            self._title_name = self._source_name
            self._draw_thumb(union, self._src_level_color)
        else:
            sid = int(self._selected.get())
            self._src_level_color = self._src_color.get(sid, (0.7, 0.7, 0.7))[:3]
            self._source_name = str(sid)
            self._title_name = f"Source {sid}"

        if not union.any():
            self._chan_list = []
            self._render()
            return

        det_chs = sorted(ch_to_mask.keys())
        rows_nz = np.where(union.any(axis=1))[0]
        cols_nz = np.where(union.any(axis=0))[0]
        cy = (int(rows_nz[0]) + int(rows_nz[-1])) // 2
        cx = (int(cols_nz[0]) + int(cols_nz[-1])) // 2
        half = max(int(rows_nz[-1]) - int(rows_nz[0]),
                   int(cols_nz[-1]) - int(cols_nz[0])) // 2 + self.PAD
        y0 = max(0, cy - half);  y1 = min(H, cy + half + 1)
        x0 = max(0, cx - half);  x1 = min(W, cx + half + 1)
        self._bbox = (y0, y1, x0, x1)
        self._center = (cx, cy)

        det_idx = np.array(det_chs)
        footprint = union[y0:y1, x0:x1]
        flux_crop = self._cube[det_idx][:, y0:y1, x0:x1]
        total_flux = flux_crop.sum(axis=0)
        ch_axis = self._sp_axis[det_idx].astype(np.float64)
        self._mom0_crop = flux_crop.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mom1 = (flux_crop * ch_axis[:, None, None]).sum(axis=0) / total_flux
        valid = _mom1_valid(total_flux, self._cube, len(det_idx)) & footprint
        self._mom1_crop = _clip_to_axis(np.where(valid, mom1, np.nan), ch_axis)
        self._footprint = footprint
        self._det_chs = det_chs

        # source integrated spectrum (each pixel once via per-channel union)
        nchan = self._cube.shape[0]
        spec = np.zeros(nchan, dtype=float)
        for ch, m in ch_to_mask.items():
            spec[ch] = float(self._cube[ch][m].sum())
        self._src_spec = spec

        self._chan_list = det_chs
        mid = len(det_chs) // 2
        self._slider.configure(to=max(len(det_chs) - 1, 0))
        self._slider.set(mid)        # triggers _render via command
        self._render()

    def _draw_thumb(self, union, color):
        """Full-field moment-0 thumbnail with the selected union mask overlaid."""
        if not hasattr(self, "_thumb_fig"):
            return
        t = _mpl_theme()
        f = self._thumb_fig
        f.clear()
        f.set_facecolor(t["fig_bg"])
        ax = f.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        v0, v1 = np.nanpercentile(self._full_mom0, [1, 99])
        ax.imshow(self._full_mom0, cmap=t["cmap"], origin="lower",
                  vmin=v0, vmax=v1)
        if union is not None and union.any():
            r, g, b = color[:3]
            ov = np.zeros(union.shape + (4,), dtype=np.float32)
            ov[union] = [r, g, b, 0.5]
            ax.imshow(ov, origin="lower", interpolation="nearest")
            ax.contour(union.astype(float), [0.5], colors=[(r, g, b)], linewidths=1.0)
        self._thumb_canvas.draw()

    @staticmethod
    def _mask_edge(ax, mask, color, lw, k=10):
        """Pixel-accurate outline of a binary mask.

        ``ax.contour`` linearly interpolates the 0.5 isoline between pixel
        centres, so it cuts across the corners of a blocky mask and the square
        edge pixels appear to poke outside it.  Outlining an up-sampled copy
        makes the line hug the pixel grid (corner cuts shrink to ~1/k px).
        """
        if mask is None or not mask.any():
            return
        H, W = mask.shape
        up = np.repeat(np.repeat(mask.astype(float), k, axis=0), k, axis=1)
        xs = (np.arange(W * k) + 0.5) / k - 0.5
        ys = (np.arange(H * k) + 0.5) / k - 0.5
        ax.contour(xs, ys, up, [0.5], colors=[color], linewidths=lw)

    def _draw_beam(self, ax, Hc, Wc):
        """Draw the synthesized-beam ellipse in the lower-left of an image panel."""
        if self._beam is None:
            return
        from ..utils import add_beam
        ps = self._pixscale
        bmaj = self._beam[0] / ps if (ps and ps != 1.0) else self._beam[0]
        bmin = self._beam[1] / ps if (ps and ps != 1.0) else self._beam[1]
        off = max(bmaj * 0.75, min(Hc, Wc) * 0.08)
        light = C._current_theme == "light"
        add_beam(ax, bmin_pix=bmin, bmaj_pix=bmaj, bpa_deg=self._beam[2],
                 xy_offset=(off, off), color="black" if light else "white")

    # ───────────────────────────── rendering ────────────────────────────────
    def _active_tracks(self):
        """Active non-source tracks (the overlays added on top of the source)."""
        return [c for i, c in enumerate(self._components)
                if not c.get("is_source")
                and self._track_active.get(i) and self._track_active[i].get()]

    def _render(self):
        t = _mpl_theme()
        if not self._chan_list or not self._union.any():
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.set_facecolor(t["fig_bg"]); ax.set_axis_off()
            ax.text(0.5, 0.5, "(no footprint)", color=t["fg"],
                    ha="center", va="center", transform=ax.transAxes)
            self._canvas.draw()
            return

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        idx = min(int(self._slider.get()), len(self._chan_list) - 1)
        ch = self._chan_list[idx]
        self._update_ch_disp(ch)

        y0, y1, x0, x1 = self._bbox
        cx, cy = self._center
        footprint = self._footprint
        show_tracks = self._show_tracks.get()
        active = self._active_tracks()
        img_cmap = self._inv_cmap(self._img_cmap.get(), self._img_invert.get())
        m1_cmap  = self._inv_cmap(self._m1_cmap.get(), self._m1_invert.get())
        per_source = self._norm_mode.get().startswith("Per-source")

        self._fig.clear()
        gs = self._fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                                    hspace=0.10, wspace=0.14,
                                    left=0.11, right=0.89, top=0.95, bottom=0.09)
        ax_ch = self._fig.add_subplot(gs[0, 0])
        ax_m0 = self._fig.add_subplot(gs[0, 1])
        ax_sp = self._fig.add_subplot(gs[1, 0])
        ax_m1 = self._fig.add_subplot(gs[1, 1])
        # All four panels are equal squares (box_aspect=1 also makes the
        # locatable colorbars exactly the subplot height).
        for ax in (ax_ch, ax_m0, ax_sp, ax_m1):
            ax.set_facecolor(t["fig_bg"])
            ax.set_box_aspect(1)
            for sp in ax.spines.values():
                sp.set_edgecolor(t["spine"]); sp.set_linewidth(0.5)
        for ax in (ax_ch, ax_m0, ax_m1):     # images: spines only, no ticks/labels
            ax.set_xticks([]); ax.set_yticks([])

        def _label(ax, text):
            ax.text(0.04, 0.96, text, transform=ax.transAxes,
                    va="top", ha="left", fontsize=9, color=t["fg"],
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=t["bbox_bg"],
                              alpha=0.5, ec="none"))

        def _overlay_footprints(ax, per_ch=None):
            """Always draw the source contour (original colour); when tracks are
            enabled, add each active track's contour in its own new colour."""
            def _crop(masks_by_ch):
                if per_ch is not None:
                    m = masks_by_ch.get(per_ch)
                    return m[y0:y1, x0:x1] if m is not None else None
                mc = np.zeros((y1 - y0, x1 - x0), dtype=bool)
                for mm in masks_by_ch.values():
                    mc |= mm[y0:y1, x0:x1]
                return mc

            # source — original colour, always shown
            if per_ch is not None:
                mm = self._ch_to_mask.get(per_ch)
                src_mc = mm[y0:y1, x0:x1] if mm is not None else None
            else:
                src_mc = footprint
            self._mask_edge(ax, src_mc, self._src_level_color, 1.2)

            # tracks — new colours, added on top
            if show_tracks:
                for comp in active:
                    self._mask_edge(ax, _crop(comp["masks_by_ch"]),
                                    comp["color"], 1.2)

        # ── [0,0] single channel (slider-driven), colorbar LEFT ──────────────
        # vmin/vmax: whole-cube nanmin/nanmax in per-source mode, or this
        # slice's nanmin/nanmax in per-channel mode.
        ch_crop = self._cube[ch][y0:y1, x0:x1]
        if per_source:
            v0, v1 = self._cube_vmin, self._cube_vmax
        else:
            finite = ch_crop[np.isfinite(ch_crop)]
            if finite.size:
                v0, v1 = float(np.nanmin(finite)), float(np.nanmax(finite))
                if v1 <= v0:
                    v1 = v0 + 1e-9
            else:
                v0, v1 = 0.0, 1.0
        im_ch = ax_ch.imshow(ch_crop, cmap=img_cmap, origin="lower",
                             vmin=v0, vmax=v1, aspect="auto")
        _overlay_footprints(ax_ch, per_ch=ch)
        self._draw_beam(ax_ch, *footprint.shape)
        _label(ax_ch, f"Channel {ch + 1}")
        cax_ch = make_axes_locatable(ax_ch).append_axes(
            "left", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb_ch = self._fig.colorbar(im_ch, cax=cax_ch)
        cax_ch.yaxis.set_ticks_position("left")
        cax_ch.yaxis.set_label_position("left")
        cb_ch.ax.tick_params(colors=t["fg"], labelsize=7, direction="out")
        cb_ch.outline.set_edgecolor(t["spine"])
        cb_ch.set_label(r"Jy beam$^{-1}$", fontsize=8, color=t["fg"], labelpad=4)

        # ── [0,1] moment 0, colorbar RIGHT ───────────────────────────────────
        ov0 = np.zeros(footprint.shape + (4,), dtype=np.float32)
        ov0[~footprint] = list(t["dim"])
        im0 = ax_m0.imshow(self._mom0_crop, cmap=img_cmap, origin="lower",
                           aspect="auto")
        ax_m0.imshow(ov0, origin="lower", interpolation="nearest", aspect="auto")
        _overlay_footprints(ax_m0)
        self._draw_beam(ax_m0, *footprint.shape)
        _label(ax_m0, "Moment 0")
        cax0 = make_axes_locatable(ax_m0).append_axes(
            "right", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb0 = self._fig.colorbar(im0, cax=cax0)
        cb0.ax.tick_params(colors=t["fg"], labelsize=7, direction="out")
        cb0.outline.set_edgecolor(t["spine"])
        cb0.set_label(r"Jy beam$^{-1}$", fontsize=8, color=t["fg"], labelpad=4)

        # ── [1,1] moment 1, colorbar RIGHT ───────────────────────────────────
        ax_m1.set_facecolor(t["m1_bg"])
        m1 = self._mom1_crop
        _v0, _v1 = _mom1_limits(m1)
        im1 = ax_m1.imshow(m1, cmap=m1_cmap, origin="lower",
                           vmin=_v0, vmax=_v1, aspect="auto")
        _overlay_footprints(ax_m1)
        self._draw_beam(ax_m1, *footprint.shape)
        _label(ax_m1, "Moment 1")
        cax1 = make_axes_locatable(ax_m1).append_axes(
            "right", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb1 = self._fig.colorbar(im1, cax=cax1)
        cb1.ax.tick_params(colors=t["fg"], labelsize=7, direction="out")
        cb1.outline.set_edgecolor(t["spine"])
        cb1.set_label(self._m1_unit, fontsize=8, color=t["fg"], labelpad=4)

        # ── [1,0] spectrum ───────────────────────────────────────────────────
        ax_sp.set_facecolor(t["fig_bg"])
        for sp in ax_sp.spines.values():
            sp.set_edgecolor(t["spine"])
        ax_sp.grid(True, color=t["spine"], alpha=0.22, linewidth=0.4)
        ax_sp.set_axisbelow(True)            # grid behind the spectrum curves
        if show_tracks:
            # individual track spectra (each active track); the dominant/source
            # track keeps the source's original colour.
            for i, comp in enumerate(self._components):
                if self._track_active.get(i) and self._track_active[i].get():
                    ax_sp.plot(self._sp_axis, comp["spec"], color=comp["color"],
                               lw=1.6 if comp.get("is_source") else 1.4,
                               label=comp["label"])
        else:
            # whole source, in its original colour
            ax_sp.plot(self._sp_axis, self._src_spec,
                       color=self._src_level_color, lw=1.6,
                       label=f"Source {self._source_name}")
        det_chs = self._det_chs
        if det_chs:
            v0_det = float(self._sp_axis[det_chs[0]])
            v1_det = float(self._sp_axis[det_chs[-1]])
            ax_sp.axvspan(min(v0_det, v1_det), max(v0_det, v1_det),
                          color=self._src_level_color, alpha=0.10)
        # vertical marker tracking the current channel
        ax_sp.axvline(float(self._sp_axis[ch]),
                      color=t["fg"], ls="--", lw=1.0, alpha=0.8)
        ax_sp.set_xlabel(self._sp_label, color=t["fg"], fontsize=9)
        ax_sp.set_ylabel(r"Integrated flux (Jy beam$^{-1}$)",
                         color=t["fg"], fontsize=8)
        ax_sp.tick_params(colors=t["fg"], labelsize=8, direction="in",
                          top=True, bottom=True, left=True, right=True)
        ax_sp.margins(x=0.02)
        leg = ax_sp.legend(fontsize=7, facecolor=t["legend_bg"],
                           edgecolor=t["spine"], labelcolor=t["fg"])
        if leg:
            for txt in leg.get_texts():
                txt.set_color(t["fg"])

        self._fig.suptitle(f"{self._title_name}  ·  pixel ({cx}, {cy})",
                           color=t["fg"], fontsize=10)
        self._canvas.draw()
