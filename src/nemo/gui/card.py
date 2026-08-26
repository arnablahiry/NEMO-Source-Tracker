import io
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

from . import _constants as C
from ._constants import (CARD_W, CARD_H, BTN_W, BTN_H, BTN_ZONE_H, BTN_TALL)
from .widgets import _FlatBtn, _QueueStream, make_slider_box
from .dialogs import WaveletParamsDialog, FalseDetParamsDialog
from .viewers import SliceViewer, ScaleViewer
from .analysis import CombinedAnalysisWindow, IndividualAnalysisWindow, _source_colors
from ..utils import clamped_bbox
from .loaders import load_cube_file, _moment0, _apply_scaling


# ---------------------------------------------------------------------------
# GIF frame builders — pure functions (safe to call from worker thread)
# ---------------------------------------------------------------------------

def _frame_to_pil(fig, dpi, hires_factor: int = 3):
    """Render *fig* at *dpi* × hires_factor then downsample to (CARD_W, CARD_H).

    Rendering at higher DPI then LANCZOS-downsampling gives crisper antialiasing
    on lines, text and contours when finally displayed at the card size.
    """
    from PIL import Image as PilImage
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi * hires_factor,
                bbox_inches="tight", pad_inches=0, facecolor="#0a0a14")
    plt.close(fig)
    buf.seek(0)
    return PilImage.open(buf).copy().resize(
        (CARD_W, CARD_H), PilImage.LANCZOS).convert("RGB")


def _subsample_for_stats(cube: np.ndarray, max_elems: int = 4_000_000):
    """Strided view (no copy) capped at ~max_elems voxels for fast statistics.

    Whole-cube min/max/percentile reductions stall the UI on large cubes; a
    strided subsample gives visually identical display stats for free.
    """
    if cube.size <= max_elems or cube.ndim != 3:
        return cube
    step = int(np.ceil(np.sqrt(cube.size / max_elems)))
    return cube[:, ::step, ::step]


def _cube_norm(cube: np.ndarray):
    sample = _subsample_for_stats(cube)
    vmin = float(np.nanmin(sample));  vmax = float(np.nanmax(sample))
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return Normalize(vmin=vmin, vmax=vmax)


def _wavelet_renderer(cube: np.ndarray, detections: list,
                      beam=None, pixscale=None):
    """Return (channels, render_fn) where render_fn(ch) -> PIL.Image for one frame.

    All shared/expensive setup happens once here; render_fn does only the
    per-channel matplotlib work, so it can be called lazily on the main thread.
    """
    norm = _cube_norm(cube)
    dpi  = 72;  fsz = CARD_W / dpi
    light = C._current_theme == "light"
    cmap = "cubehelix_r" if light else "inferno"
    contour_color = "black" if light else "white"
    H, W = cube.shape[1], cube.shape[2]
    det_by_ch = {d.channel: d for d in detections}
    channels  = sorted(det_by_ch)

    def render(ch):
        d = det_by_ch[ch]
        fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
        ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
        ax.imshow(cube[ch], cmap=cmap, norm=norm, origin="lower")
        for mask in d.footprint_masks:
            ax.contour(mask.astype(float), [0.5],
                       colors=[contour_color], linewidths=0.6, alpha=0.85)
        _draw_beam(ax, H, W, beam, pixscale, light)
        return _frame_to_pil(fig, dpi)

    return channels, render


def _build_wavelet_frames(cube: np.ndarray, detections: list,
                          beam=None, pixscale=None) -> dict:
    channels, render = _wavelet_renderer(cube, detections, beam, pixscale)
    return {ch: render(ch) for ch in channels}


def _multi_scale_renderer(cube: np.ndarray, multi_scale_dets: list,
                          beam=None, pixscale=None):
    """(channels, render_fn) for multi-scale detections with opacity-graded contours."""
    norm = _cube_norm(cube)
    dpi  = 72;  fsz = CARD_W / dpi
    light = C._current_theme == "light"
    cmap  = "cubehelix_r" if light else "inferno"
    cv    = 0.0 if light else 1.0  # black / white contour base
    H, W = cube.shape[1], cube.shape[2]

    all_scales = sorted({s for sd in multi_scale_dets for s in sd.scales.keys()})
    n = len(all_scales)

    def _alpha(scale_idx):
        rank = all_scales.index(scale_idx)
        return 0.95 if n <= 1 else 0.95 - rank * 0.70 / (n - 1)

    sd_by_ch = {sd.channel: sd for sd in multi_scale_dets}
    channels = sorted(sd_by_ch)

    def render(ch):
        sd = sd_by_ch[ch]
        fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
        ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
        ax.imshow(cube[ch], cmap=cmap, norm=norm, origin="lower")
        for scale_idx in all_scales:
            if scale_idx not in sd.scales:
                continue
            masks, _, _ = sd.scales[scale_idx]
            a = _alpha(scale_idx)
            for mask in masks:
                ax.contour(mask.astype(float), [0.5],
                           colors=[[cv, cv, cv, a]], linewidths=0.6)
        _draw_beam(ax, H, W, beam, pixscale, light)
        return _frame_to_pil(fig, dpi)

    return channels, render


def _build_multi_scale_frames(cube: np.ndarray, multi_scale_dets: list,
                              beam=None, pixscale=None) -> dict:
    channels, render = _multi_scale_renderer(cube, multi_scale_dets, beam, pixscale)
    return {ch: render(ch) for ch in channels}


def _flow_renderer(cube: np.ndarray, flow_seq: list,
                   detections: list | None = None,
                   beam=None, pixscale=None):
    """(channels, render_fn) for the optical-flow quiver + footprint overlay."""
    norm = _cube_norm(cube)
    dpi  = 72;  fsz = CARD_W / dpi
    det_by_ch = {d.channel: d for d in (detections or [])}
    light = C._current_theme == "light"
    cmap = "cubehelix_r" if light else "inferno"
    contour_color = "black" if light else "white"
    H, W = cube.shape[1], cube.shape[2]
    qs = max(min(H, W) // 28, 2)
    _qcmap = "cool" if C._current_theme == "dark" else "viridis"
    flow_by_ch = {cr: fl for cr, _ct, fl, _m in flow_seq}
    channels   = sorted(flow_by_ch)

    def render(ch):
        flow = flow_by_ch[ch]
        fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
        ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
        ax.imshow(cube[ch], cmap=cmap, norm=norm, origin="lower")
        ys = np.arange(0, H, qs);  xs = np.arange(0, W, qs)
        Xq, Yq = np.meshgrid(xs, ys)
        u = flow[1][ys[:, None], xs[None, :]].ravel()
        v = flow[0][ys[:, None], xs[None, :]].ravel()
        mag = np.hypot(u, v);  pk = float(mag.max())
        if pk > 1e-6:
            sc = qs * 0.9 / pk
            ax.quiver(Xq.ravel(), Yq.ravel(), u*sc, v*sc,
                      mag, cmap=_qcmap, angles="xy", scale_units="xy", scale=1,
                      width=0.003, headwidth=3, alpha=0.85, clim=(0, pk))
        d = det_by_ch.get(ch)
        if d:
            for mask in d.footprint_masks:
                ax.contour(mask.astype(float), [0.5],
                           colors=[contour_color], linewidths=0.5, alpha=0.4)
        _draw_beam(ax, H, W, beam, pixscale, light)
        return _frame_to_pil(fig, dpi)

    return channels, render


def _build_flow_frames(cube: np.ndarray, flow_seq: list,
                       detections: list | None = None,
                       beam=None, pixscale=None) -> dict:
    channels, render = _flow_renderer(cube, flow_seq, detections, beam, pixscale)
    return {ch: render(ch) for ch in channels}


def _draw_beam(ax, H, W, beam, pixscale, light: bool) -> None:
    """Draw only the beam ellipse."""
    if beam is None:
        return
    from ..utils import add_beam
    bmaj_pix = beam[0] / pixscale if (pixscale and pixscale != 1.0) else beam[0]
    bmin_pix = beam[1] / pixscale if (pixscale and pixscale != 1.0) else beam[1]
    corner_offset = max(bmaj_pix * 0.75, min(H, W) * 0.06)
    add_beam(ax, bmin_pix=bmin_pix, bmaj_pix=bmaj_pix, bpa_deg=beam[2],
             xy_offset=(corner_offset, corner_offset),
             color="black" if light else "white")


def _draw_annotations(ax, H, W, beam, pixscale, kpc_per_pix, light: bool) -> None:
    """Draw scalebar (in kpc if kpc_per_pix available, else arcsec) and beam."""
    from ..utils import add_beam
    bar_color = "black" if light else "white"

    if kpc_per_pix is not None:
        # kpc scalebar
        for kpc in (0.5, 1, 2, 5, 10, 20, 50, 100, 200):
            bar_px = kpc / kpc_per_pix
            if W * 0.12 <= bar_px <= W * 0.35:
                break
        x0, y0 = W * 0.68, H * 0.07
        ax.plot([x0, x0 + bar_px], [y0, y0], color=bar_color, lw=1.5)
        ax.text(x0 + bar_px / 2, y0 + H * 0.045, f'{kpc} kpc',
                color=bar_color, ha="center", va="bottom", fontsize=6)
    elif pixscale is not None:
        for arcsec in (1, 2, 5, 10, 20, 30, 60, 120):
            bar_px = arcsec / pixscale
            if W * 0.12 <= bar_px <= W * 0.35:
                break
        x0, y0 = W * 0.68, H * 0.07
        ax.plot([x0, x0 + bar_px], [y0, y0], color=bar_color, lw=1.5)
        ax.text(x0 + bar_px / 2, y0 + H * 0.045, f'{arcsec}"',
                color=bar_color, ha="center", va="bottom", fontsize=6)

    if beam is not None:
        bmaj_pix = beam[0] / pixscale if (pixscale and pixscale != 1.0) else beam[0]
        bmin_pix = beam[1] / pixscale if (pixscale and pixscale != 1.0) else beam[1]
        # Offset scales with spatial extent so the beam never overlaps the edge
        corner_offset = max(bmaj_pix * 0.75, min(H, W) * 0.06)
        add_beam(ax, bmin_pix=bmin_pix, bmaj_pix=bmaj_pix, bpa_deg=beam[2],
                 xy_offset=(corner_offset, corner_offset), color=bar_color)


def _sources_renderer(cube: np.ndarray, tracks: list, sources: list,
                      beam=None, pixscale=None):
    """(channels, render_fn) for the flat per-source contours + bboxes overlay."""
    from matplotlib.patches import Rectangle as _Rect
    norm = _cube_norm(cube)
    dpi  = 72;  fsz = CARD_W / dpi
    light = C._current_theme == "light"
    cmap = "cubehelix_r" if light else "inferno"
    H, W = cube.shape[1], cube.shape[2]
    PAD_BB = 4

    tracks_by_id = {t["id"]: t for t in tracks}
    src_color = _source_colors(sources)
    src_ch_masks: dict[int, dict[int, list]] = {}
    for s in sources:
        ch_dict: dict[int, list] = {}
        for tid in s["track_ids"]:
            t = tracks_by_id.get(tid)
            if not t:
                continue
            for ch, mask in t["masks"].items():
                ch_dict.setdefault(ch, []).append(mask)
        src_ch_masks[s["id"]] = ch_dict

    channels = sorted({ch for d in src_ch_masks.values() for ch in d})

    def render(ch):
        fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
        ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
        ax.imshow(cube[ch], cmap=cmap, norm=norm, origin="lower")

        for sid, ch_dict in src_ch_masks.items():
            masks = ch_dict.get(ch)
            if not masks:
                continue
            col  = src_color[sid]
            lcol = (min(col[0]+0.3, 1.0), min(col[1]+0.3, 1.0), min(col[2]+0.3, 1.0))
            for mask in masks:
                ax.contour(mask.astype(float), [0.5],
                           colors=[col], linewidths=0.7)
                rows, cols = np.where(mask)
                if not len(rows):
                    continue
                r0, r1 = int(rows.min()), int(rows.max())
                c0, c1 = int(cols.min()), int(cols.max())
                _bx, _by, _bw, _bh, _lx, _ly = clamped_bbox(
                    r0, r1, c0, c1, PAD_BB, mask.shape)
                ax.add_patch(_Rect(
                    (_bx, _by), _bw, _bh,
                    linewidth=0.8, edgecolor=lcol,
                    facecolor="none", zorder=4,
                ))
                ax.text(_lx, _ly, str(sid),
                        ha="center", va="center", fontsize=6,
                        color="black", fontweight="bold",
                        bbox=dict(boxstyle="circle,pad=0.2",
                                  fc=lcol, ec=lcol, lw=1.0),
                        zorder=6)
        _draw_beam(ax, H, W, beam, pixscale, light)
        return _frame_to_pil(fig, dpi)

    return channels, render


def _build_sources_frames(cube: np.ndarray, tracks: list, sources: list,
                          beam=None, pixscale=None) -> dict:
    channels, render = _sources_renderer(cube, tracks, sources, beam, pixscale)
    return {ch: render(ch) for ch in channels}


def _scale_alpha(scale, all_scales):
    """White-contour opacity: finest scale opaque, coarsest faint."""
    n = len(all_scales)
    if n <= 1:
        return 0.95
    rank = sorted(all_scales).index(scale)
    return 0.95 - rank * 0.70 / (n - 1)


def _multi_scale_flow_renderer(cube, flow_seq_per_scale, detections_per_scale,
                               coarsest_scale, beam=None, pixscale=None):
    """(channels, render_fn): per-scale footprint contours (graded opacity) for
    every scale, quiver arrows only for the *coarsest* scale."""
    norm  = _cube_norm(cube)
    dpi   = 72;  fsz = CARD_W / dpi
    light = C._current_theme == "light"
    cmap  = "cubehelix_r" if light else "inferno"
    cv    = 0.0 if light else 1.0
    _qcmap = "cool" if not light else "viridis"
    H, W  = cube.shape[1], cube.shape[2]
    qs    = max(min(H, W) // 28, 2)
    all_scales = sorted(detections_per_scale.keys())

    # per-scale {channel: detection} lookup
    det_by_scale_ch = {s: {d.channel: d for d in dets}
                       for s, dets in detections_per_scale.items()}
    # coarsest flow {ch_ref: flow}
    coarse_flow = {cr: fl for cr, _ct, fl, _m in flow_seq_per_scale.get(coarsest_scale, [])}

    channels = sorted({cr for fseq in flow_seq_per_scale.values() for cr, *_ in fseq})

    def render(ch):
        fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
        ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
        ax.imshow(cube[ch], cmap=cmap, norm=norm, origin="lower")

        # arrows for coarsest scale only
        flow = coarse_flow.get(ch)
        if flow is not None:
            ys = np.arange(0, H, qs);  xs = np.arange(0, W, qs)
            Xq, Yq = np.meshgrid(xs, ys)
            u = flow[1][ys[:, None], xs[None, :]].ravel()
            v = flow[0][ys[:, None], xs[None, :]].ravel()
            mag = np.hypot(u, v);  pk = float(mag.max())
            if pk > 1e-6:
                sc = qs * 0.9 / pk
                ax.quiver(Xq.ravel(), Yq.ravel(), u*sc, v*sc, mag, cmap=_qcmap,
                          angles="xy", scale_units="xy", scale=1,
                          width=0.003, headwidth=3, alpha=0.85, clim=(0, pk))

        # footprint contours for every scale, graded opacity
        for scale in all_scales:
            d = det_by_scale_ch.get(scale, {}).get(ch)
            if not d:
                continue
            a = _scale_alpha(scale, all_scales)
            for mask in d.footprint_masks:
                ax.contour(mask.astype(float), [0.5],
                           colors=[[cv, cv, cv, a]], linewidths=0.5)
        _draw_beam(ax, H, W, beam, pixscale, light)
        return _frame_to_pil(fig, dpi)

    return channels, render


def _build_multi_scale_flow_frames(cube, flow_seq_per_scale, detections_per_scale,
                                   coarsest_scale, beam=None, pixscale=None) -> dict:
    channels, render = _multi_scale_flow_renderer(
        cube, flow_seq_per_scale, detections_per_scale, coarsest_scale, beam, pixscale)
    return {ch: render(ch) for ch in channels}


def _hsrc_channel_masks(hierarchical_sources, tracks_per_scale):
    """Return {h_id: {channel: [mask, ...]}} for every hierarchical source."""
    tracks_by_scale_id = {}
    for scale, tracks in tracks_per_scale.items():
        for t in tracks:
            tracks_by_scale_id[(scale, t['id'])] = t
    out = {}
    for h in hierarchical_sources:
        ch_dict: dict = {}
        for tid in h.track_ids:
            t = tracks_by_scale_id.get((h.scale, tid))
            if not t:
                continue
            for ch, m in t['masks'].items():
                ch_dict.setdefault(ch, []).append(m)
        out[h.id] = ch_dict
    return out


def _hierarchical_sources_renderer(cube, hierarchical_sources, tracks_per_scale,
                                   multi_scale_dets, beam=None, pixscale=None):
    """(channels, render_fn): each tree one colour, per-scale opacity (coarse faint,
    leaves bright); bounding box only around the largest (root) contour."""
    from matplotlib.patches import Rectangle as _Rect
    from ..hierarchy import assign_tree_colors, assign_tree_names

    if not hierarchical_sources:
        return [], None
    norm  = _cube_norm(cube)
    dpi   = 72;  fsz = CARD_W / dpi
    light = C._current_theme == "light"
    cmap  = "cubehelix_r" if light else "inferno"
    H, W  = cube.shape[1], cube.shape[2]
    PAD_BB = 4

    colors   = assign_tree_colors(hierarchical_sources)
    names    = assign_tree_names(hierarchical_sources)
    h_by_id  = {h.id: h for h in hierarchical_sources}
    ch_masks = _hsrc_channel_masks(hierarchical_sources, tracks_per_scale)

    channels = sorted({ch for d in ch_masks.values() for ch in d})

    def render(ch):
        fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
        ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
        ax.imshow(cube[ch], cmap=cmap, norm=norm, origin="lower")

        for h_id, ch_dict in ch_masks.items():
            masks = ch_dict.get(ch)
            if not masks:
                continue
            (r, g, b), alpha = colors[h_id]
            is_root = h_by_id[h_id].is_root()
            for mask in masks:
                ax.contour(mask.astype(float), [0.5],
                           colors=[[r, g, b, alpha]], linewidths=0.8)
            if is_root:  # bbox only around the largest (root) contour
                union = np.zeros((H, W), dtype=bool)
                for m in masks:
                    union |= m
                rows, cols = np.where(union)
                if len(rows):
                    r0, r1 = int(rows.min()), int(rows.max())
                    c0, c1 = int(cols.min()), int(cols.max())
                    _bx, _by, _bw, _bh, _lx, _ly = clamped_bbox(
                        r0, r1, c0, c1, PAD_BB, union.shape)
                    ax.add_patch(_Rect(
                        (_bx, _by), _bw, _bh,
                        linewidth=0.9, edgecolor=(r, g, b, 0.9),
                        facecolor="none", zorder=4))
                    ax.text(_lx, _ly, names[h_id],
                            ha="center", va="center", fontsize=6,
                            color="black", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc=(r, g, b), ec=(r, g, b), lw=1.0),
                            zorder=6)
        _draw_beam(ax, H, W, beam, pixscale, light)
        return _frame_to_pil(fig, dpi)

    return channels, render


def _build_hierarchical_sources_frames(cube, hierarchical_sources, tracks_per_scale,
                                       multi_scale_dets, beam=None, pixscale=None) -> dict:
    channels, render = _hierarchical_sources_renderer(
        cube, hierarchical_sources, tracks_per_scale, multi_scale_dets, beam, pixscale)
    if render is None:
        return {}
    return {ch: render(ch) for ch in channels}


# ---------------------------------------------------------------------------
# CubeCard
# ---------------------------------------------------------------------------

class CubeCard(tk.Frame):
    def __init__(self, master, index: int, name: str, description: str,
                 app=None, on_loaded=None, **kw):
        active = index == 0
        super().__init__(master, bg=C.CARD_BG if active else C.CARD_OFF,
                         bd=0, highlightthickness=1, takefocus=False,
                         highlightbackground=C.CARD_BORDER if active else C.DIM,
                         highlightcolor=C.CARD_BORDER if active else C.DIM, **kw)
        self.index        = index
        self.name         = name
        self.description  = description
        self.enabled      = active
        self._app         = app
        self.cube_raw     = None
        self.cube         = None
        self.vel_array    = None
        self.scaling      = dict(mode="linear", gamma=0.5)
        self.filepath     = None
        self.beam         = None
        self.pixscale     = None
        self.kpc_per_pix  = None
        self.detections        = None
        self._multi_scale_dets = []
        self.flow_seq          = None
        self._wav_params   = None
        self._flow_params      = None
        self._false_det_params = None
        self._multi_scale_enabled = tk.BooleanVar(value=True)  # Default enabled
        self._on_loaded    = on_loaded
        self._gif_frames   = []
        self._gif_idx      = 0
        self._gif_job      = None
        self._log_lines: list[str] = []
        self._preview_state = "placeholder"
        self._has_figure    = False
        self._gif_frames_by_ch: dict = {}
        self._gif_tk_by_ch:     dict = {}
        self._gif_canvas             = None
        self._gif_last_ch            = None
        # Lazy frame rendering: render one channel on demand on the main thread
        # (keeps the worker free of GIL-heavy matplotlib so the UI stays smooth).
        self._gif_render             = None   # callable(ch) -> PIL.Image | None
        self._gif_channels: list     = []

        bg = C.CARD_BG if active else C.CARD_OFF

        # Horizontal body: button column on the left, preview square right
        self._body = tk.Frame(self, bg=bg)
        self._body.pack(padx=8, pady=8)

        self._btn_zone = tk.Frame(self._body, bg=bg, width=BTN_W + 12)
        self._btn_zone.pack_propagate(False)
        self._btn_zone.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        self._right_col = tk.Frame(self._body, bg=bg)
        self._right_col.pack(side=tk.LEFT)

        self._preview_frame = tk.Frame(self._right_col, bg=bg,
                                       width=CARD_W, height=CARD_H)
        self._preview_frame.pack_propagate(False)
        self._preview_frame.pack()
        self._draw_placeholder()

        self._toggle_btn = tk.Label(
            self._preview_frame, text="Show Logs",
            bg=C.BG, fg=C.LOG_TXT,
            font=("Helvetica", 7, "bold"), cursor="pointinghand",
            relief=tk.FLAT, padx=4, pady=2,
        )
        self._toggle_btn.bind("<Button-1>", lambda _e: self._toggle_view())

        self._step_label = tk.Label(self._right_col, text=name, bg=bg,
                                    fg=C.STEP_LABEL_TXT if active else C.STEP_LABEL_DIS,
                                    font=("Helvetica", 9, "bold"))
        self._step_label.pack(pady=(4, 0))

        if   index == 0: self._build_buttons_step0(bg, active)
        elif index == 1: self._build_buttons_step1(bg, active)
        elif index == 2: self._build_buttons_step2(bg, active)
        elif index == 3: self._build_buttons_step3(bg, active)
        else:            self._build_buttons_generic(bg)

    # ------------------------------------------------------------------ #
    # Button layouts
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stack_height(n: int, gap: int = 6) -> int:
        """Button height so a column of *n* buttons spans the preview square."""
        return (CARD_H - (n - 1) * gap) // n

    @staticmethod
    def _stack(buttons, gap: int = 6):
        for i, b in enumerate(buttons):
            b.pack(side=tk.TOP, pady=(0, gap if i < len(buttons) - 1 else 0))

    def _build_buttons_step0(self, bg: str, active: bool):
        # Fixed-height column spanning the preview square, so the bottom of the
        # gamma box lines up exactly with the bottom of the square.
        col = tk.Frame(self._btn_zone, bg=bg, width=BTN_W, height=CARD_H)
        col.pack_propagate(False)
        col.pack(side=tk.TOP, anchor="n")
        self._scl_col = col

        BTN_H0 = 60
        self.btn_load = _FlatBtn(col, "Load Cube", self._load_cube,
                                 bg_on=C.ACCENT, active=active, height=BTN_H0)
        self.btn_view = _FlatBtn(col, "View Slice", self._view_slice,
                                 bg_on=C.ACCENT, active=False, height=BTN_H0)
        self.btn_load.pack(side=tk.TOP, pady=(0, 6))
        self.btn_view.pack(side=tk.TOP, pady=(0, 6))
        self._build_scaling_controls(col, bg, active)

    # ------------------------------------------------------------------ #
    # Card-level cube scaling (Linear / Log / Power + gamma)
    # ------------------------------------------------------------------ #

    def _build_scaling_controls(self, parent, bg: str, active: bool):
        from tkinter import font as tkfont
        self._scl_active = active
        self._scl_bg     = bg
        self._scale_mode = tk.StringVar(value=self.scaling.get("mode", "linear"))

        GAP, PILLS_H, GAMMA_H, CAP_H = 5, 88, 24, 16

        # ── "Scaling" caption ────────────────────────────────────────────────
        cap_fr = tk.Frame(parent, bg=bg, height=CAP_H)
        cap_fr.pack_propagate(False)
        cap_fr.pack(side=tk.TOP, fill=tk.X, pady=(0, GAP))
        self._scl_caption = tk.Label(
            cap_fr, text="Scaling", bg=bg,
            fg=C.STEP_LABEL_TXT if active else C.STEP_LABEL_DIS,
            font=("Helvetica", 8, "bold"))
        self._scl_caption.pack(side=tk.LEFT, anchor="w")

        # ── pills (fixed, compact height) ────────────────────────────────────
        self._scl_font = tkfont.Font(family="Helvetica", size=9, weight="bold")
        cv = tk.Canvas(parent, bg=bg, width=BTN_W, height=PILLS_H,
                       highlightthickness=0, bd=0,
                       cursor="pointinghand" if active else "arrow")
        cv.pack(side=tk.TOP, fill=tk.X)
        self._scl_canvas = cv
        self._scl_nodes = []
        cv.bind("<Button-1>", self._on_scale_pill)
        cv.bind("<Configure>", lambda _e: self._draw_scaling_pills())

        # ── gamma box (shorter than a pill), GAP below the pills ─────────────
        gcard = tk.Frame(parent, bg=C.DIM, padx=1, pady=1, height=GAMMA_H)
        gcard.pack_propagate(False)
        gcard.pack(side=tk.TOP, fill=tk.X, pady=(GAP, 0))
        box = tk.Frame(gcard, bg=C.ACCENT, padx=1, pady=1)      # accent border
        box.pack(fill=tk.BOTH, expand=True)
        self._gamma_border = box
        row = tk.Frame(box, bg=C.CARD_BG, padx=4, pady=1)
        row.pack(fill=tk.BOTH, expand=True)
        self._gamma_cap = tk.Label(row, text="γ", bg=C.CARD_BG, fg=C.ACCENT,
                                   font=("Georgia", 11, "italic"))
        self._gamma_cap.pack(side=tk.LEFT, padx=(1, 4))
        self._gamma_sl = tk.Scale(row, from_=0.1, to=2.0, resolution=0.05,
                                  orient=tk.HORIZONTAL, command=self._on_card_gamma,
                                  bg=C.ACCENT, fg=C.ACCENT, troughcolor=C.LOG_BG,
                                  activebackground=C.ACCENT_HOVER,
                                  highlightthickness=0, sliderrelief=tk.FLAT,
                                  bd=0, showvalue=False, width=8)
        self._gamma_sl.set(float(self.scaling.get("gamma", 0.5)))
        self._gamma_sl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._gamma_divider = tk.Frame(row, bg=C.ACCENT, width=1)
        self._gamma_divider.pack(side=tk.LEFT, fill=tk.Y, padx=(3, 0))   # divider
        self._gamma_var = tk.StringVar(
            value=f"{float(self.scaling.get('gamma', 0.5)):.2f}")
        self._gamma_entry = tk.Entry(row, textvariable=self._gamma_var, width=4,
                                     justify="right", bg=C.CARD_BG,
                                     readonlybackground=C.CARD_BG, fg=C.ACCENT,
                                     relief=tk.FLAT, highlightthickness=0,
                                     font=("Courier", 7), bd=0, state="readonly")
        self._gamma_entry.pack(side=tk.RIGHT, padx=(2, 1))

        self._set_card_gamma_enabled(self._scale_mode.get() == "power")

    def _draw_scaling_pills(self):
        cv = self._scl_canvas
        cv.delete("all")
        H = cv.winfo_height()
        W = cv.winfo_width()
        if H <= 1 or W <= 1:
            return
        items = [("linear", "Linear Scale"),
                 ("log",    "Log Scale"),
                 ("power",  "Power Scale")]
        n, GAP = len(items), 5
        pill_h = (H - GAP * (n - 1)) // n
        sel    = self._scale_mode.get()
        active = getattr(self, "_scl_active", True)
        light  = C._current_theme == "light"
        self._scl_nodes = []
        y = 0
        for mode, label in items:
            y1, y2 = y, y + pill_h
            cy = (y1 + y2) // 2
            self._scl_nodes.append((mode, 0, y1, W - 1, y2, cy, label))
            if mode == sel and active:
                fill = outline = C.ACCENT
                txt_col = "#ffffff" if light else "#000000"
            elif mode == sel:
                fill = outline = C.DIM
                txt_col = C.STEP_LABEL_DIS
            else:
                fill, outline = C.CARD_BG, C.DIM_TXT
                txt_col = C.DIM_TXT if active else C.STEP_LABEL_DIS
            # clamp bottom edge so the last pill's lower border isn't clipped
            cv.create_rectangle(0, y1, W - 1, min(y2, H - 1),
                                fill=fill, outline=outline, width=1)
            cv.create_text(W // 2, cy, text=label,
                           font=self._scl_font, fill=txt_col)
            y += pill_h + GAP

    def _on_scale_pill(self, event):
        if not getattr(self, "_scl_active", True):
            return
        for mode, x1, y1, x2, y2, cy, label in self._scl_nodes:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self._scale_mode.set(mode)
                self._draw_scaling_pills()
                self._set_card_gamma_enabled(mode == "power")
                self._apply_card_scaling()
                return

    def _set_card_gamma_enabled(self, on):
        on = bool(on) and getattr(self, "_scl_active", True)
        if on:
            self._gamma_sl.configure(state="normal", fg=C.ACCENT, bg=C.ACCENT,
                                     troughcolor=C.LOG_BG,
                                     activebackground=C.ACCENT_HOVER)
            self._gamma_border.configure(bg=C.ACCENT)
            self._gamma_divider.configure(bg=C.ACCENT)
            self._gamma_cap.configure(fg=C.ACCENT)
            self._gamma_entry.configure(fg=C.ACCENT)
        else:
            self._gamma_sl.configure(state="disabled", fg=C.DIM, bg=C.DIM,
                                     troughcolor=C.CARD_BG, activebackground=C.DIM)
            self._gamma_border.configure(bg=C.DIM)
            self._gamma_divider.configure(bg=C.DIM)
            self._gamma_cap.configure(fg=C.DIM_TXT)
            self._gamma_entry.configure(fg=C.DIM_TXT)

    def _on_card_gamma(self, _v=None):
        self._gamma_var.set(f"{float(self._gamma_sl.get()):.2f}")
        self._apply_card_scaling()

    def _apply_card_scaling(self):
        self.scaling = dict(mode=self._scale_mode.get(),
                            gamma=float(self._gamma_sl.get()))
        if self.cube_raw is not None:
            self.cube = _apply_scaling(self.cube_raw, self.scaling)
            self._render_moment0()

    def _refresh_scaling_theme(self):
        """Recolor the canvas-drawn pills + gamma box after a theme switch."""
        if not hasattr(self, "_scl_canvas"):
            return
        bg = C.CARD_BG if self._scl_active else C.CARD_OFF
        self._scl_canvas.configure(bg=bg)
        self._scl_caption.configure(
            bg=bg, fg=C.STEP_LABEL_TXT if self._scl_active else C.STEP_LABEL_DIS)
        self._draw_scaling_pills()
        self._set_card_gamma_enabled(self._scale_mode.get() == "power")

    def _build_buttons_step1(self, bg: str, active: bool):
        h = self._stack_height(3)
        self.btn_configure = _FlatBtn(self._btn_zone, "Configure\nDetection",
                                      self._open_configure, bg_on=C.ACCENT, active=active,
                                      height=h)
        self.btn_decompose = self.btn_configure
        self.btn_run = _FlatBtn(self._btn_zone, "Run Source ID",
                                self._run_sourceid, bg_on=C.RUN_COLOR, active=active,
                                font=("Arial", 10, "bold"), height=h,
                                special_color="yellow")
        self.btn_det_view = _FlatBtn(self._btn_zone, "View Detections",
                                     self._view_detections, bg_on=C.ACCENT, active=False,
                                     height=h)
        self._stack((self.btn_configure, self.btn_run, self.btn_det_view))

    def _build_buttons_step2(self, bg: str, active: bool):
        # Fixed-height column spanning the preview square, so the bottom of the
        # View Flow button lines up with the bottom of the square.
        col = tk.Frame(self._btn_zone, bg=bg, width=BTN_W, height=CARD_H)
        col.pack_propagate(False)
        col.pack(side=tk.TOP, anchor="n")
        self._flow_col = col
        self._build_flow_controls(col, bg, active)

    # ------------------------------------------------------------------ #
    # Card-level flow / false-detection parameters (inline sliders)
    # ------------------------------------------------------------------ #

    def _build_flow_controls(self, parent, bg: str, active: bool):
        self._flow_active = active
        self._flow_bg     = bg
        self._flow_params = dict(min_match_overlap=5, max_gap_channels=5)
        self._flow_param_sliders: list = []   # registry for enable/disable
        self._flow_caption_lbls: list  = []   # labels that dim when disabled

        GAP, VIEW_H = 6, 34

        # ── Optical Flow Parameters: heading + description + slider per param ──
        of_specs = [
            ("flow", "min_match_overlap", "Min match overlap (px)",
             "Pixels a track's mask must share with a detection to link them "
             "across channels — higher is stricter.", 1, 100, 1, "int"),
            ("flow", "max_gap_channels", "Max channel gap bridged",
             "Channels a source may vanish for before its track is ended.",
             1, 50, 1, "int"),
        ]
        # Param box hugs its content; View Flow button fills the space below it.
        self._build_group_box(parent, "Optical Flow Parameters", of_specs,
                              bg, active)
        self.btn_flow_view = _FlatBtn(parent, "View Flow Per Channel",
                                      self._view_flow, bg_on=C.ACCENT, active=False,
                                      height=VIEW_H, font=("Arial", 9, "bold"))
        self.btn_flow_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(GAP, 0))

        self._set_flow_params_enabled(active)

    def _build_group_box(self, parent, title: str, specs: list, bg: str,
                         active: bool):
        """Titled box: each param shows its heading, a description line, then
        its slider (no per-heading cards)."""
        SL_H = 24
        fg = C.STEP_LABEL_TXT if active else C.STEP_LABEL_DIS
        outer = tk.Frame(parent, bg=C.DIM, padx=1, pady=1)
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        inner = tk.Frame(outer, bg=bg)
        inner.pack(fill=tk.BOTH, expand=True)
        title_lbl = tk.Label(inner, text=title, bg=bg, fg=fg,
                             font=("Helvetica", 8, "bold"))
        title_lbl.pack(side=tk.TOP, anchor="w", padx=5, pady=(4, 6))
        self._flow_caption_lbls.append(title_lbl)
        for group, key, name, desc, lo, hi, res, kind in specs:
            nlbl = tk.Label(inner, text=name, bg=bg, fg=fg,
                            font=("Helvetica", 8, "bold"), anchor="w")
            nlbl.pack(side=tk.TOP, anchor="w", padx=6, pady=(2, 0))
            self._flow_caption_lbls.append(nlbl)
            dlbl = tk.Label(inner, text=desc, bg=bg, fg=fg,
                            font=("Helvetica", 7), justify="left",
                            wraplength=BTN_W - 16, anchor="w")
            dlbl.pack(side=tk.TOP, anchor="w", padx=6, pady=(0, 3))
            self._flow_caption_lbls.append(dlbl)
            self._build_param_slider(inner, group, key, lo, hi, res, kind,
                                     active=active, height=SL_H, pad=4)

    def _build_param_slider(self, parent, group: str, key: str,
                            lo, hi, res, kind: str, active: bool,
                            height: int, pad: int):
        """Accent-bordered slider box with a divider and an editable value."""
        cur = (self._flow_params if group == "flow"
               else self._false_det_params)[key]

        def _on(v, g=group, k=key):
            (self._flow_params if g == "flow" else self._false_det_params)[k] = v

        parts = make_slider_box(parent, lo, hi, res, kind, cur,
                                on_change=_on, height=height, divider_padx=(7, 0))
        parts["outer"].pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, pad))
        self._flow_param_sliders.append(dict(
            group=group, key=key, kind=kind, default=cur,
            border=parts["box"], scale=parts["scale"], entry=parts["entry"],
            divider=parts["divider"], set=parts["set"]))

    def _open_false_det_params(self):
        if not getattr(self, "_flow_active", False):
            return
        FalseDetParamsDialog(self, on_save=self._on_fd_params_saved,
                             current=self._false_det_params)

    def _on_fd_params_saved(self, params: dict):
        self._false_det_params = params

    # ── enable / reset ───────────────────────────────────────────────────────

    def _set_flow_params_enabled(self, on: bool):
        self._flow_active = bool(on)
        for lbl in getattr(self, "_flow_caption_lbls", []):
            lbl.configure(fg=C.STEP_LABEL_TXT if on else C.STEP_LABEL_DIS)
        for w in getattr(self, "_flow_param_sliders", []):
            if on:
                w["scale"].configure(state="normal", fg=C.ACCENT, bg=C.ACCENT,
                                     troughcolor=C.LOG_BG,
                                     activebackground=C.ACCENT_HOVER)
                w["border"].configure(bg=C.ACCENT)
                w["divider"].configure(bg=C.ACCENT)
                w["entry"].configure(state="normal", fg=C.ACCENT)
            else:
                w["scale"].configure(state="disabled", fg=C.DIM, bg=C.DIM,
                                     troughcolor=C.CARD_BG, activebackground=C.DIM)
                w["border"].configure(bg=C.DIM)
                w["divider"].configure(bg=C.DIM)
                w["entry"].configure(state="disabled")
        btn = getattr(self, "btn_fd_params", None)
        if btn is not None:
            btn.enable() if on else btn.disable()

    def _reset_flow_controls(self):
        """Restore optical-flow sliders to defaults."""
        self._flow_params = dict(min_match_overlap=5, max_gap_channels=5)
        for w in getattr(self, "_flow_param_sliders", []):
            w["set"](w["default"])

    def _build_buttons_step3(self, bg: str, active: bool):
        # false-detection parameters now live on this (source-classification) card
        self._fd_defaults = dict(wav_abrupt_thresh=0.5, flow_iou_thresh=0.25,
                                 short_det_max=8)
        self._false_det_params = dict(self._fd_defaults)

        h = self._stack_height(4)
        self.btn_fd_params = _FlatBtn(self._btn_zone, "False Detection\nParameters",
                                      self._open_false_det_params, bg_on=C.ACCENT,
                                      active=False, height=h, font=("Arial", 9, "bold"))
        self.btn_view_sources = _FlatBtn(self._btn_zone, "View Sources",
                                         self._view_sources_per_channel,
                                         bg_on=C.RUN_COLOR, active=False,
                                         font=("Arial", 10, "bold"),
                                         height=h,
                                         special_color="yellow")
        self.btn_individual = _FlatBtn(self._btn_zone, "Individual\nAnalysis",
                                       self._individual_analysis,
                                       bg_on=C.ACCENT, active=False,
                                       height=h, font=("Arial", 10, "bold"))
        self.btn_reset = _FlatBtn(self._btn_zone, "Reset",
                                  self._app._reset_pipeline if self._app else None,
                                  bg_on=C.ACCENT, active=True,
                                  height=h, font=("Arial", 10, "bold"),
                                  special_color="red")
        self._stack((self.btn_fd_params, self.btn_view_sources,
                     self.btn_individual, self.btn_reset))

    def _build_buttons_generic(self, bg: str):
        self.btn_run = _FlatBtn(self._btn_zone, "Run Pipeline",
                                lambda: messagebox.showinfo("Coming soon", "Not yet implemented."),
                                bg_on=C.RUN_COLOR, active=False,
                                font=("Arial", 10, "bold"),
                                height=self._stack_height(1))
        self._stack((self.btn_run,))

    # ------------------------------------------------------------------ #
    # Log / preview
    # ------------------------------------------------------------------ #

    def _show_logs(self):
        self._clear_preview()
        self._log_widget = tk.Text(
            self._preview_frame,
            bg=C.LOG_BG, fg=C.LOG_TXT,
            font=("Courier", 7), wrap=tk.NONE,
            state=tk.DISABLED, relief=tk.FLAT, bd=0,
            highlightthickness=0,
            insertbackground=C.LOG_TXT,
        )
        self._log_widget.place(x=0, y=0, width=CARD_W, height=CARD_H)
        if self._log_lines:
            self._log_widget.configure(state=tk.NORMAL)
            self._log_widget.insert(tk.END, "".join(self._log_lines))
            self._log_widget.see(tk.END)
            self._log_widget.configure(state=tk.DISABLED)
        self._preview_state = "logs"
        if self._has_figure:
            self._raise_toggle("Show Figure")
        else:
            self._hide_toggle()

    def _show_figure(self):
        if not self._has_figure:
            return
        if self._gif_frames_by_ch:
            self._install_gif_canvas()
            self._preview_state = "figure"
            if self._app:
                ch = self._app.current_gif_channel()
                if ch is not None:
                    self.show_gif_for_channel(ch)
        elif getattr(self, "_cached_fig_pil", None) is not None:
            self._clear_preview()
            from PIL import ImageTk
            self._cached_fig_tk = ImageTk.PhotoImage(self._cached_fig_pil)
            lbl = tk.Label(self._preview_frame, image=self._cached_fig_tk,
                           bg=C.LOG_BG, bd=0)
            lbl.place(x=0, y=0, width=CARD_W, height=CARD_H)
            self._preview_state = "figure"
        else:
            self._render_moment0(detections=self.detections)
            self._preview_state = "figure"
        self._raise_toggle("Show Logs")

    def _toggle_view(self):
        if self._preview_state == "figure":
            self._show_logs()
        else:
            self._show_figure()

    def _raise_toggle(self, label: str):
        self._toggle_btn.configure(text=label)
        self._toggle_btn.place(
            x=CARD_W - 4, y=CARD_H - 4,
            anchor="se",
        )
        self._toggle_btn.lift()

    def _hide_toggle(self):
        self._toggle_btn.place_forget()

    def _init_log_preview(self):
        self._log_lines     = []
        self._has_figure    = False
        self._cached_fig_pil = None
        self._show_logs()

    def _append_log(self, text: str):
        self._log_lines.append(text)
        w = getattr(self, "_log_widget", None)
        if w is None or not w.winfo_exists():
            return
        if self._preview_state == "logs":
            w.configure(state=tk.NORMAL)
            w.insert(tk.END, text)
            w.see(tk.END)
            w.configure(state=tk.DISABLED)

    def _draw_placeholder(self):
        bg       = C.PLACEHOLDER_BG_EN if self.enabled else C.PLACEHOLDER_BG_DIS
        txt_fill = C.PLACEHOLDER_TXT  if self.enabled else C.DIM_TXT
        ph = tk.Canvas(self._preview_frame, width=CARD_W, height=CARD_H,
                       bg=bg, highlightthickness=0)
        ph.create_text(CARD_W // 2, CARD_H // 2,
                       text=self.description, fill=txt_fill,
                       font=("Helvetica", 9), width=CARD_W - 24, justify=tk.CENTER)
        ph.place(x=0, y=0, width=CARD_W, height=CARD_H)

    def _clear_preview(self):
        if self._gif_job:
            self.after_cancel(self._gif_job)
            self._gif_job = None
        for w in self._preview_frame.winfo_children():
            if w is not self._toggle_btn:
                w.destroy()

    def _render_moment0(self, detections=None):
        cube = self.cube if self.cube is not None else (self._app.cards[0].cube if self._app else None)
        if cube is None:
            return
        mom0 = _moment0(cube)
        np.clip(mom0, 0, None, out=mom0)
        self._clear_preview()

        dpi = 96
        cmap = "cubehelix_r" if C._current_theme == "light" else "inferno"
        contour_color = "black" if C._current_theme == "light" else "white"
        fig = plt.Figure(figsize=(CARD_W/dpi, CARD_H/dpi), dpi=dpi, facecolor="#0a0a14")
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        vmin = float(np.nanmin(mom0))
        vmax = float(np.nanmax(mom0))
        if vmax <= vmin:
            vmax = vmin + 1e-9
        ax.imshow(mom0, cmap=cmap,
                  norm=Normalize(vmin=vmin, vmax=vmax),
                  origin="lower")

        H, W = mom0.shape
        if detections:
            union = np.zeros((H, W), dtype=bool)
            for d in detections:
                for m in d.footprint_masks:
                    union |= m
            if union.any():
                ax.contour(union.astype(float), [0.5],
                           colors=[contour_color], linewidths=0.6, alpha=0.7)

        c0 = self._app.cards[0] if self._app else self
        _draw_annotations(ax, H, W, c0.beam, c0.pixscale, c0.kpc_per_pix,
                          light=C._current_theme == "light")

        canvas = FigureCanvasTkAgg(fig, master=self._preview_frame)
        canvas.draw()
        canvas.get_tk_widget().configure(highlightthickness=0)
        canvas.get_tk_widget().place(x=0, y=0, width=CARD_W, height=CARD_H)
        self._preview_frame.update_idletasks()

        try:
            from PIL import Image as PilImage
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                        pad_inches=0, facecolor="#0a0a14")
            buf.seek(0)
            self._cached_fig_pil = PilImage.open(buf).copy().resize(
                (CARD_W, CARD_H), PilImage.LANCZOS).convert("RGB")
        except Exception:
            self._cached_fig_pil = None
        plt.close(fig)
        if self._has_figure and self._log_lines:
            self._raise_toggle("Show Logs")

    # ------------------------------------------------------------------ #
    # GIF renderers — produce {channel: PIL.Image} dicts
    # ------------------------------------------------------------------ #

    @staticmethod
    def _frame_to_pil(fig, dpi):
        from PIL import Image as PilImage
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=0, facecolor="#0a0a14")
        plt.close(fig)
        buf.seek(0)
        return PilImage.open(buf).copy().resize(
            (CARD_W, CARD_H), PilImage.LANCZOS).convert("RGB")

    @staticmethod
    def _cube_norm(cube: np.ndarray):
        return _cube_norm(cube)

    def _render_wavelet_gif(self, cube: np.ndarray, detections: list):
        c0 = self._app.cards[0] if self._app else None
        self._set_lazy_gif(*_wavelet_renderer(
            cube, detections,
            beam=c0.beam if c0 else None,
            pixscale=c0.pixscale if c0 else None,
        ))

    def _render_flow_gif(self, cube: np.ndarray, flow_seq: list):
        c0  = self._app.cards[0] if self._app else None
        det = self._app.cards[1].detections if self._app else None
        self._set_lazy_gif(*_flow_renderer(
            cube, flow_seq, det,
            beam=c0.beam if c0 else None,
            pixscale=c0.pixscale if c0 else None,
        ))

    def _render_sources_gif(self, cube: np.ndarray, tracks: list, sources: list):
        c0 = self._app.cards[0] if self._app else None
        self._set_lazy_gif(*_sources_renderer(
            cube, tracks, sources,
            beam=c0.beam if c0 else None,
            pixscale=c0.pixscale if c0 else None,
        ))

    def _render_scale_preview(self, cube: np.ndarray, wav_p: dict):
        from ..detect import starlet_transform, active_channels
        n_scales = int(wav_p.get("scales", 6))
        ch_list  = active_channels(cube)
        mid_ch   = ch_list[len(ch_list) // 2]
        img      = cube[mid_ch].astype(np.float32)
        coeffs   = starlet_transform(img, scales=n_scales)

        n_panels = n_scales
        n_cols   = min(3, n_panels)
        n_rows   = (n_panels + n_cols - 1) // n_cols

        self._clear_preview()
        dpi = 72
        cmap = "cubehelix_r" if C._current_theme == "light" else "inferno"
        fig = plt.Figure(figsize=(CARD_W/dpi, CARD_H/dpi), dpi=dpi, facecolor="#0a0a14")
        for i in range(n_panels):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#0a0a14")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333355"); sp.set_linewidth(0.3)
            band = np.clip(coeffs[i], 0, None)
            vmax = float(np.nanpercentile(band, 99.5)) if band.max() > 0 else 1e-9
            ax.imshow(band, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
            label = "Coarse" if i == n_scales - 1 else f"S{i+1}"
            ax.set_title(label, color="white", fontsize=5, pad=1)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01,
                            hspace=0.25, wspace=0.05)
        canvas = FigureCanvasTkAgg(fig, master=self._preview_frame)
        canvas.draw()
        canvas.get_tk_widget().configure(highlightthickness=0)
        canvas.get_tk_widget().place(x=0, y=0, width=CARD_W, height=CARD_H)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Coordinated GIF
    # ------------------------------------------------------------------ #

    def _set_lazy_gif(self, channels, render_fn):
        """Install a lazy per-channel frame renderer.

        Nothing is rendered now; each channel's frame is produced on first
        display in ``show_gif_for_channel`` and cached.  This keeps matplotlib
        on the main thread and off the pipeline worker.
        """
        self._gif_render       = render_fn
        self._gif_channels     = list(channels)
        self._gif_frames_by_ch = {}
        self._gif_tk_by_ch     = {}

    def _set_eager_gif(self, frames: dict):
        """Install a pre-rendered {channel: PIL.Image} frame dict (no lazy render)."""
        self._gif_render       = None
        self._gif_channels     = sorted(frames)
        self._gif_frames_by_ch = frames
        self._gif_tk_by_ch     = {}

    def _install_gif_canvas(self):
        self._clear_preview()
        self._gif_canvas = tk.Canvas(self._preview_frame, width=CARD_W, height=CARD_H,
                                     bg=C.LOG_BG, highlightthickness=0)
        self._gif_canvas.place(x=0, y=0, width=CARD_W, height=CARD_H)
        self._gif_last_ch = None

    def show_gif_for_channel(self, ch: int):
        if self._preview_state != "figure":
            return
        # Render this channel lazily on first display, then cache the PIL image.
        pil = self._gif_frames_by_ch.get(ch)
        if pil is None and self._gif_render is not None and ch in self._gif_channels:
            try:
                pil = self._gif_render(ch)
            except Exception:
                pil = None
            if pil is not None:
                self._gif_frames_by_ch[ch] = pil
        if pil is None:
            return
        if self._gif_canvas is None or not self._gif_canvas.winfo_exists():
            self._install_gif_canvas()
        if ch == self._gif_last_ch:
            return
        from PIL import ImageTk
        if ch not in self._gif_tk_by_ch:
            self._gif_tk_by_ch[ch] = ImageTk.PhotoImage(pil)
        self._gif_canvas.delete("all")
        self._gif_canvas.create_image(0, 0, anchor="nw",
                                      image=self._gif_tk_by_ch[ch])
        self._gif_last_ch = ch
        if hasattr(self, "_toggle_btn") and self._toggle_btn.winfo_ismapped():
            self._toggle_btn.lift()

    # ------------------------------------------------------------------ #
    # Card 0 actions
    # ------------------------------------------------------------------ #

    def _load_cube(self):
        path = filedialog.askopenfilename(
            title="Select spectral cube",
            filetypes=[
                ("All supported", "*.npy *.npz *.fits *.fit *.h5 *.hdf5 *.hdf"),
                ("NumPy",  "*.npy *.npz"),
                ("FITS",   "*.fits *.fit"),
                ("HDF5",   "*.h5 *.hdf5 *.hdf"),
            ],
        )
        if not path:
            return
        # The file read + scaling can take seconds on large cubes; run them off
        # the UI thread so the app stays responsive, then finish on the main
        # thread (matplotlib rendering must happen there).
        self.btn_load.disable()
        self._step_label.configure(text="Loading…")

        q: queue.Queue = queue.Queue()

        def _work():
            try:
                cube, beam, pixscale, vel, kpc_per_pix = load_cube_file(path)
                scaled = _apply_scaling(cube, self.scaling)
                q.put(("ok", cube, scaled, beam, pixscale, vel, kpc_per_pix))
            except Exception as exc:  # noqa: BLE001 — surfaced to the user
                q.put(("err", str(exc)))

        threading.Thread(target=_work, daemon=True).start()
        self.after(40, lambda: self._finish_load_cube(q, path))

    def _finish_load_cube(self, q: queue.Queue, path: str):
        try:
            msg = q.get_nowait()
        except queue.Empty:
            self.after(40, lambda: self._finish_load_cube(q, path))
            return

        self.btn_load.enable()
        if msg[0] == "err":
            self._step_label.configure(text=self.name)
            messagebox.showerror("Load error", msg[1])
            return

        _, cube, scaled, beam, pixscale, vel, kpc_per_pix = msg
        self.cube_raw    = cube
        self.cube        = scaled
        self.vel_array   = vel
        self.filepath    = path
        self.beam        = beam
        self.pixscale    = pixscale
        self.kpc_per_pix = kpc_per_pix
        self._step_label.configure(text=self.name)
        self._render_moment0()
        self.btn_view.enable()
        if self._app and len(self._app.cards) > 2:
            self._app.cards[2]._set_flow_params_enabled(True)
        if self._app and len(self._app.cards) > 3:
            self._app.cards[3]._set_flow_params_enabled(True)   # enables FD-params button
        if self._app:
            self._app._disable_theme_button()
        if self._on_loaded:
            self._on_loaded(self.index)

    def _view_slice(self):
        cube = self.cube_raw if self.cube_raw is not None else self.cube
        if cube is None:
            return
        SliceViewer(self, cube,
                    detections=self._app.cards[1].detections if self._app else None,
                    mode="raw",
                    initial_norm=self.scaling.get("mode", "linear"),
                    initial_gamma=self.scaling.get("gamma", 0.5),
                    beam=self.beam, pixscale=self.pixscale, kpc_per_pix=self.kpc_per_pix,
                    vel_array=self.vel_array,
                    card_0=self)

    def _poll_pipeline(self, q: queue.Queue, log_card: int = 1, carry=None):
        # Drain everything currently available (plus any messages carried over
        # from the previous tick) into one ordered list.
        pending = list(carry) if carry else []
        try:
            while True:
                pending.append(q.get_nowait())
        except queue.Empty:
            pass

        done = False
        appended_log = False
        leftover = []
        log_buf: list[str] = []   # coalesce consecutive log chunks into one insert

        def _flush_logs():
            nonlocal appended_log
            if log_buf and self._app and len(self._app.cards) > log_card:
                self._app.cards[log_card]._append_log("".join(log_buf))
                appended_log = True
            log_buf.clear()

        i = 0
        while i < len(pending):
            msg  = pending[i]
            kind = msg[0]

            if kind == "log":
                log_buf.append(msg[1])
                i += 1
                continue

            # Paint any buffered logs in a single Text insert before handling a
            # view-changing transition (cheaper than one insert per line).
            _flush_logs()

            # If we've already appended log lines this tick, defer this
            # transition (and the rest) so the logs actually paint before the
            # preview switches away — keeps card 1's detection logs streaming.
            if appended_log:
                leftover = pending[i:]
                break

            if kind == "switch_card":
                log_card = msg[1]
                if self._app and len(self._app.cards) > log_card:
                    self._app.cards[log_card]._init_log_preview()

            elif kind == "detection_done":
                det, wav_p = msg[1], msg[2]
                ms_dets = msg[3] if len(msg) > 3 else None
                self._on_detection_done(det, wav_p, multi_scale_dets=ms_dets)
                self._step_label.configure(text="Flow…")

            elif kind == "flow_done":
                flow_seq = msg[1]
                fps = msg[2] if len(msg) > 2 else None
                dps = msg[3] if len(msg) > 3 else None
                if self._app:
                    c2 = self._app.cards[2]
                    if not c2.enabled:
                        c2.enable()
                    c2._on_flow_done(flow_seq,
                                     flow_seq_per_scale=fps,
                                     detections_per_scale=dps)

            elif kind == "tracking_done":
                flow_seq, tracks, sources = msg[1], msg[2], msg[3]
                hier = msg[4] if len(msg) > 4 else None
                tps  = msg[5] if len(msg) > 5 else None
                sps  = msg[6] if len(msg) > 6 else None
                ms_dets = msg[7] if len(msg) > 7 else None
                if self._app:
                    c3 = self._app.cards[3]
                    if not c3.enabled:
                        c3.enable()
                    c3._on_tracking_done(tracks, sources,
                                         hierarchical_sources=hier,
                                         tracks_per_scale=tps,
                                         sources_per_scale=sps,
                                         multi_scale_dets=ms_dets)
                self._step_label.configure(text=self.name)
                self.btn_run.enable()
                self.btn_decompose.enable()
                self.btn_configure.enable()
                done = True

            elif kind == "error":
                messagebox.showerror("Pipeline failed", msg[1])
                self._step_label.configure(text=self.name)
                self.btn_run.enable()
                self.btn_decompose.enable()
                self.btn_configure.enable()
                done = True

            i += 1

        # Paint any logs left in the buffer after the loop.
        _flush_logs()

        if not done:
            self.after(40, lambda: self._poll_pipeline(q, log_card, leftover))

    # ------------------------------------------------------------------ #
    # Card 1 actions
    # ------------------------------------------------------------------ #

    def _open_configure(self):
        self._view_choose_scales()

    def _on_params_saved(self, params: dict):
        self._wav_params = params

    def _on_detection_done(self, detections: list, params: dict,
                           multi_scale_dets: list | None = None):
        self.detections        = detections
        self._multi_scale_dets = multi_scale_dets or []
        self._wav_params       = params
        c0   = self._app.cards[0] if self._app else None
        cube = c0.cube if c0 else None
        # Install a lazy per-channel renderer (frames render on demand on the
        # main thread, so the pipeline worker never touches matplotlib).
        channels: list = []
        if cube is not None and self._multi_scale_dets:
            channels, render = _multi_scale_renderer(
                cube, self._multi_scale_dets, c0.beam, c0.pixscale)
            self._set_lazy_gif(channels, render)
        elif cube is not None and detections:
            channels, render = _wavelet_renderer(cube, detections, c0.beam, c0.pixscale)
            self._set_lazy_gif(channels, render)
        if channels:
            # Same behaviour as card 2 / card 3: logs stream into the preview
            # while detection runs, then switch to the GIF when it completes.
            self._has_figure = True
            self._install_gif_canvas()
            self._preview_state = "figure"
            if self._app:
                self._app.refresh_gif_clock()
            if self._log_lines:
                self._raise_toggle("Show Logs")
        self.btn_det_view.enable()
        if self._on_loaded:
            self._on_loaded(self.index)

    def _view_detections(self):
        if not self.detections and not self._multi_scale_dets:
            messagebox.showinfo("No detections", "Run Decomposition first.")
            return
        cube = self._app.cards[0].cube if self._app else None
        if cube is None:
            return
        c0 = self._app.cards[0] if self._app else None
        SliceViewer(self, cube, detections=self.detections,
                    multi_scale_dets=self._multi_scale_dets or None,
                    mode="detections",
                    beam=c0.beam if c0 else None, pixscale=c0.pixscale if c0 else None,
                    kpc_per_pix=c0.kpc_per_pix if c0 else None)

    def _view_choose_scales(self):
        cube = self._app.cards[0].cube if self._app else None
        if cube is None:
            messagebox.showwarning("No cube", "Load a cube first.")
            return

        def _on_chosen(scale: int):
            if self._wav_params is None:
                self._wav_params = {}
            self._wav_params["use_scale"] = scale

        def _on_saved(params: dict):
            self._wav_params = params

        ScaleViewer(self, cube,
                    wav_params=self._wav_params,
                    detections=self.detections,
                    on_scale_chosen=_on_chosen,
                    on_params_saved=_on_saved,
                    card_0=self._app.cards[0] if self._app else None)

    def _beam_fwhm_px(self):
        """Beam FWHM in pixels from the loaded cube's header, or None.

        The loader already extracts ``beam = (bmaj", bmin", bpa)`` and
        ``pixscale`` ("/px) for drawing the beam ellipse; this reuses them so
        the physical scale bounds need no manual entry.
        """
        c0 = self._app.cards[0] if self._app else None
        beam = getattr(c0, "beam", None)
        pixscale = getattr(c0, "pixscale", None)
        if not beam or not pixscale:
            return None
        try:
            bmaj, bmin, ps = float(beam[0]), float(beam[1]), float(pixscale)
        except (TypeError, ValueError, IndexError):
            return None
        if bmaj <= 0 or bmin <= 0 or ps <= 0:
            return None
        return float(np.sqrt(bmaj * bmin) / ps)

    def _max_scale_wav_p(self, cube) -> dict:
        """Wavelet params with ``scales`` forced to the max 2-D scales the cube
        supports (use_scale clamped to a valid detail band)."""
        from ..detect import max_2d_scales, default_detect_scales
        wav_p = dict(self._wav_params or dict(
            scales=6, k_sigma=5.0, use_scale=5,
            min_area=None, use_mean_map_sigma=True,
        ))
        # Blank beam in the dialog means "read it from the header".
        if wav_p.get("beam_fwhm_px") is None:
            wav_p["beam_fwhm_px"] = self._beam_fwhm_px()
        n = max_2d_scales(cube.shape[1], cube.shape[2])
        wav_p["scales"] = n
        wav_p["use_scale"] = min(int(wav_p.get("use_scale", n - 1)), n - 1)
        # Multi-scale detection bands: honour the configured selection (clamped
        # to valid detail bands 1…n-1, never the coarse residual), else default
        # to the three below the coarsest (Nmax-1, Nmax-2, Nmax-3).
        ds = sorted({int(s) for s in (wav_p.get("detect_scales") or [])
                     if 1 <= int(s) <= n - 1})
        wav_p["detect_scales"] = ds or default_detect_scales(n)
        return wav_p

    def _run_decomposition(self):
        cube = self._app.cards[0].cube if self._app else None
        if cube is None:
            messagebox.showwarning("No cube", "Load a cube in the Moment 0 step first.")
            return
        wav_p = self._max_scale_wav_p(cube)
        n_scales = wav_p["scales"]
        n_detail = n_scales - 1
        n_ch, H, W = cube.shape

        from ..detect import beam_area_px, admissible_scales

        beam = wav_p.get("beam_fwhm_px")
        min_area = wav_p.get("min_area", 20)

        if min_area is None:
            min_area_s = (f"{int(np.ceil(beam_area_px(beam)))} px (one beam)"
                          if beam else "10 px (no beam — fallback)")
        else:
            min_area_s = f"{min_area} px"

        if beam:
            allowed = admissible_scales(n_scales, beam)
            req = wav_p.get("detect_scales") or []
            dropped = [s for s in req if s not in set(allowed)]
            bounds_s = (f"  Admissible bands : {allowed}"
                        + (f"   dropped {dropped}" if dropped else "")
                        + f"\n    beam {beam:.2f} px\n")
        else:
            bounds_s = ("  Admissible bands : unbounded "
                        "(no beam — sub-beam bands may hold correlated noise)\n")

        thresh_s = (f"  Threshold : k-sigma = {wav_p.get('k_sigma', 5.0)} "
                    f"x per-scale noise\n")

        self._init_log_preview()
        self._append_log(
            f"Starlet (à trous IUWT) undecimated wavelet decomposition\n"
            f"  Cube : {n_ch} channels  {H}×{W} px\n"
            f"  Scales : {n_scales}  "
            f"({n_detail} detail band{'s' if n_detail != 1 else ''} + 1 coarse residual)\n"
            f"  k-sigma threshold : {wav_p.get('k_sigma', 5.0)}\n"
            f"  Min component area : {min_area_s}\n"
            + thresh_s + bounds_s +
            f"\nOpen Configure to inspect scales per channel.\n"
            f"Click Run Source Identification to run detection and tracking.\n"
        )
        self.btn_configure.enable()

    def _run_sourceid(self):
        c0   = self._app.cards[0] if self._app else None
        cube = c0.cube if c0 else None
        if cube is None:
            messagebox.showwarning("No cube", "Load a cube in the Moment 0 step first.")
            return
        _beam     = c0.beam     if c0 else None
        _pixscale = c0.pixscale if c0 else None

        wav_p  = self._max_scale_wav_p(cube)
        flow_p = (self._app.cards[2]._flow_params if self._app else None) or dict(
            min_match_overlap=5, max_gap_channels=5,
        )
        fd_p = (self._app.cards[3]._false_det_params if self._app else None) or {}

        # Read BooleanVar on the main thread — Tcl is not thread-safe
        _ms_var = getattr(c0, '_multi_scale_enabled', None)
        multi_scale_enabled = _ms_var.get() if isinstance(_ms_var, tk.BooleanVar) else True

        self.btn_run.disable()
        self.btn_decompose.disable()
        self.btn_configure.disable()
        self._step_label.configure(text="Running…")
        self._init_log_preview()

        q: queue.Queue = queue.Queue()

        def _worker():
            import sys
            stream = _QueueStream(q)
            old_stdout = sys.stdout
            sys.stdout = stream
            try:
                from ..detect import WaveletDetector, active_channels
                from ..track  import compute_flow_sequence, link_tracks, _reconcile_splits
                from ..track  import group_into_sources, classify_sources, classify_kinematic, FlowTracker

                ch_list = active_channels(cube)

                if multi_scale_enabled:
                    # Multi-scale path: detection (card 1) → per-scale flow &
                    # linking (card 2) → grouping & hierarchy (card 3).  Each
                    # phase runs verbose so its logs land on the right card.
                    detector = WaveletDetector(detect_all_scales=True, **wav_p)
                    multi_scale_dets = detector.detect(cube, channel_list=ch_list, verbose=True)

                    # Preview frames are rendered lazily on the main thread —
                    # the worker only posts the computed data (keeps the UI
                    # responsive: no GIL-heavy matplotlib in this thread).
                    det = []
                    q.put(("detection_done", det, wav_p, multi_scale_dets))
                    q.put(("switch_card", 2))

                    tracker = FlowTracker(detector=detector,
                                        min_match_overlap=flow_p["min_match_overlap"],
                                        max_gap_channels=flow_p["max_gap_channels"],
                                        **fd_p)

                    # Phase 1 — per-scale optical flow + track linking
                    dps = tracker._multi_scale_to_per_scale_dets(multi_scale_dets)
                    tracks_per_scale, flow_seq_per_scale = tracker.link_multi_scale(
                        dps, verbose=True)
                    coarsest = max(flow_seq_per_scale.keys())
                    flow_seq = flow_seq_per_scale[coarsest]
                    q.put(("flow_done", flow_seq, flow_seq_per_scale, dps))
                    q.put(("switch_card", 3))

                    # Phase 2 — grouping, hierarchy, false-detection filtering
                    result = tracker.group_multi_scale(
                        multi_scale_dets, dps, tracks_per_scale,
                        flow_seq_per_scale, verbose=True)
                    tracks = result.tracks
                    good_sources = result.sources
                    q.put(("tracking_done", flow_seq, tracks, good_sources,
                           result.hierarchical_sources, result.tracks_per_scale,
                           result.sources_per_scale, multi_scale_dets))
                else:
                    # Single-scale detection path (original)
                    det = WaveletDetector(**wav_p).detect(cube, channel_list=ch_list, verbose=True)
                    q.put(("detection_done", det, wav_p))

                    q.put(("switch_card", 2))
                    flow_seq   = compute_flow_sequence(det, verbose=True)
                    q.put(("flow_done", flow_seq))

                    q.put(("switch_card", 3))
                    tracks = link_tracks(det, flow_seq,
                                         min_match_overlap=flow_p["min_match_overlap"],
                                         max_gap_channels=flow_p["max_gap_channels"],
                                         verbose=True)
                    det_rev  = list(reversed(det))
                    flow_rev = [(b, a, -fl, mg) for (a, b, fl, mg) in reversed(flow_seq)]
                    bwd = link_tracks(det_rev, flow_rev,
                                      min_match_overlap=flow_p["min_match_overlap"],
                                      max_gap_channels=flow_p["max_gap_channels"])
                    _reconcile_splits(tracks, bwd, verbose=True)
                    classify_kinematic(tracks, verbose=True)
                    sources = group_into_sources(tracks)
                    good_sources, false_dets, _, _ = classify_sources(
                        sources, tracks, det, flow_seq,
                        verbose=True, **fd_p,
                    )
                    q.put(("tracking_done", flow_seq, tracks, good_sources))
            except Exception as exc:
                q.put(("error", str(exc)))
            finally:
                sys.stdout = old_stdout

        threading.Thread(target=_worker, daemon=True).start()
        self._poll_pipeline(q, log_card=1)

    # ------------------------------------------------------------------ #
    # Card 2 actions
    # ------------------------------------------------------------------ #

    def _on_flow_done(self, flow_seq: list,
                      flow_seq_per_scale=None, detections_per_scale=None):
        self.flow_seq = flow_seq
        self._flow_seq_per_scale   = flow_seq_per_scale
        self._flow_detections_per_scale = detections_per_scale
        c0   = self._app.cards[0] if self._app else None
        cube = c0.cube if c0 else None
        channels: list = []
        if cube is not None and flow_seq_per_scale and detections_per_scale:
            coarsest = max(flow_seq_per_scale.keys())
            channels, render = _multi_scale_flow_renderer(
                cube, flow_seq_per_scale, detections_per_scale, coarsest,
                c0.beam, c0.pixscale)
            self._set_lazy_gif(channels, render)
        elif cube is not None:
            det = self._app.cards[1].detections if self._app else None
            channels, render = _flow_renderer(cube, flow_seq, det, c0.beam, c0.pixscale)
            self._set_lazy_gif(channels, render)
        if channels:
            self._has_figure = True
            self._install_gif_canvas()
            self._preview_state = "figure"
            if self._app:
                self._app.refresh_gif_clock()
                ch = self._app.current_gif_channel()
                if ch is not None:
                    self.show_gif_for_channel(ch)
            if self._log_lines:
                self._raise_toggle("Show Logs")
        self.btn_flow_view.enable()
        if self._on_loaded:
            self._on_loaded(self.index)

    # ------------------------------------------------------------------ #
    # Card 3 actions
    # ------------------------------------------------------------------ #

    def _on_tracking_done(self, tracks: list, sources: list,
                          hierarchical_sources=None, tracks_per_scale=None,
                          sources_per_scale=None, multi_scale_dets=None):
        self.tracks  = tracks
        self.sources = sources
        self._hierarchical_sources = hierarchical_sources
        self._tracks_per_scale     = tracks_per_scale
        self._sources_per_scale    = sources_per_scale
        self._multi_scale_dets     = multi_scale_dets or []
        n_tracks  = len(tracks)
        n_sources = len(sources)
        if hierarchical_sources:
            n_roots  = sum(1 for h in hierarchical_sources if h.is_root())
            n_leaves = sum(1 for h in hierarchical_sources if h.is_leaf())
            self._append_log(
                f"\n[Hierarchy]\n"
                f"  Tracks (all scales)   : {n_tracks}\n"
                f"  Hierarchical sources  : {len(hierarchical_sources)}\n"
                f"  Trees (roots)         : {n_roots}\n"
                f"  Leaves (finest scale) : {n_leaves}\n"
            )
        else:
            self._append_log(
                f"\n[Source Grouping]\n"
                f"  Tracks   : {n_tracks}\n"
                f"  Sources  : {n_sources}\n"
            )
        c0   = self._app.cards[0] if self._app else None
        cube = c0.cube if c0 else None
        channels: list = []
        if cube is not None and hierarchical_sources and tracks_per_scale:
            channels, render = _hierarchical_sources_renderer(
                cube, hierarchical_sources, tracks_per_scale,
                self._multi_scale_dets, c0.beam, c0.pixscale)
            if render is not None:
                self._set_lazy_gif(channels, render)
        elif cube is not None and sources:
            channels, render = _sources_renderer(cube, tracks, sources, c0.beam, c0.pixscale)
            self._set_lazy_gif(channels, render)
        self._has_figure = bool(channels)
        if self._has_figure:
            self._install_gif_canvas()
            self._preview_state = "figure"
            if self._app:
                self._app.refresh_gif_clock()
                ch = self._app.current_gif_channel()
                if ch is not None:
                    self.show_gif_for_channel(ch)
            if self._log_lines:
                self._raise_toggle("Show Logs")
        self.btn_view_sources.enable()
        self.btn_individual.enable()
        if self._on_loaded:
            self._on_loaded(self.index)

    def _view_sources_per_channel(self):
        # Folded into the combined Source Analysis window.
        self._combined_analysis()

    def _combined_analysis(self):
        c0 = self._app.cards[0] if self._app else None
        cube = c0.cube if c0 else None
        hier = getattr(self, "_hierarchical_sources", None)
        if cube is None or (not getattr(self, "sources", None) and not hier):
            return
        CombinedAnalysisWindow(self, cube, self.tracks, self.sources,
                               vel_array=c0.vel_array if c0 else None,
                               hierarchical_sources=hier,
                               tracks_per_scale=getattr(self, "_tracks_per_scale", None),
                               beam=c0.beam if c0 else None,
                               pixscale=c0.pixscale if c0 else None,
                               kpc_per_pix=c0.kpc_per_pix if c0 else None)

    def _individual_analysis(self):
        c0 = self._app.cards[0] if self._app else None
        cube = c0.cube if c0 else None
        hier = getattr(self, "_hierarchical_sources", None)
        if cube is None or (not getattr(self, "sources", None) and not hier):
            return
        IndividualAnalysisWindow(self, cube, self.tracks, self.sources,
                                 vel_array=c0.vel_array if c0 else None,
                                 hierarchical_sources=hier,
                                 tracks_per_scale=getattr(self, "_tracks_per_scale", None),
                                 beam=c0.beam if c0 else None,
                                 pixscale=c0.pixscale if c0 else None)

    def _view_flow(self):
        if not self.flow_seq:
            return
        cube = self._app.cards[0].cube if self._app else None
        if cube is None:
            return
        det = self._app.cards[1].detections if self._app else None
        c0  = self._app.cards[0] if self._app else None
        SliceViewer(self, cube, detections=det, flow_seq=self.flow_seq, mode="flow",
                    flow_seq_per_scale=getattr(self, "_flow_seq_per_scale", None),
                    detections_per_scale=getattr(self, "_flow_detections_per_scale", None),
                    beam=c0.beam if c0 else None, pixscale=c0.pixscale if c0 else None,
                    kpc_per_pix=c0.kpc_per_pix if c0 else None)

    # ------------------------------------------------------------------ #
    # Enable / reset
    # ------------------------------------------------------------------ #

    def enable(self):
        if self.enabled:
            return
        self.enabled = True
        bg = C.CARD_BG
        self.configure(bg=bg, highlightbackground=C.CARD_BORDER, highlightcolor=C.CARD_BORDER)
        self._body.configure(bg=bg)
        self._right_col.configure(bg=bg)
        self._preview_frame.configure(bg=bg)
        self._btn_zone.configure(bg=bg)
        self._clear_preview()
        self._draw_placeholder()
        self._step_label.configure(fg=C.STEP_LABEL_TXT, bg=bg)
        for child in self._btn_zone.winfo_children():
            if isinstance(child, tk.Frame) and not isinstance(child, _FlatBtn):
                child.configure(bg=bg)
        if self.index == 0:
            self.btn_load.enable()
        elif self.index == 1:
            self.btn_decompose.enable()
            self.btn_configure.enable()
            self.btn_run.enable()
        elif self.index == 2:
            self._set_flow_params_enabled(True)
        elif self.index == 3:
            pass
        else:
            if hasattr(self, "btn_run"):
                self.btn_run.enable()

    def refresh_on_theme_change(self):
        """Regenerate visualization frames with new theme colormap."""
        if self.index == 0:
            self._refresh_scaling_theme()
        if self._preview_state == "placeholder":
            return
        if self._preview_state == "logs":
            return
        if self._preview_state == "figure":
            # Save current channel before regenerating
            current_ch = self._gif_last_ch
            # Regenerate frames based on what's currently displayed
            cube0 = self._app.cards[0].cube if self._app else None
            c0 = self._app.cards[0] if self._app else None
            beam = c0.beam if c0 else None
            pxs  = c0.pixscale if c0 else None
            if self.index == 0 and self.cube is not None:
                self._render_moment0(self.detections)
            elif self.index == 1 and getattr(self, "_multi_scale_dets", None) and cube0 is not None:
                self._set_lazy_gif(*_multi_scale_renderer(
                    cube0, self._multi_scale_dets, beam, pxs))
            elif self.index == 1 and self.detections is not None and cube0 is not None:
                self._render_wavelet_gif(cube0, self.detections)
            elif self.index == 2 and getattr(self, "_flow_seq_per_scale", None) and cube0 is not None:
                dps = getattr(self, "_flow_detections_per_scale", {}) or {}
                coarsest = max(self._flow_seq_per_scale.keys())
                self._set_lazy_gif(*_multi_scale_flow_renderer(
                    cube0, self._flow_seq_per_scale, dps, coarsest, beam, pxs))
            elif self.index == 2 and self.flow_seq is not None and cube0 is not None:
                self._render_flow_gif(cube0, self.flow_seq)
            elif self.index == 3 and getattr(self, "_hierarchical_sources", None) and cube0 is not None:
                channels, render = _hierarchical_sources_renderer(
                    cube0, self._hierarchical_sources, self._tracks_per_scale,
                    getattr(self, "_multi_scale_dets", []), beam, pxs)
                if render is not None:
                    self._set_lazy_gif(channels, render)
            elif self.index == 3 and self.sources is not None and cube0 is not None:
                self._render_sources_gif(cube0, self.tracks, self.sources)
            # Clear the last channel so show_gif_for_channel will update
            self._gif_last_ch = None
            # Refresh the current displayed frame
            if current_ch is not None:
                self.show_gif_for_channel(current_ch)
            elif self._app:
                ch = self._app.current_gif_channel()
                if ch is not None:
                    self.show_gif_for_channel(ch)

    def reset(self):
        """Restore card to its initial state, clearing all results and logs.

        Card 0 preserves its loaded cube; cards 1+ wipe everything.
        """
        self._clear_preview()
        if self.index != 0:
            self.cube_raw    = None
            self.cube        = None
            self.vel_array   = None
            self.scaling     = dict(mode="linear", gamma=0.5)
            self.filepath    = None
            self.beam        = None
            self.pixscale    = None
            self.kpc_per_pix = None
        self.detections        = None
        self._multi_scale_dets = []
        self.flow_seq          = None
        self._flow_seq_per_scale        = None
        self._flow_detections_per_scale = None
        self._hierarchical_sources      = None
        self._tracks_per_scale          = None
        self._sources_per_scale         = None
        self._wav_params = None
        self._flow_params= None
        self._log_lines  = []
        self._has_figure = False
        self._preview_state = "placeholder"
        self._hide_toggle()
        self._gif_pil_frames     = []
        self._gif_tk_frames      = []
        self._gif_frames_by_ch   = {}
        self._gif_tk_by_ch       = {}
        self._gif_canvas         = None
        self._gif_last_ch        = None
        self._cached_fig_pil     = None

        if self.index == 0:
            if self.cube is not None:
                self._render_moment0()
                self._has_figure = True
                self._preview_state = "figure"
            else:
                self._draw_placeholder()
        else:
            self.enabled = False
            bg = C.CARD_OFF
            self.configure(bg=bg, highlightbackground=C.DIM, highlightcolor=C.DIM)
            self._body.configure(bg=bg)
            self._right_col.configure(bg=bg)
            self._preview_frame.configure(bg=bg)
            self._btn_zone.configure(bg=bg)
            self._draw_placeholder()
            self._step_label.configure(fg=C.STEP_LABEL_DIS, bg=bg)
            for child in self._btn_zone.winfo_children():
                if isinstance(child, tk.Frame) and not isinstance(child, _FlatBtn):
                    child.configure(bg=bg)
            if self.index == 2:
                self._reset_flow_controls()
                self._set_flow_params_enabled(False)
            for attr in ("btn_decompose", "btn_configure", "btn_det_view",
                         "btn_flow_view", "btn_fd_params",
                         "btn_view_sources", "btn_individual",
                         "btn_run"):
                if hasattr(self, attr):
                    getattr(self, attr).disable()
