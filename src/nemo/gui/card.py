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

from ._constants import (ACCENT, BG, CARD_BG, CARD_BORDER, CARD_OFF,
                         CARD_W, CARD_H, DIM, DIM_TXT,
                         RUN_COLOR, BTN_W, BTN_H, BTN_ZONE_H, BTN_TALL)
from .widgets import _FlatBtn, _QueueStream
from .dialogs import WaveletParamsDialog, FlowParamsDialog, ScalingDialog
from .viewers import SliceViewer, ScaleViewer
from .analysis import CombinedAnalysisWindow, IndividualAnalysisWindow, _source_colors
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


def _cube_norm(cube: np.ndarray):
    flat = cube.ravel()
    vmin = float(np.nanmin(flat));  vmax = float(np.nanmax(flat))
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return Normalize(vmin=vmin, vmax=vmax)


def _build_wavelet_frames(cube: np.ndarray, detections: list) -> dict:
    norm = _cube_norm(cube)
    dpi  = 72;  fsz = CARD_W / dpi
    frames: dict = {}
    for d in detections:
        fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
        ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
        ax.imshow(cube[d.channel], cmap="inferno", norm=norm, origin="lower")
        for mask in d.footprint_masks:
            ax.contour(mask.astype(float), [0.5],
                       colors=["white"], linewidths=0.6, alpha=0.85)
        frames[d.channel] = _frame_to_pil(fig, dpi)
    return frames


def _build_flow_frames(cube: np.ndarray, flow_seq: list,
                       detections: list | None = None) -> dict:
    norm = _cube_norm(cube)
    dpi  = 72;  fsz = CARD_W / dpi
    det_by_ch = {d.channel: d for d in (detections or [])}
    frames: dict = {}
    for ch_ref, _ch_tgt, flow, _mask in flow_seq:
        img_data = cube[ch_ref]
        H, W = img_data.shape
        fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
        ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
        ax.imshow(img_data, cmap="inferno", norm=norm, origin="lower")
        qs = max(H // 35, 3)
        ys = np.arange(0, H, qs);  xs = np.arange(0, W, qs)
        Xq, Yq = np.meshgrid(xs, ys)
        u = flow[1][ys[:, None], xs[None, :]].ravel()
        v = flow[0][ys[:, None], xs[None, :]].ravel()
        mag = np.hypot(u, v);  pk = float(mag.max())
        if pk > 1e-6:
            sc = qs * 0.9 / pk
            ax.quiver(Xq.ravel(), Yq.ravel(), u*sc, v*sc,
                      mag, cmap="cool", angles="xy", scale_units="xy", scale=1,
                      width=0.003, headwidth=3, alpha=0.85, clim=(0, pk))
        d = det_by_ch.get(ch_ref)
        if d:
            for mask in d.footprint_masks:
                ax.contour(mask.astype(float), [0.5],
                           colors=["white"], linewidths=0.5, alpha=0.4)
        frames[ch_ref] = _frame_to_pil(fig, dpi)
    return frames


def _build_sources_frames(cube: np.ndarray, tracks: list, sources: list) -> dict:
    from matplotlib.patches import Rectangle as _Rect
    norm = _cube_norm(cube)
    dpi  = 72;  fsz = CARD_W / dpi

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

    all_channels = sorted({ch for d in src_ch_masks.values() for ch in d})
    frames: dict = {}
    PAD_BB = 4

    for ch in all_channels:
        img_data = cube[ch]
        H, W = img_data.shape
        fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
        ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
        ax.imshow(img_data, cmap="inferno", norm=norm, origin="lower")

        union = np.zeros((H, W), dtype=bool)
        for sid, ch_dict in src_ch_masks.items():
            for m in ch_dict.get(ch, []):
                union |= m
        if union.any():
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            rgba[~union] = [0, 0, 0, 0.55]
            ax.imshow(rgba, origin="lower", interpolation="nearest")

        for sid, ch_dict in src_ch_masks.items():
            masks = ch_dict.get(ch)
            if not masks:
                continue
            col  = src_color[sid]
            lcol = (0.3*col[0]+0.7, 0.3*col[1]+0.7, 0.3*col[2]+0.7)
            for mask in masks:
                ax.contour(mask.astype(float), [0.5],
                           colors=[col], linewidths=0.7)
                rows, cols = np.where(mask)
                if not len(rows):
                    continue
                r0, r1 = int(rows.min()), int(rows.max())
                c0, c1 = int(cols.min()), int(cols.max())
                ax.add_patch(_Rect(
                    (c0 - PAD_BB, r0 - PAD_BB),
                    c1 - c0 + 2*PAD_BB, r1 - r0 + 2*PAD_BB,
                    linewidth=0.8, edgecolor=lcol,
                    facecolor="none", zorder=4,
                ))
                ax.text(c1 + PAD_BB, r1 + PAD_BB, str(sid),
                        ha="center", va="center", fontsize=6,
                        color="black", fontweight="bold",
                        bbox=dict(boxstyle="circle,pad=0.2",
                                  fc=lcol, ec=lcol, lw=1.0),
                        zorder=6)
        frames[ch] = _frame_to_pil(fig, dpi)
    return frames


# ---------------------------------------------------------------------------
# CubeCard
# ---------------------------------------------------------------------------

class CubeCard(tk.Frame):
    def __init__(self, master, index: int, name: str, description: str,
                 app=None, on_loaded=None, **kw):
        active = index == 0
        super().__init__(master, bg=CARD_BG if active else CARD_OFF,
                         bd=0, highlightthickness=1, takefocus=False,
                         highlightbackground=CARD_BORDER if active else DIM,
                         highlightcolor=CARD_BORDER if active else DIM, **kw)
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
        self.detections    = None
        self.flow_seq      = None
        self._wav_params   = None
        self._flow_params  = None
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

        bg = CARD_BG if active else CARD_OFF

        self._preview_frame = tk.Frame(self, bg=bg, width=CARD_W, height=CARD_H)
        self._preview_frame.pack_propagate(False)
        self._preview_frame.pack(padx=8, pady=(8, 4))
        self._draw_placeholder()

        self._toggle_btn = tk.Label(
            self._preview_frame, text="Show Logs",
            bg="#1a1a2e", fg=ACCENT,
            font=("Helvetica", 7, "bold"), cursor="pointinghand",
            relief=tk.FLAT, padx=4, pady=2,
        )
        self._toggle_btn.bind("<Button-1>", lambda _e: self._toggle_view())

        self._step_label = tk.Label(self, text=name, bg=bg,
                                    fg=ACCENT if active else DIM_TXT,
                                    font=("Helvetica", 9, "bold"))
        self._step_label.pack(pady=(0, 4))

        self._btn_zone = tk.Frame(self, bg=bg, height=BTN_ZONE_H)
        self._btn_zone.pack_propagate(False)
        self._btn_zone.pack(fill=tk.X, padx=8, pady=(0, 8))

        if   index == 0: self._build_buttons_step0(bg, active)
        elif index == 1: self._build_buttons_step1(bg, active)
        elif index == 2: self._build_buttons_step2(bg, active)
        elif index == 3: self._build_buttons_step3(bg, active)
        else:            self._build_buttons_generic(bg)

    # ------------------------------------------------------------------ #
    # Button layouts
    # ------------------------------------------------------------------ #

    def _build_buttons_step0(self, bg: str, active: bool):
        row1 = tk.Frame(self._btn_zone, bg=bg)
        row1.pack(pady=(4, 3))
        self.btn_load     = _FlatBtn(row1, "Load Cube",     self._load_cube,    bg_on=ACCENT, active=active, height=BTN_TALL)
        self.btn_view     = _FlatBtn(row1, "View Slice",    self._view_slice,   bg_on=ACCENT, active=False,  height=BTN_TALL)
        self.btn_load.pack(side=tk.LEFT, padx=3)
        self.btn_view.pack(side=tk.LEFT, padx=3)

        row2 = tk.Frame(self._btn_zone, bg=bg)
        row2.pack()
        self.btn_spectrum = _FlatBtn(row2, "View Spectrum", self._view_spectrum, bg_on=ACCENT, active=False, height=BTN_TALL)
        self.btn_scaling  = _FlatBtn(row2, "Scaling",       self._open_scaling,  bg_on=ACCENT, active=False, height=BTN_TALL)
        self.btn_spectrum.pack(side=tk.LEFT, padx=3)
        self.btn_scaling.pack(side=tk.LEFT, padx=3)

    def _build_buttons_step1(self, bg: str, active: bool):
        _span = BTN_W * 2 + 6

        row1 = tk.Frame(self._btn_zone, bg=bg)
        row1.pack(pady=(4, 3))
        self.btn_configure = _FlatBtn(row1, "Configure & Run Decomposition",
                                      self._open_configure, bg_on=ACCENT, active=active,
                                      height=BTN_TALL, btn_width=_span)
        self.btn_configure.pack(padx=3)
        self.btn_decompose = self.btn_configure

        row2 = tk.Frame(self._btn_zone, bg=bg)
        row2.pack()
        self.btn_run = _FlatBtn(row2, "Run Source ID",
                                self._run_sourceid, bg_on=RUN_COLOR, active=active,
                                font=("Helvetica", 11, "bold"), height=BTN_TALL)
        self.btn_det_view = _FlatBtn(row2, "View Detections",
                                     self._view_detections, bg_on=ACCENT, active=False,
                                     height=BTN_TALL)
        self.btn_run.pack(side=tk.LEFT, padx=3)
        self.btn_det_view.pack(side=tk.LEFT, padx=3)

    def _build_buttons_step2(self, bg: str, active: bool):
        row1 = tk.Frame(self._btn_zone, bg=bg)
        row1.pack(pady=(4, 0))
        self.btn_flow_params = _FlatBtn(row1, "Optical Flow\nParameters",
                                        self._open_flow_params, bg_on=ACCENT, active=active,
                                        height=BTN_TALL, font=("Helvetica", 10))
        self.btn_flow_view   = _FlatBtn(row1, "View Flow\nPer Channel",
                                        self._view_flow, bg_on=ACCENT, active=False,
                                        height=BTN_TALL, font=("Helvetica", 10))
        self.btn_flow_params.pack(side=tk.LEFT, padx=3)
        self.btn_flow_view.pack(side=tk.LEFT, padx=3)

    def _build_buttons_step3(self, bg: str, active: bool):
        _span = BTN_W * 2 + 6
        row1 = tk.Frame(self._btn_zone, bg=bg)
        row1.pack(pady=(4, 3))
        self.btn_view_sources = _FlatBtn(row1, "View Sources per Channel",
                                         self._view_sources_per_channel,
                                         bg_on=RUN_COLOR, active=False,
                                         font=("Helvetica", 11, "bold"),
                                         btn_width=_span, height=BTN_TALL)
        self.btn_view_sources.pack(padx=3)

        row2 = tk.Frame(self._btn_zone, bg=bg)
        row2.pack()
        self.btn_combined    = _FlatBtn(row2, "Combined\nAnalysis",
                                        self._combined_analysis,
                                        bg_on=ACCENT, active=False,
                                        height=BTN_TALL, font=("Helvetica", 10))
        self.btn_individual  = _FlatBtn(row2, "Individual\nAnalysis",
                                        self._individual_analysis,
                                        bg_on=ACCENT, active=False,
                                        height=BTN_TALL, font=("Helvetica", 10))
        self.btn_combined.pack(side=tk.LEFT, padx=3)
        self.btn_individual.pack(side=tk.LEFT, padx=3)

    def _build_buttons_generic(self, bg: str):
        row1 = tk.Frame(self._btn_zone, bg=bg)
        row1.pack(pady=(6, 0))
        self.btn_run = _FlatBtn(row1, "Run Pipeline",
                                lambda: messagebox.showinfo("Coming soon", "Not yet implemented."),
                                bg_on=RUN_COLOR, active=False,
                                font=("Helvetica", 9, "bold"))
        self.btn_run.pack(padx=3)

    # ------------------------------------------------------------------ #
    # Log / preview
    # ------------------------------------------------------------------ #

    def _show_logs(self):
        self._clear_preview()
        self._log_widget = tk.Text(
            self._preview_frame,
            bg="#0a0a14", fg=ACCENT,
            font=("Courier", 7), wrap=tk.NONE,
            state=tk.DISABLED, relief=tk.FLAT, bd=0,
            insertbackground=ACCENT,
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
                           bg="#0a0a14", bd=0)
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
        bg       = "#2a2a4a" if self.enabled else "#111128"
        txt_fill = "#666699"  if self.enabled else DIM_TXT
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
        fig = plt.Figure(figsize=(CARD_W/dpi, CARD_H/dpi), dpi=dpi, facecolor="#0a0a14")
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        vmin = float(np.nanmin(mom0))
        vmax = float(np.nanmax(mom0))
        if vmax <= vmin:
            vmax = vmin + 1e-9
        ax.imshow(mom0, cmap="inferno",
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
                           colors=["white"], linewidths=0.6, alpha=0.7)

        beam = pixscale = None
        if self._app:
            beam     = self._app.cards[0].beam
            pixscale = self._app.cards[0].pixscale
        if pixscale:
            for arcsec in (1, 2, 5, 10, 20, 30, 60, 120):
                bar_px = arcsec / pixscale
                if W * 0.12 <= bar_px <= W * 0.35:
                    break
            x0, y0 = W * 0.68, H * 0.07
            ax.plot([x0, x0+bar_px], [y0, y0], color="white", lw=1.5)
            ax.text(x0+bar_px/2, y0+H*0.045, f'{arcsec}"',
                    color="white", ha="center", va="bottom", fontsize=6)
        if beam and pixscale:
            pad = max(beam[0]/pixscale, 5) * 0.75
            ax.add_patch(Ellipse((pad, pad),
                                 width=beam[1]/pixscale, height=beam[0]/pixscale,
                                 angle=beam[2], color="cyan", alpha=0.75))

        canvas = FigureCanvasTkAgg(fig, master=self._preview_frame)
        canvas.draw()
        canvas.get_tk_widget().configure(highlightthickness=0)
        canvas.get_tk_widget().place(x=0, y=0, width=CARD_W, height=CARD_H)

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
        flat = cube.ravel()
        vmin = float(np.nanmin(flat));  vmax = float(np.nanmax(flat))
        if vmax <= vmin:
            vmax = vmin + 1e-9
        return Normalize(vmin=vmin, vmax=vmax)

    def _render_wavelet_gif(self, cube: np.ndarray, detections: list):
        norm = self._cube_norm(cube)
        dpi  = 72;  fsz = CARD_W / dpi
        frames: dict[int, "PilImage.Image"] = {}

        for d in detections:
            img_data = cube[d.channel]
            fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
            ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
            ax.imshow(img_data, cmap="inferno", norm=norm, origin="lower")
            for mask in d.footprint_masks:
                ax.contour(mask.astype(float), [0.5],
                           colors=["white"], linewidths=0.6, alpha=0.85)
            frames[d.channel] = self._frame_to_pil(fig, dpi)

        self._gif_frames_by_ch = frames
        self._gif_tk_by_ch     = {}

    def _render_flow_gif(self, cube: np.ndarray, flow_seq: list):
        norm = self._cube_norm(cube)
        dpi  = 72;  fsz = CARD_W / dpi
        frames: dict[int, "PilImage.Image"] = {}

        det_by_ch = {}
        if self._app:
            for d in (self._app.cards[1].detections or []):
                det_by_ch[d.channel] = d

        for ch_ref, _ch_tgt, flow, _mask in flow_seq:
            img_data = cube[ch_ref]
            H, W = img_data.shape
            fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
            ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
            ax.imshow(img_data, cmap="inferno", norm=norm, origin="lower")

            qs = max(H // 35, 3)
            ys = np.arange(0, H, qs);  xs = np.arange(0, W, qs)
            Xq, Yq = np.meshgrid(xs, ys)
            u = flow[1][ys[:, None], xs[None, :]].ravel()
            v = flow[0][ys[:, None], xs[None, :]].ravel()
            mag = np.hypot(u, v);  pk = float(mag.max())
            if pk > 1e-6:
                sc = qs * 0.9 / pk
                ax.quiver(Xq.ravel(), Yq.ravel(), u*sc, v*sc,
                          mag, cmap="cool", angles="xy", scale_units="xy", scale=1,
                          width=0.003, headwidth=3, alpha=0.85, clim=(0, pk))

            d = det_by_ch.get(ch_ref)
            if d:
                for mask in d.footprint_masks:
                    ax.contour(mask.astype(float), [0.5],
                               colors=["white"], linewidths=0.5, alpha=0.4)

            frames[ch_ref] = self._frame_to_pil(fig, dpi)

        self._gif_frames_by_ch = frames
        self._gif_tk_by_ch     = {}

    def _render_sources_gif(self, cube: np.ndarray, tracks: list, sources: list):
        from matplotlib.patches import Rectangle as _Rect

        norm = self._cube_norm(cube)
        dpi  = 72;  fsz = CARD_W / dpi

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

        all_channels = sorted({ch for d in src_ch_masks.values() for ch in d})
        frames: dict[int, "PilImage.Image"] = {}
        PAD_BB = 4

        for ch in all_channels:
            img_data = cube[ch]
            H, W = img_data.shape
            fig = plt.Figure(figsize=(fsz, fsz), dpi=dpi, facecolor="#0a0a14")
            ax  = fig.add_axes([0, 0, 1, 1]);  ax.set_axis_off()
            ax.imshow(img_data, cmap="inferno", norm=norm, origin="lower")

            union = np.zeros((H, W), dtype=bool)
            for sid, ch_dict in src_ch_masks.items():
                for m in ch_dict.get(ch, []):
                    union |= m
            if union.any():
                rgba = np.zeros((H, W, 4), dtype=np.float32)
                rgba[~union] = [0, 0, 0, 0.55]
                ax.imshow(rgba, origin="lower", interpolation="nearest")

            for sid, ch_dict in src_ch_masks.items():
                masks = ch_dict.get(ch)
                if not masks:
                    continue
                col  = src_color[sid]
                lcol = (0.3*col[0]+0.7, 0.3*col[1]+0.7, 0.3*col[2]+0.7)
                for mask in masks:
                    ax.contour(mask.astype(float), [0.5],
                               colors=[col], linewidths=0.7)
                    rows, cols = np.where(mask)
                    if not len(rows):
                        continue
                    r0, r1 = int(rows.min()), int(rows.max())
                    c0, c1 = int(cols.min()), int(cols.max())
                    ax.add_patch(_Rect(
                        (c0 - PAD_BB, r0 - PAD_BB),
                        c1 - c0 + 2*PAD_BB, r1 - r0 + 2*PAD_BB,
                        linewidth=0.8, edgecolor=lcol,
                        facecolor="none", zorder=4,
                    ))
                    ax.text(c1 + PAD_BB, r1 + PAD_BB, str(sid),
                            ha="center", va="center", fontsize=6,
                            color="black", fontweight="bold",
                            bbox=dict(boxstyle="circle,pad=0.2",
                                      fc=lcol, ec=lcol, lw=1.0),
                            zorder=6)

            frames[ch] = self._frame_to_pil(fig, dpi)

        self._gif_frames_by_ch = frames
        self._gif_tk_by_ch     = {}

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
        fig = plt.Figure(figsize=(CARD_W/dpi, CARD_H/dpi), dpi=dpi, facecolor="#0a0a14")
        for i in range(n_panels):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#0a0a14")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333355"); sp.set_linewidth(0.3)
            band = np.clip(coeffs[i], 0, None)
            vmax = float(np.nanpercentile(band, 99.5)) if band.max() > 0 else 1e-9
            ax.imshow(band, cmap="seismic", origin="lower", vmin=-vmax, vmax=vmax)
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

    def _install_gif_canvas(self):
        self._clear_preview()
        self._gif_canvas = tk.Canvas(self._preview_frame, width=CARD_W, height=CARD_H,
                                     bg="#0a0a14", highlightthickness=0)
        self._gif_canvas.place(x=0, y=0, width=CARD_W, height=CARD_H)
        self._gif_last_ch = None

    def show_gif_for_channel(self, ch: int):
        if self._preview_state != "figure":
            return
        if not self._gif_frames_by_ch:
            return
        if self._gif_canvas is None or not self._gif_canvas.winfo_exists():
            self._install_gif_canvas()
        if ch == self._gif_last_ch:
            return
        if ch not in self._gif_frames_by_ch:
            return
        from PIL import ImageTk
        if ch not in self._gif_tk_by_ch:
            self._gif_tk_by_ch[ch] = ImageTk.PhotoImage(self._gif_frames_by_ch[ch])
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
        try:
            cube, beam, pixscale, vel = load_cube_file(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return
        self.cube_raw  = cube
        self.cube      = _apply_scaling(cube, self.scaling)
        self.vel_array = vel
        self.filepath  = path
        self.beam      = beam
        self.pixscale  = pixscale
        self._render_moment0()
        self.btn_view.enable()
        self.btn_spectrum.enable()
        self.btn_scaling.enable()
        if self._app and len(self._app.cards) > 2:
            self._app.cards[2].btn_flow_params.enable()
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
                    initial_gamma=self.scaling.get("gamma", 0.5))

    def _open_scaling(self):
        if self.cube_raw is None:
            return
        ScalingDialog(self, on_save=self._on_scaling_saved,
                      current=self.scaling)

    def _on_scaling_saved(self, scaling: dict):
        self.scaling = scaling
        self.cube    = _apply_scaling(self.cube_raw, scaling)
        self._render_moment0()

    def _view_spectrum(self):
        messagebox.showinfo("View Spectrum", "Spectrum viewer — coming soon.")

    def _poll_pipeline(self, q: queue.Queue, log_card: int = 1):
        done = False
        try:
            while True:
                msg  = q.get_nowait()
                kind = msg[0]

                if kind == "log":
                    if self._app and len(self._app.cards) > log_card:
                        self._app.cards[log_card]._append_log(msg[1])

                elif kind == "switch_card":
                    log_card = msg[1]
                    if self._app and len(self._app.cards) > log_card:
                        self._app.cards[log_card]._init_log_preview()

                elif kind == "detection_done":
                    _, det, wav_p, frames = msg
                    self._on_detection_done(det, wav_p, frames=frames)
                    self._step_label.configure(text="Flow…")

                elif kind == "flow_done":
                    _, flow_seq, frames = msg
                    if self._app:
                        c2 = self._app.cards[2]
                        if not c2.enabled:
                            c2.enable()
                        c2._on_flow_done(flow_seq, frames=frames)

                elif kind == "tracking_done":
                    _, flow_seq, tracks, sources, frames = msg
                    if self._app:
                        c3 = self._app.cards[3]
                        if not c3.enabled:
                            c3.enable()
                        c3._on_tracking_done(tracks, sources, frames=frames)
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

        except queue.Empty:
            pass

        if not done:
            self.after(40, lambda: self._poll_pipeline(q, log_card))

    # ------------------------------------------------------------------ #
    # Card 1 actions
    # ------------------------------------------------------------------ #

    def _open_configure(self):
        self._view_choose_scales()

    def _on_params_saved(self, params: dict):
        self._wav_params = params

    def _on_detection_done(self, detections: list, params: dict, frames: dict | None = None):
        self.detections  = detections
        self._wav_params = params
        if frames is None:
            cube = self._app.cards[0].cube if self._app else None
            if cube is not None:
                frames = _build_wavelet_frames(cube, detections)
            else:
                frames = {}
        self._gif_frames_by_ch = frames
        self._gif_tk_by_ch     = {}
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
        if not self.detections:
            messagebox.showinfo("No detections", "Run Decomposition first.")
            return
        cube = self._app.cards[0].cube if self._app else None
        if cube is None:
            return
        SliceViewer(self, cube, detections=self.detections, mode="detections")

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
                    on_params_saved=_on_saved)

    def _run_decomposition(self):
        cube = self._app.cards[0].cube if self._app else None
        if cube is None:
            messagebox.showwarning("No cube", "Load a cube in the Moment 0 step first.")
            return
        wav_p = self._wav_params or dict(
            scales=6, k_sigma=5.0, use_scale=5,
            min_area=20, thresh=None, use_mean_map_sigma=True,
        )
        n_scales = wav_p.get("scales", 6)
        n_detail = n_scales - 1
        n_ch, H, W = cube.shape

        self._init_log_preview()
        self._append_log(
            f"Starlet (à trous IUWT) undecimated wavelet decomposition\n"
            f"  Cube : {n_ch} channels  {H}×{W} px\n"
            f"  Scales : {n_scales}  "
            f"({n_detail} detail band{'s' if n_detail != 1 else ''} + 1 coarse residual)\n"
            f"  k-sigma threshold : {wav_p.get('k_sigma', 5.0)}\n"
            f"  Min component area : {wav_p.get('min_area', 20)} px\n\n"
            f"Open Configure to inspect scales per channel.\n"
            f"Click Run Source Identification to run detection and tracking.\n"
        )
        self.btn_configure.enable()

    def _run_sourceid(self):
        cube = self._app.cards[0].cube if self._app else None
        if cube is None:
            messagebox.showwarning("No cube", "Load a cube in the Moment 0 step first.")
            return

        wav_p  = self._wav_params or dict(
            scales=6, k_sigma=5.0, use_scale=5,
            min_area=20, thresh=None, use_mean_map_sigma=True,
        )
        flow_p = (self._app.cards[2]._flow_params if self._app else None) or dict(
            min_match_overlap=5, max_gap_channels=5,
        )

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
                from ..track  import group_into_sources

                ch_list = active_channels(cube)
                det = WaveletDetector(**wav_p).detect(cube, channel_list=ch_list, verbose=True)
                wav_frames = _build_wavelet_frames(cube, det)
                q.put(("detection_done", det, wav_p, wav_frames))

                q.put(("switch_card", 2))
                flow_seq   = compute_flow_sequence(det, verbose=True)
                flow_frames = _build_flow_frames(cube, flow_seq, det)
                q.put(("flow_done", flow_seq, flow_frames))

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
                sources = group_into_sources(tracks)
                src_frames = (_build_sources_frames(cube, tracks, sources)
                              if sources else {})
                q.put(("tracking_done", flow_seq, tracks, sources, src_frames))
            except Exception as exc:
                q.put(("error", str(exc)))
            finally:
                sys.stdout = old_stdout

        threading.Thread(target=_worker, daemon=True).start()
        self._poll_pipeline(q, log_card=1)

    # ------------------------------------------------------------------ #
    # Card 2 actions
    # ------------------------------------------------------------------ #

    def _open_flow_params(self):
        FlowParamsDialog(self, on_save=self._on_flow_params_saved,
                         current=self._flow_params)

    def _on_flow_params_saved(self, params: dict):
        self._flow_params = params

    def _on_flow_done(self, flow_seq: list, frames: dict | None = None):
        self.flow_seq = flow_seq
        if frames is None:
            cube = self._app.cards[0].cube if self._app else None
            det  = self._app.cards[1].detections if self._app else None
            if cube is not None:
                frames = _build_flow_frames(cube, flow_seq, det)
            else:
                frames = {}
        self._gif_frames_by_ch = frames
        self._gif_tk_by_ch     = {}
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

    def _on_tracking_done(self, tracks: list, sources: list, frames: dict | None = None):
        self.tracks  = tracks
        self.sources = sources
        n_tracks  = len(tracks)
        n_sources = len(sources)
        self._append_log(
            f"\n[Source Grouping]\n"
            f"  Tracks   : {n_tracks}\n"
            f"  Sources  : {n_sources}\n"
        )
        if frames is None:
            cube = self._app.cards[0].cube if self._app else None
            if cube is not None and sources:
                frames = _build_sources_frames(cube, tracks, sources)
            else:
                frames = {}
        self._gif_frames_by_ch = frames
        self._gif_tk_by_ch     = {}
        self._has_figure = bool(self._gif_frames_by_ch)
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
        self.btn_combined.enable()
        self.btn_individual.enable()
        if self._on_loaded:
            self._on_loaded(self.index)

    def _view_sources_per_channel(self):
        cube = self._app.cards[0].cube if self._app else None
        if cube is None or not getattr(self, "sources", None):
            return
        SliceViewer(self, cube,
                    tracks=self.tracks, sources=self.sources,
                    mode="sources")

    def _combined_analysis(self):
        c0 = self._app.cards[0] if self._app else None
        cube = c0.cube if c0 else None
        if cube is None or not getattr(self, "sources", None):
            return
        CombinedAnalysisWindow(self, cube, self.tracks, self.sources,
                               vel_array=c0.vel_array if c0 else None)

    def _individual_analysis(self):
        c0 = self._app.cards[0] if self._app else None
        cube = c0.cube if c0 else None
        if cube is None or not getattr(self, "sources", None):
            return
        IndividualAnalysisWindow(self, cube, self.tracks, self.sources,
                                 vel_array=c0.vel_array if c0 else None)

    def _view_flow(self):
        if not self.flow_seq:
            return
        cube = self._app.cards[0].cube if self._app else None
        if cube is None:
            return
        det = self._app.cards[1].detections if self._app else None
        SliceViewer(self, cube, detections=det, flow_seq=self.flow_seq, mode="flow")

    # ------------------------------------------------------------------ #
    # Enable / reset
    # ------------------------------------------------------------------ #

    def enable(self):
        if self.enabled:
            return
        self.enabled = True
        bg = CARD_BG
        self.configure(bg=bg, highlightbackground=CARD_BORDER, highlightcolor=CARD_BORDER)
        self._preview_frame.configure(bg=bg)
        self._btn_zone.configure(bg=bg)
        self._clear_preview()
        self._draw_placeholder()
        self._step_label.configure(fg=ACCENT, bg=bg)
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
            self.btn_flow_params.enable()
        elif self.index == 3:
            pass
        else:
            if hasattr(self, "btn_run"):
                self.btn_run.enable()

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
        self.detections  = None
        self.flow_seq    = None
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
            bg = CARD_OFF
            self.configure(bg=bg, highlightbackground=DIM, highlightcolor=DIM)
            self._preview_frame.configure(bg=bg)
            self._btn_zone.configure(bg=bg)
            self._draw_placeholder()
            self._step_label.configure(fg=DIM_TXT, bg=bg)
            for child in self._btn_zone.winfo_children():
                if isinstance(child, tk.Frame) and not isinstance(child, _FlatBtn):
                    child.configure(bg=bg)
            for attr in ("btn_decompose", "btn_configure", "btn_det_view",
                         "btn_flow_params", "btn_flow_view",
                         "btn_view_sources", "btn_combined", "btn_individual",
                         "btn_run"):
                if hasattr(self, attr):
                    getattr(self, attr).disable()
