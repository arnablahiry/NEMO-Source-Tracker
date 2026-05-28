"""NEMO GUI — spectral cube browser with moment-0 preview and slice viewer.

Launch with::

    python -m nemo.gui

or from Python::

    from nemo.gui import launch
    launch()
"""
from __future__ import annotations

import io
import queue
import subprocess
import threading
import tkinter as tk
import tkinter.ttk as ttk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import PowerNorm, LogNorm, Normalize
from matplotlib.patches import Ellipse

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
BG          = "#1a1a2e"
CARD_BG     = "#16213e"
CARD_OFF    = "#0f0f1a"
ACCENT      = "#4ecca3"
DIM         = "#44445a"
DIM_TXT     = "#555577"
CARD_W      = 240
CARD_H      = 240
BANNER_W    = 80
RUN_COLOR   = "#f5a623"
BTN_W       = 112
BTN_H       = 30
BTN_ZONE_H  = 120

_ASSETS = Path(__file__).parent.parent.parent / "assets"

_CMAPS = ["inferno", "viridis", "magma", "plasma", "cividis",
          "gray", "hot", "afmhot", "YlOrRd", "cubehelix"]


# ---------------------------------------------------------------------------
# Custom flat button
# ---------------------------------------------------------------------------

class _FlatBtn(tk.Frame):
    """Reliably coloured flat button — tk.Button ignores bg on macOS."""

    def __init__(self, parent, text: str, command,
                 bg_on: str, fg_on: str = "black",
                 font=("Helvetica", 11),
                 active: bool = False,
                 height: int | None = None,
                 btn_width: int | None = None,
                 **kw):
        self._bg_on  = bg_on
        self._fg_on  = fg_on
        self._cmd    = command
        self._active = False

        h  = height    if height    is not None else BTN_H
        w  = btn_width if btn_width is not None else BTN_W
        bg = bg_on if active else CARD_OFF
        fg = fg_on if active else DIM_TXT

        super().__init__(parent, width=w, height=h,
                         bg=bg, cursor="arrow", **kw)
        self.pack_propagate(False)

        self._lbl = tk.Label(self, text=text, bg=bg, fg=fg,
                             font=font, anchor="center", justify=tk.CENTER)
        self._lbl.place(relwidth=1, relheight=1)

        if active:
            self.enable()

    def enable(self, bg_on: str | None = None):
        if bg_on:
            self._bg_on = bg_on
        self._active = True
        self.configure(bg=self._bg_on, cursor="pointinghand")
        self._lbl.configure(bg=self._bg_on, fg=self._fg_on)
        for w in (self, self._lbl):
            w.bind("<Button-1>", self._click)
            w.bind("<Enter>",    self._hover)
            w.bind("<Leave>",    self._leave)

    def disable(self):
        self._active = False
        self.configure(bg=CARD_OFF, cursor="arrow")
        self._lbl.configure(bg=CARD_OFF, fg=DIM_TXT)
        for w in (self, self._lbl):
            w.unbind("<Button-1>")
            w.unbind("<Enter>")
            w.unbind("<Leave>")

    def _click(self, _e=None):
        if self._active and self._cmd:
            self._cmd()

    def _hover(self, _e=None):
        r = int(self._bg_on[1:3], 16)
        g = int(self._bg_on[3:5], 16)
        b = int(self._bg_on[5:7], 16)
        dim = f"#{int(r*.82):02x}{int(g*.82):02x}{int(b*.82):02x}"
        self.configure(bg=dim)
        self._lbl.configure(bg=dim)

    def _leave(self, _e=None):
        self.configure(bg=self._bg_on)
        self._lbl.configure(bg=self._bg_on)


# ---------------------------------------------------------------------------
# Cube I/O helpers
# ---------------------------------------------------------------------------

def _load_fits(path: str):
    from astropy.io import fits
    with fits.open(path) as hdul:
        data = np.squeeze(hdul[0].data).astype(np.float32)
        hdr  = hdul[0].header
    beam = pixscale = None
    vel_array = None
    if "BMAJ" in hdr and "BMIN" in hdr:
        beam = (float(hdr["BMAJ"])*3600, float(hdr["BMIN"])*3600,
                float(hdr.get("BPA", 0.0)))
    for key in ("CDELT1", "CD1_1"):
        if key in hdr:
            pixscale = abs(float(hdr[key])) * 3600
            break
    # Spectral WCS -> velocity (matches notebook)
    if all(k in hdr for k in ("CRVAL3", "CDELT3", "CRPIX3", "RESTFRQ")) \
            and data.ndim == 3:
        try:
            crval3, cdelt3 = float(hdr["CRVAL3"]), float(hdr["CDELT3"])
            crpix3, restf  = float(hdr["CRPIX3"]), float(hdr["RESTFRQ"])
            n_ch = data.shape[0]
            freq = crval3 + cdelt3 * (np.arange(n_ch) - (crpix3 - 1))
            vel_array = (299792.458 * (restf - freq) / restf).astype(np.float64)
        except Exception:
            vel_array = None
    return data, beam, pixscale, vel_array


def _load_h5(path: str):
    """Load an HDF5 spectral cube.

    Supports:
    - Toy-cube format: top-level dataset 'cube' (3-D), with beam under 'beam'
      attrs (bmaj_px, bmin_px, bpa_deg) and pixscale in 'spatial_resolution_kpc_per_px'.
    - Generic FITS-style attrs (BMAJ/BMIN/BPA in degrees, CDELT1 in degrees).
    - Fallback: pick the largest 3-D dataset.
    """
    import h5py
    with h5py.File(path, "r") as f:
        # 1) Prefer a top-level 3-D dataset named 'cube' (toy-cube convention)
        data = None
        key  = None
        if "cube" in f and isinstance(f["cube"], h5py.Dataset) and f["cube"].ndim == 3:
            key  = "cube"
            data = f[key][()].astype(np.float32)
        else:
            # 2) Otherwise pick the largest 3-D dataset
            cubes_3d: dict[str, tuple] = {}
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                    cubes_3d[name] = obj.shape
            f.visititems(_collect)
            if not cubes_3d:
                raise ValueError(
                    "No 3-D dataset found in HDF5 file (need shape (n_ch, H, W)).")
            key  = max(cubes_3d, key=lambda k: int(np.prod(cubes_3d[k])))
            data = f[key][()].astype(np.float32)

        beam = pixscale = None
        vel_array = None
        if "channel_velocities_km_s" in f:
            try:
                vel_array = f["channel_velocities_km_s"][()].astype(np.float64)
            except Exception:
                vel_array = None

        # --- Toy-cube beam: '/beam' group with px-unit attrs ---------------
        if "beam" in f and not isinstance(f["beam"], h5py.Dataset):
            bg = f["beam"]
            if "bmaj_px" in bg.attrs:
                # Stored already in pixels; render code expects beam in same
                # units as pixscale. To play nicely, set beam to (bmaj_px, bmin_px,
                # bpa) and pixscale to 1 so beam/pixscale ratios stay in pixels.
                bmaj_px = float(bg.attrs["bmaj_px"])
                bmin_px = float(bg.attrs.get("bmin_px", bmaj_px))
                bpa     = float(bg.attrs.get("bpa_deg", 0.0))
                beam     = (bmaj_px, bmin_px, bpa)
                pixscale = 1.0   # so beam[i] / pixscale gives px directly

        # --- Toy-cube pixscale in kpc/px overrides arcsec if found ----------
        if "spatial_resolution_kpc_per_px" in f.attrs and pixscale == 1.0:
            # Keep pixscale=1 so beam-in-px works; the renderer's scalebar
            # arcsec heuristic will pick a sensible bar length either way.
            pass

        # --- Generic FITS-style fallbacks (degrees) -------------------------
        for src in (f, f[key]):
            if beam is None and "BMAJ" in src.attrs:
                beam = (float(src.attrs["BMAJ"])*3600,
                        float(src.attrs.get("BMIN", src.attrs["BMAJ"]))*3600,
                        float(src.attrs.get("BPA", 0.0)))
            if pixscale is None and "CDELT1" in src.attrs:
                pixscale = abs(float(src.attrs["CDELT1"])) * 3600

    # Clean up NaNs
    np.nan_to_num(data, copy=False, nan=0.0)
    return data, beam, pixscale, vel_array


def load_cube_file(path: str):
    """Return (cube, beam, pixscale, vel_array). Any may be None."""
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32), None, None, None
    if ext == ".npz":
        npz = np.load(path)
        return npz[list(npz.files)[0]].astype(np.float32), None, None, None
    if ext in (".fits", ".fit"):
        return _load_fits(path)
    if ext in (".h5", ".hdf5", ".hdf"):
        return _load_h5(path)
    raise ValueError(f"Unsupported format: {ext}")


def _moment0(cube: np.ndarray) -> np.ndarray:
    return np.nansum(cube, axis=0) if cube.ndim == 3 else cube


# ---------------------------------------------------------------------------
# Wavelet parameters dialog
# ---------------------------------------------------------------------------

class WaveletParamsDialog(tk.Toplevel):
    """Modal dialog: edit and save WaveletDetector parameters."""

    def __init__(self, master, on_save, current: dict | None = None):
        super().__init__(master)
        self.title("Wavelet Detection — Parameters")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()
        self._on_save = on_save

        defaults = dict(scales=6, k_sigma=5.0, use_scale=5,
                        min_area=20, thresh="", use_mean_map_sigma=True)
        if current:
            defaults.update({k: ("" if v is None else v) for k, v in current.items()})

        pad = dict(padx=14, pady=5, sticky="w")
        tk.Label(self, text="Starlet Wavelet Detector", bg=BG, fg=ACCENT,
                 font=("Helvetica", 12, "bold")).grid(
                     row=0, column=0, columnspan=2, pady=(14, 8), padx=14)

        fields = [
            ("Scales (total starlet levels)",        "scales",             "int",   (2, 12)),
            ("k-sigma (detection threshold)",        "k_sigma",            "float", None),
            ("Use scale (1-based detail band)",      "use_scale",          "int",   (1, 11)),
            ("Min area (px, discard smaller blobs)", "min_area",           "int",   (1, 500)),
            ("Absolute threshold (blank = auto)",    "thresh",             "str",   None),
            ("Use mean-map sigma reference",         "use_mean_map_sigma", "bool",  None),
        ]
        self._vars: dict[str, tk.Variable] = {}
        for r, (label, key, typ, rng) in enumerate(fields, start=1):
            tk.Label(self, text=label, bg=BG, fg="white",
                     font=("Helvetica", 9), anchor="w").grid(row=r, column=0, **pad)
            if typ == "bool":
                v = tk.BooleanVar(value=bool(defaults[key]))
                tk.Checkbutton(self, variable=v, bg=BG, activebackground=BG,
                               selectcolor=CARD_BG, fg=ACCENT,
                               relief=tk.FLAT, bd=0).grid(
                                   row=r, column=1, padx=14, pady=5, sticky="w")
            elif typ == "int" and rng:
                v = tk.IntVar(value=int(defaults[key]))
                tk.Spinbox(self, from_=rng[0], to=rng[1], textvariable=v,
                           width=6, bg=CARD_BG, fg="white",
                           buttonbackground=CARD_BG,
                           relief=tk.FLAT, insertbackground="white").grid(
                               row=r, column=1, padx=14, pady=5, sticky="w")
            else:
                v = tk.StringVar(value=str(defaults[key]))
                tk.Entry(self, textvariable=v, width=10, bg=CARD_BG, fg="white",
                         insertbackground="white", relief=tk.FLAT).grid(
                             row=r, column=1, padx=14, pady=5, sticky="w")
            self._vars[key] = v

        btn_row = tk.Frame(self, bg=BG)
        btn_row.grid(row=len(fields)+1, column=0, columnspan=2, pady=(12, 14))
        _FlatBtn(btn_row, "Cancel", self.destroy, bg_on=DIM,    active=True).pack(side=tk.LEFT, padx=6)
        _FlatBtn(btn_row, "Save",   self._save,   bg_on=ACCENT, active=True).pack(side=tk.LEFT, padx=6)

        self.transient(master)
        self.wait_visibility()
        self.update_idletasks()
        px = master.winfo_rootx() + (master.winfo_width()  - self.winfo_width())  // 2
        py = master.winfo_rooty() + (master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")

    def _save(self):
        def _f(k): return self._vars[k].get()
        thresh_s = str(_f("thresh")).strip()
        try:
            params = dict(
                scales=int(_f("scales")), k_sigma=float(_f("k_sigma")),
                use_scale=int(_f("use_scale")), min_area=int(_f("min_area")),
                thresh=float(thresh_s) if thresh_s else None,
                use_mean_map_sigma=bool(_f("use_mean_map_sigma")),
            )
        except ValueError as exc:
            messagebox.showerror("Bad parameter", str(exc), parent=self)
            return
        self._on_save(params)
        self.destroy()


# ---------------------------------------------------------------------------
# Flow tracking parameters dialog
# ---------------------------------------------------------------------------

class FlowParamsDialog(tk.Toplevel):
    """Modal dialog: edit and save FlowTracker parameters."""

    def __init__(self, master, on_save, current: dict | None = None):
        super().__init__(master)
        self.title("Flow Tracking — Parameters")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()
        self._on_save = on_save

        defaults = dict(min_match_overlap=5, max_gap_channels=5)
        if current:
            defaults.update(current)

        pad = dict(padx=14, pady=5, sticky="w")
        tk.Label(self, text="TV-L1 Flow Tracker", bg=BG, fg=ACCENT,
                 font=("Helvetica", 12, "bold")).grid(
                     row=0, column=0, columnspan=2, pady=(14, 8), padx=14)

        fields = [
            ("Min match overlap (px)",   "min_match_overlap", "int", (1, 100)),
            ("Max gap channels",         "max_gap_channels",  "int", (1, 50)),
        ]
        self._vars: dict[str, tk.Variable] = {}
        for r, (label, key, typ, rng) in enumerate(fields, start=1):
            tk.Label(self, text=label, bg=BG, fg="white",
                     font=("Helvetica", 9), anchor="w").grid(row=r, column=0, **pad)
            if typ == "int" and rng:
                v = tk.IntVar(value=int(defaults[key]))
                tk.Spinbox(self, from_=rng[0], to=rng[1], textvariable=v,
                           width=6, bg=CARD_BG, fg="white",
                           buttonbackground=CARD_BG,
                           relief=tk.FLAT, insertbackground="white").grid(
                               row=r, column=1, padx=14, pady=5, sticky="w")
            else:
                v = tk.StringVar(value=str(defaults[key]))
                tk.Entry(self, textvariable=v, width=10, bg=CARD_BG, fg="white",
                         insertbackground="white", relief=tk.FLAT).grid(
                             row=r, column=1, padx=14, pady=5, sticky="w")
            self._vars[key] = v

        btn_row = tk.Frame(self, bg=BG)
        btn_row.grid(row=len(fields)+1, column=0, columnspan=2, pady=(12, 14))
        _FlatBtn(btn_row, "Cancel", self.destroy, bg_on=DIM,    active=True).pack(side=tk.LEFT, padx=6)
        _FlatBtn(btn_row, "Save",   self._save,   bg_on=ACCENT, active=True).pack(side=tk.LEFT, padx=6)

        self.transient(master)
        self.wait_visibility()
        self.update_idletasks()
        px = master.winfo_rootx() + (master.winfo_width()  - self.winfo_width())  // 2
        py = master.winfo_rooty() + (master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")

    def _save(self):
        def _f(k): return self._vars[k].get()
        try:
            params = dict(
                min_match_overlap=int(_f("min_match_overlap")),
                max_gap_channels=int(_f("max_gap_channels")),
            )
        except ValueError as exc:
            messagebox.showerror("Bad parameter", str(exc), parent=self)
            return
        self._on_save(params)
        self.destroy()


# ---------------------------------------------------------------------------
# Cube scaling dialog
# ---------------------------------------------------------------------------

class ScalingDialog(tk.Toplevel):
    """Choose Linear / Log / Power scaling for the cube."""

    def __init__(self, master, on_save, current: dict | None = None):
        super().__init__(master)
        self.title("Cube Scaling")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()
        self._on_save = on_save

        cur_mode  = (current or {}).get("mode", "linear")
        cur_gamma = float((current or {}).get("gamma", 0.5))

        self._mode  = tk.StringVar(value=cur_mode)
        self._gamma = tk.StringVar(value=f"{cur_gamma:g}")

        tk.Label(self, text="Cube Scaling", bg=BG, fg=ACCENT,
                 font=("Helvetica", 12, "bold")).grid(
                     row=0, column=0, columnspan=2, pady=(14, 6), padx=14)
        tk.Label(self, text="Applies to the whole cube. Downstream\n"
                            "detections use the rescaled values.",
                 bg=BG, fg="white", font=("Helvetica", 8),
                 justify=tk.LEFT).grid(row=1, column=0, columnspan=2,
                                       padx=14, pady=(0, 8), sticky="w")

        for r, label in enumerate(("linear", "log", "power")):
            tk.Radiobutton(self, text=label.capitalize(),
                           variable=self._mode, value=label,
                           bg=BG, fg="white", selectcolor=CARD_BG,
                           activebackground=BG, activeforeground=ACCENT,
                           font=("Helvetica", 10), relief=tk.FLAT,
                           anchor="w").grid(row=2+r, column=0,
                                            padx=(20, 8), pady=2, sticky="w")

        tk.Label(self, text="gamma", bg=BG, fg="white",
                 font=("Helvetica", 9)).grid(row=4, column=1, padx=(0, 4),
                                              sticky="e")
        tk.Entry(self, textvariable=self._gamma, width=6,
                 bg=CARD_BG, fg="white", font=("Helvetica", 9),
                 insertbackground="white",
                 relief=tk.FLAT).grid(row=4, column=1, padx=(60, 14), sticky="e")

        btn_row = tk.Frame(self, bg=BG)
        btn_row.grid(row=8, column=0, columnspan=2, pady=(14, 14))
        _FlatBtn(btn_row, "Cancel", self.destroy, bg_on=DIM,    active=True).pack(side=tk.LEFT, padx=6)
        _FlatBtn(btn_row, "Apply",  self._save,   bg_on=ACCENT, active=True).pack(side=tk.LEFT, padx=6)

        self.transient(master)
        self.wait_visibility()
        self.update_idletasks()
        px = master.winfo_rootx() + (master.winfo_width()  - self.winfo_width())  // 2
        py = master.winfo_rooty() + (master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")

    def _save(self):
        try:
            params = dict(mode=self._mode.get(),
                          gamma=float(self._gamma.get()))
        except ValueError as exc:
            messagebox.showerror("Bad gamma", str(exc), parent=self)
            return
        self._on_save(params)
        self.destroy()


def _apply_scaling(cube: np.ndarray, scaling: dict) -> np.ndarray:
    """Return a rescaled copy of *cube* according to *scaling* dict."""
    mode = (scaling or {}).get("mode", "linear")
    if mode == "linear":
        return cube
    if mode == "log":
        return np.log1p(np.clip(cube, 0.0, None)).astype(np.float32)
    if mode == "power":
        gamma = float((scaling or {}).get("gamma", 0.5))
        return (np.clip(cube, 0.0, None) ** gamma).astype(np.float32)
    return cube


# ---------------------------------------------------------------------------
# Slice / flow viewer
# ---------------------------------------------------------------------------

class SliceViewer(tk.Toplevel):
    """Channel-by-channel viewer with normalization controls and optional overlays.

    mode : "raw"        — plain channel images
           "detections" — white contour overlays from detection footprints
           "flow"       — quiver overlay from flow_seq
    """

    def __init__(self, master, cube: np.ndarray,
                 detections: list | None = None,
                 flow_seq: list | None = None,
                 tracks: list | None = None,
                 sources: list | None = None,
                 mode: str = "raw",
                 initial_norm: str = "linear",
                 initial_gamma: float = 0.5):
        super().__init__(master)
        self._initial_norm  = initial_norm
        self._initial_gamma = float(initial_gamma)
        self.title({
            "raw":        "Channel Viewer",
            "detections": "Wavelet Detections — Channel Viewer",
            "flow":       "Optical Flow — Channel Viewer",
            "sources":    "Sources — Channel Viewer",
        }.get(mode, "Channel Viewer"))
        self.configure(bg=BG)
        self.resizable(True, True)

        self._cube     = cube
        self._dets     = detections or []
        self._mode     = mode
        self._tracks   = tracks or []
        self._sources  = sources or []
        # build a channel→detection lookup
        self._det_by_ch = {d.channel: d for d in self._dets}
        # build flow lookup: ch_ref → flow array
        self._flow_by_ch = {}
        if flow_seq:
            for ch_ref, _ch_tgt, flow, _mask in flow_seq:
                self._flow_by_ch[ch_ref] = flow

        # ---- per-source channel→masks lookup (for "sources" mode) -----------
        # _src_masks_by_ch[src_id][ch] = list of masks for that source on that channel
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

        # ---- per-source colour --------------------------------------------
        cmap_src = plt.get_cmap("tab10")
        self._src_color = {
            s["id"]: cmap_src(i % 10) for i, s in enumerate(self._sources)
        }
        # visibility flags (one BooleanVar per source)
        self._src_visible: dict[int, tk.BooleanVar] = {}

        # channels to browse
        if self._dets and mode != "sources":
            self._channels = [d.channel for d in self._dets]
        else:
            self._channels = list(range(cube.shape[0]))

        VW = 500

        # global stats for slider defaults
        flat     = cube.ravel()
        self._data_min = float(np.nanmin(flat))
        self._data_max = float(np.nanmax(flat))
        self._p1       = max(float(np.nanpercentile(flat, 1)), 0)
        self._p99      = float(np.nanpercentile(flat, 99.5))

        # ---- matplotlib canvas ----
        self._fig = plt.Figure(figsize=(VW/96, VW/96), dpi=96, facecolor="#0a0a14")
        # Image axes — leaves a right margin for the colorbar
        self._ax    = self._fig.add_axes([0.01, 0.01, 0.82, 0.98])
        # Colorbar axes — placeholder; repositioned in _draw to match image height exactly
        self._ax_cb = self._fig.add_axes([0.86, 0.01, 0.06, 0.98])
        # Light border around the image
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#555577")
            spine.set_linewidth(0.8)
        self._ax_cb.set_facecolor("#0a0a14")
        # In sources mode wrap canvas + checkboxes side-by-side
        if self._mode == "sources" and self._sources:
            top = tk.Frame(self, bg=BG)
            top.pack(fill=tk.BOTH, expand=True)

            cb_panel = tk.Frame(top, bg=BG, width=95)
            cb_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 2), pady=4)
            cb_panel.pack_propagate(False)
            tk.Label(cb_panel, text="Sources", bg=BG, fg=ACCENT,
                     font=("Helvetica", 8, "bold")).pack(pady=(2, 4), anchor="w")
            for s in self._sources:
                sid = s["id"]
                col = self._src_color[sid]
                hex_col = "#{:02x}{:02x}{:02x}".format(
                    int(col[0]*255), int(col[1]*255), int(col[2]*255))
                var = tk.BooleanVar(value=True)
                self._src_visible[sid] = var
                row = tk.Frame(cb_panel, bg=BG)
                row.pack(fill=tk.X, pady=1, anchor="w")
                tk.Label(row, text="■", bg=BG, fg=hex_col,
                         font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 2))
                tk.Checkbutton(row,
                               text=f"S{sid}",
                               variable=var,
                               command=self._draw,
                               bg=BG, fg="white", selectcolor=CARD_BG,
                               activebackground=BG, activeforeground=ACCENT,
                               font=("Helvetica", 8), relief=tk.FLAT,
                               anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

            self._canvas = FigureCanvasTkAgg(self._fig, master=top)
            self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        else:
            self._canvas = FigureCanvasTkAgg(self._fig, master=self)
            self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---- controls ----
        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(fill=tk.X, padx=10, pady=(6, 2))

        # colormap
        self._cmap = tk.StringVar(value="inferno")
        tk.Label(ctrl, text="Colormap:", bg=BG, fg="white",
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 4))
        cm = ttk.Combobox(ctrl, textvariable=self._cmap,
                          values=_CMAPS, width=11, state="readonly")
        cm.pack(side=tk.LEFT, padx=(0, 14))
        cm.bind("<<ComboboxSelected>>", lambda _e: self._draw())

        # normalization radio buttons
        self._norm_mode = tk.StringVar(value=self._initial_norm)
        tk.Label(ctrl, text="Norm:", bg=BG, fg="white",
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 4))
        for label in ("linear", "log", "power"):
            tk.Radiobutton(ctrl, text=label, variable=self._norm_mode, value=label,
                           command=self._draw,
                           bg=BG, fg="white", selectcolor=CARD_BG,
                           activebackground=BG, activeforeground=ACCENT,
                           font=("Helvetica", 9), relief=tk.FLAT).pack(
                               side=tk.LEFT, padx=2)

        # info label — fixed width prevents window resizing as text changes
        max_ch  = max(self._channels) if self._channels else 0
        n_ch    = len(self._channels)
        # pre-compute the widest possible label text
        _sample = f"Channel {max_ch}  ({n_ch}/{n_ch})  ·  999 det. · 9999 total"
        self._ch_lbl = tk.Label(ctrl, text=_sample, bg=BG, fg=ACCENT,
                                font=("Helvetica", 9, "italic"), anchor="e")
        self._ch_lbl.pack(side=tk.RIGHT, padx=6)
        self._ch_lbl.configure(text="")

        # ---- vmin / vmax sliders ----
        vf = tk.Frame(self, bg=BG)
        vf.pack(fill=tk.X, padx=10, pady=(2, 2))

        def _slider(parent, label, from_, to_, default, row):
            txt = tk.Label(parent, text=label, bg=BG, fg="white",
                           font=("Helvetica", 8), width=5, anchor="e")
            txt.grid(row=row, column=0, padx=(0, 4), pady=1)
            s = tk.Scale(parent, from_=from_, to=to_, resolution=(to_-from_)/500,
                         orient=tk.HORIZONTAL, command=lambda _v: self._draw(),
                         bg=ACCENT, fg="black", troughcolor=CARD_BG,
                         activebackground=ACCENT, highlightthickness=0,
                         sliderrelief=tk.FLAT, bd=0, length=VW-160, showvalue=False)
            s.set(default)
            s.grid(row=row, column=1, sticky="ew", pady=1)
            val_lbl = tk.Label(parent, text="", bg=BG, fg="white",
                               font=("Helvetica", 7), width=10, anchor="w")
            val_lbl.grid(row=row, column=2, padx=(6, 0), pady=1)
            return s, val_lbl, txt

        vf.columnconfigure(1, weight=1)
        self._vmin_sl, self._vmin_lbl, self._vmin_txt = _slider(vf, "vmin", self._data_min, self._data_max, self._data_min, 0)
        self._vmax_sl, self._vmax_lbl, self._vmax_txt = _slider(vf, "vmax", self._data_min, self._data_max, self._data_max, 1)

        # ---- channel slider ----
        sf = tk.Frame(self, bg=BG)
        sf.pack(fill=tk.X, padx=10, pady=(2, 8))
        tk.Label(sf, text="Channel", bg=BG, fg="white",
                 font=("Helvetica", 9), width=7, anchor="e").pack(side=tk.LEFT, padx=(0, 6))
        self._slider = tk.Scale(
            sf, from_=0, to=len(self._channels)-1,
            orient=tk.HORIZONTAL, command=lambda _v: self._draw(),
            bg=ACCENT, fg="black", troughcolor=CARD_BG,
            activebackground=ACCENT, highlightthickness=0,
            sliderrelief=tk.FLAT, bd=0,
            length=VW - 120, showvalue=False,
        )
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._slider.set(len(self._channels) // 2)

        self._draw()
        # Pin the window width at its natural size so digit-count changes
        # in labels never cause the window to grow or shrink.
        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())
        self.maxsize(self.winfo_width(), 9999)   # allow vertical resize only

    # ------------------------------------------------------------------ #
    def _norm(self):
        vmin = float(self._vmin_sl.get())
        vmax = float(self._vmax_sl.get())
        # guard: sliders can cross — always keep vmin strictly below vmax
        if vmin >= vmax:
            vmax = vmin + 1e-9
        mode = self._norm_mode.get()
        if mode == "log":
            vmin = max(vmin, 1e-12)
            vmax = max(vmax, vmin + 1e-12)
            return LogNorm(vmin=vmin, vmax=vmax)
        elif mode == "power":
            vmin = max(vmin, 0)
            return PowerNorm(gamma=self._initial_gamma, vmin=vmin, vmax=vmax)
        else:
            return Normalize(vmin=vmin, vmax=vmax)

    def _fmt_val(self, v: float) -> str:
        if self._norm_mode.get() == "log":
            exp = np.log10(max(abs(v), 1e-30))
            return f"10^{exp:.2f}"
        return f"{v:.2e}"

    def _update_value_labels(self):
        for w in (self._vmin_txt, self._vmax_txt, self._vmin_lbl, self._vmax_lbl):
            w.configure(fg="white")
        self._vmin_lbl.configure(text=self._fmt_val(float(self._vmin_sl.get())))
        self._vmax_lbl.configure(text=self._fmt_val(float(self._vmax_sl.get())))

    def _draw(self):
        idx = int(self._slider.get())
        ch  = self._channels[idx]
        img = self._cube[ch]

        norm = self._norm()
        cmap = self._cmap.get()

        self._ax.clear()
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#555577")
            spine.set_linewidth(0.8)
        self._ax.imshow(img, cmap=cmap, norm=norm, origin="lower")

        # Sync colorbar height to match image axes exactly
        self._fig.canvas.draw()           # forces layout so get_position() is current
        pos = self._ax.get_position()
        cb_w = 0.05
        cb_gap = 0.025
        self._ax_cb.set_position([pos.x1 + cb_gap, pos.y0, cb_w, pos.height])

        # Colorbar — clear and redraw so norm/cmap changes propagate
        self._ax_cb.clear()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = self._fig.colorbar(sm, cax=self._ax_cb)
        cb.ax.tick_params(colors="white", labelsize=6, length=3)
        cb.outline.set_edgecolor("#333355")
        plt.setp(plt.getp(cb.ax, "yticklabels"), color="white", fontsize=6)

        if self._mode == "detections":
            d = self._det_by_ch.get(ch)
            if d:
                for mask in d.footprint_masks:
                    self._ax.contour(mask.astype(float), [0.5],
                                     colors=["white"], linewidths=0.8)

        elif self._mode == "sources":
            from matplotlib.patches import Rectangle as _Rect
            H_img, W_img = img.shape
            PAD_BB = 4

            # 1) footprint union overlay: dim everything OUTSIDE visible footprints
            union = np.zeros((H_img, W_img), dtype=bool)
            for sid, ch_dict in self._src_masks_by_ch.items():
                if not self._src_visible[sid].get():
                    continue
                for m in ch_dict.get(ch, []):
                    union |= m
            if union.any():
                rgba = np.zeros((H_img, W_img, 4), dtype=np.float32)
                rgba[~union] = [0, 0, 0, 0.55]
                self._ax.imshow(rgba, origin="lower", interpolation="nearest")

            # 2) per-source contour + bbox + numbered label (like notebook)
            for sid, ch_dict in self._src_masks_by_ch.items():
                if not self._src_visible[sid].get():
                    continue
                masks = ch_dict.get(ch)
                if not masks:
                    continue
                col = self._src_color[sid]
                # lightened colour for box/label background
                lcol = (0.3 * col[0] + 0.7,
                        0.3 * col[1] + 0.7,
                        0.3 * col[2] + 0.7)
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
                        linewidth=0.8, edgecolor=lcol, facecolor="none",
                        zorder=4,
                    ))
                    self._ax.text(
                        c1 + PAD_BB, r1 + PAD_BB, str(sid),
                        ha="center", va="center", fontsize=7,
                        color="black", fontweight="bold",
                        bbox=dict(boxstyle="circle,pad=0.22",
                                  fc=lcol, ec=lcol, lw=1.2),
                        zorder=6,
                    )

        elif self._mode == "flow":
            flow = self._flow_by_ch.get(ch)
            if flow is not None:
                H, W = img.shape
                qs = max(H // 30, 4)
                ys = np.arange(0, H, qs);  xs = np.arange(0, W, qs)
                Xq, Yq = np.meshgrid(xs, ys)
                u = flow[1][ys[:, None], xs[None, :]].ravel()
                v = flow[0][ys[:, None], xs[None, :]].ravel()
                mag = np.hypot(u, v)
                pk  = float(mag.max())
                if pk > 1e-6:
                    scale = qs * 0.8 / pk
                    self._ax.quiver(
                        Xq.ravel(), Yq.ravel(), u * scale, v * scale,
                        mag, cmap="cool", angles="xy", scale_units="xy", scale=1,
                        width=0.003, headwidth=4, headlength=5,
                        alpha=0.85, clim=(0, pk),
                    )
            # Overlay wavelet detection contours (faint white)
            d = self._det_by_ch.get(ch)
            if d:
                for mask in d.footprint_masks:
                    self._ax.contour(mask.astype(float), [0.5],
                                     colors=["white"], linewidths=0.6, alpha=0.7)

        self._canvas.draw()
        self._update_value_labels()
        total_det = sum(len(d.peaks) for d in self._dets) if self._dets else 0
        ch_det = len(self._det_by_ch[ch].peaks) if ch in self._det_by_ch else 0
        parts = [f"Channel {ch}  ({idx+1}/{len(self._channels)})"]
        if self._mode == "detections":
            parts.append(f"{ch_det} det. · {total_det} total")
        elif self._mode == "flow" and ch in self._flow_by_ch:
            parts.append("flow shown")
        elif self._mode == "sources":
            n_active = sum(1 for sid, v in self._src_visible.items()
                           if v.get() and self._src_masks_by_ch.get(sid, {}).get(ch))
            parts.append(f"{n_active} source(s) here")
        self._ch_lbl.configure(text="  ·  ".join(parts))


# ---------------------------------------------------------------------------
# Wavelet scale viewer
# ---------------------------------------------------------------------------

class ScaleViewer(tk.Toplevel):
    """Browse per-channel 2D wavelet coefficient maps with embedded parameters.

    Number of scales is chosen via radio buttons (2…max), where max depends on
    the image dimensions.  The figure is always laid out as 2 rows × ceil(n/2)
    cols.  Changing n_scales re-runs the starlet transform and redraws in real
    time.
    """

    N_ROWS = 2   # fixed layout: 2 rows × ceil(n/2) cols

    def __init__(self, master, cube: np.ndarray, wav_params: dict,
                 detections=None, on_scale_chosen=None, on_params_saved=None):
        super().__init__(master)
        self.title("Wavelet Scale Viewer — Configure")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._cube            = cube
        self._on_scale_chosen = on_scale_chosen
        self._on_params_saved = on_params_saved
        self._wav_params      = dict(wav_params or {})

        # Max meaningful scales depends on image dimensions.
        # B3-spline support at scale j is 5·2^(j-1); cap so support ≤ min(H,W).
        H, W = cube.shape[1], cube.shape[2]
        self._max_scales = max(2, int(np.floor(np.log2(min(H, W)))) - 1)
        n_scales = int(self._wav_params.get("scales", 6))
        n_scales = max(2, min(self._max_scales, n_scales))
        self._wav_params["scales"] = n_scales

        self._n_scales_var   = tk.IntVar(value=n_scales)
        current_scale        = int(self._wav_params.get("use_scale", 1))
        self._selected_scale = tk.IntVar(value=current_scale)

        self._channels = [d.channel for d in detections] if detections \
                         else list(range(cube.shape[0]))

        VW = 540

        # ---- top info bar ----
        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(fill=tk.X, padx=10, pady=(8, 4))
        self._ch_lbl = tk.Label(ctrl, text="", bg=BG, fg=ACCENT,
                                font=("Helvetica", 9, "italic"))
        self._ch_lbl.pack(side=tk.RIGHT, padx=6)

        # ---- figure ----
        self._fig_frame = tk.Frame(self, bg=BG)
        self._fig_frame.pack(fill=tk.BOTH, expand=True)
        self._fig = None;  self._mpl_canvas = None
        self._n_scales_last = 0

        # ---- number of scales radio row ----
        nf = tk.Frame(self, bg=BG)
        nf.pack(fill=tk.X, padx=10, pady=(4, 0))
        tk.Label(nf, text="Number of scales:", bg=BG, fg=ACCENT,
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        for s in range(2, self._max_scales + 1):
            tk.Radiobutton(nf, text=str(s),
                           variable=self._n_scales_var, value=s,
                           command=self._on_nscales_changed,
                           bg=BG, fg="white", selectcolor=CARD_BG,
                           activebackground=BG, activeforeground=ACCENT,
                           font=("Helvetica", 9), relief=tk.FLAT).pack(
                               side=tk.LEFT, padx=2)

        # ---- choose scale radio row ----
        self._rf = tk.Frame(self, bg=BG)
        self._rf.pack(fill=tk.X, padx=10, pady=(4, 2))
        self._rebuild_choose_scale_radios()

        # ---- embedded parameters ----
        pf = tk.LabelFrame(self, text="Detection Parameters", bg=BG, fg=ACCENT,
                           font=("Helvetica", 8, "bold"),
                           relief=tk.FLAT, bd=1, highlightthickness=1,
                           highlightbackground=DIM)
        pf.pack(fill=tk.X, padx=10, pady=(6, 4))

        # 3 params in a single row (Scales is now controlled by radio buttons above)
        param_fields = [
            ("k-sigma",                        "k_sigma",  "float"),
            ("Min area (px)",                  "min_area", "int"),
            ("Flux threshold (% of cube max)", "thresh",   "str"),
        ]
        self._pvars: dict[str, tk.Variable] = {}
        self._cube_max = float(np.nanmax(self._cube)) if self._cube.size else 1.0
        _stored = self._wav_params.get("thresh")
        if _stored is None or self._cube_max <= 0:
            _thresh_disp = "1.0"
        else:
            _thresh_disp = f"{100 * float(_stored) / self._cube_max:.3g}"
        defaults = dict(k_sigma=5.0, min_area=20, thresh=_thresh_disp)
        defaults.update({k: v for k, v in self._wav_params.items()
                         if k in defaults and k != "thresh"})

        inner = tk.Frame(pf, bg=BG)
        inner.pack(pady=(4, 4))
        for c, (label, key, _typ) in enumerate(param_fields):
            tk.Label(inner, text=label, bg=BG, fg="white",
                     font=("Helvetica", 8), anchor="e").grid(
                         row=0, column=c*2, padx=(10, 2), sticky="e")
            v = tk.StringVar(value=str(defaults.get(key, "")))
            tk.Entry(inner, textvariable=v, width=6,
                     bg=CARD_BG, fg="white", font=("Helvetica", 8),
                     insertbackground="white", relief=tk.FLAT).grid(
                         row=0, column=c*2+1, padx=(0, 6))
            self._pvars[key] = v

        _FlatBtn(pf, "Save Parameters", self._save_params,
                 bg_on=ACCENT, active=True).pack(pady=(0, 6))

        # ---- channel slider ----
        sf = tk.Frame(self, bg=BG)
        sf.pack(fill=tk.X, padx=10, pady=(4, 10))
        tk.Label(sf, text="Channel", bg=BG, fg="white",
                 font=("Helvetica", 9), width=7, anchor="e").pack(side=tk.LEFT, padx=(0, 6))
        self._slider = tk.Scale(
            sf, from_=0, to=len(self._channels) - 1,
            orient=tk.HORIZONTAL, command=lambda _v: self._draw(),
            bg=ACCENT, fg="black", troughcolor=CARD_BG,
            activebackground=ACCENT, highlightthickness=0,
            sliderrelief=tk.FLAT, bd=0, length=VW - 120, showvalue=False,
        )
        self._slider.set(len(self._channels) // 2)
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._VW = VW
        self._rebuild_figure()
        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())
        self.maxsize(self.winfo_width(), 9999)

    # ------------------------------------------------------------------ #
    def _on_nscales_changed(self):
        n_new = int(self._n_scales_var.get())
        # Clamp selected scale into the new range
        if int(self._selected_scale.get()) > n_new - 1:
            self._selected_scale.set(max(1, n_new - 1))
        self._wav_params["scales"]    = n_new
        self._wav_params["use_scale"] = int(self._selected_scale.get())
        self._rebuild_choose_scale_radios()
        self._rebuild_figure()

    def _rebuild_choose_scale_radios(self):
        for w in self._rf.winfo_children():
            w.destroy()
        tk.Label(self._rf, text="Choose scale:", bg=BG, fg="white",
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 6))
        n_detail = int(self._n_scales_var.get()) - 1
        for s in range(1, n_detail + 1):
            tk.Radiobutton(self._rf, text=str(s),
                           variable=self._selected_scale, value=s,
                           command=self._on_radio,
                           bg=BG, fg=ACCENT, selectcolor=CARD_BG,
                           activebackground=BG, activeforeground=ACCENT,
                           font=("Helvetica", 9), relief=tk.FLAT).pack(
                               side=tk.LEFT, padx=2)

    # ------------------------------------------------------------------ #
    def _save_params(self):
        def _f(k): return self._pvars[k].get()
        thresh_s = str(_f("thresh")).strip()
        try:
            thresh_abs = (float(thresh_s) / 100.0 * self._cube_max
                          if thresh_s else None)
            params = dict(
                scales=int(self._n_scales_var.get()),
                k_sigma=float(_f("k_sigma")),
                use_scale=int(self._selected_scale.get()),
                min_area=int(_f("min_area")),
                thresh=thresh_abs,
                use_mean_map_sigma=True,   # always on
            )
        except ValueError as exc:
            messagebox.showerror("Bad parameter", str(exc), parent=self)
            return
        if self._on_params_saved:
            self._on_params_saved(params)
        self.destroy()

    def _rebuild_figure(self):
        n_scales = int(self._n_scales_var.get())
        n_cols   = (n_scales + self.N_ROWS - 1) // self.N_ROWS   # ceil(n/2)
        n_rows   = self.N_ROWS

        if self._mpl_canvas:
            self._mpl_canvas.get_tk_widget().destroy()
        if self._fig:
            plt.close(self._fig)

        cell_px = self._VW // n_cols
        dpi     = 96
        n_slots = n_rows * n_cols
        self._fig = plt.Figure(
            figsize=(self._VW / dpi, (n_rows * cell_px) / dpi),
            dpi=dpi, facecolor="#0a0a14",
        )
        self._axes = []
        for i in range(n_slots):
            ax = self._fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_xticks([]);  ax.set_yticks([])
            ax.set_facecolor("#0a0a14")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333355");  sp.set_linewidth(0.5)
            self._axes.append(ax)
        self._fig.subplots_adjust(left=0.01, right=0.99,
                                  top=0.88, bottom=0.01,
                                  hspace=0.34, wspace=0.05)
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=self._fig_frame)
        self._mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._n_scales_last = n_scales
        self._draw()

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
        img = self._cube[ch].astype(np.float32)

        from .detect import starlet_transform
        coeffs = starlet_transform(img, scales=n_scales)
        n_detail = n_scales - 1
        chosen = int(self._selected_scale.get())

        for i, ax in enumerate(self._axes):
            ax.clear()
            ax.set_xticks([]);  ax.set_yticks([])
            ax.set_facecolor("#0a0a14")

            if i < n_scales:
                is_coarse = (i == n_detail)
                band      = np.clip(coeffs[i], 0, None)
                vmax      = float(np.nanpercentile(band, 99.5)) if band.max() > 0 else 1e-9
                ax.imshow(band, cmap="seismic", origin="lower",
                          vmin=-vmax, vmax=vmax)
                is_chosen = (not is_coarse) and (i + 1 == chosen)
                label     = "Coarse Scale" if is_coarse else f"Scale {i + 1}"
                ax.set_title(label, color=ACCENT if is_chosen else "white",
                             fontsize=8, pad=8,
                             fontweight="bold" if is_chosen else "normal")
                for sp in ax.spines.values():
                    sp.set_edgecolor(ACCENT if is_chosen else "#333355")
                    sp.set_linewidth(1.5 if is_chosen else 0.5)
            else:
                ax.set_visible(False)

        self._mpl_canvas.draw()
        self._ch_lbl.configure(
            text=f"Channel {ch}  ({idx + 1} / {len(self._channels)})")


# ---------------------------------------------------------------------------
# Helpers for analysis windows
# ---------------------------------------------------------------------------

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
    cmap = plt.get_cmap("tab10")
    return {s["id"]: cmap(i % 10) for i, s in enumerate(sources)}


# ---------------------------------------------------------------------------
# Combined analysis — full-field moments + spectra with toggleable curves
# ---------------------------------------------------------------------------

class CombinedAnalysisWindow(tk.Toplevel):
    def __init__(self, master, cube, tracks, sources, vel_array=None):
        super().__init__(master)
        self.title("Combined Analysis — Full-Field Moments & Spectra")
        self.configure(bg=BG)
        self.geometry("820x820")
        self.resizable(True, True)

        self._cube      = cube
        self._tracks    = tracks
        self._sources   = sources
        self._vel_array = (np.asarray(vel_array, dtype=np.float64)
                           if vel_array is not None else None)
        # spectral axis to use for moment-1 + spectrum x
        self._has_vel  = self._vel_array is not None
        self._sp_axis  = self._vel_array if self._has_vel \
                         else np.arange(cube.shape[0], dtype=np.float64)
        self._sp_label = r"Velocity (km s$^{-1}$)" if self._has_vel else "Channel"
        self._m1_unit  = r"km s$^{-1}$" if self._has_vel else "channel"

        H, W = cube.shape[1], cube.shape[2]
        self._H, self._W = H, W

        self._src_union, self._all_union, self._det_chs_all = \
            _build_source_unions(tracks, sources, H, W)
        self._src_color = _source_colors(sources)

        # Pre-compute spectra
        self._total_spec = cube.sum(axis=(1, 2))
        self._src_spec   = {
            s["id"]: cube[:, self._src_union[s["id"]]].sum(axis=1)
                     if self._src_union[s["id"]].any() else np.zeros(cube.shape[0])
            for s in sources
        }
        src_sum = np.zeros(cube.shape[0])
        for spec in self._src_spec.values():
            src_sum += spec
        self._diffuse_spec = self._total_spec - src_sum

        # Pre-compute moment maps
        det_idx = np.array(self._det_chs_all) if self._det_chs_all \
                  else np.arange(cube.shape[0])
        self._mom0 = cube[det_idx].sum(axis=0)
        flux_stack = cube[det_idx]
        total_flux = flux_stack.sum(axis=0)
        ch_axis = self._sp_axis[det_idx].astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            self._mom1 = np.where(
                (total_flux > 0) & self._all_union,
                (flux_stack * ch_axis[:, None, None]).sum(axis=0) / total_flux,
                np.nan,
            )

        # Side panel with checkboxes (right) + figure (left)
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)

        fig_frame = tk.Frame(body, bg=BG)
        fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._fig = plt.Figure(figsize=(6.5, 7.5), dpi=96, facecolor="#0a0a14")
        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        cb_panel = tk.Frame(body, bg=BG, width=170)
        cb_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=8)
        cb_panel.pack_propagate(False)
        tk.Label(cb_panel, text="Spectra", bg=BG, fg=ACCENT,
                 font=("Helvetica", 10, "bold")).pack(pady=(2, 6), anchor="w")

        self._show_total = tk.BooleanVar(value=True)
        self._show_diff  = tk.BooleanVar(value=True)
        self._show_src: dict[int, tk.BooleanVar] = {}

        # Total (ground truth) — always on, disabled checkbox
        row = tk.Frame(cb_panel, bg=BG); row.pack(fill=tk.X, pady=1, anchor="w")
        tk.Label(row, text="—", bg=BG, fg="white",
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(row, text="Total (always on)",
                       variable=self._show_total, state=tk.DISABLED,
                       bg=BG, fg="white", selectcolor=CARD_BG,
                       disabledforeground="white",
                       activebackground=BG, font=("Helvetica", 8),
                       relief=tk.FLAT, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Diffuse
        row = tk.Frame(cb_panel, bg=BG); row.pack(fill=tk.X, pady=1, anchor="w")
        tk.Label(row, text="■", bg=BG, fg="#8b4513",
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(row, text="Diffuse",
                       variable=self._show_diff, command=self._draw,
                       bg=BG, fg="white", selectcolor=CARD_BG,
                       activebackground=BG, activeforeground=ACCENT,
                       font=("Helvetica", 8), relief=tk.FLAT,
                       anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Per-source
        for s in sources:
            sid = s["id"]
            col = self._src_color[sid]
            hexc = "#{:02x}{:02x}{:02x}".format(
                int(col[0]*255), int(col[1]*255), int(col[2]*255))
            var = tk.BooleanVar(value=True)
            self._show_src[sid] = var
            row = tk.Frame(cb_panel, bg=BG); row.pack(fill=tk.X, pady=1, anchor="w")
            tk.Label(row, text="■", bg=BG, fg=hexc,
                     font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
            tk.Checkbutton(row, text=f"Source {sid}",
                           variable=var, command=self._draw,
                           bg=BG, fg="white", selectcolor=CARD_BG,
                           activebackground=BG, activeforeground=ACCENT,
                           font=("Helvetica", 8), relief=tk.FLAT,
                           anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._draw()

    def _draw(self):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        self._fig.clear()
        gs = self._fig.add_gridspec(2, 2, height_ratios=[1, 1.15],
                                    hspace=0.30, wspace=0.06,
                                    left=0.07, right=0.97, top=0.92, bottom=0.08)
        ax_m0 = self._fig.add_subplot(gs[0, 0])
        ax_m1 = self._fig.add_subplot(gs[0, 1])
        ax_sp = self._fig.add_subplot(gs[1, :])
        for ax in (ax_m0, ax_m1):
            ax.set_facecolor("#0a0a14")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_edgecolor("#555577"); sp.set_linewidth(0.5)
        ax_sp.set_facecolor("#0a0a14")
        for sp in ax_sp.spines.values():
            sp.set_edgecolor("#555577")

        # --- Moment 0 ---
        v0, v1 = np.nanpercentile(self._mom0, [1, 99])
        im0 = ax_m0.imshow(self._mom0, cmap="inferno", origin="lower",
                           vmin=v0, vmax=v1)
        rgba = np.zeros((self._H, self._W, 4), dtype=np.float32)
        rgba[~self._all_union] = [0, 0, 0, 0.55]
        for s in self._sources:
            sid = s["id"]
            if not self._show_src[sid].get():
                continue
            m = self._src_union[sid]
            if not m.any():
                continue
            r, g, b, _ = self._src_color[sid]
            rgba[m] = [r, g, b, 0.45]
        ax_m0.imshow(rgba, origin="lower", interpolation="nearest")
        ax_m0.text(0.04, 0.96, "Moment 0", transform=ax_m0.transAxes,
                   va="top", ha="left", fontsize=9, color="white",
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25", fc="black",
                             alpha=0.5, ec="none"))
        # Colorbar exactly the width of the image, mounted on top
        cax0 = make_axes_locatable(ax_m0).append_axes(
            "top", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb0 = self._fig.colorbar(im0, cax=cax0, orientation="horizontal")
        cax0.xaxis.set_ticks_position("top")
        cax0.xaxis.set_label_position("top")
        cb0.ax.tick_params(colors="white", labelsize=7, direction="out")
        cb0.outline.set_edgecolor("#555577")
        cb0.set_label(r"Jy beam$^{-1}$", fontsize=8, color="white", labelpad=4)

        # --- Moment 1 ---
        ax_m1.set_facecolor("#222236")
        m1_show = np.where(self._all_union, self._mom1, np.nan)
        if np.any(~np.isnan(m1_show)):
            vmax = float(np.nanmax(np.abs(m1_show)))
        else:
            vmax = 1.0
        im1 = ax_m1.imshow(m1_show, cmap="RdBu_r", origin="lower",
                           vmin=-vmax, vmax=vmax)
        ax_m1.text(0.04, 0.96, "Moment 1", transform=ax_m1.transAxes,
                   va="top", ha="left", fontsize=9, color="white",
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25", fc="black",
                             alpha=0.5, ec="none"))
        cax1 = make_axes_locatable(ax_m1).append_axes(
            "top", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb1 = self._fig.colorbar(im1, cax=cax1, orientation="horizontal")
        cax1.xaxis.set_ticks_position("top")
        cax1.xaxis.set_label_position("top")
        cb1.ax.tick_params(colors="white", labelsize=7, direction="out")
        cb1.outline.set_edgecolor("#555577")
        cb1.set_label(self._m1_unit, fontsize=8, color="white", labelpad=4)

        # --- Spectra ---
        xs = self._sp_axis
        ax_sp.plot(xs, self._total_spec,
                   color="0.7", lw=1, ls="--", label="Total")
        if self._show_diff.get():
            ax_sp.plot(xs, self._diffuse_spec,
                       color="#cd853f", lw=1, ls=":", label="Diffuse")
        for s in self._sources:
            sid = s["id"]
            if not self._show_src[sid].get():
                continue
            ax_sp.plot(xs, self._src_spec[sid],
                       color=self._src_color[sid], lw=1.2,
                       label=f"Source {sid}")
        ax_sp.set_xlabel(self._sp_label, color="white", fontsize=9)
        ax_sp.set_ylabel(r"Integrated flux (Jy beam$^{-1}$)",
                         color="white", fontsize=9)
        ax_sp.tick_params(colors="white", labelsize=8)
        leg = ax_sp.legend(fontsize=7, facecolor="#16213e",
                           edgecolor="#555577", labelcolor="white")
        for txt in leg.get_texts():
            txt.set_color("white")

        self._canvas.draw()


# ---------------------------------------------------------------------------
# Individual source analysis — per-source moments + spectrum, radio cycle
# ---------------------------------------------------------------------------

class IndividualAnalysisWindow(tk.Toplevel):
    PAD = 8

    def __init__(self, master, cube, tracks, sources, vel_array=None):
        super().__init__(master)
        self.title("Individual Source Analysis")
        self.configure(bg=BG)
        self.geometry("760x800")
        self.resizable(True, True)

        self._cube      = cube
        self._tracks    = tracks
        self._sources   = sources
        self._tracks_by_id = {t["id"]: t for t in tracks}
        self._src_color = _source_colors(sources)
        self._vel_array = (np.asarray(vel_array, dtype=np.float64)
                           if vel_array is not None else None)
        self._has_vel   = self._vel_array is not None
        self._sp_axis   = self._vel_array if self._has_vel \
                          else np.arange(cube.shape[0], dtype=np.float64)
        self._sp_label  = r"Velocity (km s$^{-1}$)" if self._has_vel else "Channel"
        self._m1_unit   = r"km s$^{-1}$" if self._has_vel else "channel"

        self._selected = tk.IntVar(value=sources[0]["id"] if sources else 0)

        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)

        fig_frame = tk.Frame(body, bg=BG)
        fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._fig = plt.Figure(figsize=(6.5, 7.5), dpi=96, facecolor="#0a0a14")
        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        rb_panel = tk.Frame(body, bg=BG, width=160)
        rb_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=8)
        rb_panel.pack_propagate(False)
        tk.Label(rb_panel, text="Choose source", bg=BG, fg=ACCENT,
                 font=("Helvetica", 10, "bold")).pack(pady=(2, 6), anchor="w")
        for s in sources:
            sid = s["id"]
            col = self._src_color[sid]
            hexc = "#{:02x}{:02x}{:02x}".format(
                int(col[0]*255), int(col[1]*255), int(col[2]*255))
            row = tk.Frame(rb_panel, bg=BG); row.pack(fill=tk.X, pady=1, anchor="w")
            tk.Label(row, text="■", bg=BG, fg=hexc,
                     font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
            tk.Radiobutton(row, text=f"Source {sid}",
                           variable=self._selected, value=sid,
                           command=self._draw,
                           bg=BG, fg="white", selectcolor=CARD_BG,
                           activebackground=BG, activeforeground=ACCENT,
                           font=("Helvetica", 9), relief=tk.FLAT,
                           anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        if sources:
            self._draw()

    def _draw(self):
        sid = int(self._selected.get())
        src = next((s for s in self._sources if s["id"] == sid), None)
        if src is None:
            return

        H, W = self._cube.shape[1], self._cube.shape[2]
        union = np.zeros((H, W), dtype=bool)
        ch_to_mask: dict[int, np.ndarray] = {}
        for tid in src["track_ids"]:
            t = self._tracks_by_id.get(tid)
            if not t:
                continue
            for ch, mask in t["masks"].items():
                union |= mask
                ch_to_mask[ch] = ch_to_mask.get(ch, np.zeros((H, W), dtype=bool)) | mask
        if not union.any():
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.set_facecolor("#0a0a14")
            ax.text(0.5, 0.5, "(no footprint)", color="white",
                    ha="center", va="center", transform=ax.transAxes)
            self._canvas.draw()
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

        det_idx = np.array(det_chs)
        footprint = union[y0:y1, x0:x1]
        mom0_crop = self._cube[det_idx].sum(axis=0)[y0:y1, x0:x1]
        flux_crop = self._cube[det_idx][:, y0:y1, x0:x1]
        total_flux = flux_crop.sum(axis=0)
        ch_axis = self._sp_axis[det_idx].astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            mom1_show = np.where(
                (total_flux > 0) & footprint,
                (flux_crop * ch_axis[:, None, None]).sum(axis=0) / total_flux,
                np.nan,
            )

        spec_chs = np.arange(self._cube.shape[0])
        spec_flux = np.array([
            float(self._cube[ch][ch_to_mask[ch]].sum()) if ch in ch_to_mask else 0.0
            for ch in spec_chs
        ])

        color = self._src_color[sid]
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        self._fig.clear()
        gs = self._fig.add_gridspec(2, 2, height_ratios=[1, 1.15],
                                    hspace=0.30, wspace=0.06,
                                    left=0.07, right=0.97, top=0.90, bottom=0.08)
        ax_m0 = self._fig.add_subplot(gs[0, 0])
        ax_m1 = self._fig.add_subplot(gs[0, 1])
        ax_sp = self._fig.add_subplot(gs[1, :])
        for ax in (ax_m0, ax_m1):
            ax.set_facecolor("#0a0a14")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_edgecolor("#555577"); sp.set_linewidth(0.5)

        # --- Moment 0 ---
        ov0 = np.zeros(footprint.shape + (4,), dtype=np.float32)
        ov0[~footprint] = [0, 0, 0, 0.6]
        im0 = ax_m0.imshow(mom0_crop, cmap="inferno", origin="lower")
        ax_m0.imshow(ov0, origin="lower", interpolation="nearest")
        ax_m0.contour(footprint.astype(float), [0.5],
                      colors=[color], linewidths=1.2)
        ax_m0.text(0.04, 0.96, "Moment 0", transform=ax_m0.transAxes,
                   va="top", ha="left", fontsize=9, color="white",
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25", fc="black",
                             alpha=0.5, ec="none"))
        cax0 = make_axes_locatable(ax_m0).append_axes(
            "top", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb0 = self._fig.colorbar(im0, cax=cax0, orientation="horizontal")
        cax0.xaxis.set_ticks_position("top")
        cax0.xaxis.set_label_position("top")
        cb0.ax.tick_params(colors="white", labelsize=7, direction="out")
        cb0.outline.set_edgecolor("#555577")
        cb0.set_label(r"Jy beam$^{-1}$", fontsize=8, color="white", labelpad=4)

        # --- Moment 1 ---
        ax_m1.set_facecolor("#222236")
        if np.any(~np.isnan(mom1_show)):
            vmax = float(np.nanmax(np.abs(mom1_show)))
        else:
            vmax = 1.0
        im1 = ax_m1.imshow(mom1_show, cmap="RdBu_r", origin="lower",
                           vmin=-vmax, vmax=vmax)
        ax_m1.contour(footprint.astype(float), [0.5],
                      colors="white", linewidths=0.8, alpha=0.7)
        ax_m1.text(0.04, 0.96, "Moment 1", transform=ax_m1.transAxes,
                   va="top", ha="left", fontsize=9, color="white",
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25", fc="black",
                             alpha=0.5, ec="none"))
        cax1 = make_axes_locatable(ax_m1).append_axes(
            "top", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb1 = self._fig.colorbar(im1, cax=cax1, orientation="horizontal")
        cax1.xaxis.set_ticks_position("top")
        cax1.xaxis.set_label_position("top")
        cb1.ax.tick_params(colors="white", labelsize=7, direction="out")
        cb1.outline.set_edgecolor("#555577")
        cb1.set_label(self._m1_unit, fontsize=8, color="white", labelpad=4)

        # --- Spectrum ---
        ax_sp.set_facecolor("#0a0a14")
        for sp in ax_sp.spines.values():
            sp.set_edgecolor("#555577")
        ax_sp.plot(self._sp_axis, spec_flux, color=color, lw=1.5)
        if det_chs:
            v0_det = float(self._sp_axis[det_chs[0]])
            v1_det = float(self._sp_axis[det_chs[-1]])
            ax_sp.axvspan(min(v0_det, v1_det), max(v0_det, v1_det),
                          color=color, alpha=0.12,
                          label=f"Detected: [{v0_det:.1f}, {v1_det:.1f}] {self._m1_unit}")
        ax_sp.set_xlabel(self._sp_label, color="white", fontsize=9)
        ax_sp.set_ylabel(r"Integrated flux (Jy beam$^{-1}$)",
                         color="white", fontsize=9)
        ax_sp.tick_params(colors="white", labelsize=8)
        leg = ax_sp.legend(fontsize=8, facecolor="#16213e",
                           edgecolor="#555577", labelcolor="white")
        for txt in leg.get_texts():
            txt.set_color("white")

        self._fig.suptitle(f"Source {sid}  ·  pixel ({cx}, {cy})",
                           color="white", fontsize=10)
        self._canvas.draw()


# ---------------------------------------------------------------------------
# CubeCard
# ---------------------------------------------------------------------------

BTN_TALL = 54   # taller button height for multi-line labels


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
# Pipeline log window + stdout bridge
# ---------------------------------------------------------------------------

class _QueueStream:
    """File-like object that forwards write() calls into a queue."""
    def __init__(self, q: queue.Queue):
        self._q = q
    def write(self, text: str):
        if text:
            self._q.put(("log", text))
    def flush(self):
        pass


# ---------------------------------------------------------------------------
# CubeCard
# ---------------------------------------------------------------------------

class CubeCard(tk.Frame):
    def __init__(self, master, index: int, name: str, description: str,
                 app=None, on_loaded=None, **kw):
        active = index == 0
        super().__init__(master, bg=CARD_BG if active else CARD_OFF,
                         bd=0, highlightthickness=1,
                         highlightbackground=ACCENT if active else DIM, **kw)
        self.index        = index
        self.name         = name
        self.description  = description
        self.enabled      = active
        self._app         = app
        self.cube_raw     = None     # original cube (card 0 only)
        self.cube         = None     # scaled cube (used everywhere downstream)
        self.vel_array    = None     # per-channel velocities (None if unknown)
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
        self._log_lines: list[str] = []   # accumulated log text
        self._preview_state = "placeholder"  # "placeholder" | "logs" | "figure"
        self._has_figure    = False
        # Coordinated GIF state (new system)
        self._gif_frames_by_ch: dict = {}     # {channel: PIL.Image}
        self._gif_tk_by_ch:     dict = {}     # {channel: ImageTk.PhotoImage}
        self._gif_canvas             = None   # tk.Canvas when actively showing
        self._gif_last_ch            = None

        bg = CARD_BG if active else CARD_OFF

        # preview square
        self._preview_frame = tk.Frame(self, bg=bg, width=CARD_W, height=CARD_H)
        self._preview_frame.pack_propagate(False)
        self._preview_frame.pack(padx=8, pady=(8, 4))
        self._draw_placeholder()

        # Toggle button — lives inside the preview frame, always on top
        self._toggle_btn = tk.Label(
            self._preview_frame, text="Show Logs",
            bg="#1a1a2e", fg=ACCENT,
            font=("Helvetica", 7, "bold"), cursor="pointinghand",
            relief=tk.FLAT, padx=4, pady=2,
        )
        self._toggle_btn.bind("<Button-1>", lambda _e: self._toggle_view())
        # Hidden until both a figure and logs are available

        # step label
        self._step_label = tk.Label(self, text=name, bg=bg,
                                    fg=ACCENT if active else DIM_TXT,
                                    font=("Helvetica", 9, "bold"))
        self._step_label.pack(pady=(0, 4))

        # button zone — fixed height so all cards are the same total size
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
        _span = BTN_W * 2 + 6   # full width matching two side-by-side buttons + gap

        row1 = tk.Frame(self._btn_zone, bg=bg)
        row1.pack(pady=(4, 3))
        self.btn_configure = _FlatBtn(row1, "Configure & Run Decomposition",
                                      self._open_configure, bg_on=ACCENT, active=active,
                                      height=BTN_TALL, btn_width=_span)
        self.btn_configure.pack(padx=3)
        # alias for code paths that previously toggled btn_decompose
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
    # Placeholder / preview
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # In-square log display (shown while pipeline is running)
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Log / figure toggle
    # ------------------------------------------------------------------ #

    def _show_logs(self):
        """Display accumulated log text in the preview square."""
        self._clear_preview()
        self._log_widget = tk.Text(
            self._preview_frame,
            bg="#0a0a14", fg=ACCENT,
            font=("Courier", 9), wrap=tk.WORD,
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
        """Restore the rendered figure in the preview square."""
        if not self._has_figure:
            return
        if self._gif_frames_by_ch:
            # GIF cards — install canvas and let master clock drive frames
            self._install_gif_canvas()
            self._preview_state = "figure"
            # Show current master frame immediately
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
        """Update toggle button text and bring it to the top of the z-order."""
        self._toggle_btn.configure(text=label)
        self._toggle_btn.place(
            x=CARD_W - 4, y=CARD_H - 4,
            anchor="se",
        )
        self._toggle_btn.lift()

    def _hide_toggle(self):
        self._toggle_btn.place_forget()

    # ------------------------------------------------------------------ #

    def _init_log_preview(self):
        """Switch the preview square to live log display, clearing stale figure."""
        self._log_lines     = []
        self._has_figure    = False
        self._cached_fig_pil = None
        self._show_logs()   # toggle visibility handled inside based on _has_figure

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

    # ------------------------------------------------------------------ #
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
        canvas.get_tk_widget().place(x=0, y=0, width=CARD_W, height=CARD_H)

        # Cache as a PIL image so toggling back is instant
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
        # Keep toggle on top if both views are available
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
        """Per-channel image with white footprint contours + source bboxes."""
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
        """Per-channel image with quivers + faint wavelet contours."""
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
        """Per-channel image with per-source contours + bboxes + numbered labels."""
        from matplotlib.patches import Rectangle as _Rect

        norm = self._cube_norm(cube)
        dpi  = 72;  fsz = CARD_W / dpi

        # Build per-source per-channel mask lookup
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

            # union dim overlay
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
        """Show a mini scale-grid for the middle active channel in the card square."""
        from .detect import starlet_transform, active_channels
        n_scales = int(wav_p.get("scales", 6))
        ch_list  = active_channels(cube)
        mid_ch   = ch_list[len(ch_list) // 2]
        img      = cube[mid_ch].astype(np.float32)
        coeffs   = starlet_transform(img, scales=n_scales)   # (n_scales, H, W)

        n_panels = n_scales           # detail bands + coarse
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
        canvas.get_tk_widget().place(x=0, y=0, width=CARD_W, height=CARD_H)
        plt.close(fig)

    def _animate_gif(self):
        if not self._gif_tk_frames:
            return
        self._gif_canvas.delete("all")
        self._gif_canvas.create_image(0, 0, anchor="nw",
                                      image=self._gif_tk_frames[self._gif_idx])
        self._gif_idx = (self._gif_idx + 1) % len(self._gif_tk_frames)
        self._gif_job = self.after(200, self._animate_gif)

    # ------------------------------------------------------------------ #
    # Coordinated GIF — frames indexed by channel, driven by NemoGUI clock
    # ------------------------------------------------------------------ #

    def _install_gif_canvas(self):
        """Create the tk.Canvas that will display the per-channel GIF frames."""
        self._clear_preview()
        self._gif_canvas = tk.Canvas(self._preview_frame, width=CARD_W, height=CARD_H,
                                     bg="#0a0a14", highlightthickness=0)
        self._gif_canvas.place(x=0, y=0, width=CARD_W, height=CARD_H)
        self._gif_last_ch = None

    def show_gif_for_channel(self, ch: int):
        """Display the cached frame for *ch* on the GIF canvas, if visible."""
        if self._preview_state != "figure":
            return
        if not self._gif_frames_by_ch:
            return
        if self._gif_canvas is None or not self._gif_canvas.winfo_exists():
            self._install_gif_canvas()
        if ch == self._gif_last_ch:
            return   # already displayed
        if ch not in self._gif_frames_by_ch:
            return
        from PIL import ImageTk
        if ch not in self._gif_tk_by_ch:
            self._gif_tk_by_ch[ch] = ImageTk.PhotoImage(self._gif_frames_by_ch[ch])
        self._gif_canvas.delete("all")
        self._gif_canvas.create_image(0, 0, anchor="nw",
                                      image=self._gif_tk_by_ch[ch])
        self._gif_last_ch = ch
        # Keep toggle button on top of the new canvas
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
        # allow flow params to be configured immediately after cube load
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
        """Drain the pipeline queue every 40 ms in the main thread."""
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
        """Open ScaleViewer (which embeds all wavelet parameters)."""
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
        """Run wavelet detection only and update card 1 preview."""
        cube = self._app.cards[0].cube if self._app else None
        if cube is None:
            messagebox.showwarning("No cube", "Load a cube in the Moment 0 step first.")
            return
        wav_p = self._wav_params or dict(
            scales=6, k_sigma=5.0, use_scale=5,
            min_area=20, thresh=None, use_mean_map_sigma=True,
        )
        # Fast — just run the starlet transform on one channel; no detection
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
        """Run full source identification: wavelet → flow → source grouping."""
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
                from .detect import WaveletDetector, active_channels
                from .track  import compute_flow_sequence, link_tracks, _reconcile_splits
                from .track  import group_into_sources

                # ---- Wavelet detection (logs → card 1) ----
                ch_list = active_channels(cube)
                det = WaveletDetector(**wav_p).detect(cube, channel_list=ch_list, verbose=True)
                # Render wavelet frames in worker thread (off main thread)
                wav_frames = _build_wavelet_frames(cube, det)
                q.put(("detection_done", det, wav_p, wav_frames))

                # ---- Optical flow computation (logs → card 2) ----
                q.put(("switch_card", 2))
                flow_seq   = compute_flow_sequence(det, verbose=True)
                flow_frames = _build_flow_frames(cube, flow_seq, det)
                q.put(("flow_done", flow_seq, flow_frames))

                # ---- Track linking + source grouping (logs → card 3) ----
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
    # Enable (unlock card)
    # ------------------------------------------------------------------ #

    def enable(self):
        if self.enabled:
            return
        self.enabled = True
        bg = CARD_BG
        self.configure(bg=bg, highlightbackground=ACCENT)
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
            pass   # buttons activate only after tracking completes
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
            # Keep the loaded cube — just wipe pipeline results
            if self.cube is not None:
                self._render_moment0()          # re-render clean moment-0
                self._has_figure = True
                self._preview_state = "figure"
                # view/spectrum buttons stay active
            else:
                self._draw_placeholder()
        else:
            # Cards 1+ go back to greyed-out state
            self.enabled = False
            bg = CARD_OFF
            self.configure(bg=bg, highlightbackground=DIM)
            self._preview_frame.configure(bg=bg)
            self._btn_zone.configure(bg=bg)
            self._draw_placeholder()
            self._step_label.configure(fg=DIM_TXT, bg=bg)
            for child in self._btn_zone.winfo_children():
                if isinstance(child, tk.Frame) and not isinstance(child, _FlatBtn):
                    child.configure(bg=bg)
            # disable all buttons on this card
            for attr in ("btn_decompose", "btn_configure", "btn_det_view",
                         "btn_flow_params", "btn_flow_view",
                         "btn_view_sources", "btn_combined", "btn_individual",
                         "btn_run"):
                if hasattr(self, attr):
                    getattr(self, attr).disable()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class NemoGUI(tk.Tk):
    N_CARDS = 4
    GIF_INTERVAL_MS = 220

    def __init__(self):
        super().__init__()
        self.title("N.E.M.O : Graphical Interface")
        self.configure(bg=BG)
        self.resizable(False, False)
        try:
            from PIL import Image, ImageTk
            _ico = Image.open(_ASSETS / "nemo_ico.png").convert("RGBA")
            self._icon_img = ImageTk.PhotoImage(_ico)
            self.wm_iconphoto(True, self._icon_img)
        except Exception:
            pass

        # Coordinated GIF clock — drives all cards' GIF frames together
        self._gif_master_chs: list[int] = []
        self._gif_master_idx = 0
        self._gif_job = None

        _STEPS = [
            ("Moment 0",
             "Moment 0"),
            ("Wavelet Detections",
             "This step computes the per-spectral channel "
             "unidentified detections of emissions with "
             "2D Starlet transform and thresholding "
             "in wavelet space."),
            ("Flow Tracking",
             "TV-L1 masked optical flow is computed between "
             "consecutive channel pairs and used to link "
             "source detections across channels via "
             "Hungarian assignment."),
            ("Source Grouping",
             "Tracks are linked across spectral channels via "
             "Hungarian overlap assignment and grouped into "
             "physical sources via union-find on split and "
             "merge relationships."),
        ]

        grid = tk.Frame(self, bg=BG)
        self.cards: list[CubeCard] = []
        for i in range(self.N_CARDS):
            name, desc = _STEPS[i]
            card = CubeCard(grid, index=i, name=name, description=desc,
                            app=self, on_loaded=self._on_card_loaded)
            card.grid(row=0, column=i, padx=6, pady=6)
            self.cards.append(card)

        self.update_idletasks()
        content_h = grid.winfo_reqheight()

        banner = tk.Frame(self, bg=BG, height=content_h)
        banner.pack_propagate(False)
        banner_w = self._build_banner(banner, content_h)
        banner.configure(width=banner_w)
        banner.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)

        tk.Frame(self, bg=DIM, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=10)
        grid.pack(side=tk.LEFT, padx=10, pady=10)

        # Force the natural content size and position centred on screen.
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    # --- Banner layout knobs ------------------------------------------------
    # >>> ADJUST THIS to move the GitHub / Docs row up or down on the banner.
    #     Value is a fraction of the total banner height (0.0 = top, 1.0 = bottom).
    BANNER_LINKS_TOP_FRAC = 0.24   # GitHub / Docs row vertical offset
    def _build_banner(self, frame: tk.Frame, height: int) -> int:
        import webbrowser
        p = _ASSETS / "nemo_vertical.png"
        try:
            from PIL import Image, ImageTk
            # Pillow ≥ 9.1 prefers Image.Resampling.LANCZOS; fall back if older
            try:
                _LANCZOS = Image.Resampling.LANCZOS
            except AttributeError:
                _LANCZOS = Image.LANCZOS

            img = Image.open(p)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")

            # On-screen logical size
            disp_h = height
            disp_w = max(1, int(img.width * disp_h / img.height))

            # Progressive halving with LANCZOS — much crisper than one big jump.
            src_w, src_h = img.size
            while src_h > disp_h * 2:
                img = img.resize((src_w // 2, src_h // 2), _LANCZOS)
                src_w, src_h = img.size
            img = img.resize((disp_w, disp_h), _LANCZOS)

            # Composite over the dark background to avoid alpha fringes
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, BG)
                bg.paste(img, mask=img.split()[3])
                img = bg

            self._banner_img = ImageTk.PhotoImage(img)
            new_w = disp_w
            tk.Label(frame, image=self._banner_img, bg=BG, bd=0,
                     anchor="n").place(x=0, y=0, width=disp_w, height=disp_h)
        except Exception:
            new_w = BANNER_W
            tk.Label(frame, text="N\nE\nM\nO", bg=BG, fg=ACCENT,
                     font=("Helvetica", 16, "bold")).place(relx=0.5, rely=0.5, anchor="center")

        btn_w = new_w - 8
        btn_h = BTN_H
        gap   = 6
        cx    = new_w // 2

        def _mklink(label, url, x, y, w):
            lbl = tk.Label(
                frame, text=label, bg=BG, fg=ACCENT,
                font=("Helvetica", 10),
                cursor="pointinghand", relief=tk.FLAT,
            )
            lbl.place(x=x, y=y, anchor="n", width=w, height=btn_h)
            lbl.bind("<Button-1>", lambda _e: webbrowser.open(url))
            lbl.bind("<Enter>",    lambda _e: lbl.configure(bg=CARD_BG))
            lbl.bind("<Leave>",    lambda _e: lbl.configure(bg=BG))
            lbl.lift()
            return lbl

        # ---- External links (top section, side-by-side) ----
        # Vertical position is controlled by self.BANNER_LINKS_TOP_FRAC (above)
        y_links  = int(height * self.BANNER_LINKS_TOP_FRAC)
        half_w   = (btn_w - gap) // 2
        x_left   = 4 + half_w // 2                 # centre of left half
        x_right  = 4 + half_w + gap + half_w // 2  # centre of right half
        _mklink("GitHub", "https://github.com/arnablahiry/NEMO-Source-Tracker",
                x_left,  y_links, half_w)
        _mklink("Docs",   "https://arnablahiry.github.io/software/nemo",
                x_right, y_links, half_w)

        # ---- Bottom: View All Logs + Reset ----
        y_rst  = height - btn_h - 6
        y_log  = y_rst  - btn_h - gap

        logs_btn = tk.Label(
            frame, text="View All Logs", bg=BG, fg=ACCENT,
            font=("Helvetica", 10), cursor="pointinghand", relief=tk.FLAT,
        )
        logs_btn.place(x=cx, y=y_log, anchor="n", width=btn_w, height=btn_h)
        logs_btn.bind("<Button-1>", lambda _e: self._view_all_logs())
        logs_btn.bind("<Enter>",    lambda _e: logs_btn.configure(bg=CARD_BG))
        logs_btn.bind("<Leave>",    lambda _e: logs_btn.configure(bg=BG))

        reset_btn = tk.Label(
            frame, text="Reset", bg=BG, fg=ACCENT,
            font=("Helvetica", 10, "bold"), cursor="pointinghand", relief=tk.FLAT,
        )
        reset_btn.place(x=cx, y=y_rst, anchor="n", width=btn_w, height=btn_h)
        reset_btn.bind("<Button-1>", lambda _e: self._reset_pipeline())
        reset_btn.bind("<Enter>",    lambda _e: reset_btn.configure(bg=CARD_BG))
        reset_btn.bind("<Leave>",    lambda _e: reset_btn.configure(bg=BG))

        logs_btn.lift()
        reset_btn.lift()
        return new_w

    def _view_all_logs(self):
        all_sections = []
        for card in self.cards:
            if card._log_lines:
                all_sections.append(f"\n── {card.name} ──\n")
                all_sections.extend(card._log_lines)

        if not all_sections:
            messagebox.showinfo("No Logs", "No pipeline logs yet. Run the pipeline first.")
            return

        win = tk.Toplevel(self)
        win.title("All Pipeline Logs")
        win.configure(bg=BG)
        win.geometry("700x500")
        win.resizable(True, True)

        frame = tk.Frame(win, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frame, bg=CARD_BG, fg=ACCENT, font=("Courier", 9),
                      wrap=tk.WORD, state=tk.DISABLED,
                      yscrollcommand=sb.set, relief=tk.FLAT, bd=0)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.configure(command=txt.yview)
        txt.tag_configure("header", foreground="white", font=("Courier", 9, "bold"))

        txt.configure(state=tk.NORMAL)
        for line in all_sections:
            if line.startswith("\n──"):
                txt.insert(tk.END, line, "header")
            else:
                txt.insert(tk.END, line)
        txt.see(tk.END)
        txt.configure(state=tk.DISABLED)

    def _reset_pipeline(self):
        if not messagebox.askyesno("Reset", "Reset all pipeline results?"):
            return
        # stop GIF clock
        if self._gif_job is not None:
            self.after_cancel(self._gif_job)
            self._gif_job = None
        self._gif_master_chs = []
        self._gif_master_idx = 0
        for card in self.cards:
            card.reset()
        # If a cube is still loaded, re-enable downstream cards
        if self.cards[0].cube is not None:
            self.cards[1].enable()
            if len(self.cards) > 2:
                self.cards[2].btn_flow_params.enable()

    # ------------------------------------------------------------------ #
    # Coordinated GIF master clock
    # ------------------------------------------------------------------ #

    def current_gif_channel(self):
        if not self._gif_master_chs:
            return None
        return self._gif_master_chs[self._gif_master_idx]

    def refresh_gif_clock(self):
        """Rebuild the master channel list from all cards' GIF frames."""
        # Union of channels across cards, in sorted order — keeps everyone aligned
        union = set()
        for card in self.cards:
            union.update(card._gif_frames_by_ch.keys())
        new_chs = sorted(union)
        if not new_chs:
            return
        # Preserve current position if possible
        prev_ch = self.current_gif_channel()
        self._gif_master_chs = new_chs
        if prev_ch in new_chs:
            self._gif_master_idx = new_chs.index(prev_ch)
        else:
            self._gif_master_idx = 0
        # Push current frame to every card immediately
        for card in self.cards:
            card.show_gif_for_channel(new_chs[self._gif_master_idx])
        # Start clock if not running
        if self._gif_job is None:
            self._tick_gif()

    def _tick_gif(self):
        if not self._gif_master_chs:
            self._gif_job = None
            return
        ch = self._gif_master_chs[self._gif_master_idx]
        for card in self.cards:
            card.show_gif_for_channel(ch)
        self._gif_master_idx = (self._gif_master_idx + 1) % len(self._gif_master_chs)
        self._gif_job = self.after(self.GIF_INTERVAL_MS, self._tick_gif)

    # ------------------------------------------------------------------ #

    def _on_card_loaded(self, index: int):
        nxt = index + 1
        if nxt < self.N_CARDS:
            self.cards[nxt].enable()


# ---------------------------------------------------------------------------

def launch():
    app = NemoGUI()
    app.mainloop()


if __name__ == "__main__":
    launch()
