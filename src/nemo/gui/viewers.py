import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import PowerNorm, LogNorm, Normalize

from ._constants import (ACCENT, BG, CARD_BG, DIM, DIM_TXT,
                         CARD_W, CARD_H, _CMAPS)
from .widgets import _FlatBtn
from .dialogs import ScalingDialog


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
        self._det_by_ch = {d.channel: d for d in self._dets}
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

        cmap_src = plt.get_cmap("tab10")
        self._src_color = {
            s["id"]: cmap_src(i % 10) for i, s in enumerate(self._sources)
        }
        self._src_visible: dict[int, tk.BooleanVar] = {}

        if self._dets and mode != "sources":
            self._channels = [d.channel for d in self._dets]
        else:
            self._channels = list(range(cube.shape[0]))

        VW = 500

        flat     = cube.ravel()
        self._data_min = float(np.nanmin(flat))
        self._data_max = float(np.nanmax(flat))
        self._p1       = max(float(np.nanpercentile(flat, 1)), 0)
        self._p99      = float(np.nanpercentile(flat, 99.5))

        self._fig = plt.Figure(figsize=(VW/96, VW/96), dpi=96, facecolor="#0a0a14")
        self._ax    = self._fig.add_axes([0.01, 0.01, 0.82, 0.98])
        self._ax_cb = self._fig.add_axes([0.86, 0.01, 0.06, 0.98])
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#555577")
            spine.set_linewidth(0.8)
        self._ax_cb.set_facecolor("#0a0a14")
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

        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(fill=tk.X, padx=10, pady=(6, 2))

        self._cmap = tk.StringVar(value="inferno")
        tk.Label(ctrl, text="Colormap:", bg=BG, fg="white",
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 4))
        cm = ttk.Combobox(ctrl, textvariable=self._cmap,
                          values=_CMAPS, width=11, state="readonly")
        cm.pack(side=tk.LEFT, padx=(0, 14))
        cm.bind("<<ComboboxSelected>>", lambda _e: self._draw())

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

        max_ch  = max(self._channels) if self._channels else 0
        n_ch    = len(self._channels)
        _sample = f"Channel {max_ch}  ({n_ch}/{n_ch})  ·  999 det. · 9999 total"
        self._ch_lbl = tk.Label(ctrl, text=_sample, bg=BG, fg=ACCENT,
                                font=("Helvetica", 9, "italic"), anchor="e")
        self._ch_lbl.pack(side=tk.RIGHT, padx=6)
        self._ch_lbl.configure(text="")

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
        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())
        self.maxsize(self.winfo_width(), 9999)

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

        self._fig.canvas.draw()
        pos = self._ax.get_position()
        cb_w = 0.05
        cb_gap = 0.025
        self._ax_cb.set_position([pos.x1 + cb_gap, pos.y0, cb_w, pos.height])

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

            for sid, ch_dict in self._src_masks_by_ch.items():
                if not self._src_visible[sid].get():
                    continue
                masks = ch_dict.get(ch)
                if not masks:
                    continue
                col = self._src_color[sid]
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


class ScaleViewer(tk.Toplevel):
    """Browse per-channel 2D wavelet coefficient maps with embedded parameters.

    Number of scales is chosen via radio buttons (2…max), where max depends on
    the image dimensions.  The figure is always laid out as 2 rows × ceil(n/2)
    cols.  Changing n_scales re-runs the starlet transform and redraws in real
    time.
    """

    N_ROWS = 2

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

        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(fill=tk.X, padx=10, pady=(8, 4))
        self._ch_lbl = tk.Label(ctrl, text="", bg=BG, fg=ACCENT,
                                font=("Helvetica", 9, "italic"))
        self._ch_lbl.pack(side=tk.RIGHT, padx=6)

        self._fig_frame = tk.Frame(self, bg=BG)
        self._fig_frame.pack(fill=tk.BOTH, expand=True)
        self._fig = None;  self._mpl_canvas = None
        self._n_scales_last = 0

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

        self._rf = tk.Frame(self, bg=BG)
        self._rf.pack(fill=tk.X, padx=10, pady=(4, 2))
        self._rebuild_choose_scale_radios()

        pf = tk.LabelFrame(self, text="Detection Parameters", bg=BG, fg=ACCENT,
                           font=("Helvetica", 8, "bold"),
                           relief=tk.FLAT, bd=1, highlightthickness=1,
                           highlightbackground=DIM)
        pf.pack(fill=tk.X, padx=10, pady=(6, 4))

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

    def _on_nscales_changed(self):
        n_new = int(self._n_scales_var.get())
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
                use_mean_map_sigma=True,
            )
        except ValueError as exc:
            messagebox.showerror("Bad parameter", str(exc), parent=self)
            return
        if self._on_params_saved:
            self._on_params_saved(params)
        self.destroy()

    def _rebuild_figure(self):
        n_scales = int(self._n_scales_var.get())
        n_cols   = (n_scales + self.N_ROWS - 1) // self.N_ROWS
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

        from ..detect import starlet_transform
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
