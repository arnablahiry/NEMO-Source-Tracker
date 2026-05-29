import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ._constants import ACCENT, BG, CARD_BG


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

        row = tk.Frame(cb_panel, bg=BG); row.pack(fill=tk.X, pady=1, anchor="w")
        tk.Label(row, text="—", bg=BG, fg="white",
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(row, text="Total (always on)",
                       variable=self._show_total, state=tk.DISABLED,
                       bg=BG, fg="white", selectcolor=CARD_BG,
                       disabledforeground="white",
                       activebackground=BG, font=("Helvetica", 8),
                       relief=tk.FLAT, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        row = tk.Frame(cb_panel, bg=BG); row.pack(fill=tk.X, pady=1, anchor="w")
        tk.Label(row, text="■", bg=BG, fg="#8b4513",
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(row, text="Diffuse",
                       variable=self._show_diff, command=self._draw,
                       bg=BG, fg="white", selectcolor=CARD_BG,
                       activebackground=BG, activeforeground=ACCENT,
                       font=("Helvetica", 8), relief=tk.FLAT,
                       anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

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
        cax0 = make_axes_locatable(ax_m0).append_axes(
            "top", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb0 = self._fig.colorbar(im0, cax=cax0, orientation="horizontal")
        cax0.xaxis.set_ticks_position("top")
        cax0.xaxis.set_label_position("top")
        cb0.ax.tick_params(colors="white", labelsize=7, direction="out")
        cb0.outline.set_edgecolor("#555577")
        cb0.set_label(r"Jy beam$^{-1}$", fontsize=8, color="white", labelpad=4)

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
