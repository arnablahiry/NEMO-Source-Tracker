import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from . import _constants as C
from .source_tree import SourceTreePanel


def _mpl_theme():
    """Matplotlib colours for the current GUI theme."""
    light = C._current_theme == "light"
    return dict(
        fig_bg="#ffffff" if light else "#0a0a14",
        fg="#1a1a2a" if light else "white",
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


def _shade(base, alpha):
    """Blend an RGB colour toward the current figure background by *alpha*."""
    bg = (1.0, 1.0, 1.0) if C._current_theme == "light" else (0.04, 0.04, 0.08)
    return tuple(c * alpha + b * (1 - alpha) for c, b in zip(base, bg))


class CombinedAnalysisWindow(tk.Toplevel):
    def __init__(self, master, cube, tracks, sources, vel_array=None,
                 hierarchical_sources=None, tracks_per_scale=None):
        super().__init__(master)
        self.title("Combined Analysis — Full-Field Moments & Spectra")
        self.configure(bg=C.BG)
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
        self._hier = bool(hierarchical_sources and tracks_per_scale)

        body = tk.Frame(self, bg=C.BG)
        body.pack(fill=tk.BOTH, expand=True)
        fig_frame = tk.Frame(body, bg=C.BG)
        fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._fig = plt.Figure(figsize=(6.5, 7.5), dpi=96, facecolor=_mpl_theme()["fig_bg"])
        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if self._hier:
            self._tree = SourceTreePanel(body, hierarchical_sources,
                                         tracks_per_scale, on_change=self._draw,
                                         mode="multi", title="Source tree",
                                         default_active_letter="B")
            self._tree.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=8)
            # per-source union masks from the resolved per-scale channel masks
            self._units = {}
            for hid, ch_dict in self._tree.masks_by_ch.items():
                m = np.zeros((H, W), dtype=bool)
                for masks in ch_dict.values():
                    for mm in masks:
                        m |= mm
                self._units[hid] = m
            det_chs = sorted({ch for cd in self._tree.masks_by_ch.values() for ch in cd})
        else:
            self._src_union, _all, self._det_chs_all = \
                _build_source_unions(tracks, sources, H, W)
            self._src_color = _source_colors(sources)
            self._units = self._src_union
            det_chs = self._det_chs_all
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
        # Diffuse = total minus the flux inside the *union* of all source pixels
        # (each pixel counted once).  Summing per-source spectra would multi-
        # count the overlapping/nested hierarchical masks and flip it negative.
        src_flux = (cube[:, self._all_union].sum(axis=1)
                    if self._all_union.any() else np.zeros(cube.shape[0]))
        self._diffuse_spec = self._total_spec - src_flux
        # legacy alias used by the flat moment code below
        self._src_spec = self._spec

        det_idx = np.array(det_chs) if det_chs else np.arange(cube.shape[0])
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

        self._draw()

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
        self._fig.clear()
        gs = self._fig.add_gridspec(2, 2, height_ratios=[1, 1.15],
                                    hspace=0.30, wspace=0.06,
                                    left=0.07, right=0.97, top=0.92, bottom=0.08)
        ax_m0 = self._fig.add_subplot(gs[0, 0])
        ax_m1 = self._fig.add_subplot(gs[0, 1])
        ax_sp = self._fig.add_subplot(gs[1, :])
        for ax in (ax_m0, ax_m1):
            ax.set_facecolor(t["fig_bg"])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_edgecolor(t["spine"]); sp.set_linewidth(0.5)
        ax_sp.set_facecolor(t["fig_bg"])
        for sp in ax_sp.spines.values():
            sp.set_edgecolor(t["spine"])

        # Union of the currently-chosen sources drives both moment maps.
        vis_union = np.zeros((self._H, self._W), dtype=bool)
        vis_items = list(self._iter_visible())
        for _label, _sid, m, _rgb in vis_items:
            vis_union |= m
        if not vis_union.any():
            vis_union = self._all_union

        v0, v1 = np.nanpercentile(self._mom0, [1, 99])
        im0 = ax_m0.imshow(self._mom0, cmap=t["cmap"], origin="lower",
                           vmin=v0, vmax=v1)
        rgba = np.zeros((self._H, self._W, 4), dtype=np.float32)
        rgba[~vis_union] = list(t["dim"])
        for _label, _sid, m, rgb in vis_items:
            if not m.any():
                continue
            r, g, b = rgb[:3]
            rgba[m] = [r, g, b, 0.45]
        ax_m0.imshow(rgba, origin="lower", interpolation="nearest")
        ax_m0.text(0.04, 0.96, "Moment 0", transform=ax_m0.transAxes,
                   va="top", ha="left", fontsize=9, color=t["fg"],
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25", fc=t["bbox_bg"],
                             alpha=0.5, ec="none"))
        cax0 = make_axes_locatable(ax_m0).append_axes(
            "top", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb0 = self._fig.colorbar(im0, cax=cax0, orientation="horizontal")
        cax0.xaxis.set_ticks_position("top")
        cax0.xaxis.set_label_position("top")
        cb0.ax.tick_params(colors=t["fg"], labelsize=7, direction="out")
        cb0.outline.set_edgecolor(t["spine"])
        cb0.set_label(r"Jy beam$^{-1}$", fontsize=8, color=t["fg"], labelpad=4)

        ax_m1.set_facecolor(t["m1_bg"])
        m1_show = np.where(vis_union, self._mom1, np.nan)
        if np.any(~np.isnan(m1_show)):
            vmax = float(np.nanmax(np.abs(m1_show)))
        else:
            vmax = 1.0
        im1 = ax_m1.imshow(m1_show, cmap="RdBu_r", origin="lower",
                           vmin=-vmax, vmax=vmax)
        ax_m1.text(0.04, 0.96, "Moment 1", transform=ax_m1.transAxes,
                   va="top", ha="left", fontsize=9, color=t["fg"],
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25", fc=t["bbox_bg"],
                             alpha=0.5, ec="none"))
        cax1 = make_axes_locatable(ax_m1).append_axes(
            "top", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb1 = self._fig.colorbar(im1, cax=cax1, orientation="horizontal")
        cax1.xaxis.set_ticks_position("top")
        cax1.xaxis.set_label_position("top")
        cb1.ax.tick_params(colors=t["fg"], labelsize=7, direction="out")
        cb1.outline.set_edgecolor(t["spine"])
        cb1.set_label(self._m1_unit, fontsize=8, color=t["fg"], labelpad=4)

        xs = self._sp_axis
        ax_sp.plot(xs, self._total_spec,
                   color=t["total"], lw=1, ls="--", label="Total")
        if getattr(self, "_show_diff", None) is None or self._show_diff.get():
            ax_sp.plot(xs, self._diffuse_spec,
                       color="#cd853f", lw=1, ls=":", label="Diffuse")
        for label, sid, m, rgb in self._iter_visible():
            ax_sp.plot(xs, self._spec[sid], color=rgb, lw=1.2, label=label)
        ax_sp.set_xlabel(self._sp_label, color=t["fg"], fontsize=9)
        ax_sp.set_ylabel(r"Integrated flux (Jy beam$^{-1}$)",
                         color=t["fg"], fontsize=9)
        ax_sp.tick_params(colors=t["fg"], labelsize=8)
        leg = ax_sp.legend(fontsize=7, facecolor=t["legend_bg"],
                           edgecolor=t["spine"], labelcolor=t["fg"])
        for txt in leg.get_texts():
            txt.set_color(t["fg"])

        self._canvas.draw()


class IndividualAnalysisWindow(tk.Toplevel):
    PAD = 8

    def __init__(self, master, cube, tracks, sources, vel_array=None,
                 hierarchical_sources=None, tracks_per_scale=None):
        super().__init__(master)
        self.title("Individual Source Analysis")
        self.configure(bg=C.BG)
        self.geometry("880x800")
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
        self._hier = bool(hierarchical_sources and tracks_per_scale)

        body = tk.Frame(self, bg=C.BG)
        body.pack(fill=tk.BOTH, expand=True)

        fig_frame = tk.Frame(body, bg=C.BG)
        fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._fig = plt.Figure(figsize=(6.5, 7.5), dpi=96, facecolor=_mpl_theme()["fig_bg"])
        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if self._hier:
            self._full_mom0 = cube.sum(axis=0)
            right = tk.Frame(body, bg=C.BG)
            right.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=8)
            self._tree = SourceTreePanel(right, hierarchical_sources,
                                         tracks_per_scale, on_change=self._draw,
                                         mode="single", title="Choose source")
            self._tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            right.configure(width=self._tree.tree_width + 18)
            right.pack_propagate(False)

            # accent-bordered moment-0 thumbnail showing the source's position
            border = tk.Frame(right, bg=C.ACCENT)
            border.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
            inner = tk.Frame(border, bg=C.BG)
            inner.pack(fill=tk.BOTH, padx=2, pady=2)
            tk.Label(inner, text="Position", bg=C.BG, fg=C.STEP_LABEL_TXT,
                     font=("Helvetica", 7)).pack(anchor="w")
            self._thumb_fig = plt.Figure(figsize=(1.3, 1.3), dpi=96,
                                         facecolor=_mpl_theme()["fig_bg"])
            self._thumb_canvas = FigureCanvasTkAgg(self._thumb_fig, master=inner)
            self._thumb_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._draw()
        else:
            self._selected = tk.IntVar(value=sources[0]["id"] if sources else 0)
            rb_panel = tk.Frame(body, bg=C.BG, width=160)
            rb_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=8)
            rb_panel.pack_propagate(False)
            tk.Label(rb_panel, text="Choose source", bg=C.BG, fg=C.ACCENT,
                     font=("Helvetica", 10, "bold")).pack(pady=(2, 6), anchor="w")
            for s in sources:
                sid = s["id"]
                col = self._src_color[sid]
                hexc = "#{:02x}{:02x}{:02x}".format(
                    int(col[0]*255), int(col[1]*255), int(col[2]*255))
                row = tk.Frame(rb_panel, bg=C.BG); row.pack(fill=tk.X, pady=1, anchor="w")
                tk.Label(row, text="■", bg=C.BG, fg=hexc,
                         font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
                tk.Radiobutton(row, text=f"Source {sid}",
                               variable=self._selected, value=sid,
                               command=self._draw,
                               bg=C.BG, fg=C.STEP_LABEL_TXT, selectcolor=C.CARD_BG,
                               activebackground=C.BG, activeforeground=C.ACCENT,
                               font=("Helvetica", 9), relief=tk.FLAT,
                               anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
            if sources:
                self._draw()

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

    def _draw(self):
        t = _mpl_theme()
        H, W = self._cube.shape[1], self._cube.shape[2]
        union = np.zeros((H, W), dtype=bool)
        ch_to_mask: dict[int, np.ndarray] = {}

        if self._hier:
            sid = self._tree.selected_id()
            base = self._tree.stable_color.get(sid, (0.7, 0.7, 0.7))
            color = base  # full tree colour for the selected subtree
            title_name = self._tree.name.get(sid, str(sid))
            # union over the selected source *and all its children*
            for hid in self._tree.descendants(sid):
                for ch, masks in self._tree.masks_by_ch.get(hid, {}).items():
                    for mask in masks:
                        union |= mask
                        ch_to_mask[ch] = ch_to_mask.get(ch, np.zeros((H, W), dtype=bool)) | mask
            self._draw_thumb(union, color)
        else:
            sid = int(self._selected.get())
            src = next((s for s in self._sources if s["id"] == sid), None)
            if src is None:
                return
            color = self._src_color[sid]
            title_name = f"Source {sid}"
            for tid in src["track_ids"]:
                trk = self._tracks_by_id.get(tid)
                if not trk:
                    continue
                for ch, mask in trk["masks"].items():
                    union |= mask
                    ch_to_mask[ch] = ch_to_mask.get(ch, np.zeros((H, W), dtype=bool)) | mask
        if not union.any():
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.set_facecolor(t["fig_bg"])
            ax.text(0.5, 0.5, "(no footprint)", color=t["fg"],
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

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        self._fig.clear()
        gs = self._fig.add_gridspec(2, 2, height_ratios=[1, 1.15],
                                    hspace=0.30, wspace=0.06,
                                    left=0.07, right=0.97, top=0.90, bottom=0.08)
        ax_m0 = self._fig.add_subplot(gs[0, 0])
        ax_m1 = self._fig.add_subplot(gs[0, 1])
        ax_sp = self._fig.add_subplot(gs[1, :])
        for ax in (ax_m0, ax_m1):
            ax.set_facecolor(t["fig_bg"])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_edgecolor(t["spine"]); sp.set_linewidth(0.5)

        ov0 = np.zeros(footprint.shape + (4,), dtype=np.float32)
        ov0[~footprint] = list(t["dim"])
        im0 = ax_m0.imshow(mom0_crop, cmap=t["cmap"], origin="lower")
        ax_m0.imshow(ov0, origin="lower", interpolation="nearest")
        ax_m0.contour(footprint.astype(float), [0.5],
                      colors=[color], linewidths=1.2)
        ax_m0.text(0.04, 0.96, "Moment 0", transform=ax_m0.transAxes,
                   va="top", ha="left", fontsize=9, color=t["fg"],
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25", fc=t["bbox_bg"],
                             alpha=0.5, ec="none"))
        cax0 = make_axes_locatable(ax_m0).append_axes(
            "top", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb0 = self._fig.colorbar(im0, cax=cax0, orientation="horizontal")
        cax0.xaxis.set_ticks_position("top")
        cax0.xaxis.set_label_position("top")
        cb0.ax.tick_params(colors=t["fg"], labelsize=7, direction="out")
        cb0.outline.set_edgecolor(t["spine"])
        cb0.set_label(r"Jy beam$^{-1}$", fontsize=8, color=t["fg"], labelpad=4)

        ax_m1.set_facecolor(t["m1_bg"])
        if np.any(~np.isnan(mom1_show)):
            vmax = float(np.nanmax(np.abs(mom1_show)))
        else:
            vmax = 1.0
        im1 = ax_m1.imshow(mom1_show, cmap="RdBu_r", origin="lower",
                           vmin=-vmax, vmax=vmax)
        ax_m1.contour(footprint.astype(float), [0.5],
                      colors=t["fg"], linewidths=0.8, alpha=0.7)
        ax_m1.text(0.04, 0.96, "Moment 1", transform=ax_m1.transAxes,
                   va="top", ha="left", fontsize=9, color=t["fg"],
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25", fc=t["bbox_bg"],
                             alpha=0.5, ec="none"))
        cax1 = make_axes_locatable(ax_m1).append_axes(
            "top", size="5%", pad=0.05, axes_class=plt.matplotlib.axes.Axes)
        cb1 = self._fig.colorbar(im1, cax=cax1, orientation="horizontal")
        cax1.xaxis.set_ticks_position("top")
        cax1.xaxis.set_label_position("top")
        cb1.ax.tick_params(colors=t["fg"], labelsize=7, direction="out")
        cb1.outline.set_edgecolor(t["spine"])
        cb1.set_label(self._m1_unit, fontsize=8, color=t["fg"], labelpad=4)

        ax_sp.set_facecolor(t["fig_bg"])
        for sp in ax_sp.spines.values():
            sp.set_edgecolor(t["spine"])
        ax_sp.plot(self._sp_axis, spec_flux, color=color, lw=1.5)
        if det_chs:
            v0_det = float(self._sp_axis[det_chs[0]])
            v1_det = float(self._sp_axis[det_chs[-1]])
            ax_sp.axvspan(min(v0_det, v1_det), max(v0_det, v1_det),
                          color=color, alpha=0.12,
                          label=f"Detected: [{v0_det:.1f}, {v1_det:.1f}] {self._m1_unit}")
        ax_sp.set_xlabel(self._sp_label, color=t["fg"], fontsize=9)
        ax_sp.set_ylabel(r"Integrated flux (Jy beam$^{-1}$)",
                         color=t["fg"], fontsize=9)
        ax_sp.tick_params(colors=t["fg"], labelsize=8)
        leg = ax_sp.legend(fontsize=8, facecolor=t["legend_bg"],
                           edgecolor=t["spine"], labelcolor=t["fg"])
        for txt in leg.get_texts():
            txt.set_color(t["fg"])

        self._fig.suptitle(f"{title_name}  ·  pixel ({cx}, {cy})",
                           color=t["fg"], fontsize=10)
        self._canvas.draw()
