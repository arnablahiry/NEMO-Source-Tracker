"""nemo.visualise — interactive channel animation and publication contour figures."""
from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _source_footprints(result, h_sources, H: int, W: int) -> dict[int, np.ndarray]:
    """Build union-footprint mask (H, W) bool for every HierarchicalSourceGroup."""
    masks: dict[int, np.ndarray] = {}
    for h in h_sources:
        m = np.zeros((H, W), dtype=bool)
        tps = (result.tracks_per_scale or {}).get(h.scale, [])
        for t in (tps or result.tracks):
            if t["id"] in h.track_ids:
                for mask in t["masks"].values():
                    m |= mask
        masks[h.id] = m
    return masks


# ---------------------------------------------------------------------------
# interactive — animated channel viewer
# ---------------------------------------------------------------------------

def interactive(
    result,
    cube: np.ndarray,
    vel_array: np.ndarray | None = None,
    beam=None,
    pixscale=None,
    kpc_per_pix=None,
    block: bool = True,
):
    """Open the NEMO source viewer — the same window the GUI shows.

    Launches the GUI :class:`~nemo.gui.viewers.SliceViewer` in ``"sources"``
    mode (per-channel footprints coloured by hierarchical source, with
    normalization controls and overlays), **always in light mode**.

    Parameters
    ----------
    result      : TrackingResult
    cube        : (n_ch, H, W) float32 — raw cube for the background images.
    vel_array   : 1-D velocity axis in km/s (``None`` → channel index only).
    beam        : optional beam tuple/object forwarded to the viewer.
    pixscale    : optional pixel scale forwarded to the viewer.
    kpc_per_pix : optional kpc/pixel forwarded to the viewer.
    block       : run a blocking Tk ``mainloop`` until the window is closed.
                  Set ``False`` when an event loop is already running (e.g. an
                  existing Tk app); the viewer is returned without blocking.

    Returns
    -------
    nemo.gui.viewers.SliceViewer
    """
    import tkinter as tk
    from .gui.viewers import SliceViewer
    from .gui._theme import apply_theme

    if not result.sources and not result.hierarchical_sources:
        raise ValueError(
            "result has no sources — nothing to view.  "
            "Ensure the pipeline completed successfully."
        )

    # Reuse a running Tk root if one exists, else create our own hidden root.
    existing_root = tk._default_root
    if existing_root is not None:
        root = existing_root
        own_root = False
    else:
        root = tk.Tk()
        root.withdraw()
        own_root = True

    # Force light mode (resets the module colour constants the viewer reads).
    apply_theme(root, "light")

    viewer = SliceViewer(
        root, cube,
        tracks=result.tracks,
        sources=result.sources,
        hierarchical_sources=result.hierarchical_sources,
        tracks_per_scale=result.tracks_per_scale,
        mode="sources",
        beam=beam, pixscale=pixscale, kpc_per_pix=kpc_per_pix,
        vel_array=vel_array,
    )

    if own_root and block:
        # Tear down the hidden root once the viewer window is closed.
        viewer.protocol("WM_DELETE_WINDOW",
                         lambda: (viewer.destroy(), root.destroy()))
        root.mainloop()

    return viewer


# ---------------------------------------------------------------------------
# publication — hierarchical contour figure
# ---------------------------------------------------------------------------

def publication(
    result,
    source_info: dict[str, list[int]],
    cube: np.ndarray | None = None,
    vel_array: np.ndarray | None = None,
    bbox: bool = False,
    ax=None,
    figsize: tuple = (6, 6),
    cmap: str = "inferno",
):
    """Publication-quality contour figure for selected hierarchical sources.

    Draws sources as filled contours with opacity and linewidth scaling by
    wavelet scale: coarsest (A) = faint/thin, finest (C) = opaque/thick.
    All members of the same tree share one base colour.

    Parameters
    ----------
    result      : TrackingResult with ``hierarchical_sources`` populated.
    source_info : dict mapping scale letter → list of 1-based indices, e.g.
                  ``{'A': [1, 2], 'B': [1, 2]}`` selects A1, A2, B1, B2.
    cube        : (n_ch, H, W) float32 used for the moment-0 background.
                  ``None`` → black background.
    vel_array   : velocity axis — reserved for future axis labelling.
    bbox        : draw a dashed bounding box around the coarsest-scale
                  (root / A-level) footprint for each source tree, with
                  the source name labelled just outside it.
    ax          : existing ``Axes`` to draw on.  ``None`` → new figure.
    figsize     : figure size when ``ax`` is ``None``.
    cmap        : background image colormap.

    Returns
    -------
    (fig, ax) when ``ax`` is ``None``; just ``ax`` otherwise.
    """
    from .hierarchy import assign_tree_names, assign_tree_colors

    h_sources = result.hierarchical_sources
    if not h_sources:
        raise ValueError(
            "result.hierarchical_sources is empty — run FlowTracker with "
            "detect_all_scales=True to produce hierarchical sources."
        )

    names    = assign_tree_names(h_sources)
    h_colors = assign_tree_colors(h_sources)
    by_name  = {v: k for k, v in names.items()}   # 'A1' → h.id
    by_id    = {h.id: h for h in h_sources}

    # ── spatial dimensions ────────────────────────────────────────────────
    H = W = None
    if cube is not None:
        H, W = cube.shape[1], cube.shape[2]
    elif result.detections:
        H, W = result.detections[0].image.shape
    elif result.detections_per_scale:
        first = next(iter(result.detections_per_scale.values()))
        H, W = first[0].image.shape

    if H is None:
        raise ValueError("Cannot determine spatial dimensions — pass cube=.")

    src_masks = _source_footprints(result, h_sources, H, W)

    # ── parse source_info ─────────────────────────────────────────────────
    requested: list[tuple[str, int]] = []   # (name_str, h_id)
    for level_key, indices in source_info.items():
        for idx in indices:
            name_str = f"{level_key}{idx}"
            h_id     = by_name.get(name_str)
            if h_id is None:
                warnings.warn(f"Source {name_str!r} not found in result — skipping.",
                              stacklevel=2)
                continue
            requested.append((name_str, h_id))

    if not requested:
        raise ValueError("No matching sources found for the given source_info.")

    # ── moment-0 background ───────────────────────────────────────────────
    mom0 = None
    if cube is not None:
        all_chs: set[int] = set()
        for _, h_id in requested:
            all_chs.update(by_id[h_id].channels)
        if all_chs:
            mom0 = cube[np.array(sorted(all_chs))].sum(axis=0)

    # ── axes ──────────────────────────────────────────────────────────────
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    if mom0 is not None:
        ax.imshow(mom0, cmap=cmap, origin="lower")
    else:
        ax.set_facecolor("black")

    ax.set_xticks([]); ax.set_yticks([])

    # ── opacity / linewidth by scale ──────────────────────────────────────
    req_scales = sorted({by_id[h_id].scale for _, h_id in requested})
    s_min, s_max = req_scales[0], req_scales[-1]

    def _alpha(scale: int) -> float:
        if s_max == s_min:
            return 0.9
        t = (s_max - scale) / (s_max - s_min)   # 0 = finest, 1 = coarsest
        return 0.9 - 0.65 * t                    # finest → 0.9, coarsest → 0.25

    def _lw(scale: int) -> float:
        if s_max == s_min:
            return 1.5
        t = (s_max - scale) / (s_max - s_min)
        return 2.2 - 1.4 * t                     # finest → 2.2, coarsest → 0.8

    # draw coarsest first so finer contours sit on top
    sorted_req = sorted(requested, key=lambda x: -by_id[x[1]].scale)
    drawn_bbox_roots: set[int] = set()

    for name_str, h_id in sorted_req:
        h     = by_id[h_id]
        mask  = src_masks[h_id]
        if not mask.any():
            warnings.warn(f"Source {name_str!r} has an empty footprint — skipping.",
                          stacklevel=2)
            continue

        base_rgb, _ = h_colors[h_id]
        a  = _alpha(h.scale)
        lw = _lw(h.scale)

        # low-opacity fill
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[mask] = (*base_rgb, a * 0.18)
        ax.imshow(rgba, origin="lower", interpolation="nearest")

        # contour
        ax.contour(mask.astype(float), [0.5],
                   colors=[base_rgb], linewidths=lw, alpha=a)

        # ── bbox on root-level (coarsest / A) footprint ──────────────────
        if bbox:
            root = h
            while root.parent_id is not None and root.parent_id in by_id:
                root = by_id[root.parent_id]

            if root.id not in drawn_bbox_roots:
                drawn_bbox_roots.add(root.id)
                rmask = src_masks[root.id]
                if rmask.any():
                    rrows, rcols = np.where(rmask)
                    r0, r1 = int(rrows.min()), int(rrows.max())
                    c0, c1 = int(rcols.min()), int(rcols.max())
                    pad       = 5
                    root_rgb, _ = h_colors[root.id]
                    rect = mpatches.Rectangle(
                        (c0 - pad, r0 - pad),
                        c1 - c0 + 2 * pad,
                        r1 - r0 + 2 * pad,
                        linewidth=1.0, edgecolor=root_rgb,
                        facecolor="none", linestyle="--", alpha=0.7, zorder=5,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        c1 + pad + 3, r1 + pad + 3, names[root.id],
                        ha="left", va="bottom", fontsize=10,
                        fontfamily="serif", fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc=root_rgb, ec="none", alpha=0.85),
                        zorder=6,
                    )

    if own_fig:
        plt.tight_layout()
        return fig, ax
    return ax
