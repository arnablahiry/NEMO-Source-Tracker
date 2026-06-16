"""Hierarchical multi-scale source organization."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ScaleDetection:
    """Detection results for a single scale within a single channel."""
    scale: int
    channel: int
    peaks: list[tuple[int, int]]
    footprint_masks: list[np.ndarray]
    boxes: list[tuple[int, int, int, int]]
    detect_coeffs: np.ndarray


@dataclass
class PerChannelScaleDetections:
    """All scale detections for one channel."""
    channel: int
    image: np.ndarray
    scales: Dict[int, tuple[list[np.ndarray], list[tuple[int, int]], list[tuple[int, int, int, int]]]]
    detect_coeffs: np.ndarray
    component_to_source: Dict[Tuple[int, int], int] = field(default_factory=dict)


@dataclass
class HierarchicalSourceGroup:
    """Multi-scale source with parent-child hierarchy."""
    id: int
    scale: int
    channels: List[int]
    track_ids: List[int]

    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)

    spatial_overlap: float = 0.0
    spectral_overlap: float = 0.0
    match_confidence: float = 0.0

    velocity_min: float = 0.0
    velocity_max: float = 0.0
    centroid_velocity: float = 0.0

    kinematic: bool = False
    has_split: bool = False
    split_events: List[int] = field(default_factory=list)
    merge_events: List[int] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    def is_leaf(self) -> bool:
        return self.scale == 1

    def is_root(self) -> bool:
        return self.parent_id is None

    def depth(self) -> int:
        return 4 - self.scale


TREE_PALETTE = [
    (0.95, 0.35, 0.35), (0.35, 0.65, 0.95), (0.45, 0.85, 0.45),
    (0.95, 0.75, 0.30), (0.75, 0.50, 0.92), (0.40, 0.85, 0.85),
    (0.95, 0.55, 0.80), (0.70, 0.70, 0.40), (0.55, 0.75, 0.95),
    (0.90, 0.50, 0.35),
]


def _tree_letter(i: int) -> str:
    """0→A, 25→Z, 26→AA, 27→AB, …"""
    s = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def assign_tree_names(hierarchical_sources):
    """Name every source by its hierarchical level (scale).

    One letter per scale level — the coarsest scale (largest ``scale``) is A,
    the next finer is B, and so on — and a 1-based index within that level.
    So the coarsest sources are A1, A2, A3, …; the next level B1, B2, …; etc.

    Returns
    -------
    dict[int, str]   h_id → name (e.g. "A1", "A2", "B1", …)
    """
    by_id = {h.id: h for h in hierarchical_sources}
    scales = sorted({h.scale for h in hierarchical_sources}, reverse=True)  # coarse→fine
    letter_for_scale = {s: _tree_letter(i) for i, s in enumerate(scales)}
    counters = {s: 0 for s in scales}
    names = {}

    # Number each level in tree-chronological (pre-order DFS) order, so the
    # whole of the first tree is numbered before the next.  Within a level the
    # first tree's members come first (B1, B2 under A1; then B3 under A2, …).
    roots = [h for h in hierarchical_sources if h.is_root()]
    roots.sort(key=lambda h: (-h.scale, h.id))

    def _walk(h):
        counters[h.scale] += 1
        names[h.id] = f"{letter_for_scale[h.scale]}{counters[h.scale]}"
        kids = [by_id[c] for c in h.children_ids if c in by_id]
        kids.sort(key=lambda k: (-k.scale, k.id))
        for k in kids:
            _walk(k)
    for r in roots:
        _walk(r)
    for h in hierarchical_sources:  # safety net for any unreachable node
        names.setdefault(h.id, f"?{h.id}")
    return names


def assign_tree_colors(hierarchical_sources, palette=None):
    """Assign one base RGB colour per source tree and a per-scale opacity.

    A *tree* is a root (a source with ``parent_id is None``) plus all of its
    descendants.  Every member of a tree shares the root's base colour.  Within
    a tree the coarsest scale (the root, highest ``scale`` index) is drawn at
    low opacity and finer scales (towards the leaves) progressively brighter
    and more opaque.

    Returns
    -------
    dict[int, tuple[tuple[float, float, float], float]]
        h_id → (base_rgb, alpha)
    """
    if palette is None:
        palette = [
            (0.95, 0.35, 0.35), (0.35, 0.65, 0.95), (0.45, 0.85, 0.45),
            (0.95, 0.75, 0.30), (0.75, 0.50, 0.92), (0.40, 0.85, 0.85),
            (0.95, 0.55, 0.80), (0.70, 0.70, 0.40), (0.55, 0.75, 0.95),
            (0.90, 0.50, 0.35),
        ]
    by_id = {h.id: h for h in hierarchical_sources}

    def root_of(h):
        seen = set()
        while h.parent_id is not None and h.parent_id in by_id and h.id not in seen:
            seen.add(h.id)
            h = by_id[h.parent_id]
        return h

    # Stable root ordering → stable colour assignment
    roots = [h for h in hierarchical_sources if h.is_root()]
    root_color = {r.id: palette[i % len(palette)] for i, r in enumerate(roots)}

    scales = sorted({h.scale for h in hierarchical_sources})
    s_min, s_max = (min(scales), max(scales)) if scales else (1, 1)

    def alpha_for(scale):
        if s_max == s_min:
            return 1.0
        # coarsest (s_max) → 0.35, finest (s_min) → 1.0
        frac = (s_max - scale) / (s_max - s_min)
        return 0.35 + 0.65 * frac

    result = {}
    for h in hierarchical_sources:
        base = root_color.get(root_of(h).id, (0.7, 0.7, 0.7))
        result[h.id] = (base, alpha_for(h.scale))
    return result


def build_hierarchical_sources(
    multi_scale_dets: List[PerChannelScaleDetections],
    tracks_per_scale: Dict[int, list],
    sources_per_scale: Dict[int, list],
    spatial_overlap_threshold: float = 0.3,
    velocity_tolerance: float = 10.0,
    vel_array: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[List[HierarchicalSourceGroup], Dict[Tuple[int, int], int]]:
    """Build parent-child relationships between sources at different scales.

    Parameters
    ----------
    multi_scale_dets : list[PerChannelScaleDetections]
        Output of detect_all_scales()
    tracks_per_scale : dict[int, list[dict]]
        Tracks grouped by scale
    sources_per_scale : dict[int, list[dict]]
        Sources grouped by scale
    spatial_overlap_threshold : float
        Minimum Jaccard IoU to consider overlap
    velocity_tolerance : float
        Max velocity difference (km/s or channels)
    vel_array : np.ndarray or None
        1-D velocity array for velocity range extraction
    verbose : bool

    Returns
    -------
    hierarchical_sources : list[HierarchicalSourceGroup]
    old_to_hierarchy : dict[(scale, src_id) → h_id]
    """
    # Build per-scale source masks
    src_masks_by_scale = {}
    for scale, sources in sources_per_scale.items():
        tracks = tracks_per_scale[scale]
        tracks_by_id = {t['id']: t for t in tracks}
        src_masks = {}

        if multi_scale_dets:
            H, W = multi_scale_dets[0].image.shape
        else:
            H, W = 0, 0

        for src in sources:
            m = np.zeros((H, W), dtype=bool)
            for tid in src['track_ids']:
                t = tracks_by_id.get(tid)
                if t:
                    for ch_mask in t['masks'].values():
                        m |= ch_mask
            src_masks[src['id']] = m
        src_masks_by_scale[scale] = src_masks

    # Compute velocity ranges per source
    src_velocities_by_scale = {}
    for scale, sources in sources_per_scale.items():
        velocities = {}
        for src in sources:
            channels = src['channels']
            if vel_array is not None and len(channels) > 0:
                vels = vel_array[np.array(channels)]
                v_min, v_max = float(vels.min()), float(vels.max())
                v_centroid = float(np.mean(vels))
            else:
                v_min = float(channels[0]) if channels else 0.0
                v_max = float(channels[-1]) if channels else 0.0
                v_centroid = float(np.mean(channels)) if channels else 0.0
            velocities[src['id']] = (v_min, v_max, v_centroid)
        src_velocities_by_scale[scale] = velocities

    def containment(a, b):
        """Fraction of *a* that lies inside *b* (intersection / area(a))."""
        area_a = a.sum()
        if area_a == 0:
            return 0.0
        return float(np.logical_and(a, b).sum()) / float(area_a)

    def spectral_containment(f_vmin, f_vmax, f_vcent, c_vmin, c_vmax):
        """Fraction of the fine velocity range inside the coarse range."""
        if (f_vmax - f_vmin) < 1e-9:  # single-channel fine source
            return 1.0 if (c_vmin - 1e-9) <= f_vcent <= (c_vmax + 1e-9) else 0.0
        overlap = max(0.0, min(f_vmax, c_vmax) - max(f_vmin, c_vmin))
        return overlap / (f_vmax - f_vmin)

    # Build hierarchy: each fine source attaches to the *nearest* coarser scale
    # that spatially & spectrally contains it.  Containment (fine-inside-coarse)
    # is used rather than IoU so that a small detail blob nested inside a large
    # coarse blob is correctly recognised as its child.
    parent_map = {}
    match_metrics = {}

    scales_sorted = sorted(src_masks_by_scale.keys())  # fine → coarse
    for i, fine_scale in enumerate(scales_sorted):
        if i == len(scales_sorted) - 1:
            continue

        fine_sources = sources_per_scale[fine_scale]
        fine_masks = src_masks_by_scale[fine_scale]
        fine_vels = src_velocities_by_scale[fine_scale]

        for fine_src in fine_sources:
            fine_id = fine_src['id']
            fine_mask = fine_masks.get(fine_id)
            if fine_mask is None or not fine_mask.any():
                continue
            f_vmin, f_vmax, f_vcent = fine_vels.get(fine_id, (0, 0, 0))

            matched = False
            for coarse_scale in scales_sorted[i+1:]:  # nearest coarser first
                best_match = None
                best_spatial_ov = 0.0
                best_spectral_ov = 0.0
                best_confidence = 0.0

                for coarse_src in sources_per_scale[coarse_scale]:
                    coarse_id = coarse_src['id']
                    coarse_mask = src_masks_by_scale[coarse_scale].get(coarse_id)
                    if coarse_mask is None or not coarse_mask.any():
                        continue

                    spatial_ov = containment(fine_mask, coarse_mask)
                    if spatial_ov < spatial_overlap_threshold:
                        continue

                    c_vmin, c_vmax, _ = src_velocities_by_scale[coarse_scale].get(
                        coarse_id, (0, 0, 0))
                    spectral_ov = spectral_containment(
                        f_vmin, f_vmax, f_vcent, c_vmin, c_vmax)
                    if spectral_ov < 0.5:
                        continue

                    confidence = float(np.sqrt(spatial_ov * spectral_ov))
                    if confidence > best_confidence:
                        best_match = (coarse_scale, coarse_id)
                        best_spatial_ov = spatial_ov
                        best_spectral_ov = spectral_ov
                        best_confidence = confidence

                if best_match:
                    parent_map[(fine_scale, fine_id)] = best_match
                    match_metrics[(fine_scale, fine_id, *best_match)] = (
                        best_spatial_ov, best_spectral_ov)
                    matched = True
                    break  # attach to the nearest containing coarser scale only
            _ = matched

    # Build HierarchicalSourceGroup objects
    hierarchical_sources = []
    h_id_counter = 0
    old_to_hierarchy = {}

    for scale in reversed(scales_sorted):
        sources = sources_per_scale[scale]
        tracks = tracks_per_scale[scale]
        tracks_by_id = {t['id']: t for t in tracks}

        for src in sources:
            src_id = src['id']
            key = (scale, src_id)

            parent_info = parent_map.get(key)
            parent_h_id = old_to_hierarchy.get((parent_info[0], parent_info[1])) \
                if parent_info else None

            channels = src['channels']
            if vel_array is not None and len(channels) > 0:
                vels = vel_array[np.array(channels)]
                v_min, v_max = float(vels.min()), float(vels.max())
                v_cent = float(np.mean(vels))
            else:
                v_min = float(channels[0]) if channels else 0.0
                v_max = float(channels[-1]) if channels else 0.0
                v_cent = (v_min + v_max) / 2.0

            h_src = HierarchicalSourceGroup(
                id=h_id_counter,
                scale=scale,
                channels=src['channels'],
                track_ids=src['track_ids'],
                parent_id=parent_h_id,
                children_ids=[],
                spatial_overlap=(match_metrics.get((*key, *parent_info), (0, 0))[0]
                               if parent_info else 1.0),
                spectral_overlap=(match_metrics.get((*key, *parent_info), (0, 0))[1]
                                if parent_info else 1.0),
                velocity_min=v_min,
                velocity_max=v_max,
                centroid_velocity=v_cent,
                kinematic=any(tracks_by_id.get(tid, {}).get('kinematic', False)
                            for tid in src['track_ids']),
                has_split=any(tracks_by_id.get(tid, {}).get('has_split', False)
                            for tid in src['track_ids']),
                split_events=src.get('split_events', []),
                merge_events=src.get('merge_events', []),
            )
            hierarchical_sources.append(h_src)
            old_to_hierarchy[key] = h_id_counter
            h_id_counter += 1

    # Second pass: populate children_ids from the now-complete parent links
    by_id = {h.id: h for h in hierarchical_sources}
    for h in hierarchical_sources:
        if h.parent_id is not None and h.parent_id in by_id:
            by_id[h.parent_id].children_ids.append(h.id)

    if verbose:
        n_roots = sum(1 for h in hierarchical_sources if h.is_root())
        n_leaves = sum(1 for h in hierarchical_sources if h.is_leaf())
        print(f"\n[Hierarchy building]  {len(hierarchical_sources)} sources  "
              f"({n_roots} roots, {n_leaves} leaves)")
        for scale in scales_sorted:
            n_at_scale = sum(1 for h in hierarchical_sources if h.scale == scale)
            print(f"  scale {scale}: {n_at_scale} sources")

    return hierarchical_sources, old_to_hierarchy
