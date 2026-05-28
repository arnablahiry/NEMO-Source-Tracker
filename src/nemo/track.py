"""Flow-guided source tracker for per-channel wavelet detections.

Takes the list of per-channel detections produced by
``wavelet_detections.detect_cube_per_channel`` and runs a four-stage pipeline:

Stage 1 — Masked optical flow
    TV-L1 flow is computed between every consecutive channel pair, but only
    inside the intersection of the two channels' union footprint masks.
    Zeroing the images outside detected sources prevents artefact-level flow
    vectors from leaking into the tracking step.

Stage 2 — Track linking with split/merge detection
    Two-pass approach for symmetric split and merge detection.

    Forward pass (ch_0 → ch_N): advected masks are propagated through the flow and
    matched to component detections via Hungarian assignment.  Unmatched predictions
    that overlap an already-claimed component are flagged as **merges**.  Unmatched
    detections start new independent tracks.

    Backward pass (ch_N → ch_0): the same algorithm runs on reversed detections
    and negated flow.  A forward split (one parent → two children) appears as a
    backward merge (two tracks → one component) and is captured symmetrically by the
    same merge-detection step.

    Reconciliation maps backward merge events onto forward tracks to annotate
    ``split_from`` and ``split_at`` fields.  No Euclidean distance is used
    anywhere in split or merge detection.

Stage 3 — Kinematic classification
    A track is **kinematically active** if its cumulative centroid displacement
    across channels exceeds MIN_DISPLACEMENT pixels, or if it was involved in
    a split or merge event.

Stage 4 — Source grouping
    Tracks connected by split_from / merge_into relationships are grouped into
    **sources** via union-find.  A source represents one physical object whose
    emission footprint may split into several components across channels (due to
    kinematics / Doppler shear) and later rejoin.

Output
------
``run_flow_tracker`` returns a :class:`TrackingResult` dataclass with:

``detections``
    list[ChannelDetection], one per processed channel.
``flow_seq``
    list of (ch_ref, ch_tgt, flow (2,H,W), joint_mask) tuples.
``tracks``
    list of track dicts.  Each dict contains ``id``, ``source_id``,
    ``trajectory``, ``masks``, ``split_at``, ``split_from``,
    ``merge_into``, ``displacement``, ``has_split``, ``kinematic``.
``sources``
    list of source dicts.  Each dict contains ``id``, ``track_ids``,
    ``channels``, ``n_channels``, ``split_events``, ``merge_events``.

Usage (standalone)::

    python flow_tracker.py \\
        --cube  data/clean_cube.npy \\
        --out   /tmp/tracks \\
        --channels 70,74 \\
        --min-match-overlap 5 --min-displacement 3
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment
from skimage.registration import optical_flow_tvl1

from .detect import (
    ChannelDetection,
    active_channels,
    detect_cube_per_channel,
    load_cube,
)


# ---------------------------------------------------------------------------
# Stage 1 — Masked optical flow
# ---------------------------------------------------------------------------

def masked_flow_tvl1(
    img_ref: np.ndarray,
    img_tgt: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """TV-L1 optical flow restricted to *mask* pixels.

    Both images are zeroed outside *mask* before the solver runs, so emission
    structure outside detected source footprints never influences the flow
    estimate inside them.

    Parameters
    ----------
    img_ref, img_tgt :
        2-D float32 channel images, shape (H, W).
    mask :
        Boolean (H, W) — True where flow should be estimated.

    Returns
    -------
    np.ndarray
        Shape (2, H, W) float32.  ``flow[0]`` = v (row displacement),
        ``flow[1]`` = u (col displacement).  Zero everywhere outside *mask*.
    """
    r = (img_ref * mask).astype(np.float64)
    t = (img_tgt * mask).astype(np.float64)
    v, u = optical_flow_tvl1(r, t)
    flow = np.stack([v, u], axis=0).astype(np.float32)
    flow[:, ~mask] = 0.0
    return flow


def compute_flow_sequence(
    detections: list[ChannelDetection],
    verbose: bool = False,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Compute masked TV-L1 flow for every consecutive detection pair.

    The joint mask is the *union* of the source footprints from both channels.
    Using the union (rather than the intersection) is critical for split
    detection: when a source splits into a new spatial location between
    channels, the two components may not overlap at all.  With an intersection
    mask the flow would be zero everywhere and the predicted centroid would
    not move — causing the split-off component to be mis-classified as a new
    independent source.  With the union mask the TV-L1 solver sees the
    source signal on both sides and produces flow vectors that point from
    the pre-split footprint toward the post-split footprint, allowing
    :func:`link_tracks` to attribute the new component to the correct parent.

    Parameters
    ----------
    detections :
        Ordered list of :class:`~wavelet_detections.ChannelDetection` objects.

    Returns
    -------
    list of (ch_ref, ch_tgt, flow, joint_mask) tuples.
    """
    H, W = detections[0].image.shape
    n_pairs = len(detections) - 1
    if verbose:
        print(f"[Stage 1] Masked TV-L1 optical flow  ({n_pairs} channel pairs, "
              f"image {H}×{W})")
    results = []

    zero_flow_count = 0
    mag_max_all = 0.0

    for i in range(len(detections) - 1):
        d_ref, d_tgt = detections[i], detections[i + 1]

        union_ref = np.zeros((H, W), dtype=bool)
        for m in d_ref.footprint_masks:
            union_ref |= m

        union_tgt = np.zeros((H, W), dtype=bool)
        for m in d_tgt.footprint_masks:
            union_tgt |= m

        joint_mask = union_ref | union_tgt

        if joint_mask.any():
            flow = masked_flow_tvl1(d_ref.image, d_tgt.image, joint_mask)
            mask_px = int(joint_mask.sum())
            u, v = flow[1], flow[0]
            mag = float(np.hypot(u[joint_mask], v[joint_mask]).max()) if joint_mask.any() else 0.0
            mag_max_all = max(mag_max_all, mag)
        else:
            flow = np.zeros((2, H, W), dtype=np.float32)
            mask_px = 0
            mag = 0.0
            zero_flow_count += 1

        results.append((d_ref.channel, d_tgt.channel, flow, joint_mask))

        if verbose:
            flag = "·" if mag < 0.1 else ("▸" if mag < 1.0 else "▶")
            print(f"  {flag} ch {d_ref.channel:4d}→{d_tgt.channel:<4d}  "
                  f"mask {mask_px:6d} px  peak flow {mag:.3f} px")

    if verbose:
        active_pairs = n_pairs - zero_flow_count
        print(f"\n  → {n_pairs} pairs  |  {active_pairs} with flow  |  "
              f"{zero_flow_count} zero-mask  |  global peak {mag_max_all:.3f} px\n")
    return results


# ---------------------------------------------------------------------------
# Catmull-Rom flow sampling helper
# ---------------------------------------------------------------------------

def _sample_flow(field: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Catmull-Rom cubic interpolation of a 2-D scalar field at (ys, xs).

    Uses scipy.ndimage.map_coordinates with order=3 (cubic spline, equivalent
    to Catmull-Rom for smooth fields).  ys and xs are 1-D float arrays.
    """
    coords = np.stack([
        np.clip(ys, 0, field.shape[0] - 1),
        np.clip(xs, 0, field.shape[1] - 1),
    ])
    return map_coordinates(field, coords, order=3, mode='nearest')


def _extrapolate_centroid(cy: float, cx: float, flow: np.ndarray) -> tuple[float, float]:
    """Flow-extrapolate a centroid one step forward."""
    ys = np.array([cy], dtype=float)
    xs = np.array([cx], dtype=float)
    return (float(cy + _sample_flow(flow[0], ys, xs)[0]),
            float(cx + _sample_flow(flow[1], ys, xs)[0]))


# ---------------------------------------------------------------------------
# Advected mask propagation helper
# ---------------------------------------------------------------------------

def _advect_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Advect a boolean source footprint forward through a flow field.

    Every True pixel at (y, x) in *mask* is displaced by the Catmull-Rom
    sampled flow vector at that pixel.  Displaced pixel positions are
    accumulated into a float32 weight map (raw hit counts, not normalised,
    not dilated).

    Overlap between this weight map and a component footprint mask is computed as
        (weight_map * component_mask).sum()
    A non-zero overlap means the flow carries source pixels into the component.

    Parameters
    ----------
    mask : (H, W) bool
    flow : (2, H, W) float32 — flow[0]=row disp, flow[1]=col disp

    Returns
    -------
    weight_map : (H, W) float32 — raw advection hit counts
    """
    H, W = mask.shape
    ys, xs = np.where(mask)
    weight_map = np.zeros((H, W), dtype=np.float32)
    if ys.size == 0:
        return weight_map

    v = _sample_flow(flow[0], ys.astype(float), xs.astype(float))
    u = _sample_flow(flow[1], ys.astype(float), xs.astype(float))

    pred_ys = np.clip(np.round(ys + v).astype(int), 0, H - 1)
    pred_xs = np.clip(np.round(xs + u).astype(int), 0, W - 1)

    # np.add.at handles duplicate destination pixels correctly (unlike +=)
    np.add.at(weight_map, (pred_ys, pred_xs), 1.0)
    return weight_map


# ---------------------------------------------------------------------------
# Stage 2 — Track linking with split/merge detection
# ---------------------------------------------------------------------------

def link_tracks(
    detections: list[ChannelDetection],
    flow_seq: list[tuple[int, int, np.ndarray, np.ndarray]],
    min_match_overlap: int = 5,
    max_gap_channels: int = 5,
    verbose: bool = False,
) -> list[dict]:
    """Link per-channel component detections into multi-channel tracks.

    Uses advected masks and pixel-overlap matching — no Euclidean distance.

    Algorithm
    ---------
    Each track maintains an *advected mask*: its most recently known wavelet
    footprint, advected channel-by-channel through the flow via Catmull-Rom
    cubic interpolation.  All matching uses pixel-overlap only.

    For each consecutive channel pair (ref → tgt):

    A. Advect every active track's footprint mask through the flow → adv_maps.
    B. Hungarian matching on negative-overlap cost matrix.  Pairs with overlap
       ≥ min_match_overlap are matched; advected mask reset to the matched
       component footprint.
    C. Unmatched predictions: check for merge via overlap, then freeze advected
       mask and extrapolate centroid; deactivate if gap exceeds max_gap_channels.
    D. Unmatched detections: always start new independent tracks.  Split
       attribution is performed separately by :func:`_reconcile_splits` using
       the backward pass, where splits look like merges and are detected
       symmetrically.

    Parameters
    ----------
    detections :
        Per-channel detection results in channel order.
    flow_seq :
        Output of :func:`compute_flow_sequence`.
    min_match_overlap :
        Minimum pixel overlap (advected mask ∩ component footprint) to accept a
        continuation match.
    max_gap_channels :
        Maximum number of consecutive unmatched channels before a track is
        deactivated.

    Returns
    -------
    list[dict]
        One dict per track with keys: ``id``, ``trajectory``, ``masks``,
        ``split_at``, ``split_from``, ``merge_into``, ``active``.
    """
    def _new_track(tid, ch, y, x, mask):
        return dict(
            id=tid, trajectory=[(ch, y, x)], masks={ch: mask},
            split_at=[], split_from=None, merge_into=[], active=True,
            gap_age=0,
        )

    tracks: list[dict] = []

    # Seed one track per component in the first channel.
    d0 = detections[0]
    for mask, (y, x) in zip(d0.footprint_masks, d0.peaks):
        tracks.append(_new_track(len(tracks), d0.channel, float(y), float(x), mask))

    if verbose:
        print(f"[Stage 2] Track linking  ({len(detections)} channels, "
              f"{len(flow_seq)} transitions, "
              f"min_match_overlap={min_match_overlap}, "
              f"max_gap_channels={max_gap_channels})")
        print(f"  Seeded {len(tracks)} track(s) from channel {d0.channel} "
              f"({len(d0.peaks)} component(s))")

    # Advected masks: each source's current advected footprint.
    adv_masks: dict[int, np.ndarray] = {
        t['id']: t['masks'][d0.channel].copy() for t in tracks
    }

    for fi, (ch_ref, ch_tgt, flow, _) in enumerate(flow_seq):
        d_tgt  = detections[fi + 1]
        active = [t for t in tracks if t['active']]

        # Per-channel event counters for verbose reporting.
        _v_matched_b = 0
        _v_merges    = 0
        _v_new       = 0
        _v_deact     = 0
        _v_frozen    = 0

        # A. Advect every active source's footprint mask through the flow field.
        adv_maps: dict[int, np.ndarray] = {
            t['id']: _advect_mask(adv_masks[t['id']], flow)
            for t in active
        }

        # No detections: freeze advected masks, extrapolate centroids.
        if not d_tgt.peaks:
            for t in active:
                t['gap_age'] += 1
                if t['gap_age'] > max_gap_channels:
                    t['active'] = False
                    continue
                cy, cx = t['trajectory'][-1][1:]
                t['trajectory'].append((ch_tgt, cy, cx))
            continue

        # B. Overlap cost matrix → Hungarian matching (continuation).
        n_active = len(active)
        n_comps  = len(d_tgt.footprint_masks)
        cost = np.zeros((n_active, n_comps), dtype=float)
        for r, t in enumerate(active):
            for c, comp_mask in enumerate(d_tgt.footprint_masks):
                cost[r, c] = -float((adv_maps[t['id']] * comp_mask).sum())

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_pred: set[int] = set()
        matched_det:  set[int] = set()
        det_to_track: dict[int, dict] = {}

        for r, c in zip(row_ind, col_ind):
            if -cost[r, c] >= min_match_overlap:
                t = active[r]
                t['trajectory'].append(
                    (ch_tgt, float(d_tgt.peaks[c][0]), float(d_tgt.peaks[c][1]))
                )
                t['masks'][ch_tgt]   = d_tgt.footprint_masks[c]
                adv_masks[t['id']] = d_tgt.footprint_masks[c]
                t['gap_age'] = 0
                matched_pred.add(r)
                matched_det.add(c)
                det_to_track[c] = t
                _v_matched_b += 1

        # C. Unmatched predictions: merge check + advected mask freeze + centroid extrapolation.
        for r, t in enumerate(active):
            if r in matched_pred:
                continue
            cy, cx = t['trajectory'][-1][1:]
            py, px = _extrapolate_centroid(cy, cx, flow)

            # Merge: advected mask overlaps a detection already owned by another track.
            merged = False
            for c_det, owner in det_to_track.items():
                ov = float((adv_maps[t['id']] * d_tgt.footprint_masks[c_det]).sum())
                if ov >= min_match_overlap:
                    t['merge_into'].append((ch_tgt, owner['id']))
                    t['active'] = False   # stop propagating once merged
                    merged = True
                    _v_merges += 1
                    break

            if merged:
                continue

            t['gap_age'] += 1
            if t['gap_age'] > max_gap_channels:
                t['active'] = False
                _v_deact += 1
                continue

            t['trajectory'].append((ch_tgt, py, px))
            _v_frozen += 1

            # Freeze the advected mask in place — sources don't move spatially between
            # spectral channels; advecting through a flow field with no signal
            # from this source pushes the mask in the wrong direction.
            # adv_masks[t['id']] is unchanged: the frozen last-known footprint.

        # D. Unmatched detections: start new independent tracks.
        for c, (dy, dx) in enumerate(d_tgt.peaks):
            if c in matched_det:
                continue
            comp_mask = d_tgt.footprint_masks[c]
            _v_new += 1
            new_t = _new_track(
                len(tracks), ch_tgt, float(dy), float(dx), comp_mask,
            )
            adv_masks[new_t['id']] = comp_mask.copy()
            tracks.append(new_t)

        if verbose:
            n_active_now = sum(1 for t in tracks if t['active'])
            print(f"  ch {ch_ref:4d}→{ch_tgt:<4d}  "
                  f"matched {_v_matched_b:2d}  "
                  f"merge {_v_merges:2d}  "
                  f"new {_v_new:2d}  "
                  f"frozen {_v_frozen:2d}  "
                  f"deact {_v_deact:2d}  "
                  f"active {n_active_now:3d}")

    n_active_final  = sum(1 for t in tracks if t['active'])
    n_merged        = sum(1 for t in tracks if t['merge_into'])
    if verbose:
        print(f"\n  → {len(tracks)} tracks total  |  "
              f"{n_active_final} still active  |  "
              f"{n_merged} merged\n")
    return tracks


def _reconcile_splits(
    fwd_tracks: list[dict],
    bwd_tracks: list[dict],
    tol_px: float = 5.0,
    verbose: bool = False,
) -> list[dict]:
    """Annotate split events in *fwd_tracks* using merge events from *bwd_tracks*.

    A forward split (one parent → two children) looks like a backward merge
    (two tracks → one component) when channels are processed in reverse.  This
    function reads the ``merge_into`` events recorded in *bwd_tracks* and
    transfers them as ``split_from`` / ``split_at`` annotations on the
    corresponding forward tracks.

    Track correspondence is established by trajectory voting: for each point
    ``(ch, y, x)`` in a backward track's trajectory, the forward track whose
    trajectory contains a point at the same channel within *tol_px* pixels
    gets a vote.  The forward track with the most votes is the match.

    Parameters
    ----------
    fwd_tracks :
        Tracks from the forward pass (modified in-place).
    bwd_tracks :
        Tracks from the backward pass (read-only).
    tol_px :
        Spatial tolerance (pixels) for trajectory-point matching.

    Returns
    -------
    list[dict]
        The same *fwd_tracks* list, with ``split_from`` and ``split_at``
        fields populated from the backward pass.
    """
    tol2 = tol_px ** 2

    # Build lookup: ch -> [(y, x, fwd_track_id), ...]
    ch_pts: dict[int, list] = defaultdict(list)
    for t in fwd_tracks:
        for (ch, y, x) in t['trajectory']:
            ch_pts[ch].append((y, x, t['id']))

    fwd_by_id = {t['id']: t for t in fwd_tracks}
    bwd_by_id = {t['id']: t for t in bwd_tracks}

    # Match each backward track to its forward correspondent by trajectory vote.
    bwd_to_fwd: dict[int, int] = {}
    for bt in bwd_tracks:
        votes: dict[int, int] = defaultdict(int)
        for (ch, by, bx) in bt['trajectory']:
            for (fy, fx, fid) in ch_pts.get(ch, []):
                if (by - fy) ** 2 + (bx - fx) ** 2 <= tol2:
                    votes[fid] += 1
        if votes:
            bwd_to_fwd[bt['id']] = max(votes, key=votes.__getitem__)

    matched = sum(1 for bid in bwd_to_fwd if bid in {bt['id'] for bt in bwd_tracks})
    if verbose:
        print(f"[Split reconciliation]  "
              f"fwd {len(fwd_tracks)} tracks  bwd {len(bwd_tracks)} tracks  "
              f"tol={tol_px} px")
        print(f"  Backward→forward matches: {matched} / {len(bwd_tracks)}")

    n_splits = 0
    n_split_parents = 0

    # Transfer merge events from backward pass as split annotations in forward.
    for bt in bwd_tracks:
        if bt['id'] not in bwd_to_fwd:
            continue
        b_fwd = fwd_by_id[bwd_to_fwd[bt['id']]]

        for (ch_merge, p_bwd_id) in bt['merge_into']:
            if p_bwd_id not in bwd_to_fwd:
                continue
            p_fwd = fwd_by_id[bwd_to_fwd[p_bwd_id]]

            if b_fwd['id'] == p_fwd['id']:
                continue

            if b_fwd['split_from'] is None:
                b_fwd['split_from'] = p_fwd['id']
                n_splits += 1
                if verbose:
                    print(f"  split: track {b_fwd['id']} ← parent {p_fwd['id']} "
                          f"at ch {ch_merge}")
            if ch_merge not in p_fwd['split_at']:
                p_fwd['split_at'].append(ch_merge)
                n_split_parents += 1

    if verbose:
        print(f"  → {n_splits} split child(ren) annotated  |  "
              f"{n_split_parents} parent split-point(s)\n")

    return fwd_tracks


# ---------------------------------------------------------------------------
# Stage 3 — Kinematic classification
# ---------------------------------------------------------------------------

def classify_kinematic(
    tracks: list[dict],
    min_displacement: float = 3.0,
    verbose: bool = False,
) -> list[dict]:
    """Add kinematic classification fields to each track dict (in-place).

    A track is **kinematically active** if:
    - Its cumulative centroid displacement across channels ≥ *min_displacement*, or
    - It was involved in a split event (either as parent or as split-off child).

    Adds keys ``displacement`` (float, px), ``has_split`` (bool),
    and ``kinematic`` (bool) to each track dict.
    """
    if verbose:
        print(f"[Stage 3] Kinematic classification  (min_displacement={min_displacement} px)")
    for t in tracks:
        traj = t['trajectory']
        # Sum of step-wise displacements — captures curved trajectories better
        # than straight-line start-to-end distance.
        disp = sum(
            np.hypot(traj[i+1][1] - traj[i][1], traj[i+1][2] - traj[i][2])
            for i in range(len(traj) - 1)
        )
        split = bool(t['split_at']) or t['split_from'] is not None or bool(t['merge_into'])
        t['displacement'] = float(disp)
        t['has_split']    = split
        t['kinematic']    = disp >= min_displacement or split
        if verbose:
            reason = []
            if disp >= min_displacement:
                reason.append(f"disp={disp:.1f} px")
            if t['split_at']:
                reason.append(f"split_at={t['split_at']}")
            if t['split_from'] is not None:
                reason.append(f"split_from={t['split_from']}")
            if t['merge_into']:
                reason.append(f"merge_into={[tid for _, tid in t['merge_into']]}")
            flag = "kinematic" if t['kinematic'] else "static   "
            ch_range = f"ch {traj[0][0]}–{traj[-1][0]}"
            print(f"  track {t['id']:>2}  {flag}  {ch_range}  "
                  + (", ".join(reason) if reason else "displacement below threshold"))
    if verbose:
        n_kin = sum(1 for t in tracks if t['kinematic'])
        n_sta = len(tracks) - n_kin
        print(f"  → {n_kin} kinematic  |  {n_sta} static\n")


# ---------------------------------------------------------------------------
# Stage 4 — Source grouping
# ---------------------------------------------------------------------------

def group_into_sources(tracks: list[dict]) -> list[dict]:
    """Group related tracks into sources via union-find over split/merge edges.

    Two tracks belong to the same source if they are connected by any chain of
    ``split_from`` (child→parent) or ``merge_into`` (merging track → target)
    relationships.  The result is one source per connected component.

    Annotates each track dict in-place with a ``source_id`` key.

    Parameters
    ----------
    tracks :
        Output of :func:`classify_kinematic` (or :func:`link_tracks`).

    Returns
    -------
    list[dict]
        One dict per source, sorted by ascending ``id``, with keys:
        ``id``, ``track_ids``, ``channels``, ``n_channels``,
        ``split_events``, ``merge_events``.
    """
    # Path-compressed union-find.
    parent = {t['id']: t['id'] for t in tracks}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for t in tracks:
        if t['split_from'] is not None:
            union(t['id'], t['split_from'])
        for _ch, target_id in t['merge_into']:
            union(t['id'], target_id)

    track_by_id = {t['id']: t for t in tracks}
    groups: dict[int, list[dict]] = defaultdict(list)
    for t in tracks:
        groups[find(t['id'])].append(t)

    sources = []
    for sid, group_tracks in enumerate(groups.values()):
        track_ids, channels, split_events, merge_events = [], set(), set(), set()
        for t in group_tracks:
            track_ids.append(t['id'])
            for ch, *_ in t['trajectory']: channels.add(ch)
            split_events.update(t['split_at'])
            merge_events.update(ch for ch, _ in t['merge_into'])
        track_ids    = sorted(track_ids)
        channels     = sorted(channels)
        split_events = sorted(split_events)
        merge_events = sorted(merge_events)
        src = dict(
            id=sid,
            track_ids=track_ids,
            channels=channels,
            n_channels=len(channels),
            split_events=split_events,
            merge_events=merge_events,
        )
        sources.append(src)
        for tid in track_ids:
            track_by_id[tid]['source_id'] = sid

    return sources


# ---------------------------------------------------------------------------
# Source classification (real vs false detections)
# ---------------------------------------------------------------------------

def classify_sources(
    sources: list[dict],
    tracks: list[dict],
    detections: list[ChannelDetection],
    flow_seq: list[tuple],
    wav_scale_idx: int = 3,
    wav_abrupt_thresh: float = 0.5,
    flow_iou_thresh: float = 0.25,
    short_det_max: int = 8,
    verbose: bool = True,
    plot: bool = False,
    vel_array: np.ndarray | None = None,
    results_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], dict, dict]:
    """Classify sources as real detections or false positives.

    Uses two complementary metrics computed from wavelet coefficients and
    optical flow:

    - **flow_iou**: advect each source's footprint through the flow field
      (backward warp) and measure IoU with the next channel's footprint.
      Real sources follow the flow and score high; artefacts don't move
      coherently and score low.
    - **wav_abrupt**: ratio of the edge wavelet flux (first or last channel
      of the detection) to the peak flux.  Step-function artefacts that
      appear or disappear abruptly score ≈ 1; real sources fade in/out
      and score lower.

    A source is classified as a false detection if:
        wav_abrupt > wav_abrupt_thresh
        OR (flow_iou < flow_iou_thresh AND n_detected_channels < short_det_max)

    Parameters
    ----------
    sources, tracks, detections, flow_seq :
        Direct outputs of :func:`run_flow_tracker`.
    wav_scale_idx :
        0-based wavelet scale index to use for spectral profile extraction.
        Should match ``use_scale - 1`` used in :func:`run_flow_tracker`.
    wav_abrupt_thresh, flow_iou_thresh, short_det_max :
        Classification thresholds (see above).
    verbose :
        Print a formatted table of sources and false detections.
    plot :
        Render and save the two-panel separation figure
        (IoU scatter + normalised wavelet profiles).  Requires
        ``vel_array`` and ``results_dir``.
    vel_array :
        1-D velocity array aligned with cube channels (km/s).  Required
        when ``plot=True``.
    results_dir :
        Directory for saved figures.  Created if absent.  Required when
        ``plot=True``.

    Returns
    -------
    good_sources : list[dict]
    false_dets   : list[dict]
    src_data     : dict  {source_id → metric dict}
    src_colors   : dict  {source_id → rgba tuple}  (tab10, cycled mod 10)
    """
    from scipy.ndimage import map_coordinates as _map_coords

    # Cube spatial dimensions inferred from the first detection.
    _nH = detections[0].image.shape[0]
    _nW = detections[0].image.shape[1]

    _flow_by_pair = {(cr, ct): ff for cr, ct, ff, _ in flow_seq}
    _det_by_ch    = {d.channel: d for d in detections}

    # ── tab10 source colours ──────────────────────────────────────────────
    try:
        import matplotlib.cm as _cm
        import matplotlib
        _src_cmap = matplotlib.colormaps['tab10']
    except (AttributeError, KeyError):
        _src_cmap = _cm.get_cmap('tab10')
    src_colors = {src['id']: _src_cmap(src['id'] % 10) for src in sources}

    # ── Backward-warp helper (advect footprint through flow) ─────────────
    def _advect(mask, flow):
        H, W = mask.shape
        ys, xs = np.mgrid[0:H, 0:W].astype(float)
        return _map_coords(
            mask.astype(float),
            [(ys - flow[0]).ravel(), (xs - flow[1]).ravel()],
            order=1, mode='nearest',
        ).reshape(H, W)

    # ── Per-source metrics ────────────────────────────────────────────────
    src_data: dict[int, dict] = {}
    for src in sources:
        src_tracks = [t for t in tracks if t['id'] in src['track_ids']]
        ch_to_mask: dict[int, np.ndarray] = {}
        for t in src_tracks:
            for ch, mask in t['masks'].items():
                ch_to_mask[ch] = (
                    ch_to_mask.get(ch, np.zeros((_nH, _nW), dtype=bool)) | mask
                )
        if not ch_to_mask:
            continue
        det_chs = sorted(ch_to_mask.keys())

        # Flow IoU: backward-warp footprint and overlap with next channel.
        iou_vals = []
        for ii in range(len(det_chs) - 1):
            key = (det_chs[ii], det_chs[ii + 1])
            if key not in _flow_by_pair:
                continue
            adv   = _advect(ch_to_mask[det_chs[ii]], _flow_by_pair[key]) > 0.3
            tgt   = ch_to_mask[det_chs[ii + 1]]
            union = int((adv | tgt).sum())
            if union > 0:
                iou_vals.append(float((adv & tgt).sum()) / union)
        flow_iou = float(np.mean(iou_vals)) if iou_vals else 0.0

        # Wavelet spectral profile and abruptness.
        wav_prof = np.array([
            float(_det_by_ch[ch].detect_coeffs[wav_scale_idx][ch_to_mask[ch]].sum())
            if ch in _det_by_ch else 0.0
            for ch in det_chs
        ])
        pk = wav_prof.max() + 1e-30
        wav_abrupt = float(wav_prof[0]) / pk

        # Centroid jitter.
        traj_pts = [
            (ry, rx) for t in src_tracks
            for ch, ry, rx in t['trajectory'] if ch in ch_to_mask
        ]
        jitter = (
            float(np.sqrt(np.var([r for r, _ in traj_pts]) +
                          np.var([c for _, c in traj_pts])))
            if len(traj_pts) > 1 else 0.0
        )

        src_data[src['id']] = dict(
            n_det=len(det_chs), flow_iou=flow_iou, wav_abrupt=wav_abrupt,
            jitter=jitter, det_chs=det_chs, wav_prof=wav_prof,
        )

    # ── Classification ────────────────────────────────────────────────────
    def _is_false_detection(m: dict) -> bool:
        return (m['wav_abrupt'] > wav_abrupt_thresh
                or (m['flow_iou'] < flow_iou_thresh and m['n_det'] < short_det_max))

    false_det_ids = {sid for sid, m in src_data.items() if _is_false_detection(m)}
    good_sources  = [s for s in sources if s['id'] not in false_det_ids]
    false_dets    = [s for s in sources if s['id']     in false_det_ids]

    # ── Chronological ordering ────────────────────────────────────────────
    def _first_ch(s: dict) -> int:
        chs = [ch for t in tracks if t['id'] in s['track_ids'] for ch in t['masks']]
        return min(chs) if chs else 9999

    good_chrono   = sorted(good_sources, key=_first_ch)
    chrono_label  = {s['id']: i + 1 for i, s in enumerate(good_chrono)}
    n_real        = len(good_sources)
    fd_chrono     = sorted(false_dets, key=_first_ch)
    chrono_lbl_fd = {
        **chrono_label,
        **{s['id']: n_real + i + 1 for i, s in enumerate(fd_chrono)},
    }
    fd_ch_min = {s['id']: _first_ch(s) for s in sources}

    # Attach chronological label to src_colors so downstream cells can use it.
    # (returned as part of src_colors via the closure over chrono_label)

    # ── Verbose table ─────────────────────────────────────────────────────
    if verbose:
        n_real_tracks = sum(1 for t in tracks if t['source_id'] not in false_det_ids)
        print(f"\n{n_real_tracks} tracks  →  {len(good_sources)} sources"
              f"  ({len(false_dets)} false detection(s) removed)")

        hdr = (f"{'Src':>4}  {'Color':>7}  {'ch start':>8}  {'ch end':>6}  "
               f"{'IoU':>5}  {'Abrupt':>6}  {'Tracks':>10}  {'Disp(px)':>14}  "
               f"{'Type':>9}  {'Splits':>8}  Merges")
        print("\n" + hdr)
        print("─" * (len(hdr) + 10))
        for src in good_chrono:
            sid  = src['id']
            rc, gc, bc = [int(v * 255) for v in src_colors[sid][:3]]
            m    = src_data[sid]
            stracks   = [t for t in tracks if t['id'] in src['track_ids']]
            track_ids = ', '.join(str(t['id']) for t in stracks)
            disps     = ', '.join(f"{t['displacement']:.1f}" for t in stracks)
            is_kin    = any(t['kinematic'] for t in stracks)
            kind      = 'kinematic' if is_kin else 'static'
            all_splt  = sorted({c for t in stracks for c in t['split_at']})
            all_mrg   = sorted({c for t in stracks for c, _ in t['merge_into']})
            splt_str  = str(all_splt) if all_splt else '—'
            mrg_str   = str(all_mrg)  if all_mrg  else '—'
            print(f"{chrono_label[sid]:>4}  #{rc:02x}{gc:02x}{bc:02x}  "
                  f"{src['channels'][0]:>8}  {src['channels'][-1]:>6}  "
                  f"{m['flow_iou']:>5.2f}  {m['wav_abrupt']:>6.2f}  "
                  f"{track_ids:>10}  {disps:>14}  "
                  f"{kind:>9}  {splt_str:>8}  {mrg_str}")

        if false_dets:
            print(f"\nFalse detections ({len(false_dets)}):")
            for fi, src in enumerate(fd_chrono, 1):
                sid = src['id']
                m   = src_data[sid]
                print(f"  {fi}.  ch {src['channels'][0]}–{src['channels'][-1]}"
                      f"  IoU={m['flow_iou']:.2f}  abrupt={m['wav_abrupt']:.2f}"
                      f"  n_det={m['n_det']}")

    # ── Two-panel figure ──────────────────────────────────────────────────
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.patches
        from matplotlib.lines import Line2D
        from matplotlib.legend_handler import HandlerBase

        if results_dir is not None:
            Path(results_dir).mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(9.5, 10))
        gs  = fig.add_gridspec(2, 1, hspace=0.18)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Panel 1: IoU vs abruptness scatter
        ax1.axhspan(wav_abrupt_thresh, 1.3, color='#fce8e8', zorder=0)
        ax1.axvspan(-0.1, flow_iou_thresh, color='#e8eeff', zorder=0, alpha=0.5)

        ann_data = []
        for sid, m in sorted(src_data.items()):
            c     = src_colors[sid]
            is_fd = sid in false_det_ids
            lbl   = chrono_lbl_fd[sid]
            if is_fd:
                ax1.scatter(m['flow_iou'], m['wav_abrupt'],
                            color=c, marker='x', s=110, zorder=4, linewidths=2)
            else:
                ax1.scatter(m['flow_iou'], m['wav_abrupt'],
                            facecolors='none', edgecolors=c, marker='o',
                            s=280, linewidths=1.4, zorder=3)
                ax1.scatter(m['flow_iou'], m['wav_abrupt'],
                            color=c, marker='o', s=28, zorder=4)
            ann_data.append((m['flow_iou'], m['wav_abrupt'], str(lbl), is_fd))

        # Group nearby annotations.
        ax1.set_xlim(-0.1, 1.0)
        ax1.set_ylim(-0.2, 1.3)
        _xspan = ax1.get_xlim()[1] - ax1.get_xlim()[0]
        _yspan = ax1.get_ylim()[1] - ax1.get_ylim()[0]
        _thresh_ann = 0.01
        used = [False] * len(ann_data)
        for i, (xi, yi, li, fdi) in enumerate(ann_data):
            if used[i]:
                continue
            gx, gy, gl = [xi], [yi], [li]
            used[i] = True
            for j, (xj, yj, lj, fdj) in enumerate(ann_data):
                if used[j] or fdj != fdi:
                    continue
                if (abs(xj - xi) / _xspan < _thresh_ann and
                        abs(yj - yi) / _yspan < _thresh_ann):
                    gx.append(xj); gy.append(yj); gl.append(lj)
                    used[j] = True
            cx, cy = float(np.mean(gx)), float(np.mean(gy))
            txt = ', '.join(sorted(gl, key=int))
            if fdi:
                ax1.annotate(txt, (cx, cy), fontsize=11, fontfamily='serif',
                             xytext=(10, 5), textcoords='offset points',
                             ha='left', va='bottom')
            else:
                ax1.annotate(txt, (cx, cy), fontsize=11, fontfamily='serif',
                             xytext=(0, 12), textcoords='offset points',
                             ha='center', va='bottom')

        ax1.axhline(wav_abrupt_thresh, color='0.5', ls='--', lw=0.9)
        ax1.axvline(flow_iou_thresh,   color='0.5', ls='--', lw=0.9)
        ax1.set_xlabel('Flow-advection IoU', fontsize=13)
        ax1.set_ylabel('Wavelet abruptness', fontsize=13)
        ax1.tick_params(which='both', direction='in', top=True, right=True)
        ax1.text(0.97, 0.03, 'Real Detections',   transform=ax1.transAxes,
                 ha='right', va='bottom', fontsize=11, color='0.4', fontfamily='serif')
        ax1.text(0.03, 0.97, 'False Detections',  transform=ax1.transAxes,
                 ha='left',  va='top',    fontsize=11, color='0.4', fontfamily='serif')

        # Panel 2: normalised wavelet profiles
        for sid, m in sorted(src_data.items(), key=lambda kv: fd_ch_min[kv[0]]):
            c     = src_colors[sid]
            is_fd = sid in false_det_ids
            lbl   = chrono_lbl_fd[sid]
            pk    = m['wav_prof'].max() + 1e-30
            norm  = m['wav_prof'] / pk
            vels  = (vel_array[np.array(m['det_chs'])]
                     if vel_array is not None else np.array(m['det_chs'], dtype=float))
            if is_fd:
                if m['n_det'] > 1:
                    ax2.plot(vels, norm, color=c, lw=1.3, ls='--', alpha=0.45, zorder=2)
            else:
                ax2.plot(vels, norm, color=c, lw=2.0, ls='-', alpha=1.0, zorder=3,
                         label=f'Source {lbl}')

        ax2.set_ylim(0.0, 1.1)
        xlabel = 'Velocity  (km s$^{-1}$)' if vel_array is not None else 'Channel'
        ax2.set_xlabel(xlabel, fontsize=13)
        ax2.set_ylabel('Normalised wavelet flux', fontsize=13)
        ax2.tick_params(which='both', direction='in', top=True, right=True)
        ax2.minorticks_on()

        # False detection peak markers.
        fig.canvas.draw()
        for sid, m in sorted(src_data.items()):
            if sid not in false_det_ids:
                continue
            pk       = m['wav_prof'].max() + 1e-30
            norm     = m['wav_prof'] / pk
            peak_idx = int(np.argmax(m['wav_prof']))
            vels     = (vel_array[np.array(m['det_chs'])]
                        if vel_array is not None else np.array(m['det_chs'], dtype=float))
            v_pk = float(vels[peak_idx])
            n_pk = float(norm[peak_idx])
            ax2.plot(v_pk, n_pk, 'o', color='red', ms=4, zorder=6)
            bb = ax2.get_window_extent()
            xr = ax2.get_xlim(); yr = ax2.get_ylim()
            r  = 12
            rx = r * (xr[1] - xr[0]) / bb.width
            ry = r * (yr[1] - yr[0]) / bb.height
            ax2.add_patch(matplotlib.patches.Ellipse(
                (v_pk, n_pk), width=2*rx, height=2*ry,
                fill=False, edgecolor='red', lw=1.0, ls='--', alpha=0.4, zorder=5,
            ))

        # Legend with custom FD proxy.
        class _DotCircleHandler(HandlerBase):
            def create_artists(self, legend, orig_handle,
                               xdescent, ydescent, width, height, fontsize, trans):
                cx = width / 2 - xdescent
                cy = height / 2 - ydescent
                r  = height * 0.5
                dot  = Line2D([cx], [cy], marker='o', color='red', ms=4,
                              linestyle='none', transform=trans)
                ring = matplotlib.patches.Ellipse(
                    (cx, cy), 4 * r, 4 * r,
                    fill=False, edgecolor='red', lw=1.0, ls='--', alpha=0.5,
                    transform=trans)
                return [ring, dot]

        fd_proxy = Line2D([], [])
        real_handles, real_labels = ax2.get_legend_handles_labels()
        ax2.legend(
            handles=real_handles + [fd_proxy],
            labels=real_labels   + ['False detections'],
            handler_map={fd_proxy: _DotCircleHandler()},
            fontsize=11, ncol=1, loc='upper left',
        )

        if results_dir is not None:
            fig.savefig(f'{results_dir}/false_detection_separation.png',
                        dpi=130, bbox_inches='tight')
            fig.savefig(f'{results_dir}/false_detection_separation.pdf',
                        dpi=130, bbox_inches='tight')
        plt.show()

    return good_sources, false_dets, src_data, src_colors


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_flow_tracker(
    cube: np.ndarray,
    channel_list: list[int] | None = None,
    scales: int = 6,
    k_sigma: float = 5.0,
    use_scale: int = 5,
    min_area: int = 20,
    thresh: float | None = None,
    use_mean_map_sigma: bool = True,
    min_match_overlap: int = 5,
    max_gap_channels: int = 5,
    min_displacement: float = 3.0,
    # Stage 5 — source classification
    wav_scale_idx: int = 3,
    wav_abrupt_thresh: float = 0.5,
    flow_iou_thresh: float = 0.25,
    short_det_max: int = 8,
    vel_array: np.ndarray | None = None,
    results_dir: str | Path | None = None,
    plot: bool = False,
    verbose: bool = False,
) -> tuple[
    list[ChannelDetection],
    list[tuple],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    dict,
    dict,
]:
    """Detect → flow → track → classify → group → classify sources.

    Runs all five pipeline stages and returns their combined outputs.

    Parameters
    ----------
    use_mean_map_sigma :
        Passed to :func:`~wavelet_detections.detect_cube_per_channel`.
        When ``True`` (default) the wavelet threshold is anchored to the
        per-scale noise from the mean map, preventing spurious detections
        on near-empty channels.
    wav_scale_idx :
        0-based wavelet scale index for source classification (should equal
        ``use_scale - 1``).
    wav_abrupt_thresh, flow_iou_thresh, short_det_max :
        Thresholds for :func:`classify_sources`.
    vel_array :
        1-D velocity array (km/s, length = cube.shape[0]).  Passed to
        :func:`classify_sources` for axis labelling when ``plot=True``.
    results_dir :
        Output directory for saved figures.  Passed to
        :func:`classify_sources` when ``plot=True``.
    plot :
        Render and save the false-detection separation figure.
    verbose :
        Print per-step progress and summary tables.

    Returns
    -------
    detections    : list[ChannelDetection]
    flow_seq      : list of (ch_ref, ch_tgt, flow, joint_mask)
    tracks        : list of classified track dicts (each annotated with source_id)
    sources       : list of all source dicts
    good_sources  : list of source dicts that passed the false-detection filter
    false_dets    : list of source dicts flagged as false detections
    src_data      : dict {source_id → metric dict (flow_iou, wav_abrupt, …)}
    src_colors    : dict {source_id → rgba tuple}  (tab10, cycled mod 10)
    """
    if channel_list is None:
        channel_list = list(range(cube.shape[0]))

    if verbose:
        print(f"[run_flow_tracker]  cube={cube.shape}  channels={len(channel_list)}"
              f"  (ch {channel_list[0]}–{channel_list[-1]})")
        print(f"  wavelet: scales={scales}  k_sigma={k_sigma}  use_scale={use_scale}"
              f"  min_area={min_area}  thresh={thresh}  mean_map_sigma={use_mean_map_sigma}")
        print(f"  tracker: min_match_overlap={min_match_overlap}"
              f"  max_gap_channels={max_gap_channels}"
              f"  min_displacement={min_displacement}\n")

    if verbose:
        print("[Stage 0] Per-channel wavelet detection...")
    detections = detect_cube_per_channel(
        cube, channel_list=channel_list,
        scales=scales, k_sigma=k_sigma,
        use_scale=use_scale, min_area=min_area, thresh=thresh,
        use_mean_map_sigma=use_mean_map_sigma,
    )
    if verbose:
        n_with_comps = sum(1 for d in detections if d.peaks)
        total_comps  = sum(len(d.peaks) for d in detections)
        print(f"  → {len(detections)} channels processed  |  "
              f"{n_with_comps} with detections  |  {total_comps} total components\n")

    flow_seq = compute_flow_sequence(detections, verbose=verbose)
    tracks = link_tracks(
        detections, flow_seq,
        min_match_overlap=min_match_overlap,
        max_gap_channels=max_gap_channels,
        verbose=verbose,
    )
    det_rev  = list(reversed(detections))
    flow_rev = [(b, a, -fl, mg) for (a, b, fl, mg) in reversed(flow_seq)]
    bwd_tracks = link_tracks(
        det_rev, flow_rev,
        min_match_overlap=min_match_overlap,
        max_gap_channels=max_gap_channels,
    )
    _reconcile_splits(tracks, bwd_tracks, verbose=verbose)
    classify_kinematic(tracks, min_displacement=min_displacement, verbose=verbose)
    sources = group_into_sources(tracks)

    if verbose:
        print("[Stage 5] Grouping into sources and removing false detections...")
    good_sources, false_dets, src_data, src_colors = classify_sources(
        sources, tracks, detections, flow_seq,
        wav_scale_idx=wav_scale_idx,
        wav_abrupt_thresh=wav_abrupt_thresh,
        flow_iou_thresh=flow_iou_thresh,
        short_det_max=short_det_max,
        verbose=verbose,
        plot=plot,
        vel_array=vel_array,
        results_dir=results_dir,
    )

    return detections, flow_seq, tracks, sources, good_sources, false_dets, src_data, src_colors


# ---------------------------------------------------------------------------
# FlowTracker — class-based API
# ---------------------------------------------------------------------------

@dataclass
class TrackingResult:
    """Output of :meth:`FlowTracker.run`.

    Wraps the full pipeline output — detections, flow, tracks, sources,
    false-detection list, and per-source metrics — in a single object.
    """
    detections: list
    flow_seq: list
    tracks: list
    sources: list
    false_detections: list
    src_data: dict
    src_colors: dict


class FlowTracker:
    """Full STORM pipeline: wavelet detection → optical flow → track linking
    → kinematic classification → source grouping → false-detection removal.

    Parameters
    ----------
    detector : WaveletDetector or None
        Wavelet detector instance.  ``None`` uses default settings.
    min_match_overlap : int
        Minimum pixel overlap to accept an advected→component continuation match.
    max_gap_channels : int
        Maximum consecutive unmatched channels before a track is deactivated.
    min_displacement : float
        Minimum cumulative centroid travel (px) to call a track kinematic.
    wav_scale_idx : int
        0-based wavelet scale index for source classification metrics.
    wav_abrupt_thresh : float
        Abruptness threshold above which a source is flagged as a false
        detection.
    flow_iou_thresh : float
        Flow-IoU threshold below which a short source is flagged as a false
        detection.
    short_det_max : int
        Maximum channel span for the flow-IoU false-detection criterion.

    Examples
    --------
    >>> from storm.detect import WaveletDetector
    >>> from storm.track import FlowTracker
    >>>
    >>> detector = WaveletDetector(scales=6, k_sigma=5.0, use_scale=5)
    >>> tracker  = FlowTracker(detector, min_match_overlap=5)
    >>> result   = tracker.run(cube, channel_list, verbose=True)
    >>>
    >>> result.sources          # real sources
    >>> result.false_detections # flagged false positives
    >>> result.tracks           # all individual tracks
    """

    def __init__(
        self,
        detector=None,
        min_match_overlap: int = 5,
        max_gap_channels: int = 5,
        min_displacement: float = 3.0,
        wav_scale_idx: int = 3,
        wav_abrupt_thresh: float = 0.5,
        flow_iou_thresh: float = 0.25,
        short_det_max: int = 8,
    ) -> None:
        from .detect import WaveletDetector
        self.detector = detector if detector is not None else WaveletDetector()
        self.min_match_overlap = min_match_overlap
        self.max_gap_channels = max_gap_channels
        self.min_displacement = min_displacement
        self.wav_scale_idx = wav_scale_idx
        self.wav_abrupt_thresh = wav_abrupt_thresh
        self.flow_iou_thresh = flow_iou_thresh
        self.short_det_max = short_det_max

    def run(
        self,
        cube: np.ndarray,
        channel_list: list[int] | None = None,
        vel_array: np.ndarray | None = None,
        results_dir=None,
        plot: bool = False,
        verbose: bool = False,
    ) -> TrackingResult:
        """Detect sources in *cube* and run the full tracking pipeline.

        Parameters
        ----------
        cube : (n_ch, H, W) float32
        channel_list : list of int or None
            Channels to process.  ``None`` processes all channels.
        vel_array : 1-D array or None
            Velocity axis (km/s) for plot axis labelling.
        results_dir : path-like or None
            Directory for saved figures when ``plot=True``.
        plot : bool
            Render and save the false-detection separation figure.
        verbose : bool
            Print per-step progress and summary tables.

        Returns
        -------
        TrackingResult
        """
        if verbose:
            print(
                "\n"
                "╔═══════════════════════════════╗\n"
                "║ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀ ║\n"
                "║ ⠀⠀⠀⠀⠀⠀⠀⣀⣀⡀⠀⠀⠀⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀ ║\n"
                "║ ⠀⠀⠀⠀⠀⠀⢰⣿⣿⣿⣷⡆⢀⣿⣿⣿⣿⣿⡿⠁⣠⣄⠀⠀⠀⠀ ║\n"
                "║ ⠀⣠⣀⠀⠀⠀⠈⢛⣿⣿⣿⠁⣸⣿⣿⣿⣿⣿⠃⢰⣿⣿⣷⣄⠀⠀ ║\n"
                "║ ⢰⣿⣿⣿⠀⢰⣶⣿⣿⣿⡇⠀⣿⣿⣿⣿⣿⣿⠀⢸⡟⠉⠙⣿⣷⠀ ║\n"
                "║ ⢸⣿⣿⣿⠀⢸⣿⣿⣿⣿⡇⢰⣿⣿⣿⣿⣿⣿⠀⢸⣷⣤⣴⣿⣿⡇ ║\n"
                "║ ⠸⣿⣿⣿⠀⠸⢿⣿⣿⣿⡇⠀⣿⣿⣿⣿⣿⣿⠀⢸⣿⣿⣿⣿⡿⠀ ║\n"
                "║ ⠀⠙⠉⠀⠀⢠⣾⣿⣿⣿⣿⡀⢻⣿⣿⣿⣿⣿⡄⠸⣿⣿⣿⠛⠀⠀ ║\n"
                "║ ⠀⠀⠀⠀⠀⠸⣿⣿⣿⣿⡿⠇⠘⣿⣿⣿⣿⣿⡷⠀⠙⠋⠀⠀⠀⠀ ║\n"
                "║ ⠀⠀⠀⠀⠀⠀⠈⠉⠉⠁⠀⠀⠀⣿⣿⣿⡿⠟⠀⠀⠀⠀⠀⠀⠀⠀ ║\n"
                "║ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ║\n"
                "╚══════════════════════════════╝\n"
            )
        detections = self.detector.detect(cube, channel_list)
        return self.run_from_detections(
            detections,
            vel_array=vel_array,
            results_dir=results_dir,
            plot=plot,
            verbose=verbose,
        )

    def run_from_detections(
        self,
        detections: list,
        vel_array: np.ndarray | None = None,
        results_dir=None,
        plot: bool = False,
        verbose: bool = False,
    ) -> TrackingResult:
        """Run the tracking pipeline on pre-computed *detections*.

        Useful when you want to inspect or filter detections before tracking.

        Parameters
        ----------
        detections : list[ChannelDetection]
            Output of :meth:`WaveletDetector.detect`.
        """
        flow_seq = compute_flow_sequence(detections, verbose=verbose)
        tracks = link_tracks(
            detections, flow_seq,
            min_match_overlap=self.min_match_overlap,
            max_gap_channels=self.max_gap_channels,
            verbose=verbose,
        )
        det_rev  = list(reversed(detections))
        flow_rev = [(b, a, -fl, mg) for (a, b, fl, mg) in reversed(flow_seq)]
        bwd_tracks = link_tracks(
            det_rev, flow_rev,
            min_match_overlap=self.min_match_overlap,
            max_gap_channels=self.max_gap_channels,
        )
        _reconcile_splits(tracks, bwd_tracks, verbose=verbose)
        classify_kinematic(tracks, min_displacement=self.min_displacement, verbose=verbose)
        all_sources = group_into_sources(tracks)
        good_sources, false_dets, src_data, src_colors = classify_sources(
            all_sources, tracks, detections, flow_seq,
            wav_scale_idx=self.wav_scale_idx,
            wav_abrupt_thresh=self.wav_abrupt_thresh,
            flow_iou_thresh=self.flow_iou_thresh,
            short_det_max=self.short_det_max,
            verbose=verbose,
            plot=plot,
            vel_array=vel_array,
            results_dir=results_dir,
        )
        return TrackingResult(
            detections=detections,
            flow_seq=flow_seq,
            tracks=tracks,
            sources=good_sources,
            false_detections=false_dets,
            src_data=src_data,
            src_colors=src_colors,
        )

    def __repr__(self) -> str:
        return (
            f"FlowTracker(detector={self.detector!r}, "
            f"min_match_overlap={self.min_match_overlap}, "
            f"max_gap_channels={self.max_gap_channels})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--cube",             required=True,
                    help="Cube file: .h5/.hdf5, .fits/.fit, .npy, .npz")
    ap.add_argument("--out",              required=True,
                    help="Output directory")
    ap.add_argument("--channels",         default=None,
                    help="Comma-separated channel indices; default: auto active")
    ap.add_argument("--active-threshold", type=float, default=0.05)
    ap.add_argument("--scales",           type=int,   default=6)
    ap.add_argument("--k-sigma",          type=float, default=5.0)
    ap.add_argument("--use-scale",        type=int,   default=5)
    ap.add_argument("--min-area",         type=int,   default=20)
    ap.add_argument("--thresh",           type=float, default=None)
    ap.add_argument("--min-match-overlap", type=int,   default=5,
                    help="Min pixel overlap (advected mask ∩ component) to match a continuation")
    ap.add_argument("--min-displacement", type=float, default=3.0,
                    help="Min centroid travel (px) to call a track kinematic")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cube = load_cube(args.cube)
    print(f"Cube: {cube.shape}  range [{cube.min():.3e}, {cube.max():.3e}]")

    if args.channels:
        channel_list = [int(c) for c in args.channels.split(",")]
    else:
        channel_list = active_channels(cube, threshold_frac=args.active_threshold)
        print(f"Auto-selected {len(channel_list)} active channels "
              f"(ch {channel_list[0]}–{channel_list[-1]})")

    (detections, flow_seq, tracks, sources,
     good_sources, false_dets, src_data, src_colors) = run_flow_tracker(
        cube, channel_list=channel_list,
        scales=args.scales, k_sigma=args.k_sigma,
        use_scale=args.use_scale, min_area=args.min_area, thresh=args.thresh,
        min_match_overlap=args.min_match_overlap,
        min_displacement=args.min_displacement,
        verbose=True,
    )

    n_kin = sum(1 for t in tracks if t['kinematic'])
    print(f"\n{len(tracks)} tracks  ({n_kin} kinematic)  →  "
          f"{len(good_sources)} real sources  +  {len(false_dets)} false detections")

    # Write tracks CSV — one row per (track, channel) pair.
    with open(out / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "track_id", "channel", "y", "x",
                    "displacement", "kinematic", "has_split", "split_from"])
        for t in tracks:
            for ch, y, x in t['trajectory']:
                w.writerow([
                    t.get('source_id', ""), t['id'], ch, f"{y:.2f}", f"{x:.2f}",
                    f"{t['displacement']:.3f}",
                    int(t['kinematic']), int(t['has_split']),
                    "" if t['split_from'] is None else t['split_from'],
                ])

    # Write sources CSV — one row per source.
    with open(out / "sources.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "track_ids", "ch_start", "ch_end",
                    "n_channels", "split_events", "merge_events"])
        for src in sources:
            w.writerow([
                src['id'],
                ";".join(str(i) for i in src['track_ids']),
                src['channels'][0], src['channels'][-1],
                src['n_channels'],
                ";".join(str(c) for c in src['split_events']),
                ";".join(str(c) for c in src['merge_events']),
            ])

    summary = {
        "cube": str(args.cube),
        "channels": channel_list,
        "n_tracks": len(tracks),
        "n_kinematic": n_kin,
        "n_sources": len(sources),
        "params": {k: v for k, v in vars(args).items()
                   if k not in ("cube", "out", "channels")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved tracks.csv, sources.csv, summary.json → {out}")


if __name__ == "__main__":
    main()
