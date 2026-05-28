"""Per-channel starlet (à trous IUWT) source detection for 3-D spectral cubes.

Strategy
--------
The starlet transform decomposes each 2-D spectral slice into detail bands at
increasing angular scales plus a coarse residual.  Compact sources produce
strong coefficients in the fine-scale bands; diffuse emission sits in the
coarser bands and is naturally suppressed.

For each channel slice, ``wavelet_footprints``:

1. Computes the starlet transform via a manual PyTorch à trous convolution with
   the B3-spline scaling function.
2. Thresholds each detail scale independently using a per-scale MAD noise
   estimate, so the threshold adapts to the actual signal level at that scale
   rather than collapsing on noise-free or low-signal data.
3. Applies an absolute floor of 10 % of the detection-plane peak to suppress
   float32 rounding artefacts in noise-free cubes.
4. Extracts connected emission components on the chosen detection scale and returns peak
   coordinates, binary footprint masks, and bounding boxes.

Input formats
-------------
``load_cube`` accepts .h5/.hdf5, .fits/.fit, .npy, .npz.

Usage (standalone)::

    python wavelet_detections.py \\
        --cube  data/clean_cube.npy \\
        --out   /tmp/detections \\
        --channels 70,74 \\
        --k-sigma 5 --scales 6 --use-scale 5 --min-area 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from skimage.measure import regionprops


# ---------------------------------------------------------------------------
# Starlet (à trous IUWT) transform — pure PyTorch
# ---------------------------------------------------------------------------

# B3-spline scaling function coefficients
_B3 = torch.tensor([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16], dtype=torch.float32)


def _atrous_conv2d(plane: torch.Tensor, dilation: int) -> torch.Tensor:
    """Separable B3-spline à trous convolution on a (1, 1, H, W) tensor."""
    h = _B3.to(plane.device)
    pad = 2 * dilation
    out = F.conv2d(plane, h.view(1, 1, 1, 5), padding=(0, pad), dilation=(1, dilation))
    out = F.conv2d(out,   h.view(1, 1, 5, 1), padding=(pad, 0), dilation=(dilation, 1))
    return out


def starlet_transform(image: np.ndarray, scales: int) -> np.ndarray:
    """Starlet (à trous IUWT) transform of a 2-D image.

    Parameters
    ----------
    image : (H, W) float32
    scales : total planes = (scales-1) detail bands + 1 coarse residual

    Returns
    -------
    np.ndarray, shape (scales, H, W)
        Planes 0 … scales-2 are detail bands (finest → coarsest).
        Plane scales-1 is the coarse residual.
    """
    c = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    planes: list[np.ndarray] = []
    for j in range(scales - 1):
        c_next = _atrous_conv2d(c, dilation=2 ** j)
        planes.append((c - c_next).squeeze().numpy())
        c = c_next
    planes.append(c.squeeze().numpy())   # coarse residual
    return np.stack(planes, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Multi-format cube loader
# ---------------------------------------------------------------------------

def load_cube(path: str | Path) -> np.ndarray:
    """Load a spectral cube from HDF5, FITS, .npy, or .npz.

    Always returns float32 (n_ch, H, W).  NaNs are replaced with 0.
    """
    path = Path(path)
    suf  = path.suffix.lower()

    if suf in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            cube = f["cube"][:].astype(np.float32)

    elif suf in (".fits", ".fit"):
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = hdul[0].data
        if data is None:
            raise ValueError(f"No data in primary HDU of {path}")
        data = np.squeeze(data).astype(np.float32)
        if data.ndim == 2:
            data = data[np.newaxis]
        if data.ndim != 3:
            raise ValueError(
                f"Cannot interpret FITS array with shape {data.shape} as (n_ch,H,W)"
            )
        cube = data

    elif suf == ".npy":
        cube = np.load(path).astype(np.float32)
        if cube.ndim == 2:
            cube = cube[np.newaxis]

    elif suf == ".npz":
        arch = np.load(path)
        key  = "cube" if "cube" in arch else list(arch.keys())[0]
        cube = arch[key].astype(np.float32)

    else:
        raise ValueError(
            f"Unsupported extension {suf!r}.  Use .h5/.hdf5, .fits/.fit, .npy, or .npz"
        )

    if cube.ndim != 3:
        raise ValueError(f"Loaded array has shape {cube.shape}; expected (n_ch, H, W)")

    np.nan_to_num(cube, copy=False, nan=0.0)
    return cube


def active_channels(cube: np.ndarray, threshold_frac: float = 0.05) -> list[int]:
    """Return indices of channels whose positive flux exceeds *threshold_frac* × max.

    Uses only positive flux so noise-dominated channels (where positive and
    negative values roughly cancel) do not inflate the total.
    """
    flux   = np.nansum(np.clip(cube, 0.0, None), axis=(1, 2))
    thresh = threshold_frac * float(flux.max())
    return [int(i) for i in np.where(flux >= thresh)[0]]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ChannelDetection(NamedTuple):
    """All detection results for a single spectral channel."""

    channel: int
    # Raw channel slice (H, W) float32.
    image: np.ndarray
    # One (H, W) bool mask per detected emission component.
    footprint_masks: list
    # (row, col) integer tuples, one per component.
    peaks: list
    # (y0, x0, y1, x1) integer tuples, one per component.
    boxes: list
    # Thresholded starlet coefficient cube (n_scales+1, H, W) float32.
    detect_coeffs: np.ndarray


# ---------------------------------------------------------------------------
# Per-channel detection
# ---------------------------------------------------------------------------

def reference_sigmas_from_mean_map(
    cube: np.ndarray,
    channel_list: list[int] | None,
    scales: int,
) -> np.ndarray:
    """Per-scale noise reference derived from the mean map across *channel_list*.

    The mean map has noise σ/√N relative to a single channel.  Per-scale MAD
    of its starlet coefficients is therefore multiplied by √N to recover an
    estimate of the single-channel noise at each wavelet scale.  This gives a
    stable, channel-independent threshold that is immune to the per-channel MAD
    collapse that occurs on near-empty channels (where residuals are tiny and
    deterministic, making per-channel σ → 0 and the threshold meaninglessly low).

    Parameters
    ----------
    cube : (n_ch, H, W) float32
    channel_list : list of channel indices to include; ``None`` uses all.
    scales : total number of starlet scales (including coarse residual).

    Returns
    -------
    np.ndarray, shape (scales - 1,)
        Single-channel noise estimate for each detail scale.
    """
    if channel_list is None:
        channel_list = list(range(cube.shape[0]))
    mean_map = cube[channel_list].mean(axis=0).astype(np.float32)
    coeffs   = starlet_transform(mean_map, scales=scales)
    n_detail = coeffs.shape[0] - 1
    sigmas   = np.empty(n_detail, dtype=np.float64)
    for i in range(n_detail):
        c = coeffs[i]
        sigmas[i] = 1.4826 * np.median(np.abs(c - np.median(c))) + 1e-12
    return sigmas


def wavelet_footprints(
    image: np.ndarray,
    scales: int = 4,
    k_sigma: float = 2.3,
    use_scale: int = 2,
    min_area: int = 10,
    thresh: float | None = None,
    sigma_per_scale: np.ndarray | None = None,
) -> ChannelDetection:
    """Detect compact-source footprints in a single 2-D image via starlet thresholding.

    Parameters
    ----------
    image :
        2-D spectral slice (H, W).
    scales :
        Total number of starlet scales (including the coarse residual plane).
    k_sigma :
        Detection threshold in units of per-scale noise.
    use_scale :
        1-based index of the detail band used for component detection.
        Scale 1 is the finest (sub-pixel structure); higher scales capture
        progressively larger compact sources.
    min_area :
        Minimum component area in pixels; smaller components are discarded as artefacts.
    thresh :
        Absolute lower bound on the detection-plane value.  ``None`` (default)
        sets it to 10 % of the detection-plane maximum.
    sigma_per_scale :
        Pre-computed per-scale noise array, shape (scales-1,).  When provided,
        these values replace the per-channel MAD estimate so that the threshold
        is anchored to a stable global reference (typically derived from the
        mean map via :func:`reference_sigmas_from_mean_map`).  Passing ``None``
        falls back to the original per-channel MAD behaviour.

    Returns
    -------
    ChannelDetection
        channel is set to -1 here; callers should replace it via ``._replace``.

    Notes
    -----
    When ``sigma_per_scale`` is ``None`` each scale is thresholded with its own
    MAD estimate.  This is adaptive but can collapse to near-zero on nearly
    empty channels (where denoised residuals are tiny), causing spurious
    detections.  Providing ``sigma_per_scale`` from the mean-map decomposition
    pins the threshold to the cube-wide noise floor and eliminates this failure.
    """
    img    = np.asarray(image, dtype=np.float32)
    coeffs = starlet_transform(img, scales=scales)

    detect = np.zeros_like(coeffs)
    for i in range(coeffs.shape[0] - 1):
        if sigma_per_scale is not None:
            sigma_i = float(sigma_per_scale[i]) + 1e-12
        else:
            sigma_i = 1.4826 * np.median(np.abs(coeffs[i] - np.median(coeffs[i]))) + 1e-12
        detect[i] = np.where(np.abs(coeffs[i]) > k_sigma * sigma_i, coeffs[i], 0.0)
    detect[-1] = coeffs[-1]   # coarse residual kept as-is
    detect[detect < 0] = 0    # positive emission only

    scale_idx = int(np.clip(use_scale - 1, 0, detect.shape[0] - 1))
    plane = detect[scale_idx]

    # 10 % of peak prevents float32 rounding artefacts (~1e-7) from
    # triggering detections when a signal-free channel makes sigma → 0.
    effective_thresh = 0.1 * float(plane.max()) if thresh is None else thresh
    binary           = plane > effective_thresh
    labeled, _       = label(binary)
    regions = [
        r for r in regionprops(labeled, intensity_image=plane) if r.area >= min_area
    ]

    peaks, footprint_masks, boxes = [], [], []
    for r in regions:
        y0, x0, y1, x1 = r.bbox
        patch = plane[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        py, px = np.unravel_index(np.argmax(patch), patch.shape)
        peaks.append((int(y0 + py), int(x0 + px)))
        footprint_masks.append(labeled == r.label)
        boxes.append((y0, x0, y1, x1))

    return ChannelDetection(
        channel=-1, image=img,
        footprint_masks=footprint_masks, peaks=peaks, boxes=boxes,
        detect_coeffs=detect,
    )


def detect_cube_per_channel(
    cube: np.ndarray,
    channel_list: list[int] | None = None,
    scales: int = 4,
    k_sigma: float = 2.3,
    use_scale: int = 2,
    min_area: int = 10,
    thresh: float | None = None,
    use_mean_map_sigma: bool = True,
    verbose: bool = False,
) -> list[ChannelDetection]:
    """Run ``wavelet_footprints`` on every channel in *channel_list*.

    Parameters
    ----------
    use_mean_map_sigma :
        When ``True`` (default), compute per-scale noise reference from the
        mean map across *channel_list* via :func:`reference_sigmas_from_mean_map`
        and pass it to every per-channel call.  This prevents the per-channel
        MAD from collapsing on near-empty channels and eliminates spurious
        detections in signal-free parts of the cube.  Set to ``False`` to
        revert to the original per-channel MAD behaviour.
    verbose :
        Print progress and summary statistics to stdout.

    Returns
    -------
    list[ChannelDetection]
        One entry per channel, with the ``channel`` field set to the cube channel index.
    """
    if channel_list is None:
        channel_list = list(range(cube.shape[0]))

    n_ch = len(channel_list)

    if verbose:
        print(f"[WaveletDetector] cube {cube.shape}  "
              f"range [{cube.min():.3e}, {cube.max():.3e}]")
        print(f"  scales={scales}  k_sigma={k_sigma}  use_scale={use_scale}  "
              f"min_area={min_area}  thresh={thresh}  "
              f"use_mean_map_sigma={use_mean_map_sigma}")
        print(f"  Processing {n_ch} channels: {channel_list[0]}–{channel_list[-1]}")

    sigma_ref = (
        reference_sigmas_from_mean_map(cube, channel_list, scales)
        if use_mean_map_sigma else None
    )

    if verbose and sigma_ref is not None:
        print(f"  Mean-map per-scale σ: "
              + "  ".join(f"s{i+1}={sigma_ref[i]:.3e}" for i in range(len(sigma_ref))))

    results = []
    total_det = 0
    channels_with_det = 0
    max_det = 0
    max_det_ch = channel_list[0]

    for idx, ch in enumerate(channel_list):
        det = wavelet_footprints(
            cube[ch],
            scales=scales, k_sigma=k_sigma,
            use_scale=use_scale, min_area=min_area, thresh=thresh,
            sigma_per_scale=sigma_ref,
        )
        det = det._replace(channel=ch)
        results.append(det)

        n = len(det.peaks)
        total_det += n
        if n > 0:
            channels_with_det += 1
        if n > max_det:
            max_det = n
            max_det_ch = ch

        if verbose:
            bar = "█" * n + "·" * max(0, 5 - n)
            print(f"  ch {ch:4d}  [{bar}]  {n:2d} detection(s)")

    if verbose:
        print(f"\n[WaveletDetector] Done.")
        print(f"  Channels processed : {n_ch}")
        print(f"  Channels with dets : {channels_with_det} / {n_ch}  "
              f"({100*channels_with_det/max(n_ch,1):.1f}%)")
        print(f"  Total detections   : {total_det}")
        print(f"  Peak channel       : ch {max_det_ch}  ({max_det} detections)")
        if n_ch > 0:
            print(f"  Mean per channel   : {total_det/n_ch:.2f}")

    return results


# ---------------------------------------------------------------------------
# WaveletDetector — class-based API
# ---------------------------------------------------------------------------

class WaveletDetector:
    """Starlet-wavelet per-channel source detector for 3-D spectral cubes.

    Parameters
    ----------
    scales : int
        Total number of starlet scales (including coarse residual).
    k_sigma : float
        Detection threshold in units of per-scale noise.
    use_scale : int
        1-based detail band used for component detection.
    min_area : int
        Minimum component area in pixels.
    thresh : float or None
        Absolute lower bound on detection-plane value.  ``None`` uses 10 % of
        the channel peak.
    use_mean_map_sigma : bool
        Anchor the noise estimate to the mean map across all channels rather
        than computing it per-channel.  Prevents spurious detections on nearly
        empty channels.

    Examples
    --------
    >>> detector = WaveletDetector(scales=6, k_sigma=5.0, use_scale=5)
    >>> detections = detector.detect(cube, channel_list)
    """

    def __init__(
        self,
        scales: int = 6,
        k_sigma: float = 5.0,
        use_scale: int = 5,
        min_area: int = 20,
        thresh: float | None = None,
        use_mean_map_sigma: bool = True,
    ) -> None:
        self.scales = scales
        self.k_sigma = k_sigma
        self.use_scale = use_scale
        self.min_area = min_area
        self.thresh = thresh
        self.use_mean_map_sigma = use_mean_map_sigma

    def detect(
        self,
        cube: np.ndarray,
        channel_list: list[int] | None = None,
        verbose: bool = False,
    ) -> list[ChannelDetection]:
        """Run per-channel wavelet detection on *cube*.

        Parameters
        ----------
        cube : (n_ch, H, W) float32
        channel_list : list of int or None
            Channel indices to process.  ``None`` processes all channels.
        verbose : bool
            Print per-channel progress and summary statistics.

        Returns
        -------
        list[ChannelDetection]
            One entry per channel in *channel_list*, in order.
        """
        return detect_cube_per_channel(
            cube,
            channel_list=channel_list,
            scales=self.scales,
            k_sigma=self.k_sigma,
            use_scale=self.use_scale,
            min_area=self.min_area,
            thresh=self.thresh,
            use_mean_map_sigma=self.use_mean_map_sigma,
            verbose=verbose,
        )

    def __repr__(self) -> str:
        return (
            f"WaveletDetector(scales={self.scales}, k_sigma={self.k_sigma}, "
            f"use_scale={self.use_scale}, min_area={self.min_area})"
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

    detections = detect_cube_per_channel(
        cube, channel_list=channel_list,
        scales=args.scales, k_sigma=args.k_sigma,
        use_scale=args.use_scale, min_area=args.min_area, thresh=args.thresh,
    )

    for det in detections:
        print(f"  ch {det.channel:4d}  {len(det.peaks)} components")

    summary = {
        "cube": str(args.cube),
        "channels": channel_list,
        "n_detections_per_channel": [len(d.peaks) for d in detections],
        "params": vars(args),
    }
    (out / "detections_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {out}/detections_summary.json")


if __name__ == "__main__":
    main()
