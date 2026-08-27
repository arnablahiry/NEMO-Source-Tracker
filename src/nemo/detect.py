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


def beam_fwhm_px(path: str | Path) -> float | None:
    """Synthesised beam FWHM in pixels, read from a FITS header.

    Returns the geometric mean of the major and minor axes, ``sqrt(BMAJ·BMIN)``,
    converted to pixels via the spatial pixel scale.  This is the resolution
    limit of the data: no structure smaller than this exists in the image, and
    noise is correlated on exactly this scale, so wavelet bands finer than the
    beam contain correlated noise rather than sky signal.

    Handles the two ways ALMA cubes carry the beam: keywords in the primary
    header, and the per-plane ``CASAMBM`` binary table written when the beam
    varies across channels (in which case the median over channels is used).

    Returns
    -------
    float or None
        None when the file is not FITS or carries no beam information, so
        callers can fall back rather than fail.
    """
    path = Path(path)
    if path.suffix.lower() not in (".fits", ".fit"):
        return None

    from astropy.io import fits

    with fits.open(path) as hdul:
        hdr = hdul[0].header

        # Pixel scale (deg/px) — CDELT2, or the CD/PC matrix diagonal.
        scale = None
        for key in ("CDELT2", "CD2_2", "PC2_2"):
            if key in hdr and float(hdr[key]) != 0.0:
                scale = abs(float(hdr[key]))
                break
        if scale is None:
            return None

        bmaj = bmin = None
        if "BMAJ" in hdr and "BMIN" in hdr:
            bmaj, bmin = abs(float(hdr["BMAJ"])), abs(float(hdr["BMIN"]))
        else:
            # CASA per-plane beam table: BMAJ/BMIN columns in arcsec.
            for hdu in hdul[1:]:
                cols = getattr(getattr(hdu, "columns", None), "names", None)
                if cols and "BMAJ" in cols and "BMIN" in cols:
                    bmaj = float(np.median(hdu.data["BMAJ"])) / 3600.0
                    bmin = float(np.median(hdu.data["BMIN"])) / 3600.0
                    break

    if not bmaj or not bmin:
        return None
    return float(np.sqrt(bmaj * bmin) / scale)


def beam_area_px(fwhm_px: float) -> float:
    """Solid angle of a Gaussian beam, in pixels.

    ``Ω = π·BMAJ·BMIN / (4 ln 2) ≈ 1.1331·BMAJ·BMIN``.  Since
    :func:`beam_fwhm_px` returns the geometric mean ``sqrt(BMAJ·BMIN)``, the
    product is just its square.

    This is the natural floor for :func:`detect_all_scales`'s ``min_area``: a
    component smaller than the beam cannot be a resolved structure, because the
    instrument cannot record structure at that scale.

    .. note::

       This is a *physical validity* floor, not a false-positive filter.
       Interferometric noise is correlated on the beam scale, so noise
       fluctuations are themselves beam-sized and are **not** rejected by it.
       Controlling false positives is the threshold's job.
    """
    return float(np.pi * fwhm_px ** 2 / (4.0 * np.log(2.0)))


def active_channels(cube: np.ndarray, threshold_frac: float = 0.05) -> list[int]:
    """Return indices of channels whose positive flux exceeds *threshold_frac* × max.

    Uses only positive flux so noise-dominated channels (where positive and
    negative values roughly cancel) do not inflate the total.
    """
    flux   = np.nansum(np.clip(cube, 0.0, None), axis=(1, 2))
    thresh = threshold_frac * float(flux.max())
    return [int(i) for i in np.where(flux >= thresh)[0]]


def max_2d_scales(height: int, width: int) -> int:
    """Maximum number of starlet planes (detail bands + coarse residual) that
    fit a *height* × *width* image.

    The à-trous B3-spline kernel at detail band ``j`` spans ~``4·2^(j-1)+1``
    pixels, so the coarsest meaningful band has ``j ≈ log2(min(H, W))``.  That
    many detail bands plus the coarse residual gives the total plane count.
    """
    m = max(int(min(height, width)), 2)
    return max(int(np.floor(np.log2(m))), 3)


def default_detect_scales(scales: int) -> list[int]:
    """Default multi-scale detection bands: the three below the coarsest.

    Detail bands run ``1 … scales-1`` (the last plane is the coarse residual).
    The coarsest detail band, ``Nmax = scales-1``, blends neighbouring sources
    into one blob, so it is *excluded* by default; the default selection is the
    three bands below it — ``Nmax-1, Nmax-2, Nmax-3`` — and never the coarse
    residual.
    """
    nmax = scales - 1                      # coarsest detail band
    return sorted({s for s in (nmax - 1, nmax - 2, nmax - 3) if s >= 1})


def noise_maps_from_channels(
    cube: np.ndarray,
    channel_list: list[int] | None,
    scales: int,
    block: int = 32,
) -> np.ndarray:
    """Per-scale, spatially varying noise maps, from line-free channels.

    A single scalar σ per scale cannot describe a primary-beam-corrected image:
    pbcor divides out the beam response, so the noise rises toward the field
    edge (measured on IC5179, 1.25× higher at the edge than at centre).
    Thresholding such an image against one number either misses real structure
    at the centre or admits noise at the edge — and it is the edge that fills up
    first, because that is where σ is most underestimated.

    Two things are estimated together here:

    * **level** — measured on the quietest (line-free) channels, so the source
      does not inflate it.  Averaging signal-bearing channels leaves extended
      emission in the mean map and the ×√N rescaling then multiplies it up; on
      IC5179 that ran 1.26× too high at the finest scale and 11.7× at the
      coarsest.
    * **shape** — block-wise MAD on a coarse grid, then smoothed and resampled,
      so the map follows the primary-beam response without tracking individual
      sources.

    Returns
    -------
    np.ndarray, shape (scales - 1, H, W)
        Noise estimate per detail band at every pixel.
    """
    from scipy.ndimage import gaussian_filter, zoom

    if channel_list is None:
        channel_list = list(range(cube.shape[0]))
    n_detail = scales - 1
    H, W = cube.shape[1], cube.shape[2]

    planes = [starlet_transform(cube[ch].astype(np.float32), scales=scales)
              for ch in channel_list]

    ny, nx = max(H // block, 1), max(W // block, 1)
    out = np.empty((n_detail, H, W), dtype=np.float32)
    for i in range(n_detail):
        grid = np.empty((ny, nx), dtype=np.float64)
        for by in range(ny):
            y0, y1 = by * block, (by + 1) * block if by < ny - 1 else H
            for bx in range(nx):
                x0, x1 = bx * block, (bx + 1) * block if bx < nx - 1 else W
                vals = np.concatenate([p[i][y0:y1, x0:x1].ravel() for p in planes])
                grid[by, bx] = 1.4826 * np.median(np.abs(vals - np.median(vals)))
        # Smooth the coarse grid before resampling: the primary-beam response is
        # smooth, so block-to-block scatter is estimator noise, not structure.
        grid = gaussian_filter(grid, sigma=1.0, mode="nearest")
        big = zoom(grid, (H / ny, W / nx), order=1, mode="nearest")
        out[i] = np.clip(big[:H, :W], 1e-12, None).astype(np.float32)
    return out


def resolve_k_sigma(k_sigma, scale: int, default: float = 3.0) -> float:
    """Detection threshold in σ for one 1-based detail *scale*.

    ``k_sigma`` may be a single number applied to every band, or a
    ``{scale: k}`` mapping for per-scale control.

    Per-scale control matters because a fixed k does *not* give a fixed
    false-positive rate across scales.  Normalising by the per-scale σ equalises
    the per-*pixel* rate, but at coarse scales the à trous kernel is wider, so a
    noise excursion covers more area and clears a beam-sized ``min_area`` far
    more easily.  Measured on IC5179 with correct per-scale noise, detections
    falling in blank sky ran 8.4% / 15.2% / 19.2% at scales 2/3/4 for k=3
    (18% = indistinguishable from pure noise), and only flattened to
    7.0% / 7.9% / 8.5% once k was raised to 7.  Coarse bands need a stricter k.
    """
    if isinstance(k_sigma, dict):
        v = k_sigma.get(scale, k_sigma.get(str(scale)))
        return float(default if v is None else v)
    return float(default if k_sigma is None else k_sigma)


def kernel_radius_px(scale: int) -> float:
    """Support radius of the B3-spline à trous kernel at 1-based detail *scale*.

    The separable B3 kernel spans 5 taps, so at dilation ``2**(scale-1)`` it
    reaches ``2 * 2**(scale-1)`` pixels either side of centre.
    """
    return 2.0 * (2 ** (scale - 1))


def admissible_scales(scales: int, beam_fwhm_px: float | None = None) -> list[int]:
    """1-based detail bands coarse enough to carry real sky signal.

    Interferometric noise is spatially correlated on the beam scale, so bands
    finer than the beam contain correlated noise that mimics compact sources —
    detecting there manufactures features rather than finding them.

    Parameters
    ----------
    scales : total starlet planes (detail bands are ``1 … scales-1``).
    beam_fwhm_px : synthesised beam FWHM in pixels, or None to skip the check.
    """
    bands = list(range(1, scales))
    if beam_fwhm_px is not None:
        bands = [j for j in bands if kernel_radius_px(j) >= 0.5 * beam_fwhm_px]
    return bands


def hysteresis_components(band: np.ndarray, seed: np.ndarray,
                          grow: np.ndarray) -> np.ndarray:
    """Grow *seed* detections out to their true extent within *grow*.

    Dual-threshold (hysteresis) segmentation: label the permissive *grow* mask,
    then keep only components containing at least one confident *seed* pixel.
    Faint pixels are admitted solely when connected to something already
    confirmed, so isolated noise — which has no seed — is never promoted.

    This is what a single threshold cannot do: low-surface-brightness extended
    emission can sit below the per-pixel cut at every single pixel while being
    strongly significant integrated over many.  Measured on IC5179, orphaned
    fine detections (no coarse-scale parent) fell from 21.5% to 5.8% with a
    1σ grow level, while the coarse component *count* went down — real
    fragments merged to their true extent rather than new detections appearing.

    Parameters
    ----------
    band : (H, W) positive wavelet coefficients for one detail scale.
    seed : boolean mask of confidently-detected pixels.
    grow : boolean mask of plausible-emission pixels (a lower threshold).

    Returns
    -------
    np.ndarray
        Boolean mask: the union of every *grow* component holding a seed.
    """
    if not seed.any():
        return np.zeros_like(seed, dtype=bool)
    labeled, _ = label(grow)
    keep = np.unique(labeled[seed])
    keep = keep[keep != 0]
    if keep.size == 0:
        return np.zeros_like(seed, dtype=bool)
    return np.isin(labeled, keep)


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

def quietest_channels(cube: np.ndarray, channel_list: list[int] | None = None,
                      frac: float = 0.25) -> list[int]:
    """The *frac* of channels carrying least positive flux — a line-free set.

    Noise must be measured where the source is not.  :func:`active_channels`
    cannot supply that: its threshold is a fraction of *peak* flux, so on a
    cube whose faintest channel still holds ~10% of the peak it returns every
    channel (measured on IC5179: 60 of 60), and the "noise" reference is then
    built from signal-dominated data.

    Ranking by positive flux and taking the bottom slice gives the line-free
    channels directly, with no absolute threshold to tune.
    """
    if channel_list is None:
        channel_list = list(range(cube.shape[0]))
    flux = np.nansum(np.clip(cube[channel_list], 0.0, None), axis=(1, 2))
    n = max(1, int(round(frac * len(channel_list))))
    order = np.argsort(flux)[:n]
    return sorted(int(channel_list[i]) for i in order)


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
    n_ch     = max(len(channel_list), 1)
    mean_map = cube[channel_list].mean(axis=0).astype(np.float32)
    coeffs   = starlet_transform(mean_map, scales=scales)
    n_detail = coeffs.shape[0] - 1
    # Mean-map noise is σ/√N, so scale the per-scale MAD back up by √N to
    # recover the single-channel noise estimate at each wavelet scale.
    root_n   = np.sqrt(n_ch)
    sigmas   = np.empty(n_detail, dtype=np.float64)
    for i in range(n_detail):
        c = coeffs[i]
        sigmas[i] = 1.4826 * np.median(np.abs(c - np.median(c))) * root_n + 1e-12
    return sigmas


def wavelet_footprints(
    image: np.ndarray,
    scales: int = 4,
    k_sigma: float = 2.3,
    use_scale: int = 2,
    min_area: int | None = None,
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

    # `min_area=None` means "one beam", but this single-plane entry point has
    # no beam information; callers that do (detect_all_scales, WaveletDetector)
    # resolve it before getting here.
    if min_area is None:
        min_area = 10

    # Detection threshold: the per-scale noise gate, and nothing else.
    scale_idx = int(np.clip(use_scale - 1, 0, coeffs.shape[0] - 1))
    band = np.clip(coeffs[scale_idx], 0.0, None)   # positive coefficients
    if sigma_per_scale is not None:
        sig = float(sigma_per_scale[scale_idx]) + 1e-12
    else:
        cj = coeffs[scale_idx]
        sig = 1.4826 * np.median(np.abs(cj - np.median(cj))) + 1e-12
    binary = band > (k_sigma * sig)
    labeled, _       = label(binary)
    regions = [
        r for r in regionprops(labeled, intensity_image=band) if r.area >= min_area
    ]

    peaks, footprint_masks, boxes = [], [], []
    for r in regions:
        y0, x0, y1, x1 = r.bbox
        patch = band[y0:y1, x0:x1]
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
    min_area: int | None = None,
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
              f"min_area={min_area}  "
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
            use_scale=use_scale, min_area=min_area,
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
# Multi-scale detection
# ---------------------------------------------------------------------------

def detect_all_scales(
    cube: np.ndarray,
    channel_list: list[int] | None = None,
    scales: int = 6,
    k_sigma: float | dict[int, float] = 5.0,
    detect_scales: list[int] | None = None,
    min_area: int | None = None,
    use_mean_map_sigma: bool = True,
    beam_fwhm_px: float | None = None,
    grow_sigma: float | dict[int, float] | None = None,
    sigma_floor_frac: float = 0.3,
    noise_channel_frac: float = 1.0,
    spatial_noise: bool = False,
    noise_block: int = 16,
    verbose: bool = False,
):
    """Detect sources at multiple wavelet scales per channel.

    Parameters
    ----------
    cube : (n_ch, H, W) float32
    detect_scales : list of int or None
        1-based scale indices to detect at. None → [1,2,3,4] (skip residual).
    beam_fwhm_px : float or None
        Synthesised beam FWHM in pixels (see :func:`beam_fwhm_px`).  When given,
        bands finer than the beam are dropped: interferometric noise is
        correlated on the beam scale, so sub-beam bands contain correlated noise
        that mimics compact sources — detecting there manufactures features.
    min_area : int or None
        Minimum component area in pixels.  Pass ``None`` to derive it from the
        beam via :func:`beam_area_px`, which is the physical floor: nothing
        smaller than the beam can be a resolved structure.  ``None`` without
        *beam_fwhm_px* falls back to 10.
    grow_sigma : float, dict[int, float], or None
        Hysteresis grow level, in units of per-scale noise.  When set, the
        normal threshold becomes a *seed* and each detection is grown into
        connected pixels above ``grow_sigma · σ`` (see
        :func:`hysteresis_components`).  ``None`` keeps single-threshold
        behaviour.

        Pass a **dict** ``{scale: level}`` to grow only chosen bands — this is
        usually what you want.  Growing *every* band is counterproductive for
        the hierarchy: measured on IC5179, growing all bands cut parentless
        fine detections only 21.5% → 14.4%, because the fine detections swell
        past the coarse component's boundary and *lose* containment, whereas
        growing the coarse band alone reached 4.2%.  The coarse envelopes are
        what is under-detected; the fine detections are already fine.
    """
    from .hierarchy import PerChannelScaleDetections

    if detect_scales is None:
        detect_scales = default_detect_scales(scales)

    if beam_fwhm_px is not None:
        # The coarse residual (plane index `scales`) is the coarsest plane
        # there is, so the sub-beam floor can never exclude it.
        allowed = set(admissible_scales(scales, beam_fwhm_px)) | {scales}
        kept = [s for s in detect_scales if s in allowed]
        dropped = [s for s in detect_scales if s not in allowed]
        if dropped and verbose:
            print(f"[detect_all_scales] dropping sub-beam bands {dropped} "
                  f"(beam={beam_fwhm_px} px)")
        if not kept:
            raise ValueError(
                f"No detection band survives the beam floor for "
                f"beam_fwhm_px={beam_fwhm_px}: requested {detect_scales}, "
                f"admissible {sorted(allowed)}.  The data cannot support "
                f"detection at the requested scales."
            )
        detect_scales = kept

    if min_area is None:
        if beam_fwhm_px is not None:
            min_area = int(np.ceil(beam_area_px(beam_fwhm_px)))
            if verbose:
                print(f"[detect_all_scales] min_area = {min_area} px "
                      f"(one beam, FWHM {beam_fwhm_px:.2f} px)")
        else:
            min_area = 10
    if channel_list is None:
        channel_list = list(range(cube.shape[0]))

    # Compute global noise reference
    sigma_ref = None
    if use_mean_map_sigma:
        # Build the noise reference from line-free channels only.  Averaging
        # signal-bearing channels leaves the source in the mean map — extended
        # emission survives averaging while noise does not — and the ×√N
        # rescaling then inflates it.  Measured on IC5179 that ran 1.25× too
        # high at the finest scale and 11.4× at the coarsest; restricting to
        # the quietest channels gives 0.82–0.88× across all scales.
        #
        # DEFAULT IS OFF (1.0) DELIBERATELY.  Correcting σ alone is not safe on
        # primary-beam-corrected images: noise there is spatially non-uniform
        # (measured on IC5179, 1.25× higher at the field edge than at centre),
        # and this module thresholds against a single scalar σ per scale.  The
        # inflated estimate was masking that.  With σ corrected but still
        # scalar, detections at scale 4 spread uniformly over the field —
        # 17.8% of detected area fell in blank corners that occupy 18% of the
        # image, i.e. noise.  Enable this only together with a spatially
        # varying noise model, or by detecting on the non-pbcor image.
        _ref_chans = (quietest_channels(cube, channel_list, noise_channel_frac)
                      if noise_channel_frac and noise_channel_frac < 1.0
                      else channel_list)
        sigma_ref = reference_sigmas_from_mean_map(
            cube, _ref_chans, scales
        )
        if verbose and _ref_chans is not channel_list:
            print(f"[detect_all_scales] noise reference from {len(_ref_chans)} "
                  f"quietest of {len(channel_list)} channels")

    # Robustness floor for the per-channel estimator: the median per-channel σ
    # at each scale, sampled across channels, scaled down by `sigma_floor_frac`.
    # A channel whose own MAD collapses well below the cube-wide typical value
    # is not genuinely quiet — its residuals are near-deterministic — so the
    # floor keeps its threshold meaningful instead of letting it fall to ~0.
    # Spatially varying noise: the only correct model for a pbcor image, where
    # σ rises toward the field edge.  Built once from line-free channels.
    noise_map = None
    if spatial_noise:
        _nchans = quietest_channels(cube, channel_list, noise_channel_frac
                                    if 0.0 < noise_channel_frac < 1.0 else 0.25)
        noise_map = noise_maps_from_channels(cube, _nchans, scales,
                                             block=noise_block)
        if verbose:
            print(f"[detect_all_scales] spatial noise map from "
                  f"{len(_nchans)} line-free channels, {noise_block}px blocks")

    sigma_floor = None
    if sigma_ref is None and sigma_floor_frac > 0.0 and len(channel_list) > 1:
        sample = channel_list[:: max(1, len(channel_list) // 12)]
        acc = []
        for _ch in sample:
            _co = starlet_transform(cube[_ch].astype(np.float32), scales=scales)
            acc.append([1.4826 * np.median(np.abs(_co[i] - np.median(_co[i])))
                        for i in range(_co.shape[0] - 1)])
        sigma_floor = sigma_floor_frac * np.median(np.asarray(acc), axis=0)

    if verbose:
        print(f"[detect_all_scales] cube {cube.shape}  "
              f"range [{cube.min():.3e}, {cube.max():.3e}]")
        print(f"  scales={scales}  k_sigma={k_sigma}  detect_scales={detect_scales}  "
              f"min_area={min_area}  "
              f"use_mean_map_sigma={use_mean_map_sigma}")
        print(f"  Processing {len(channel_list)} channels: "
              f"{channel_list[0]}–{channel_list[-1]}")
        if sigma_ref is not None:
            print(f"  Mean-map per-scale σ: "
                  + "  ".join(f"s{i+1}={sigma_ref[i]:.3e}"
                              for i in range(len(sigma_ref))))

    # Per-scale tallies for the closing per-scale sections
    per_scale_total = {s: 0 for s in detect_scales}
    per_scale_chans = {s: 0 for s in detect_scales}
    per_scale_ch_n  = {s: [] for s in detect_scales}  # [(ch, n_det), ...]

    results = []
    for ch in channel_list:
        img = cube[ch].astype(np.float32)
        coeffs = starlet_transform(img, scales=scales)

        # Per-scale noise σ (for the optional noise gate on noisy cubes)
        detect = np.zeros_like(coeffs)
        sigma_per_scale = []
        for i in range(coeffs.shape[0] - 1):
            if noise_map is not None:
                # (H, W) array — broadcasting makes every downstream
                # comparison position-dependent with no other change.
                sigma_i = noise_map[i]
            elif sigma_ref is not None:
                sigma_i = float(sigma_ref[i]) + 1e-12
            else:
                # Per-channel, per-scale MAD measured in wavelet space — the
                # same space the threshold is applied in.  Unlike the mean-map
                # reference this carries no scale-dependent bias: measured
                # against blank sky on IC5179 it sits at 0.81–0.87× truth
                # across all five scales (spread 1.1×), where the mean-map
                # estimate ran 1.26× → 11.73× (spread 9.3×) because extended
                # emission survives channel-averaging and is then multiplied
                # by √N.
                sigma_i = 1.4826 * np.median(np.abs(coeffs[i] - np.median(coeffs[i]))) + 1e-12
                # Guard the failure this estimator is prone to: on a near-empty
                # channel the residuals are tiny and near-deterministic, MAD
                # collapses toward zero and the threshold becomes meaningless.
                # Floor it against the typical σ at this scale across channels.
                floor = sigma_floor[i] if sigma_floor is not None else 0.0
                if floor > 0.0 and sigma_i < floor:
                    sigma_i = floor
            sigma_per_scale.append(sigma_i)
            _k_i = resolve_k_sigma(k_sigma, i + 1)
            detect[i] = np.where(np.abs(coeffs[i]) > _k_i * sigma_i, coeffs[i], 0.0)
        # The coarse residual is a selectable detection band too, so it needs a
        # noise estimate like any other plane.  It has no mean-map/spatial
        # reference (both cover detail bands only), so measure it in place.
        _c = coeffs[-1]
        sigma_per_scale.append(
            1.4826 * np.median(np.abs(_c - np.median(_c))) + 1e-12)

        detect[-1] = coeffs[-1]
        detect[detect < 0] = 0

        # Detection threshold: the per-scale noise gate, and nothing else.
        scale_dets = {}
        for scale_idx in detect_scales:
            plane_idx = int(np.clip(scale_idx - 1, 0, coeffs.shape[0] - 1))
            band = np.clip(coeffs[plane_idx], 0.0, None)   # positive coefficients

            sig = (sigma_per_scale[plane_idx]
                   if plane_idx < len(sigma_per_scale) else 0.0)
            _k = resolve_k_sigma(k_sigma, scale_idx)
            binary = band > (_k * sig)        # sig may be scalar or (H, W)

            # Hysteresis: treat the above as *seeds* and grow them into
            # connected lower-significance emission.  Without this, extended
            # low-surface-brightness structure is discarded even when strongly
            # significant integrated — the cause of fine detections having no
            # coarse-scale parent.
            _gs = (grow_sigma.get(scale_idx) if isinstance(grow_sigma, dict)
                   else grow_sigma)
            if _gs is not None and plane_idx < len(sigma_per_scale):
                grow = band > (_gs * sigma_per_scale[plane_idx])
                binary = hysteresis_components(band, binary, grow)

            labeled, _ = label(binary)
            regions = [
                r for r in regionprops(labeled, intensity_image=band)
                if r.area >= min_area
            ]

            peaks, masks, boxes = [], [], []
            for r in regions:
                y0, x0, y1, x1 = r.bbox
                patch = band[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                py, px = np.unravel_index(np.argmax(patch), patch.shape)
                peaks.append((int(y0 + py), int(x0 + px)))
                masks.append((labeled == r.label).astype(bool))
                boxes.append((y0, x0, y1, x1))

            scale_dets[scale_idx] = (masks, peaks, boxes)
            n_det = len(peaks)
            per_scale_total[scale_idx] += n_det
            per_scale_ch_n[scale_idx].append((ch, n_det))
            if n_det:
                per_scale_chans[scale_idx] += 1

        # Stream a per-channel line as each channel is processed, so the log
        # fills in progressively during the (slow) detection pass.
        if verbose:
            counts = "  ".join(f"j{s}:{len(scale_dets[s][1]):>2}" for s in detect_scales)
            print(f"  ch {ch:4d}   {counts}")

        results.append(PerChannelScaleDetections(
            channel=ch,
            image=img,
            scales={s: scale_dets[s] for s in detect_scales},
            detect_coeffs=detect,
        ))

    if verbose:
        n_ch = len(channel_list)
        coarsest = max(detect_scales)
        # Comprehensive per-scale summary for each chosen scale
        for s in detect_scales:
            label_s = "coarsest" if s == coarsest else "detail"
            ch_n = per_scale_ch_n[s]
            peak_ch, peak_n = (max(ch_n, key=lambda t: t[1]) if ch_n
                               else (channel_list[0], 0))
            print(f"\n  SCALE j={s} ({label_s}):  "
                  f"{per_scale_total[s]} detection(s)  |  "
                  f"{per_scale_chans[s]}/{n_ch} channels with dets  |  "
                  f"peak ch {peak_ch} ({peak_n})")

        print(f"\n[detect_all_scales] Done — {n_ch} channels, scales {detect_scales}.")

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
        k_sigma: float | dict[int, float] = 5.0,
        use_scale: int = 5,
        min_area: int | None = None,
            use_mean_map_sigma: bool = True,
        detect_all_scales: bool = False,
        detect_scales: list[int] | None = None,
        beam_fwhm_px: float | None = None,
        grow_sigma: float | None = None,
    ) -> None:
        self.scales = scales
        self.k_sigma = k_sigma
        self.use_scale = use_scale
        self.min_area = min_area
        self.use_mean_map_sigma = use_mean_map_sigma
        self.detect_all_scales = detect_all_scales
        self.detect_scales = (detect_scales if detect_scales is not None
                              else default_detect_scales(scales))
        self.beam_fwhm_px = beam_fwhm_px
        self.grow_sigma = grow_sigma

    def detect(
        self,
        cube: np.ndarray,
        channel_list: list[int] | None = None,
        verbose: bool = False,
    ):
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
        list[ChannelDetection] (single-scale) or list[PerChannelScaleDetections] (multi-scale)
            One entry per channel in *channel_list*, in order.
        """
        if not self.detect_all_scales:
            # Legacy single-scale detection.  The beam/MRS bounds and FDR
            # thresholding are only wired into the multi-scale path.
            min_area = self.min_area
            if min_area is None:
                min_area = (int(np.ceil(beam_area_px(self.beam_fwhm_px)))
                            if self.beam_fwhm_px is not None else 10)
            return detect_cube_per_channel(
                cube,
                channel_list=channel_list,
                scales=self.scales,
                k_sigma=self.k_sigma,
                use_scale=self.use_scale,
                min_area=min_area,
                use_mean_map_sigma=self.use_mean_map_sigma,
                verbose=verbose,
            )
        else:
            # Multi-scale detection
            return detect_all_scales(
                cube,
                channel_list=channel_list,
                scales=self.scales,
                k_sigma=self.k_sigma,
                detect_scales=self.detect_scales,
                min_area=self.min_area,
                use_mean_map_sigma=self.use_mean_map_sigma,
                beam_fwhm_px=self.beam_fwhm_px,
                grow_sigma=self.grow_sigma,
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
        use_scale=args.use_scale, min_area=args.min_area,
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
