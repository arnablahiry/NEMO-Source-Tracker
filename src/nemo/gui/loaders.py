from pathlib import Path

import numpy as np


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
        data = None
        key  = None
        if "cube" in f and isinstance(f["cube"], h5py.Dataset) and f["cube"].ndim == 3:
            key  = "cube"
            data = f[key][()].astype(np.float32)
        else:
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

        if "beam" in f and not isinstance(f["beam"], h5py.Dataset):
            bg = f["beam"]
            if "bmaj_px" in bg.attrs:
                bmaj_px = float(bg.attrs["bmaj_px"])
                bmin_px = float(bg.attrs.get("bmin_px", bmaj_px))
                bpa     = float(bg.attrs.get("bpa_deg", 0.0))
                beam     = (bmaj_px, bmin_px, bpa)
                pixscale = 1.0

        if "spatial_resolution_kpc_per_px" in f.attrs and pixscale == 1.0:
            pass

        for src in (f, f[key]):
            if beam is None and "BMAJ" in src.attrs:
                beam = (float(src.attrs["BMAJ"])*3600,
                        float(src.attrs.get("BMIN", src.attrs["BMAJ"]))*3600,
                        float(src.attrs.get("BPA", 0.0)))
            if pixscale is None and "CDELT1" in src.attrs:
                pixscale = abs(float(src.attrs["CDELT1"])) * 3600

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
