"""Shared fixtures and synthetic-data helpers for NEMO tests."""
import numpy as np
import pytest

from nemo.detect import ChannelDetection


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def gaussian_blob(H: int, W: int, cy: float, cx: float,
                  sigma: float = 2.5, amplitude: float = 1.0) -> np.ndarray:
    """2-D Gaussian centred at (cy, cx)."""
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    return (amplitude * np.exp(-((ys - cy)**2 + (xs - cx)**2) / (2 * sigma**2))
            ).astype(np.float32)


def disk_mask(H: int, W: int, cy: int, cx: int, radius: int = 4) -> np.ndarray:
    """Boolean disk mask."""
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    return ((ys - cy)**2 + (xs - cx)**2) <= radius**2


def make_detection(channel: int, H: int, W: int,
                   centers: list[tuple[int, int]],
                   radius: int = 4) -> ChannelDetection:
    """Build a ChannelDetection with circular footprints at *centers*."""
    image = np.zeros((H, W), dtype=np.float32)
    masks, peaks, boxes = [], [], []
    for cy, cx in centers:
        m = disk_mask(H, W, cy, cx, radius)
        image[m] = 1.0
        ys, xs = np.where(m)
        masks.append(m)
        peaks.append((cy, cx))
        boxes.append((int(ys.min()), int(xs.min()),
                      int(ys.max()) + 1, int(xs.max()) + 1))
    return ChannelDetection(
        channel=channel, image=image,
        footprint_masks=masks, peaks=peaks, boxes=boxes,
        detect_coeffs=np.zeros((4, H, W), dtype=np.float32),
    )


def zero_flow_seq(detections: list[ChannelDetection]) -> list[tuple]:
    """Build a flow sequence of all-zero flow fields matching *detections*."""
    seq = []
    H, W = detections[0].image.shape
    for i in range(len(detections) - 1):
        flow = np.zeros((2, H, W), dtype=np.float32)
        mask = np.ones((H, W), dtype=bool)
        seq.append((detections[i].channel, detections[i + 1].channel, flow, mask))
    return seq


# ---------------------------------------------------------------------------
# Cube fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def static_cube():
    """10-channel 32×32 cube with a single stationary Gaussian blob (ch 2-8)."""
    n_ch, H, W = 10, 32, 32
    cube = np.zeros((n_ch, H, W), dtype=np.float32)
    for ch in range(2, 9):
        cube[ch] = gaussian_blob(H, W, cy=16, cx=16, sigma=3.0)
    return cube


@pytest.fixture
def moving_cube():
    """8-channel 32×32 cube — blob shifts 2 px/channel along the x-axis."""
    n_ch, H, W = 8, 32, 32
    cube = np.zeros((n_ch, H, W), dtype=np.float32)
    for ch in range(n_ch):
        cube[ch] = gaussian_blob(H, W, cy=16, cx=8 + ch * 2, sigma=2.5)
    return cube


@pytest.fixture
def split_cube():
    """6-channel 32×32 cube — source splits into two at channel 3."""
    n_ch, H, W = 6, 32, 32
    cube = np.zeros((n_ch, H, W), dtype=np.float32)
    for ch in range(3):                       # single blob in channels 0-2
        cube[ch] = gaussian_blob(H, W, cy=16, cx=16, sigma=2.5)
    for ch in range(3, n_ch):                 # two blobs in channels 3-5
        cube[ch] = (gaussian_blob(H, W, cy=16, cx=10, sigma=2.5) +
                    gaussian_blob(H, W, cy=16, cx=22, sigma=2.5))
    return cube
