"""Tests for nemo.detect — starlet transform, active channels, wavelet detection."""
import numpy as np
import pytest

from nemo.detect import (
    ChannelDetection,
    WaveletDetector,
    active_channels,
    detect_cube_per_channel,
    reference_sigmas_from_mean_map,
    starlet_transform,
    wavelet_footprints,
)
from tests.conftest import gaussian_blob


# ---------------------------------------------------------------------------
# starlet_transform
# ---------------------------------------------------------------------------

class TestStarletTransform:
    def test_output_shape(self):
        img = np.random.rand(32, 32).astype(np.float32)
        for scales in (3, 5, 6):
            out = starlet_transform(img, scales=scales)
            assert out.shape == (scales, 32, 32), f"scales={scales}"

    def test_partition_of_unity(self):
        """Sum of all planes must reconstruct the original image."""
        img = np.random.rand(32, 32).astype(np.float32)
        planes = starlet_transform(img, scales=5)
        reconstructed = planes.sum(axis=0)
        np.testing.assert_allclose(reconstructed, img, atol=1e-4)

    def test_zero_input(self):
        img = np.zeros((16, 16), dtype=np.float32)
        out = starlet_transform(img, scales=4)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_output_dtype(self):
        img = np.ones((16, 16), dtype=np.float32)
        out = starlet_transform(img, scales=4)
        assert out.dtype == np.float32

    def test_coarse_residual_is_smooth(self):
        """Coarse residual should be smoother than the original image."""
        rng = np.random.default_rng(0)
        img = (rng.standard_normal((32, 32))).astype(np.float32)
        out = starlet_transform(img, scales=5)
        coarse_std = float(out[-1].std())
        original_std = float(img.std())
        assert coarse_std < original_std

    def test_known_source_has_nonzero_detail(self):
        """A bright point source should produce nonzero fine-scale detail."""
        img = np.zeros((32, 32), dtype=np.float32)
        img[16, 16] = 10.0
        out = starlet_transform(img, scales=4)
        assert out[0].max() > 0.0  # scale 1 (finest) should detect it

    def test_uniform_image_finest_band_zero_interior(self):
        """Finest detail band (scale 0) is zero in the image interior.

        The scale-0 B₃ kernel has support ±2 px; boundary contamination stays
        within 2 px of the edges.  Higher scales accumulate support and are
        not tested here.
        """
        img = np.full((32, 32), 3.14, dtype=np.float32)
        out = starlet_transform(img, scales=4)
        np.testing.assert_allclose(out[0, 4:-4, 4:-4], 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# active_channels
# ---------------------------------------------------------------------------

class TestActiveChannels:
    def test_bright_channels_selected(self):
        cube = np.zeros((10, 16, 16), dtype=np.float32)
        cube[3] = 1.0
        cube[5] = 1.0
        cube[7] = 0.5   # half peak — above 5% threshold
        channels = active_channels(cube, threshold_frac=0.05)
        assert 3 in channels
        assert 5 in channels
        assert 7 in channels

    def test_dim_channels_excluded(self):
        cube = np.zeros((10, 16, 16), dtype=np.float32)
        cube[4] = 1.0
        cube[0] = 0.01  # 1% of peak — below 5% threshold
        channels = active_channels(cube, threshold_frac=0.05)
        assert 4 in channels
        assert 0 not in channels

    def test_zero_threshold_returns_all(self):
        cube = np.ones((5, 8, 8), dtype=np.float32)
        channels = active_channels(cube, threshold_frac=0.0)
        assert len(channels) == 5

    def test_returns_sorted_list(self):
        cube = np.zeros((8, 16, 16), dtype=np.float32)
        for ch in (7, 2, 5):
            cube[ch] = 1.0
        channels = active_channels(cube)
        assert channels == sorted(channels)

    def test_returns_int_indices(self):
        cube = np.ones((4, 8, 8), dtype=np.float32)
        channels = active_channels(cube)
        assert all(isinstance(c, int) for c in channels)


# ---------------------------------------------------------------------------
# reference_sigmas_from_mean_map
# ---------------------------------------------------------------------------

class TestReferenceSignmas:
    def test_output_shape(self):
        cube = np.random.rand(5, 16, 16).astype(np.float32)
        for scales in (3, 5):
            sigmas = reference_sigmas_from_mean_map(cube, None, scales)
            assert sigmas.shape == (scales - 1,)

    def test_positive_values(self):
        cube = np.random.rand(6, 16, 16).astype(np.float32)
        sigmas = reference_sigmas_from_mean_map(cube, None, scales=5)
        assert (sigmas > 0).all()

    def test_channel_list_subset(self):
        cube = np.random.rand(10, 16, 16).astype(np.float32)
        sigmas_all = reference_sigmas_from_mean_map(cube, list(range(10)), scales=4)
        sigmas_sub = reference_sigmas_from_mean_map(cube, [0, 1, 2], scales=4)
        assert sigmas_all.shape == sigmas_sub.shape  # same shape regardless


# ---------------------------------------------------------------------------
# wavelet_footprints
# ---------------------------------------------------------------------------

class TestWaveletFootprints:
    def test_returns_channel_detection(self):
        img = gaussian_blob(32, 32, cy=16, cx=16, sigma=3.0)
        result = wavelet_footprints(img, scales=4, k_sigma=2.0, min_area=4)
        assert isinstance(result, ChannelDetection)

    def test_zero_image_no_detections(self):
        img = np.zeros((32, 32), dtype=np.float32)
        result = wavelet_footprints(img, scales=4, k_sigma=2.0, min_area=1)
        assert len(result.peaks) == 0

    def test_single_blob_one_detection(self):
        img = gaussian_blob(32, 32, cy=16, cx=16, sigma=3.0, amplitude=10.0)
        result = wavelet_footprints(img, scales=5, k_sigma=2.0, use_scale=3, min_area=4)
        assert len(result.peaks) == 1

    def test_peak_near_true_center(self):
        cy, cx = 16, 16
        img = gaussian_blob(32, 32, cy=cy, cx=cx, sigma=3.0, amplitude=10.0)
        result = wavelet_footprints(img, scales=5, k_sigma=2.0, use_scale=3, min_area=4)
        assert len(result.peaks) >= 1
        py, px = result.peaks[0]
        assert abs(py - cy) <= 3
        assert abs(px - cx) <= 3

    def test_min_area_filters_small_blobs(self):
        img = np.zeros((32, 32), dtype=np.float32)
        img[10, 10] = 5.0   # single pixel — too small
        img[15:20, 15:20] = 5.0  # 5×5 = 25 px region
        # With min_area=20, only the larger blob should survive
        result = wavelet_footprints(img, scales=4, k_sigma=0.5, use_scale=2, min_area=20)
        # All detections should be ≥ 20 px
        for mask in result.footprint_masks:
            assert mask.sum() >= 20

    def test_result_fields_consistent(self):
        img = gaussian_blob(32, 32, cy=16, cx=16, sigma=3.0, amplitude=10.0)
        result = wavelet_footprints(img, scales=4, k_sigma=2.0, use_scale=2, min_area=4)
        assert len(result.peaks) == len(result.footprint_masks) == len(result.boxes)

    def test_footprint_masks_are_boolean(self):
        img = gaussian_blob(32, 32, cy=16, cx=16, sigma=3.0, amplitude=10.0)
        result = wavelet_footprints(img, scales=4, k_sigma=2.0, use_scale=2, min_area=4)
        for mask in result.footprint_masks:
            assert mask.dtype == bool

    def test_with_sigma_per_scale(self):
        """Providing sigma_per_scale should not crash and still detect sources."""
        img = gaussian_blob(32, 32, cy=16, cx=16, sigma=3.0, amplitude=10.0)
        sigma_ref = np.full(3, 0.001, dtype=np.float64)  # tiny sigma → aggressive threshold
        result = wavelet_footprints(img, scales=4, k_sigma=2.0, use_scale=2,
                                    min_area=4, sigma_per_scale=sigma_ref)
        assert isinstance(result, ChannelDetection)


# ---------------------------------------------------------------------------
# WaveletDetector
# ---------------------------------------------------------------------------

class TestWaveletDetector:
    def test_repr_contains_params(self):
        d = WaveletDetector(scales=5, k_sigma=3.0, use_scale=3, min_area=15)
        r = repr(d)
        assert "5" in r and "3.0" in r and "3" in r

    def test_detect_returns_list_of_channel_detections(self, static_cube):
        det = WaveletDetector(scales=4, k_sigma=2.5, use_scale=2,
                              min_area=5).detect(static_cube)
        assert isinstance(det, list)
        assert all(isinstance(d, ChannelDetection) for d in det)

    def test_detect_channel_list_length(self, static_cube):
        channel_list = [2, 4, 6]
        det = WaveletDetector(scales=4, k_sigma=2.5, use_scale=2,
                              min_area=5).detect(static_cube, channel_list=channel_list)
        assert len(det) == len(channel_list)

    def test_detect_channel_fields_match(self, static_cube):
        channel_list = [2, 5, 8]
        det = WaveletDetector(scales=4, k_sigma=2.5, use_scale=2,
                              min_area=5).detect(static_cube, channel_list=channel_list)
        assert [d.channel for d in det] == channel_list

    def test_detect_finds_blob_in_bright_channels(self, static_cube):
        """Channels 2-8 have a source; the detector should find it."""
        det = WaveletDetector(scales=4, k_sigma=2.0, use_scale=2,
                              min_area=5).detect(static_cube, channel_list=list(range(2, 9)))
        detections_with_source = [d for d in det if len(d.peaks) > 0]
        assert len(detections_with_source) > 0

    def test_detect_no_blobs_in_empty_channels(self, static_cube):
        """Channels 0 and 1 are zero; no detections expected."""
        det = WaveletDetector(scales=4, k_sigma=2.0, use_scale=2,
                              min_area=5).detect(static_cube, channel_list=[0, 1])
        assert all(len(d.peaks) == 0 for d in det)

    def test_detect_all_channels_by_default(self, static_cube):
        det = WaveletDetector(scales=4, k_sigma=2.0,
                              use_scale=2, min_area=5).detect(static_cube)
        assert len(det) == static_cube.shape[0]

    def test_detect_cube_per_channel_equivalent(self, static_cube):
        """detect() wraps detect_cube_per_channel — results must match."""
        channel_list = [2, 4, 6]
        d = WaveletDetector(scales=4, k_sigma=2.5, use_scale=2, min_area=5)
        via_class  = d.detect(static_cube, channel_list=channel_list)
        via_func   = detect_cube_per_channel(
            static_cube, channel_list=channel_list,
            scales=4, k_sigma=2.5, use_scale=2, min_area=5,
        )
        assert len(via_class) == len(via_func)
        for a, b in zip(via_class, via_func):
            assert a.channel == b.channel
            assert len(a.peaks) == len(b.peaks)
