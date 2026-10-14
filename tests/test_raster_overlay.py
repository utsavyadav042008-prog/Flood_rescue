import struct
import unittest
import zlib

import numpy as np

from dashboard import raster_overlay as ro
from pipeline.geo_math import lonlat_to_utm, lonlat_to_utm_array, utm_to_lonlat

EPSG = 32645


def decode_png_rgba(data: bytes) -> np.ndarray:
    """Minimal decoder for what encode_png_rgba writes (filter 0 only)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag, payload = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        assert crc == zlib.crc32(tag + payload) & 0xFFFFFFFF, "bad CRC"
        if tag == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        elif tag == b"IDAT":
            idat += payload
        pos += 12 + length
    raw = zlib.decompress(idat)
    rows = [np.frombuffer(raw[i * (1 + width * 4) + 1:(i + 1) * (1 + width * 4)], dtype=np.uint8)
            for i in range(height)]
    return np.stack(rows).reshape(height, width, 4)


class PngTests(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (7, 11, 4), dtype=np.uint8)
        np.testing.assert_array_equal(decode_png_rgba(ro.encode_png_rgba(img)), img)

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            ro.encode_png_rgba(np.zeros((3, 3, 3), dtype=np.uint8))

    def test_pillow_can_open_it_when_available(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        import io
        img = np.zeros((5, 9, 4), dtype=np.uint8)
        img[2, 3] = (255, 0, 0, 255)
        im = Image.open(io.BytesIO(ro.encode_png_rgba(img)))
        self.assertEqual(im.size, (9, 5))
        self.assertEqual(im.convert("RGBA").getpixel((3, 2)), (255, 0, 0, 255))


class VectorisedUtmTests(unittest.TestCase):
    def test_matches_scalar_version(self):
        lon, lat = np.array([85.0, 85.3, 86.9]), np.array([27.8, 28.2, 28.5])
        e, n = lonlat_to_utm_array(lon, lat, EPSG)
        for i in range(3):
            se, sn = lonlat_to_utm(lon[i], lat[i], EPSG)
            self.assertAlmostEqual(e[i], se, places=6)
            self.assertAlmostEqual(n[i], sn, places=6)


class WarpTests(unittest.TestCase):
    def setUp(self):
        # 400 x 300 px grid, 50 m pixels, north-up, around Trishuli
        self.x0, self.y0 = 320_000.0, 3_115_000.0
        self.transform = (50.0, 0.0, self.x0, 0.0, -50.0, self.y0)
        self.array = np.zeros((300, 400), dtype=np.uint8)

    def test_footprint_contains_the_corners(self):
        w, s, e, n = ro.footprint_lonlat(self.transform, self.array.shape, EPSG)
        lon, lat = utm_to_lonlat(self.x0 + 10_000, self.y0 - 7_500, EPSG)   # centre
        self.assertTrue(w < lon < e and s < lat < n)

    def test_a_blob_lands_at_the_right_longitude_and_latitude(self):
        # blob centred 5 km east, 3 km south of the top-left corner
        cx, cy = self.x0 + 5_000, self.y0 - 3_000
        col, row = int((cx - self.x0) / 50), int((self.y0 - cy) / 50)
        self.array[row - 4:row + 5, col - 4:col + 5] = 1
        out, inside, (w, s, e, n) = ro.warp_to_lonlat(self.array, self.transform, EPSG, max_px=600)
        lon, lat = utm_to_lonlat(cx, cy, EPSG)
        out_col = int((lon - w) / (e - w) * out.shape[1])
        out_row = int((n - lat) / (n - s) * out.shape[0])
        self.assertEqual(out[out_row, out_col], 1)
        # and the blob does not smear far away
        ys, xs = np.nonzero(out)
        self.assertLess(abs(xs.mean() - out_col), 3)
        self.assertLess(abs(ys.mean() - out_row), 3)

    def test_output_is_capped_and_corners_of_the_lonlat_box_are_outside(self):
        out, inside, _ = ro.warp_to_lonlat(self.array, self.transform, EPSG, max_px=200)
        self.assertLessEqual(max(out.shape), 200)
        # UTM north is rotated against true north, so a lon/lat box has empty corners
        self.assertFalse(inside.all())
        self.assertTrue(inside.mean() > 0.9)

    def test_rotated_transform_is_rejected(self):
        with self.assertRaises(ValueError):
            ro.warp_to_lonlat(self.array, (50.0, 1.0, 0.0, 0.0, -50.0, 0.0), EPSG)


class ColourTests(unittest.TestCase):
    def test_mask_rgba_is_transparent_outside_and_blue_inside(self):
        values = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        inside = np.array([[True, True], [True, False]])
        rgba = ro.mask_rgba(values, inside)
        self.assertEqual(rgba[0, 0, 3], 0)
        self.assertEqual(tuple(rgba[0, 1]), (31, 120, 180, 170))
        self.assertEqual(rgba[1, 1, 3], 0)

    def test_probability_alpha_scales_and_low_values_vanish(self):
        values = np.array([[0.0, 0.04, 0.5, 1.0]])
        rgba = ro.probability_rgba(values, np.ones((1, 4), dtype=bool))
        self.assertEqual(list(rgba[0, :, 3]), [0, 0, 100, 200])

    def test_gray_ignores_nan_and_stretches(self):
        values = np.linspace(-20, 0, 100).reshape(10, 10)
        values[0, 0] = np.nan
        rgba = ro.gray_rgba(values)
        self.assertTrue((rgba[..., 3] == 255).all())
        self.assertEqual(int(rgba[..., 0].max()), 255)

    def test_downsample(self):
        self.assertEqual(max(ro.downsample(np.zeros((2500, 1000)), 900).shape), 834)
        self.assertEqual(ro.downsample(np.zeros((10, 10)), 900).shape, (10, 10))

    def test_overlay_png_end_to_end(self):
        arr = np.zeros((120, 160), dtype=np.uint8)
        arr[40:80, 60:100] = 1
        data, bounds = ro.overlay_png("mask", arr, (50.0, 0.0, 320_000.0, 0.0, -50.0, 3_115_000.0), EPSG, 300)
        img = decode_png_rgba(data)
        self.assertGreater(int((img[..., 3] > 0).sum()), 100)
        self.assertEqual(len(bounds), 4)


if __name__ == "__main__":
    unittest.main()
