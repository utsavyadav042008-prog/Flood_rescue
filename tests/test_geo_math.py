import math
import unittest

from pipeline.geo_math import (central_meridian, epsg_for_utm, lonlat_to_utm, utm_to_lonlat,
                               utm_zone_from_epsg)

A, F = 6378137.0, 1 / 298.257223563
E2 = F * (2 - F)


def meridian_arc(lat_deg: float, steps: int = 20000) -> float:
    """Independent reference: numerical integral of the WGS84 meridian arc (Simpson)."""
    phi = math.radians(lat_deg)
    f = lambda p: A * (1 - E2) / (1 - E2 * math.sin(p) ** 2) ** 1.5
    h = phi / steps
    s = f(0) + f(phi) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, steps))
    return s * h / 3


class GeoMathTests(unittest.TestCase):
    def test_central_meridian(self):
        self.assertEqual(central_meridian(45), 87.0)

    def test_epsg_helpers(self):
        self.assertEqual(utm_zone_from_epsg(32645), (45, True))
        self.assertEqual(utm_zone_from_epsg(32733), (33, False))
        self.assertEqual(epsg_for_utm(45, True), 32645)
        with self.assertRaises(ValueError):
            utm_zone_from_epsg(4326)

    def test_central_meridian_easting_is_false_easting(self):
        e, _ = lonlat_to_utm(87.0, 28.0, 32645)
        self.assertAlmostEqual(e, 500000.0, places=6)

    def test_northing_on_central_meridian_matches_meridian_arc(self):
        for lat in (10.0, 28.0, 45.0):
            _, n = lonlat_to_utm(87.0, lat, 32645)
            self.assertAlmostEqual(n, 0.9996 * meridian_arc(lat), delta=0.001)   # 1 mm

    def test_round_trip(self):
        for lon, lat in ((85.2, 28.1), (84.0, 27.0), (89.9, 30.5)):
            e, n = lonlat_to_utm(lon, lat, 32645)
            lon2, lat2 = utm_to_lonlat(e, n, 32645)
            self.assertAlmostEqual(lon, lon2, places=8)
            self.assertAlmostEqual(lat, lat2, places=8)

    def test_southern_hemisphere_round_trip(self):
        e, n = lonlat_to_utm(10.5, -19.5, 32732)
        self.assertGreater(n, 5_000_000)         # false northing applied
        lon, lat = utm_to_lonlat(e, n, 32732)
        self.assertAlmostEqual(lon, 10.5, places=8)
        self.assertAlmostEqual(lat, -19.5, places=8)


if __name__ == "__main__":
    unittest.main()
