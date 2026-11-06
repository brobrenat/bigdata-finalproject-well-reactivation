#!/usr/bin/env python3
"""Uji pengurai UWI dan konversi DLS ke koordinat.

Sengaja memakai unittest dari pustaka standar supaya bisa dijalankan tanpa
memasang apa pun.

Pakai:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.dls import parse_uwi, uwi_to_latlon, uwi_string_to_latlon  # noqa: E402

# Batas kasar provinsi Alberta, dipakai sebagai uji kewarasan koordinat.
ALBERTA_LAT = (48.9, 60.1)
ALBERTA_LON = (-120.5, -109.9)


class TestParseUwi(unittest.TestCase):
    def test_uwi_biasa(self):
        parsed = parse_uwi("100010206504W400")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.lsd, 1)
        self.assertEqual(parsed.section, 2)
        self.assertEqual(parsed.township, 65)
        self.assertEqual(parsed.range_, 4)
        self.assertEqual(parsed.meridian, "W4")
        self.assertEqual(parsed.dls_location, "01-02-065-04W4")

    def test_prefix_berhuruf(self):
        # Sebagian sumur memakai prefix seperti 1S0 atau 1W0.
        parsed = parse_uwi("1S0021406504W400")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.lsd, 2)
        self.assertEqual(parsed.section, 14)

    def test_bentuk_tidak_valid(self):
        for bad in ("", "abc", "100010206504X400", "10001020650", None):
            self.assertIsNone(parse_uwi(bad))

    def test_komponen_di_luar_rentang(self):
        # LSD 17 dan section 37 tidak ada dalam sistem DLS.
        self.assertIsNone(parse_uwi("100170206504W400"))
        self.assertIsNone(parse_uwi("100013706504W400"))


class TestKoordinat(unittest.TestCase):
    def test_di_dalam_alberta(self):
        for uwi in (
            "100010206504W400",
            "102023606404W400",
            "100011706503W400",
            "100013506404W400",
        ):
            with self.subTest(uwi=uwi):
                coords = uwi_string_to_latlon(uwi)
                self.assertIsNotNone(coords)
                lat, lon = coords
                self.assertTrue(ALBERTA_LAT[0] <= lat <= ALBERTA_LAT[1], lat)
                self.assertTrue(ALBERTA_LON[0] <= lon <= ALBERTA_LON[1], lon)

    def test_township_naik_berarti_ke_utara(self):
        south = uwi_string_to_latlon("100010201004W400")
        north = uwi_string_to_latlon("100010209004W400")
        self.assertLess(south[0], north[0])

    def test_range_naik_berarti_ke_barat(self):
        east = uwi_string_to_latlon("100010206502W400")
        west = uwi_string_to_latlon("100010206520W400")
        self.assertLess(west[1], east[1])

    def test_meridian_w5_lebih_barat_dari_w4(self):
        w4 = uwi_string_to_latlon("100010206501W400")
        w5 = uwi_string_to_latlon("100010206501W500")
        self.assertLess(w5[1], w4[1])

    def test_section_satu_di_tenggara(self):
        # Section 1 ada di sudut tenggara, section 31 di sudut barat laut.
        se = uwi_string_to_latlon("100010106504W400")
        nw = uwi_string_to_latlon("100013106504W400")
        self.assertLess(se[0], nw[0])  # lebih selatan
        self.assertGreater(se[1], nw[1])  # lebih timur

    def test_penomoran_berkelok_section(self):
        # Section 6 ada di ujung barat baris pertama, section 7 tepat di
        # atasnya karena penomoran berbalik arah. Jadi section 7 lebih utara
        # pada kolom yang sama.
        s6 = uwi_string_to_latlon("100010606504W400")
        s7 = uwi_string_to_latlon("100010706504W400")
        self.assertLess(s6[0], s7[0])

        # Bujurnya tidak persis sama. Lebar 6 mil menyempit ke arah utara
        # mengikuti kosinus lintang, jadi selisih kecil justru yang benar.
        # Yang diuji: keduanya masih di kolom yang sama, bukan bergeser satu
        # section yang jaraknya sekitar 0,025 derajat.
        self.assertLess(abs(s6[1] - s7[1]), 0.005)


if __name__ == "__main__":
    unittest.main()
