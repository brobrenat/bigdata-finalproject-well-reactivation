#!/usr/bin/env python3
"""Uji ekstraksi zip bersarang Petrinex.

Endpoint mengembalikan zip di dalam zip. Lapis kedua mudah terlewat, dan
kalau terlewat maka yang masuk ke bronze adalah berkas zip biner yang
dikira CSV. Uji ini mengunci perilaku itu.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.b1_extract import extract_csv_bytes  # noqa: E402

CSV_BODY = b"ProductionMonth,Volume\n2022-01,12.5\n"


def build_nested_zip(path: Path) -> None:
    """Tiru bentuk asli: zip berisi zip berisi CSV."""
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w", zipfile.ZIP_DEFLATED) as inner:
        inner.writestr("Vol_2022-01-AB.CSV", CSV_BODY)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as outer:
        outer.writestr("Vol_2022-01-AB.csv.zip", inner_buffer.getvalue())


def build_flat_zip(path: Path) -> None:
    """Bentuk satu lapis, untuk berjaga kalau format berubah."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as outer:
        outer.writestr("Vol_2022-01-AB.CSV", CSV_BODY)


class TestExtract(unittest.TestCase):
    def test_zip_bersarang(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested.zip"
            build_nested_zip(path)
            name, data = extract_csv_bytes(path)
            self.assertTrue(name.upper().endswith(".CSV"))
            self.assertEqual(data, CSV_BODY)

    def test_zip_satu_lapis(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flat.zip"
            build_flat_zip(path)
            name, data = extract_csv_bytes(path)
            self.assertTrue(name.upper().endswith(".CSV"))
            self.assertEqual(data, CSV_BODY)

    def test_hasil_bukan_zip(self):
        # Kalau lapis kedua terlewat, isi yang keluar akan diawali "PK",
        # yaitu tanda berkas zip. Pastikan itu tidak terjadi.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested.zip"
            build_nested_zip(path)
            _, data = extract_csv_bytes(path)
            self.assertFalse(data.startswith(b"PK"))

    def test_zip_kosong(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.zip"
            with zipfile.ZipFile(path, "w"):
                pass
            with self.assertRaises(ValueError):
                extract_csv_bytes(path)


if __name__ == "__main__":
    unittest.main()
