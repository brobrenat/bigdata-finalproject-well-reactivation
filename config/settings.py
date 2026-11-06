"""Konfigurasi terpusat untuk pipeline data sumur idle.

Semua angka dan path yang bisa berubah dikumpulkan di sini supaya tidak
tersebar di dalam kode pipeline.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# --- Lokasi data -----------------------------------------------------------
# Data mentah TIDAK boleh ditaruh di folder proyek. Folder proyek ada di
# /mnt/c yang ruangnya tipis dan lambat dibaca dari WSL.
DATA_ROOT = Path(os.environ.get("BIGDATA_DATA_ROOT", Path.home() / "bigdata-data"))

RAW_DIR = DATA_ROOT / "raw"
BRONZE_DIR = DATA_ROOT / "bronze"
SILVER_DIR = DATA_ROOT / "silver"
GOLD_DIR = DATA_ROOT / "gold"
MANIFEST_PATH = RAW_DIR / "manifest.json"

# --- Sumber data Petrinex --------------------------------------------------
PETRINEX_URL = (
    "https://www.petrinex.gov.ab.ca/publicdata/API/Files/{jurisdiction}/Vol/{month}/CSV"
)

# BC tidak tersedia lewat endpoint ini (HTTP 400), jadi hanya AB dan SK.
JURISDICTIONS = ("AB", "SK")

# Endpoint memakai jendela bergulir. Bulan sebelum 2022-01 mengembalikan 404.
# Nilai ini sengaja dibuat lebih longgar dari yang diketahui tersedia; bulan
# yang tidak ada akan dicatat sebagai "missing" dan dilewati, bukan gagal.
START_MONTH = "2021-01"
END_MONTH = None  # None berarti sampai bulan berjalan

# --- Parameter unduhan -----------------------------------------------------
DOWNLOAD_TIMEOUT_SEC = 300
DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_SEC = 5
USER_AGENT = "bigdata-idle-well-coursework/1.0"

# --- Aturan domain ---------------------------------------------------------
# AER menganggap sumur tidak aktif bila tidak ada aktivitas volumetrik selama
# enam bulan berturut-turut (tiga bulan untuk critical sour).
IDLE_THRESHOLD_MONTHS = 6

# Jendela pengamatan setelah idle spell berakhir. Idle spell yang masih
# berjalan dan lebih pendek dari ini ditandai "disensor", bukan negatif.
CENSOR_WINDOW_MONTHS = 12

# --- Data warehouse --------------------------------------------------------
# DuckDB membaca Parquet langsung, jadi tidak ada proses muat terpisah.
DUCKDB_PATH = DATA_ROOT / "warehouse.duckdb"


def month_range(start: str = START_MONTH, end: str | None = END_MONTH) -> list[str]:
    """Daftar bulan "YYYY-MM" dari start sampai end inklusif."""
    sy, sm = (int(x) for x in start.split("-"))
    if end is None:
        today = date.today()
        ey, em = today.year, today.month
    else:
        ey, em = (int(x) for x in end.split("-"))

    months: list[str] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return months
