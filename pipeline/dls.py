"""Pengurai UWI Alberta dan konversi Dominion Land Survey ke lintang bujur.

Data volumetrik Petrinex mengidentifikasi sumur lewat UWI sepanjang 16
karakter, bukan koordinat. Peta pada dashboard butuh lintang bujur, jadi
koordinat diturunkan dari komponen DLS yang terkandung di dalam UWI itu.

Contoh UWI: 100010206504W400

    100  prefix survey dan kode location exception
    01   legal subdivision, 1 sampai 16
    02   section, 1 sampai 36
    065  township, 1 sampai 126
    04   range, 1 sampai 34
    W4   meridian
    00   nomor urut event

Koordinat yang dihasilkan bersifat PERKIRAAN, yaitu titik tengah legal
subdivision. Sistem DLS punya road allowance dan koreksi yang membuat
township tidak persis seragam, dan itu tidak dimodelkan di sini. Untuk
keperluan peta sebaran sumur, ketelitian ini memadai. Jangan dipakai untuk
keperluan survei.

Tata letak penomoran DLS berbentuk boustrophedon, yaitu berkelok seperti
bajak sawah. Section 1 berada di sudut tenggara township, lalu penomoran
bergerak ke barat sampai section 6, naik satu baris, dan berbalik arah.
Legal subdivision memakai pola yang sama di dalam satu section.
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple

# Satu township setinggi 6 mil, satu section 1 mil, satu LSD seperempat mil.
MILE_KM = 1.609344
DEG_LAT_PER_KM = 1.0 / 111.32
TOWNSHIP_HEIGHT_DEG = 6 * MILE_KM * DEG_LAT_PER_KM

# Baris LSD per township: 6 section dikali 4 LSD.
LSD_ROWS_PER_TOWNSHIP = 24

# Township 1 bermula di perbatasan Amerika Serikat.
BASE_LATITUDE = 49.0

# Bujur meridian acuan. Range dihitung ke arah barat dari meridian ini.
MERIDIAN_LONGITUDE = {"W1": -97.457, "W2": -102.0, "W3": -106.0, "W4": -110.0, "W5": -114.0, "W6": -118.0}

# Prefix boleh memuat huruf. Sebagian sumur memakai kode seperti 1S0 atau 1W0
# untuk menandai jenis survei khusus, dan komponen DLS sesudahnya tetap sama.
UWI_PATTERN = re.compile(r"^([0-9A-Z]{3})(\d{2})(\d{2})(\d{3})(\d{2})(W[1-6])(\d{2})$")


class Uwi(NamedTuple):
    """Komponen UWI Alberta yang sudah terurai."""

    raw: str
    prefix: str
    lsd: int
    section: int
    township: int
    range_: int
    meridian: str
    event: str

    @property
    def dls_location(self) -> str:
        """Bentuk baca manusia, contoh 01-02-065-04W4."""
        return (
            f"{self.lsd:02d}-{self.section:02d}-"
            f"{self.township:03d}-{self.range_:02d}{self.meridian}"
        )


def parse_uwi(value: str) -> Uwi | None:
    """Urai UWI 16 karakter. Mengembalikan None bila bentuknya tidak cocok."""
    if not value:
        return None
    text = value.strip().upper()
    match = UWI_PATTERN.match(text)
    if not match:
        return None

    prefix, lsd, section, township, range_, meridian, event = match.groups()
    lsd_i, section_i = int(lsd), int(section)
    township_i, range_i = int(township), int(range_)

    # Nilai di luar rentang menandakan UWI rusak, bukan sekadar tidak biasa.
    if not 1 <= lsd_i <= 16:
        return None
    if not 1 <= section_i <= 36:
        return None
    if not 1 <= township_i <= 126:
        return None
    if not 1 <= range_i <= 34:
        return None

    return Uwi(text, prefix, lsd_i, section_i, township_i, range_i, meridian, event)


def _snake_position(index_one_based: int, width: int) -> tuple[int, int]:
    """Posisi (baris dari selatan, kolom dari timur) pada grid berkelok.

    Nomor 1 berada di sudut tenggara. Baris genap dihitung dari timur ke
    barat, baris ganjil berbalik arah.
    """
    zero = index_one_based - 1
    row = zero // width
    offset = zero % width
    col_from_east = offset if row % 2 == 0 else (width - 1 - offset)
    return row, col_from_east


def uwi_to_latlon(uwi: Uwi) -> tuple[float, float] | None:
    """Titik tengah perkiraan dari sebuah LSD, dalam (lintang, bujur)."""
    base_longitude = MERIDIAN_LONGITUDE.get(uwi.meridian)
    if base_longitude is None:
        return None

    section_row, section_col_east = _snake_position(uwi.section, 6)
    lsd_row, lsd_col_east = _snake_position(uwi.lsd, 4)

    # Lintang: hitung jumlah baris LSD dari perbatasan selatan.
    rows_north = (uwi.township - 1) * LSD_ROWS_PER_TOWNSHIP + section_row * 4 + lsd_row
    latitude = BASE_LATITUDE + (rows_north + 0.5) * (
        TOWNSHIP_HEIGHT_DEG / LSD_ROWS_PER_TOWNSHIP
    )

    # Bujur: lebar 6 mil menyusut mengikuti kosinus lintang, jadi dihitung
    # pada lintang yang baru saja diperoleh.
    km_per_deg_lon = 111.32 * math.cos(math.radians(latitude))
    if km_per_deg_lon <= 0:
        return None
    range_width_deg = 6 * MILE_KM / km_per_deg_lon

    cols_west = (uwi.range_ - 1) * LSD_ROWS_PER_TOWNSHIP + section_col_east * 4 + lsd_col_east
    longitude = base_longitude - (cols_west + 0.5) * (
        range_width_deg / LSD_ROWS_PER_TOWNSHIP
    )

    return latitude, longitude


def uwi_string_to_latlon(value: str) -> tuple[float, float] | None:
    """Jalan pintas dari string UWI langsung ke koordinat."""
    parsed = parse_uwi(value)
    return uwi_to_latlon(parsed) if parsed else None
