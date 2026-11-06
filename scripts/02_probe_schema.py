#!/usr/bin/env python3
"""Periksa isi satu bulan data Petrinex sebelum pipeline dibangun di atasnya.

Keputusan penting yang diambil dari keluaran skrip ini: tipe fasilitas mana
yang dianggap sumur. Jangan menebak, baca hasilnya dulu.

Skrip ini juga mencetak nilai unik untuk ActivityID dan ProductID, karena
data Petrinex berformat panjang. Satu baris adalah satu kombinasi fasilitas,
aktivitas, dan produk, jadi minyak, gas, dan air menempati baris terpisah
untuk bulan yang sama.

Pakai:
    python3 scripts/02_probe_schema.py
    python3 scripts/02_probe_schema.py --file ~/bigdata-data/bronze/petrinex/AB/2022-01.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import paths  # noqa: E402

# Kolom yang nilai uniknya ingin dilihat, beserta batas jumlah yang dicetak.
PROFILE_COLUMNS = {
    "ReportingFacilityType": 40,
    "ReportingFacilitySubType": 25,
    "ReportingFacilitySubTypeDesc": 25,
    "ActivityID": 40,
    "ProductID": 40,
    "ReportingFacilityProvinceState": 10,
    "ProductionMonth": 5,
}


def pick_default_file() -> Path | None:
    bronze = paths.local_dir("bronze") / "petrinex"
    candidates = sorted(bronze.glob("*/*.csv"))
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Batasi baris yang dibaca, 0 berarti seluruh berkas",
    )
    args = parser.parse_args()

    path = args.file or pick_default_file()
    if path is None or not path.exists():
        print("Tidak ada CSV bronze. Jalankan unduhan lalu pipeline/b1_extract.py.")
        return 1

    print(f"Berkas : {path}")
    print(f"Ukuran : {path.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    counters = {col: Counter() for col in PROFILE_COLUMNS}
    facility_by_type: Counter = Counter()
    blank_volume = 0
    rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []

        print(f"Jumlah kolom: {len(header)}")
        for name in header:
            print(f"  - {name}")
        print()

        missing = [c for c in PROFILE_COLUMNS if c not in header]
        if missing:
            print(f"PERINGATAN kolom tidak ditemukan: {missing}")
            print()

        for row in reader:
            rows += 1
            for col in counters:
                if col in row:
                    counters[col][row[col] or "(kosong)"] += 1

            ftype = row.get("ReportingFacilityType") or "(kosong)"
            facility_by_type[(ftype, row.get("ReportingFacilityID") or "")] += 0
            if not (row.get("Volume") or "").strip():
                blank_volume += 1

            if args.max_rows and rows >= args.max_rows:
                break

    print(f"Total baris dibaca: {rows:,}")
    print(f"Baris dengan Volume kosong: {blank_volume:,}")
    print()

    for col, limit in PROFILE_COLUMNS.items():
        counter = counters[col]
        if not counter:
            continue
        print(f"{col}  ({len(counter)} nilai unik)")
        for value, count in counter.most_common(limit):
            print(f"    {value:<45} {count:>10,}")
        if len(counter) > limit:
            print(f"    ... dan {len(counter) - limit} nilai lain")
        print()

    # Jumlah fasilitas unik per tipe. Ini yang menentukan tipe mana yang
    # benar-benar merepresentasikan sumur individual.
    unique_facilities: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    for ftype, fid in facility_by_type:
        if (ftype, fid) not in seen:
            seen.add((ftype, fid))
            unique_facilities[ftype] += 1

    print("Jumlah fasilitas unik per tipe")
    for ftype, count in unique_facilities.most_common(40):
        print(f"    {ftype:<45} {count:>10,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
