#!/usr/bin/env python3
"""Hitung cepat jumlah label idle yang bisa dipakai, tanpa Spark.

Gunanya menjawab gerbang keputusan sebelum kode model ditulis: apakah jumlah
sumur yang sempat idle lalu kembali berproduksi cukup untuk dilatih.

Skrip ini membaca CSV bronze secara streaming dan hanya mengambil empat hal
per baris produksi sumur, yaitu bulan, UWI, produk, dan volume. Sisanya
dibuang, sehingga seluruh dataset bisa diringkas menjadi peta aktivitas
sumur terhadap bulan yang muat di memori.

Hasilnya harus cocok dengan keluaran pipeline/g1_idle_spells.py. Kalau tidak
cocok, salah satu dari keduanya keliru.

Pakai:
    python3 scripts/03_label_feasibility.py
    python3 scripts/03_label_feasibility.py --jurisdiction AB
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import paths, settings  # noqa: E402

# Posisi kolom pada CSV Petrinex. Dipakai langsung supaya jauh lebih cepat
# daripada DictReader saat memproses puluhan juta baris.
COL_PRODUCTION_MONTH = 0
COL_ACTIVITY_ID = 17
COL_PRODUCT_ID = 18
COL_FROM_TO_TYPE = 21
COL_FROM_TO_IDENTIFIER = 22
COL_VOLUME = 23

EXPECTED_HEADER = {
    COL_PRODUCTION_MONTH: "ProductionMonth",
    COL_ACTIVITY_ID: "ActivityID",
    COL_PRODUCT_ID: "ProductID",
    COL_FROM_TO_TYPE: "FromToIDType",
    COL_FROM_TO_IDENTIFIER: "FromToIDIdentifier",
    COL_VOLUME: "Volume",
}

COUNTED_PRODUCTS = {"OIL", "GAS", "WATER", "COND"}


def month_to_index(month: str) -> int:
    year, mon = month.split("-")
    return int(year) * 12 + int(mon)


def index_to_month(index: int) -> str:
    year, mon = divmod(index - 1, 12)
    return f"{year:04d}-{mon + 1:02d}"


def verify_header(header: list[str], path: Path) -> None:
    """Pastikan posisi kolom masih sesuai sebelum membaca dengan indeks."""
    for position, name in EXPECTED_HEADER.items():
        if position >= len(header) or header[position] != name:
            actual = header[position] if position < len(header) else "(tidak ada)"
            raise SystemExit(
                f"Skema berubah di {path.name}: kolom {position} "
                f"seharusnya {name}, ternyata {actual}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jurisdiction", default=None, help="Contoh: AB, SK")
    parser.add_argument("--threshold", type=int, default=settings.IDLE_THRESHOLD_MONTHS)
    parser.add_argument("--censor-window", type=int, default=settings.CENSOR_WINDOW_MONTHS)
    args = parser.parse_args()

    bronze = paths.local_dir("bronze") / "petrinex"
    pattern = f"{args.jurisdiction}/*.csv" if args.jurisdiction else "*/*.csv"
    files = sorted(bronze.glob(pattern))
    if not files:
        print(f"Tidak ada CSV di {bronze}/{pattern}")
        return 1

    print(f"Berkas diproses: {len(files)}")

    # uwi -> himpunan indeks bulan yang sumurnya berproduksi
    activity: dict[str, set[int]] = defaultdict(set)
    rows_read = 0
    rows_kept = 0
    rows_short = 0
    short_examples: list[tuple[str, int, list[str]]] = []

    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                continue
            verify_header(header, path)

            for row in reader:
                rows_read += 1

                # Tiap berkas diakhiri satu baris kosong, jadi ada tepat 110
                # baris berkolom nol untuk 110 berkas. Itu bukan kerusakan
                # data, hanya newline penutup. Tetap dihitung supaya kalau
                # suatu saat muncul baris pendek yang sungguhan, angkanya
                # naik dan terlihat.
                if len(row) <= COL_VOLUME:
                    rows_short += 1
                    if len(short_examples) < 5:
                        short_examples.append((path.name, len(row), row[:6]))
                    continue

                if row[COL_FROM_TO_TYPE] != "WI":
                    continue
                if row[COL_ACTIVITY_ID] != "PROD":
                    continue
                if row[COL_PRODUCT_ID] not in COUNTED_PRODUCTS:
                    continue
                volume = row[COL_VOLUME]
                if not volume or volume == "0":
                    continue
                try:
                    if float(volume) <= 0:
                        continue
                except ValueError:
                    continue

                activity[row[COL_FROM_TO_IDENTIFIER]].add(
                    month_to_index(row[COL_PRODUCTION_MONTH])
                )
                rows_kept += 1

        print(f"  {path.parent.name}/{path.stem}: total {rows_read:,} baris dibaca")

    if not activity:
        print("Tidak ada baris produksi sumur yang ditemukan.")
        return 1

    all_months = {m for months in activity.values() for m in months}
    min_idx, max_idx = min(all_months), max(all_months)

    print()
    print(f"Baris dibaca        : {rows_read:,}")
    print(f"Baris kolom kurang  : {rows_short:,}")
    if short_examples:
        print("  contoh baris pendek:")
        for name, width, head in short_examples:
            print(f"    {name} kolom={width} awal={head}")
    print(f"Baris produksi sumur: {rows_kept:,}")
    print(f"Sumur unik          : {len(activity):,}")
    print(
        f"Rentang bulan       : {index_to_month(min_idx)} sampai "
        f"{index_to_month(max_idx)} ({max_idx - min_idx + 1} bulan)"
    )
    print()

    positive = negative = censored = 0
    idle_durations: list[int] = []
    resumed_durations: list[int] = []
    wells_with_spell = 0

    for months in activity.values():
        ordered = sorted(months)
        had_spell = False

        # Periode idle di antara dua bulan aktif, artinya sumur kembali hidup.
        for previous, current in zip(ordered, ordered[1:]):
            gap = current - previous - 1
            if gap >= args.threshold:
                positive += 1
                idle_durations.append(gap)
                resumed_durations.append(gap)
                had_spell = True

        # Periode idle yang masih berjalan saat data berakhir.
        trailing = max_idx - ordered[-1]
        if trailing >= args.threshold:
            idle_durations.append(trailing)
            had_spell = True
            if trailing >= args.threshold + args.censor_window:
                negative += 1
            else:
                censored += 1

        if had_spell:
            wells_with_spell += 1

    usable = positive + negative

    print("Hasil pelabelan")
    print(f"  positive (hidup lagi)      : {positive:,}")
    print(f"  negative (tetap mati)      : {negative:,}")
    print(f"  censored (belum bisa dinilai): {censored:,}")
    print(f"  sumur punya periode idle   : {wells_with_spell:,}")
    print()
    print(f"  bisa dilatih: {usable:,}")
    if usable:
        print(f"  proporsi positive: {positive / usable * 100:.1f}%")

    if idle_durations:
        idle_durations.sort()
        n = len(idle_durations)
        print()
        print("Durasi idle, dalam bulan")
        print(f"  min    : {idle_durations[0]}")
        print(f"  median : {idle_durations[n // 2]}")
        print(f"  maks   : {idle_durations[-1]}")

    print()
    print("Gerbang keputusan")
    if positive >= 5000:
        print("  LANJUT. Label positive memadai untuk melatih klasifikasi.")
    elif positive >= 1000:
        print("  LANJUT DENGAN HATI-HATI. Label positive cukup tetapi tidak banyak.")
        print("  Pakai validasi berbasis waktu dan laporkan interval kepercayaan.")
    else:
        print("  ALIHKAN SASARAN. Label positive terlalu sedikit.")
        print("  Pilihan: prediksi sumur mana yang akan menjadi idle, atau")
        print("  perkiraan sisa cadangan dari decline curve.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
