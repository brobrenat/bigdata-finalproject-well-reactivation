#!/usr/bin/env python3
"""Ekstrak zip Petrinex menjadi CSV di layer bronze.

Endpoint Petrinex mengembalikan zip bersarang. Urutannya:

    respons.zip  ->  Vol_2022-01-AB.csv.zip  ->  Vol_2022-01-AB.CSV

Dua lapis, dan lapis kedua mudah terlewat. Skrip ini menangani keduanya, dan
juga menerima kasus zip satu lapis kalau formatnya berubah di kemudian hari.

CSV disimpan apa adanya tanpa perubahan isi, supaya layer bronze tetap bisa
dibandingkan dengan sumber aslinya.

Pakai:
    python3 pipeline/b1_extract.py
    python3 pipeline/b1_extract.py --force
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import paths  # noqa: E402


def human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def extract_csv_bytes(zip_path: Path) -> tuple[str, bytes]:
    """Buka zip yang mungkin bersarang, kembalikan (nama_csv, isi)."""
    with zipfile.ZipFile(zip_path) as outer:
        members = outer.infolist()
        if not members:
            raise ValueError(f"zip kosong: {zip_path}")

        first = members[0]
        payload = outer.read(first.filename)

        if first.filename.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                inner_members = inner.infolist()
                if not inner_members:
                    raise ValueError(f"zip dalam kosong: {zip_path}")
                name = inner_members[0].filename
                return name, inner.read(name)

        return first.filename, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Ekstrak ulang walau CSV sudah ada"
    )
    args = parser.parse_args()

    raw_root = paths.local_dir("raw") / "petrinex"
    bronze_root = paths.local_dir("bronze") / "petrinex"

    if not raw_root.exists():
        print(f"Belum ada data mentah di {raw_root}. Jalankan skrip unduhan dulu.")
        return 1

    zips = sorted(raw_root.glob("*/*.zip"))
    if not zips:
        print(f"Tidak ada berkas zip di {raw_root}.")
        return 1

    print(f"Ditemukan {len(zips)} berkas zip")
    print()

    extracted = skipped = failed = 0
    total_zip = total_csv = 0

    for zip_path in zips:
        jurisdiction = zip_path.parent.name
        month = zip_path.stem
        out_dir = bronze_root / jurisdiction
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{month}.csv"

        zip_size = zip_path.stat().st_size
        total_zip += zip_size

        if dest.exists() and not args.force:
            skipped += 1
            total_csv += dest.stat().st_size
            continue

        try:
            _, data = extract_csv_bytes(zip_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  {jurisdiction}/{month}: GAGAL {exc}")
            failed += 1
            continue

        tmp = dest.with_suffix(".csv.tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)

        total_csv += len(data)
        extracted += 1
        print(f"  {jurisdiction}/{month}: {human(zip_size)} -> {human(len(data))}")

    print()
    print("Ringkasan")
    print(f"  diekstrak     : {extracted}")
    print(f"  sudah ada     : {skipped}")
    print(f"  gagal         : {failed}")
    print(f"  total zip     : {human(total_zip)}")
    print(f"  total csv     : {human(total_csv)}")
    if total_zip:
        print(f"  rasio ekspansi: {total_csv / total_zip:.1f}x")
    print(f"  bronze        : {bronze_root}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
