#!/usr/bin/env python3
"""Periksa kesiapan lingkungan sebelum pipeline dijalankan.

Memeriksa Java, PySpark, ruang disk, jumlah core, dan lokasi data root.
Dijalankan lebih dulu supaya kegagalan muncul di sini, bukan di tengah
pemrosesan yang sudah berjalan lama.

Pakai:
    python3 scripts/00_setup_check.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import paths, settings  # noqa: E402

OK = "  OK   "
FAIL = " GAGAL "
WARN = "  CEK  "


def human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def check_java() -> bool:
    binary = shutil.which("java")
    if not binary:
        print(f"[{FAIL}] Java tidak ditemukan")
        print("         Tidak perlu dipasang kalau Spark dijalankan di Docker.")
        print("         Lihat docker/check.sh dan docker/run.sh.")
        print("         Kalau memang ingin di host: sudo apt-get install -y openjdk-17-jdk-headless")
        return False
    try:
        out = subprocess.run(
            [binary, "-version"], capture_output=True, text=True, timeout=30
        )
        version = (out.stderr or out.stdout).splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        print(f"[{FAIL}] Java ada tapi gagal dijalankan: {exc}")
        return False
    print(f"[{OK}] Java  : {version}")
    return True


def check_pyspark() -> bool:
    try:
        import pyspark  # noqa: PLC0415
    except ImportError:
        print(f"[{FAIL}] PySpark belum terpasang")
        print("         Tidak perlu dipasang kalau Spark dijalankan di Docker.")
        print("         Kalau memang ingin di host: python3 -m pip install -r requirements.txt")
        return False
    print(f"[{OK}] PySpark: {pyspark.__version__}")
    return True


def check_spark_runs() -> bool:
    try:
        from pipeline.spark_session import build_spark  # noqa: PLC0415

        spark = build_spark("setup_check", driver_memory="1g")
        count = spark.range(1000).count()
        parallelism = spark.sparkContext.defaultParallelism
        spark.stop()
    except Exception as exc:  # noqa: BLE001
        print(f"[{FAIL}] Spark gagal start: {exc}")
        return False
    if count != 1000:
        print(f"[{FAIL}] Spark start tapi hasil hitung salah: {count}")
        return False
    print(f"[{OK}] Spark  : jalan, paralelisme {parallelism}")
    return True


def check_disk() -> bool:
    root = settings.DATA_ROOT
    probe = root if root.exists() else root.parent
    usage = shutil.disk_usage(probe)
    enough = usage.free > 40 * 1024**3
    tag = OK if enough else WARN
    print(f"[{tag}] Disk   : sisa {human(usage.free)} di {probe}")
    if not enough:
        print("         Perkiraan kebutuhan sekitar 20 sampai 40 GB untuk bronze dan silver.")
    return True


def check_data_root() -> bool:
    root = settings.DATA_ROOT
    if str(root).startswith("/mnt/"):
        print(f"[{WARN}] Data root ada di {root}")
        print("         Drive Windows lambat dibaca dari WSL dan ruangnya tipis.")
        print("         Sebaiknya pakai filesystem Linux, contoh ~/bigdata-data.")
    else:
        print(f"[{OK}] Data   : {root}")
    paths.ensure_local_dirs()
    return True


def check_cpu() -> bool:
    print(f"[{OK}] CPU    : {os.cpu_count()} core")
    return True


def main() -> int:
    print("Pemeriksaan lingkungan")
    print("=" * 60)

    results = [
        check_cpu(),
        check_disk(),
        check_data_root(),
        check_java(),
        check_pyspark(),
    ]

    if results[-1] and results[-2]:
        results.append(check_spark_runs())

    print("=" * 60)
    if all(results):
        print("Semua siap.")
        return 0
    print("Ada yang belum siap. Perbaiki yang bertanda GAGAL di atas.")
    print()
    print("Catatan: Java dan PySpark hanya dibutuhkan kalau Spark dijalankan")
    print("langsung di host. Untuk jalur Docker, pakai bash docker/check.sh.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
