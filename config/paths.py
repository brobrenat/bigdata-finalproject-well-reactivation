"""Satu-satunya tempat path untuk Spark dibentuk.

Aturan yang dipegang: tidak ada path yang ditulis langsung di dalam kode
pipeline. Semuanya lewat fungsi di sini, sehingga memindahkan data root cukup
mengubah satu variabel lingkungan.
"""

from __future__ import annotations

from pathlib import Path

from config import settings

LAYERS = ("raw", "bronze", "silver", "gold")


def local_dir(layer: str) -> Path:
    """Direktori sebuah layer di filesystem."""
    if layer not in LAYERS:
        raise ValueError(f"layer tidak dikenal: {layer}")
    return settings.DATA_ROOT / layer


def spark_path(layer: str, *parts: str) -> str:
    """URI yang diberikan ke Spark untuk membaca atau menulis."""
    return local_dir(layer).joinpath(*parts).as_uri()


def ensure_local_dirs() -> None:
    for layer in LAYERS:
        local_dir(layer).mkdir(parents=True, exist_ok=True)
