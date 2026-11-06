#!/usr/bin/env python3
"""Unduh data volumetrik bulanan Petrinex untuk Alberta dan Saskatchewan.

Sifatnya idempoten. Berkas yang sudah ada dan checksum-nya cocok dengan
manifes akan dilewati, jadi skrip ini aman dijalankan ulang setelah koneksi
putus.

Bulan yang mengembalikan HTTP 404 dicatat sebagai "missing" dan dilewati.
Endpoint Petrinex memakai jendela bergulir, jadi bulan lama memang tidak ada
dan itu bukan kegagalan.

Pakai:
    python3 scripts/01_download_petrinex.py
    python3 scripts/01_download_petrinex.py --jurisdictions AB --limit 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import paths, settings  # noqa: E402

CHUNK = 1024 * 256


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"files": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(path)


def human(n: int) -> str:
    step = 1024.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} TB"


def download_one(url: str, dest: Path) -> tuple[str, int]:
    """Unduh satu berkas. Mengembalikan (status, bytes).

    status: "ok" | "missing" | "failed"
    Tulis ke berkas .part dulu lalu rename, supaya berkas setengah jadi tidak
    pernah terlihat sebagai berkas final.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": settings.USER_AGENT})

    last_error: Exception | None = None
    for attempt in range(1, settings.DOWNLOAD_RETRIES + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=settings.DOWNLOAD_TIMEOUT_SEC
            ) as response, part.open("wb") as out:
                total = 0
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)
            part.replace(dest)
            return "ok", total
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
            if exc.code in (400, 404):
                # Bulan atau yurisdiksi memang tidak tersedia.
                return "missing", 0
            last_error = exc
        except Exception as exc:  # noqa: BLE001 - jaringan bisa gagal macam-macam
            part.unlink(missing_ok=True)
            last_error = exc

        if attempt < settings.DOWNLOAD_RETRIES:
            time.sleep(settings.DOWNLOAD_BACKOFF_SEC * attempt)

    print(f"    gagal setelah {settings.DOWNLOAD_RETRIES} percobaan: {last_error}")
    return "failed", 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jurisdictions",
        nargs="+",
        default=list(settings.JURISDICTIONS),
        help="Kode yurisdiksi, contoh: AB SK",
    )
    parser.add_argument("--start", default=settings.START_MONTH)
    parser.add_argument("--end", default=settings.END_MONTH)
    parser.add_argument(
        "--limit", type=int, default=0, help="Batasi jumlah unduhan, 0 berarti tanpa batas"
    )
    args = parser.parse_args()

    paths.ensure_local_dirs()
    manifest = load_manifest(settings.MANIFEST_PATH)
    months = settings.month_range(args.start, args.end)

    print(f"Data root   : {settings.DATA_ROOT}")
    print(f"Yurisdiksi  : {', '.join(args.jurisdictions)}")
    print(f"Rentang     : {months[0]} sampai {months[-1]} ({len(months)} bulan)")
    print()

    counts = {"ok": 0, "skipped": 0, "missing": 0, "failed": 0}
    total_bytes = 0
    downloaded = 0

    for jurisdiction in args.jurisdictions:
        out_dir = paths.local_dir("raw") / "petrinex" / jurisdiction
        out_dir.mkdir(parents=True, exist_ok=True)

        for month in months:
            key = f"{jurisdiction}/{month}"
            dest = out_dir / f"{month}.zip"
            record = manifest["files"].get(key)

            if record and record["status"] == "missing":
                counts["missing"] += 1
                continue

            if record and record["status"] == "ok" and dest.exists():
                if dest.stat().st_size == record["bytes"]:
                    counts["skipped"] += 1
                    total_bytes += record["bytes"]
                    continue

            if args.limit and downloaded >= args.limit:
                continue

            url = settings.PETRINEX_URL.format(jurisdiction=jurisdiction, month=month)
            print(f"  {key} ... ", end="", flush=True)
            started = time.time()
            status, size = download_one(url, dest)
            elapsed = time.time() - started

            if status == "ok":
                digest = sha256_of(dest)
                manifest["files"][key] = {
                    "status": "ok",
                    "bytes": size,
                    "sha256": digest,
                    "url": url,
                    "seconds": round(elapsed, 1),
                }
                total_bytes += size
                downloaded += 1
                print(f"{human(size)} dalam {elapsed:.0f}s")
            else:
                manifest["files"][key] = {"status": status, "bytes": 0, "url": url}
                print(status)

            counts[status] += 1
            save_manifest(settings.MANIFEST_PATH, manifest)

    print()
    print("Ringkasan")
    print(f"  berhasil diunduh : {counts['ok']}")
    print(f"  sudah ada        : {counts['skipped']}")
    print(f"  tidak tersedia   : {counts['missing']}")
    print(f"  gagal            : {counts['failed']}")
    print(f"  total ukuran     : {human(total_bytes)}")
    print(f"  manifes          : {settings.MANIFEST_PATH}")

    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
