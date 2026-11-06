#!/usr/bin/env python3
"""Ubah CSV bronze menjadi Parquet di layer silver.

Skema ditulis eksplisit, tidak mengandalkan inferensi Spark. Dengan lebih
dari seratus berkas, inferensi berisiko menghasilkan tipe kolom yang
berbeda antar berkas dan baru ketahuan jauh di hilir.

Isi baris tidak diubah. Yang ditambahkan hanya kolom turunan untuk
partisi, yaitu jurisdiction dan production_year.

Catatan tentang duplikat. Di dalam satu berkas bulanan terdapat baris
faktor prorasi yang ActivityID dan ProductID-nya kosong, dan baris seperti
itu bisa berulang untuk satu fasilitas. Baris volumetrik yang sebenarnya
tidak berduplikat. Tiap berkas bulanan juga hanya memuat satu
ProductionMonth, jadi tidak ada koreksi bulan lama yang bercampur di sini.

Pakai:
    python3 pipeline/s1_to_silver.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import StringType, StructField, StructType  # noqa: E402

from config import paths  # noqa: E402
from pipeline.spark_session import build_spark  # noqa: E402

# Ke-30 kolom dibaca sebagai string lebih dulu, lalu dikonversi secara
# terkendali. Membaca angka langsung sebagai double membuat satu sel rusak
# bisa mendiamkan seluruh kolom menjadi null tanpa jejak.
RAW_COLUMNS = [
    "ProductionMonth",
    "OperatorBAID",
    "OperatorName",
    "ReportingFacilityID",
    "ReportingFacilityProvinceState",
    "ReportingFacilityType",
    "ReportingFacilityIdentifier",
    "ReportingFacilityName",
    "ReportingFacilitySubType",
    "ReportingFacilitySubTypeDesc",
    "ReportingFacilityLocation",
    "FacilityLegalSubdivision",
    "FacilitySection",
    "FacilityTownship",
    "FacilityRange",
    "FacilityMeridian",
    "SubmissionDate",
    "ActivityID",
    "ProductID",
    "FromToID",
    "FromToIDProvinceState",
    "FromToIDType",
    "FromToIDIdentifier",
    "Volume",
    "Energy",
    "Hours",
    "CCICode",
    "ProrationProduct",
    "ProrationFactor",
    "Heat",
]

NUMERIC_COLUMNS = ("Volume", "Energy", "Hours", "ProrationFactor", "Heat")

SCHEMA = StructType([StructField(name, StringType(), True) for name in RAW_COLUMNS])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="overwrite",
        choices=("overwrite", "append"),
        help="Mode penulisan Parquet",
    )
    args = parser.parse_args()

    bronze_root = paths.local_dir("bronze") / "petrinex"
    if not bronze_root.exists():
        print(f"Bronze belum ada di {bronze_root}. Jalankan pipeline/b1_extract.py.")
        return 1

    csv_glob = str(bronze_root / "*" / "*.csv")
    out_path = paths.spark_path("silver", "volumetrics")

    spark = build_spark("s1_to_silver")
    spark.sparkContext.setLogLevel("WARN")

    print(f"Baca   : {csv_glob}")
    print(f"Tulis  : {out_path}")
    print()

    # Tiap CSV diakhiri satu baris kosong. Mode PERMISSIVE disebut eksplisit
    # supaya baris seperti itu menjadi null dan tidak menggagalkan pembacaan.
    # Jumlahnya tetap dihitung setelah penulisan, sehingga kalau suatu saat
    # ada kerusakan yang sungguhan, angkanya naik dan terlihat.
    raw = (
        spark.read.option("header", True)
        .option("encoding", "UTF-8")
        .option("mode", "PERMISSIVE")
        .schema(SCHEMA)
        .csv(csv_glob)
        .withColumn("source_file", F.input_file_name())
    )

    # Yurisdiksi diambil dari nama direktori, bukan dari kolom provinsi,
    # karena kolom provinsi mengacu pada lokasi fasilitas dan bisa berbeda
    # dari yurisdiksi pelaporannya.
    enriched = raw.withColumn(
        "jurisdiction", F.regexp_extract("source_file", r"petrinex/([A-Z]{2})/", 1)
    ).withColumn(
        "production_year", F.substring("ProductionMonth", 1, 4).cast("int")
    )

    for column in NUMERIC_COLUMNS:
        enriched = enriched.withColumn(column, F.col(column).cast("double"))

    (
        enriched.write.mode(args.mode)
        .partitionBy("jurisdiction", "production_year")
        .parquet(out_path)
    )

    written = spark.read.parquet(out_path)
    total = written.count()

    # Baris yang kolom kuncinya null berarti baris pendek atau rusak. Angkanya
    # dicetak agar bisa dilaporkan sebagai temuan kualitas data.
    truncated = written.filter(
        F.col("ProductionMonth").isNull() | F.col("ReportingFacilityID").isNull()
    ).count()

    print(f"Baris ditulis : {total:,}")
    print(f"Baris rusak atau pendek: {truncated:,}")
    print("Baris per yurisdiksi dan tahun:")
    (
        written.groupBy("jurisdiction", "production_year")
        .count()
        .orderBy("jurisdiction", "production_year")
        .show(200, truncate=False)
    )

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
