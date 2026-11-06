#!/usr/bin/env python3
"""Bentuk produksi bulanan per sumur dari layer silver.

Tiga hal yang dikerjakan di sini.

Pertama, menyaring baris tingkat sumur. Data Petrinex melaporkan pada
tingkat fasilitas, dan tipe fasilitasnya hanya BT, GS, IF, dan GP. Sumur
tidak muncul sebagai fasilitas pelapor, melainkan sebagai lawan aliran di
kolom FromToID dengan FromToIDType bernilai WI. Baris produksi sumur adalah
kombinasi FromToIDType sama dengan WI dan ActivityID sama dengan PROD.

Kedua, melakukan pivot. Data berformat panjang, jadi minyak, gas, air, dan
kondensat menempati baris terpisah untuk bulan yang sama. Pivot mengubahnya
menjadi satu baris per sumur per bulan.

Ketiga, menurunkan koordinat. UWI memuat komponen Dominion Land Survey,
sehingga lintang bujur bisa dihitung tanpa sumber data tambahan. Pemetaan
dilakukan pada daftar UWI unik lalu digabungkan kembali, bukan baris per
baris, karena jumlah sumur unik jauh lebih kecil daripada jumlah baris.

Keluaran:
    silver/wells_monthly  satu baris per sumur per bulan produksi
    silver/wells_dim      satu baris per sumur, berisi koordinat dan operator

Pakai:
    python3 pipeline/s2_wells_monthly.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from config import paths  # noqa: E402
from pipeline.dls import parse_uwi, uwi_to_latlon  # noqa: E402
from pipeline.spark_session import build_spark  # noqa: E402

# Produk yang dijadikan kolom. Sisanya jarang dan tidak dipakai untuk
# analisis sumur idle, jadi dibiarkan di layer silver saja.
PIVOT_PRODUCTS = ["OIL", "GAS", "WATER", "COND"]

WELL_DIM_SCHEMA = StructType(
    [
        StructField("uwi", StringType(), False),
        StructField("dls_location", StringType(), True),
        StructField("lsd", IntegerType(), True),
        StructField("section", IntegerType(), True),
        StructField("township", IntegerType(), True),
        StructField("range", IntegerType(), True),
        StructField("meridian", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
    ]
)


def build_well_dim(spark, wells_df):
    """Tabel dimensi sumur, satu baris per UWI, dengan koordinat DLS.

    Penguraian dilakukan di driver atas daftar UWI unik. Jumlahnya ratusan
    ribu, jadi muat di memori, dan cara ini menghindari UDF yang dipanggil
    jutaan kali.
    """
    unique = [row.uwi for row in wells_df.select("uwi").distinct().collect()]
    print(f"  UWI unik: {len(unique):,}")

    records = []
    failed = 0
    for uwi in unique:
        parsed = parse_uwi(uwi)
        if parsed is None:
            failed += 1
            records.append((uwi, None, None, None, None, None, None, None, None))
            continue
        coords = uwi_to_latlon(parsed)
        latitude, longitude = coords if coords else (None, None)
        records.append(
            (
                uwi,
                parsed.dls_location,
                parsed.lsd,
                parsed.section,
                parsed.township,
                parsed.range_,
                parsed.meridian,
                latitude,
                longitude,
            )
        )

    if failed:
        print(f"  PERINGATAN: {failed:,} UWI gagal diurai, koordinatnya null")

    return spark.createDataFrame(records, schema=WELL_DIM_SCHEMA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="overwrite", choices=("overwrite", "append"))
    args = parser.parse_args()

    silver_in = paths.spark_path("silver", "volumetrics")
    monthly_out = paths.spark_path("silver", "wells_monthly")
    dim_out = paths.spark_path("silver", "wells_dim")

    spark = build_spark("s2_wells_monthly")
    spark.sparkContext.setLogLevel("WARN")

    print(f"Baca  : {silver_in}")
    volumetrics = spark.read.parquet(silver_in)

    well_rows = volumetrics.filter(
        (F.col("FromToIDType") == "WI")
        & (F.col("ActivityID") == "PROD")
        & F.col("Volume").isNotNull()
        & (F.col("ProductID").isin(PIVOT_PRODUCTS))
    ).select(
        F.col("FromToIDIdentifier").alias("uwi"),
        F.col("ProductionMonth").alias("production_month"),
        F.col("production_year"),
        F.col("jurisdiction"),
        F.col("OperatorBAID").alias("operator_id"),
        F.col("OperatorName").alias("operator_name"),
        F.col("ReportingFacilityID").alias("facility_id"),
        F.col("ReportingFacilitySubTypeDesc").alias("facility_subtype"),
        F.col("ProductID").alias("product"),
        F.col("Volume").alias("volume"),
        F.col("Hours").alias("hours"),
    )

    # Satu sumur bisa melapor lewat lebih dari satu fasilitas dalam sebulan,
    # jadi volume dijumlahkan per produk. Operator dan fasilitas diambil
    # yang volumenya terbesar agar deterministik.
    aggregated = well_rows.groupBy(
        "uwi", "production_month", "production_year", "jurisdiction", "product"
    ).agg(
        F.sum("volume").alias("volume"),
        F.sum("hours").alias("hours"),
        F.max(F.struct("volume", "operator_id", "operator_name", "facility_id", "facility_subtype")).alias(
            "top"
        ),
    )

    pivoted = (
        aggregated.groupBy("uwi", "production_month", "production_year", "jurisdiction")
        .pivot("product", PIVOT_PRODUCTS)
        .agg(F.first("volume"))
        .na.fill(0.0, [p for p in PIVOT_PRODUCTS])
    )

    # Atribut non-volumetrik diambil dari baris dengan volume terbesar di
    # bulan itu, supaya satu sumur punya satu operator per bulan.
    attributes = (
        aggregated.groupBy("uwi", "production_month")
        .agg(F.max("top").alias("top"))
        .select(
            "uwi",
            "production_month",
            F.col("top.operator_id").alias("operator_id"),
            F.col("top.operator_name").alias("operator_name"),
            F.col("top.facility_id").alias("facility_id"),
            F.col("top.facility_subtype").alias("facility_subtype"),
        )
    )

    hours = aggregated.groupBy("uwi", "production_month").agg(
        F.max("hours").alias("hours")
    )

    monthly = (
        pivoted.join(attributes, ["uwi", "production_month"], "left")
        .join(hours, ["uwi", "production_month"], "left")
        .withColumnRenamed("OIL", "oil_m3")
        .withColumnRenamed("GAS", "gas_e3m3")
        .withColumnRenamed("WATER", "water_m3")
        .withColumnRenamed("COND", "cond_m3")
    )

    monthly = monthly.withColumn(
        "water_cut",
        F.when(
            (F.col("oil_m3") + F.col("water_m3")) > 0,
            F.col("water_m3") / (F.col("oil_m3") + F.col("water_m3")),
        ).otherwise(F.lit(None).cast("double")),
    )

    monthly.cache()

    print("Bangun tabel dimensi sumur")
    well_dim = build_well_dim(spark, monthly)
    well_dim.write.mode(args.mode).parquet(dim_out)

    (
        monthly.write.mode(args.mode)
        .partitionBy("jurisdiction", "production_year")
        .parquet(monthly_out)
    )

    total = monthly.count()
    wells = monthly.select("uwi").distinct().count()
    months = monthly.select("production_month").distinct().count()

    print()
    print(f"Baris sumur-bulan : {total:,}")
    print(f"Sumur unik        : {wells:,}")
    print(f"Bulan tercakup    : {months}")
    print(f"Tulis monthly     : {monthly_out}")
    print(f"Tulis dim         : {dim_out}")

    # Kunci sumur-bulan wajib unik. Kalau tidak, seluruh deteksi idle di
    # hilir ikut salah, jadi diperiksa di sini.
    duplicates = (
        monthly.groupBy("uwi", "production_month").count().filter(F.col("count") > 1).count()
    )
    if duplicates:
        print(f"GAGAL: ada {duplicates:,} kunci sumur-bulan berduplikat")
        spark.stop()
        return 1
    print("Kunci sumur-bulan unik: OK")

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
