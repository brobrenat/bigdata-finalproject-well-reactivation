#!/usr/bin/env python3
"""Deteksi periode idle per sumur dan beri label hasilnya.

Definisi idle mengikuti aturan AER, yaitu tidak ada aktivitas volumetrik
selama enam bulan berturut-turut. Bulan yang tidak muncul di data produksi
berarti sumur tidak melapor, dan justru ketiadaan baris itulah sinyalnya.

Cara deteksinya tidak memakai cross join antara sumur dan seluruh bulan,
karena itu akan menggelembungkan data tanpa perlu. Sebagai gantinya, tiap
bulan aktif dibandingkan dengan bulan aktif sebelumnya memakai window
function. Selisih lebih dari ambang berarti ada periode idle di antaranya.

Label yang dihasilkan ada tiga, bukan dua:

    positive  sumur kembali berproduksi setelah periode idle
    negative  sumur tetap idle sampai akhir data, dan sudah cukup lama
              diamati untuk disimpulkan tidak kembali
    censored  sumur masih idle saat data berakhir tetapi belum cukup lama
              diamati. Baris ini WAJIB dikeluarkan dari data latih.

Penanganan censored itu yang paling sering dilewatkan. Kalau sumur yang
baru berhenti dianggap negatif, model akan belajar bahwa sumur yang baru
berhenti selalu gagal hidup lagi, padahal itu hanya akibat data yang belum
cukup panjang.

Pakai:
    python3 pipeline/g1_idle_spells.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from config import paths, settings  # noqa: E402
from pipeline.spark_session import build_spark  # noqa: E402

PRODUCT_COLUMNS = ("oil_m3", "gas_e3m3", "water_m3", "cond_m3")


def month_index(column: str):
    """Ubah "YYYY-MM" menjadi indeks bulan berurutan supaya bisa dikurangi."""
    year = F.substring(F.col(column), 1, 4).cast("int")
    month = F.substring(F.col(column), 6, 2).cast("int")
    return year * F.lit(12) + month


def index_to_month(column: str):
    """Kebalikan dari month_index, kembali ke bentuk "YYYY-MM"."""
    year = F.floor((F.col(column) - F.lit(1)) / F.lit(12))
    month = F.col(column) - year * F.lit(12)
    return F.concat(
        year.cast("int").cast("string"),
        F.lit("-"),
        F.lpad(month.cast("int").cast("string"), 2, "0"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=settings.IDLE_THRESHOLD_MONTHS)
    parser.add_argument("--censor-window", type=int, default=settings.CENSOR_WINDOW_MONTHS)
    args = parser.parse_args()

    monthly_in = paths.spark_path("silver", "wells_monthly")
    spells_out = paths.spark_path("gold", "idle_spells")

    spark = build_spark("g1_idle_spells")
    spark.sparkContext.setLogLevel("WARN")

    print(f"Baca          : {monthly_in}")
    print(f"Ambang idle   : {args.threshold} bulan")
    print(f"Jendela sensor: {args.censor_window} bulan")
    print()

    monthly = spark.read.parquet(monthly_in)

    total_production = sum(F.coalesce(F.col(c), F.lit(0.0)) for c in PRODUCT_COLUMNS)

    active = (
        monthly.withColumn("total_volume", total_production)
        .filter(F.col("total_volume") > 0)
        .withColumn("m_idx", month_index("production_month"))
    )

    # Batas akhir data. Semua penilaian "masih idle sampai sekarang" diukur
    # terhadap bulan ini.
    max_idx = active.agg(F.max("m_idx")).collect()[0][0]
    min_idx = active.agg(F.min("m_idx")).collect()[0][0]
    print(f"Rentang data  : indeks {min_idx} sampai {max_idx} ({max_idx - min_idx + 1} bulan)")

    by_well = Window.partitionBy("uwi").orderBy("m_idx")

    marked = (
        active.withColumn("prev_idx", F.lag("m_idx").over(by_well))
        .withColumn("prev_oil", F.lag("oil_m3").over(by_well))
        .withColumn("prev_gas", F.lag("gas_e3m3").over(by_well))
        .withColumn("prev_water_cut", F.lag("water_cut").over(by_well))
        .withColumn("gap", F.col("m_idx") - F.col("prev_idx") - F.lit(1))
    )

    # Periode idle yang berakhir karena sumur kembali berproduksi.
    resumed = (
        marked.filter(F.col("gap") >= args.threshold)
        .select(
            "uwi",
            "jurisdiction",
            F.col("prev_idx").alias("last_active_idx"),
            F.col("m_idx").alias("resume_idx"),
            F.col("gap").alias("idle_months"),
            F.col("prev_oil").alias("oil_before_idle"),
            F.col("prev_gas").alias("gas_before_idle"),
            F.col("prev_water_cut").alias("water_cut_before_idle"),
            F.col("oil_m3").alias("oil_after_resume"),
            F.col("gas_e3m3").alias("gas_after_resume"),
            F.col("operator_name"),
        )
        .withColumn("label", F.lit("positive"))
        .withColumn("resumed", F.lit(True))
    )

    # Periode idle yang masih berjalan saat data berakhir.
    last_active = (
        active.groupBy("uwi", "jurisdiction")
        .agg(
            F.max("m_idx").alias("last_active_idx"),
            F.count(F.lit(1)).alias("active_months"),
        )
        .withColumn("idle_months", F.lit(max_idx) - F.col("last_active_idx"))
        .filter(F.col("idle_months") >= args.threshold)
    )

    last_values = (
        marked.withColumn("rn", F.row_number().over(Window.partitionBy("uwi").orderBy(F.col("m_idx").desc())))
        .filter(F.col("rn") == 1)
        .select(
            "uwi",
            F.col("oil_m3").alias("oil_before_idle"),
            F.col("gas_e3m3").alias("gas_before_idle"),
            F.col("water_cut").alias("water_cut_before_idle"),
            "operator_name",
        )
    )

    ongoing = (
        last_active.join(last_values, "uwi", "left")
        .withColumn("resume_idx", F.lit(None).cast("int"))
        .withColumn("oil_after_resume", F.lit(None).cast("double"))
        .withColumn("gas_after_resume", F.lit(None).cast("double"))
        .withColumn(
            "label",
            F.when(
                F.col("idle_months") >= (args.threshold + args.censor_window),
                F.lit("negative"),
            ).otherwise(F.lit("censored")),
        )
        .withColumn("resumed", F.lit(False))
        .drop("active_months")
    )

    columns = [
        "uwi",
        "jurisdiction",
        "last_active_idx",
        "resume_idx",
        "idle_months",
        "oil_before_idle",
        "gas_before_idle",
        "water_cut_before_idle",
        "oil_after_resume",
        "gas_after_resume",
        "operator_name",
        "label",
        "resumed",
    ]

    spells = (
        resumed.select(*columns)
        .unionByName(ongoing.select(*columns))
        .withColumn("idle_start_month", index_to_month("last_active_idx"))
        .withColumn(
            "idle_start_year", F.substring(F.col("idle_start_month"), 1, 4).cast("int")
        )
    )

    spells.write.mode("overwrite").partitionBy("jurisdiction").parquet(spells_out)

    written = spark.read.parquet(spells_out)
    print()
    print("Jumlah periode idle per label")
    written.groupBy("label").count().orderBy("label").show(truncate=False)

    print("Sebaran durasi idle")
    written.select("idle_months").summary("count", "min", "25%", "50%", "75%", "max").show(
        truncate=False
    )

    positives = written.filter(F.col("label") == "positive").count()
    negatives = written.filter(F.col("label") == "negative").count()
    censored = written.filter(F.col("label") == "censored").count()

    print()
    print("Gerbang keputusan")
    print(f"  positive : {positives:,}")
    print(f"  negative : {negatives:,}")
    print(f"  censored : {censored:,}  (dikeluarkan dari data latih)")
    usable = positives + negatives
    print(f"  bisa dilatih: {usable:,}")
    if usable and positives:
        print(f"  proporsi positive: {positives / usable * 100:.1f}%")
    if positives < 1000:
        print()
        print("  CATATAN: label positive sedikit. Pertimbangkan mengubah sasaran")
        print("  prediksi, misalnya memperkirakan sumur mana yang akan menjadi")
        print("  idle, atau memperkirakan sisa cadangan dari decline curve.")

    print()
    print(f"Tulis: {spells_out}")

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
