"""Pembuat SparkSession yang dipakai seluruh pipeline.

Disetel untuk mesin satu node dengan RAM terbatas. Jumlah partisi shuffle
sengaja dikecilkan dari bawaan 200, karena pada data sebesar ini bawaan itu
menghasilkan ratusan berkas kecil dan justru memperlambat.
"""

from __future__ import annotations

import os

from pyspark.sql import SparkSession


def build_spark(app_name: str, driver_memory: str | None = None) -> SparkSession:
    memory = driver_memory or os.environ.get("SPARK_DRIVER_MEMORY", "4g")

    return (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "16"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
        # Biarkan Spark menggabungkan partisi kecil setelah shuffle.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
