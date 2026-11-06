#!/usr/bin/env python3
"""Demo Spark UI: jalankan beberapa job kecil, lalu biarkan UI tetap hidup.

Buka http://localhost:4040 selama skrip berjalan. Tab Jobs, Stages (DAG),
SQL, dan Storage menunjukkan apa yang Spark kerjakan di tiap langkah.

Pakai:
    bash scripts/run_local.sh scripts/06_spark_ui_demo.py
Hentikan dengan Ctrl+C.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.ml.classification import LogisticRegression  # noqa: E402
from pyspark.ml.feature import VectorAssembler  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from config import paths  # noqa: E402
from pipeline.ml_common import LABEL, WELL_FEATURES  # noqa: E402
from pipeline.spark_session import build_spark  # noqa: E402

spark = build_spark("demo-spark-ui")
sc = spark.sparkContext


def step(desc: str):
    sc.setJobDescription(desc)
    print(f"-> {desc}")


step("1. Baca Parquet silver, hitung baris (scan saja, tanpa shuffle)")
wm = spark.read.parquet(paths.spark_path("silver", "wells_monthly"))
print("   baris:", wm.count())

step("2. groupBy per bulan (ada shuffle: lihat 2 stage di tab Stages)")
wm.groupBy("production_month").count().orderBy("production_month").show(5)

step("3. cache lalu hitung ulang (lihat tab Storage)")
wm.cache().count()

step("4. Spark MLlib: Logistic Regression pada fitur gold")
df = spark.read.parquet(paths.spark_path("gold", "ml_features")).na.fill(0, WELL_FEATURES)
X = VectorAssembler(inputCols=WELL_FEATURES, outputCol="features").transform(df)
model = LogisticRegression(labelCol=LABEL, maxIter=20).fit(X)
print("   AUC latih:", round(model.summary.areaUnderROC, 4))

print("\nSelesai. Spark UI: http://localhost:4040  (Ctrl+C untuk berhenti)")
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    spark.stop()
