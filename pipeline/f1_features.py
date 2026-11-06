#!/usr/bin/env python3
"""Bangun tabel fitur dan label untuk machine learning.

Satu baris adalah satu periode idle. Titik keputusan adalah bulan ketika sumur
genap enam bulan tanpa produksi, yaitu L + 6, dengan L bulan aktif terakhir.
Semua fitur hanya memakai informasi yang sudah tersedia di titik keputusan,
supaya tidak ada kebocoran informasi dari masa depan.

Label memakai horizon tetap:

    y = 1     sumur berproduksi lagi paling lambat 12 bulan setelah titik
              keputusan, yaitu sampai bulan L + 18
    y = 0     belum berproduksi sampai L + 18, termasuk yang baru hidup lagi
              sesudahnya
    censored  data berakhir sebelum L + 18, dikeluarkan apa pun hasilnya.
              Kalau yang hidup lagi tetap dipakai, periode terakhir hanya
              berisi label 1 dan data uji menjadi bias.

Horizon tetap membuat label adil antarwaktu: setiap baris diamati selama
jangka yang sama, sehingga pembagian data latih dan uji menurut waktu tidak
bias.

Target regresi: rata-rata produksi minyak tiga bulan pertama setelah sumur
hidup lagi, dalam skala log.

Keluaran: gold/ml_features dan results/ml_dataset_summary.json

Pakai:
    python3 pipeline/f1_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from config import paths, settings  # noqa: E402
from pipeline.spark_session import build_spark  # noqa: E402

T = settings.IDLE_THRESHOLD_MONTHS          # 6
H = T + settings.CENSOR_WINDOW_MONTHS       # 18
TOP_OPERATORS = 30
TEST_SHARE = 0.2
RESULTS = Path(__file__).resolve().parents[1] / "results"


def month_index(column: str):
    return F.substring(F.col(column), 1, 4).cast("int") * 12 + F.substring(F.col(column), 6, 2).cast("int")


def idx_to_month(c):
    y = F.floor((c - 1) / 12)
    return F.concat(y.cast("string"), F.lit("-"), F.lpad((c - y * 12).cast("string"), 2, "0"))


def log1p(c):
    return F.log1p(F.greatest(F.coalesce(c, F.lit(0.0)), F.lit(0.0)))


def main() -> int:
    spark = build_spark("f1_features")
    spark.sparkContext.setLogLevel("WARN")

    monthly = (spark.read.parquet(paths.spark_path("silver", "wells_monthly"))
               .withColumn("m_idx", month_index("production_month"))
               .withColumn("total", F.col("oil_m3") + F.col("gas_e3m3") + F.col("water_m3") + F.col("cond_m3"))
               .cache())
    max_idx = monthly.agg(F.max("m_idx")).collect()[0][0]

    # --- Label horizon tetap --------------------------------------------------
    spells = (spark.read.parquet(paths.spark_path("gold", "idle_spells"))
              .select("uwi", "jurisdiction", F.col("last_active_idx").alias("L"), F.col("resume_idx").alias("R"),
                      "idle_months", "resumed")
              .withColumn("prior_spells", F.row_number().over(Window.partitionBy("uwi").orderBy("L")) - 1)
              # jendela pengamatan penuh wajib untuk kedua kelas; tanpa ini periode akhir hanya menyisakan yang hidup lagi
              .withColumn("y", F.when(F.col("L") + H <= F.lit(max_idx),
                                      F.when(F.col("resumed") & (F.col("idle_months") < H), 1).otherwise(0)))
              .withColumn("dec_idx", F.col("L") + T))
    counts = {r["status"]: r["count"] for r in spells.withColumn(
        "status", F.when(F.col("y") == 1, "positive").when(F.col("y") == 0, "negative").otherwise("censored"))
        .groupBy("status").count().collect()}
    spells = spells.filter(F.col("y").isNotNull())

    # --- Riwayat produksi sampai L --------------------------------------------
    hist = (spells.select("uwi", "L").join(monthly.select("uwi", "m_idx", "oil_m3", "gas_e3m3", "water_m3", "hours", "total"), "uwi")
            .filter(F.col("m_idx") <= F.col("L")))
    w = lambda lo, hi, c: F.sum(F.when((F.col("m_idx") >= F.col("L") - lo) & (F.col("m_idx") <= F.col("L") - hi), F.coalesce(F.col(c), F.lit(0.0))).otherwise(0.0))
    agg = hist.groupBy("uwi", "L").agg(
        (w(2, 0, "oil_m3") / 3).alias("oil_3"), (w(11, 0, "oil_m3") / 12).alias("oil_12"), (w(5, 3, "oil_m3") / 3).alias("oil_prev3"),
        (w(2, 0, "gas_e3m3") / 3).alias("gas_3"), (w(11, 0, "gas_e3m3") / 12).alias("gas_12"), (w(5, 3, "gas_e3m3") / 3).alias("gas_prev3"),
        (w(11, 0, "water_m3") / 12).alias("water_12"), (w(11, 0, "hours") / 12).alias("hours_12"),
        F.sum(F.when((F.col("m_idx") >= F.col("L") - 11) & (F.col("total") > 0), 1).otherwise(0)).alias("active_12"),
        F.sum("oil_m3").alias("cum_oil"), F.sum("gas_e3m3").alias("cum_gas"), F.sum("water_m3").alias("cum_water"),
        F.min(F.when(F.col("total") > 0, F.col("m_idx"))).alias("first_idx"),
    )
    last = monthly.select("uwi", F.col("m_idx").alias("L"), F.col("oil_m3").alias("oil_last"), F.col("gas_e3m3").alias("gas_last"),
                          F.col("water_m3").alias("water_last"), F.col("cond_m3").alias("cond_last"),
                          F.col("water_cut").alias("water_cut_last"), F.col("hours").alias("hours_last"),
                          "facility_id", "facility_subtype", "operator_id")

    # --- Fitur graf: jaringan sumur-fasilitas dan sumur-operator per bulan ---
    act = monthly.filter(F.col("total") > 0)
    fac_m = act.groupBy("facility_id", "m_idx").agg(F.countDistinct("uwi").alias("n"))
    op_m = act.groupBy("operator_id", "m_idx").agg(F.countDistinct("uwi").alias("n"))

    def at(table, key, offset, name):
        return table.select(key, (F.col("m_idx") - offset).alias("L"), F.col("n").alias(name))

    graph = spark.read.parquet(paths.spark_path("gold", "graph_facility")).select(
        "facility_id", F.col("pagerank").alias("fac_pagerank"), F.col("downstream_pr").alias("fac_downstream_pr"),
        F.col("out_degree").alias("fac_out_degree"), "community_size")
    dim = spark.read.parquet(paths.spark_path("silver", "wells_dim")).select("uwi", "latitude", "longitude", "township", "range", "meridian")

    wti = (spark.read.option("header", True).csv(str(settings.RAW_DIR / "wti" / "wti_monthly.csv"))
           .select(month_index("observation_date").alias("m_idx"), F.col("MCOILWTICO").cast("double").alias("wti")))

    df = (spells.join(agg, ["uwi", "L"]).join(last, ["uwi", "L"], "left")
          .join(at(fac_m, "facility_id", 0, "fac_wells_L"), ["facility_id", "L"], "left")
          .join(at(fac_m, "facility_id", -12, "fac_wells_L12"), ["facility_id", "L"], "left")
          .join(at(fac_m, "facility_id", T, "fac_wells_dec"), ["facility_id", "L"], "left")
          .join(at(op_m, "operator_id", 0, "op_wells_L"), ["operator_id", "L"], "left")
          .join(at(op_m, "operator_id", T, "op_wells_dec"), ["operator_id", "L"], "left")
          .join(graph, "facility_id", "left").join(dim, "uwi", "left")
          .join(wti.withColumnRenamed("m_idx", "L").withColumnRenamed("wti", "wti_L"), "L", "left")
          .join(wti.withColumnRenamed("m_idx", "dec_idx").withColumnRenamed("wti", "wti_dec"), "dec_idx", "left"))

    # operator besar dipertahankan namanya, sisanya OTHER
    top_ops = [r.operator_id for r in df.groupBy("operator_id").count().orderBy(F.desc("count")).limit(TOP_OPERATORS).collect()]
    ratio = lambda a, b: F.when(F.col(b) > 0, F.col(a) / F.col(b))
    df = (df
          .withColumn("log_oil_last", log1p(F.col("oil_last"))).withColumn("log_gas_last", log1p(F.col("gas_last")))
          .withColumn("log_water_last", log1p(F.col("water_last"))).withColumn("log_cond_last", log1p(F.col("cond_last")))
          .withColumn("water_cut_last", F.coalesce("water_cut_last", F.lit(-1.0)))
          .withColumn("log_gor_last", F.coalesce(F.log1p(ratio("gas_last", "oil_last") * 1000), F.lit(-1.0)))
          .withColumn("hours_last", F.coalesce("hours_last", F.lit(0.0)))
          .withColumn("log_oil_3", log1p(F.col("oil_3"))).withColumn("log_oil_12", log1p(F.col("oil_12")))
          .withColumn("log_gas_3", log1p(F.col("gas_3"))).withColumn("log_gas_12", log1p(F.col("gas_12")))
          .withColumn("log_water_12", log1p(F.col("water_12")))
          .withColumn("oil_decline", log1p(F.col("oil_3")) - log1p(F.col("oil_prev3")))
          .withColumn("gas_decline", log1p(F.col("gas_3")) - log1p(F.col("gas_prev3")))
          .withColumn("log_cum_oil", log1p(F.col("cum_oil"))).withColumn("log_cum_gas", log1p(F.col("cum_gas")))
          .withColumn("log_cum_water", log1p(F.col("cum_water")))
          .withColumn("months_observed", F.col("L") - F.col("first_idx") + 1)
          .withColumn("is_oil", (F.col("cum_oil") > 0).cast("int"))
          .withColumn("dec_month", ((F.col("dec_idx") - 1) % 12) + 1)
          .withColumn("wti_change", F.coalesce(ratio("wti_dec", "wti_L"), F.lit(1.0)))
          .withColumn("log_fac_wells_L", log1p(F.col("fac_wells_L")))
          .withColumn("fac_trend", F.coalesce(ratio("fac_wells_L", "fac_wells_L12"), F.lit(-1.0)))
          .withColumn("fac_share_dec", F.coalesce(ratio("fac_wells_dec", "fac_wells_L"), F.lit(0.0)))
          .withColumn("log_op_wells_L", log1p(F.col("op_wells_L")))
          .withColumn("op_share_dec", F.coalesce(ratio("op_wells_dec", "op_wells_L"), F.lit(0.0)))
          .withColumn("log_community_size", log1p(F.col("community_size")))
          .withColumn("operator_top", F.when(F.col("operator_id").isin(top_ops), F.col("operator_id")).otherwise("OTHER"))
          .withColumn("decision_month", idx_to_month(F.col("dec_idx")))
          .fillna({"facility_subtype": "UNKNOWN", "meridian": "UNKNOWN", "fac_pagerank": 0.0, "fac_downstream_pr": 0.0,
                   "fac_out_degree": 0.0, "latitude": 0.0, "longitude": 0.0, "township": 0, "range": 0}))

    # --- Target regresi: rata-rata minyak tiga bulan pertama setelah hidup lagi
    after = (spells.filter(F.col("y") == 1).select("uwi", "L", "R")
             .join(monthly.select("uwi", "m_idx", "oil_m3"), "uwi")
             .filter((F.col("m_idx") >= F.col("R")) & (F.col("m_idx") <= F.col("R") + 2))
             .groupBy("uwi", "L").agg((F.sum("oil_m3") / F.least(F.lit(3), F.lit(max_idx) - F.first("R") + 1)).alias("oil_after")))
    df = df.join(after, ["uwi", "L"], "left").withColumn("log_oil_after", F.when(F.col("oil_after").isNotNull(), log1p(F.col("oil_after"))))

    # --- Pembagian menurut waktu: 20% titik keputusan terakhir menjadi data uji
    cutoff = df.approxQuantile("dec_idx", [1 - TEST_SHARE], 0.0)[0]
    df = df.withColumn("split", F.when(F.col("dec_idx") > cutoff, "test").otherwise("train")).cache()

    out = paths.spark_path("gold", "ml_features")
    df.coalesce(1).write.mode("overwrite").parquet(out)

    by = {f"{r['split']}_{r['y']}": r["count"] for r in df.groupBy("split", "y").count().collect()}
    summary = {
        "spells_by_label": counts, "rows": df.count(), "cutoff_decision_month": df.filter(F.col("dec_idx") == cutoff).select("decision_month").first()[0],
        "decision_range": list(df.agg(F.min("decision_month"), F.max("decision_month")).collect()[0]),
        "by_split_label": by, "regression_rows": df.filter((F.col("y") == 1) & (F.col("is_oil") == 1) & F.col("oil_after").isNotNull()).count(),
        "top_operators": len(top_ops),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ml_dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
