#!/usr/bin/env python3
"""Latih dan evaluasi model dengan Spark MLlib.

Klasifikasi: akankah sumur idle berproduksi lagi dalam 12 bulan setelah
ditandai idle. Tiga algoritma dibandingkan: Logistic Regression (baseline
linear), Random Forest, dan Gradient-Boosted Trees. Hiperparameter GBT dipilih
dengan TrainValidationSplit di atas data latih saja.

Regresi: laju minyak tiga bulan pertama setelah hidup lagi (skala log),
dengan GBTRegressor, dibandingkan baseline "laju sebelum berhenti".

Ablasi: GBT dilatih ulang tanpa fitur graf untuk mengukur kontribusinya.

Keluaran: results/spark_metrics.json, spark_predictions.parquet,
spark_feature_importance.csv

Pakai:
    python3 pipeline/m1_train_spark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.ml.regression import GBTRegressor  # noqa: E402
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from config import paths  # noqa: E402
from pipeline.ml_common import CATEGORICAL, GRAPH_FEATURES, LABEL, NUMERIC, REG_TARGET, evaluate_binary, evaluate_regression  # noqa: E402
from pipeline.spark_session import build_spark  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED = 42


def indexers():
    return [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL]


def tree_stages(numeric):
    return indexers() + [VectorAssembler(inputCols=numeric + [f"{c}_idx" for c in CATEGORICAL], outputCol="features")]


def predict(model, df, name):
    return (model.transform(df).select("uwi", "L", F.col(LABEL).alias("label"),
                                       vector_to_array("probability")[1].alias("p")).toPandas().assign(model=name))


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    spark = build_spark("m1_train_spark")
    spark.sparkContext.setLogLevel("WARN")
    df = spark.read.parquet(paths.spark_path("gold", "ml_features")).withColumn(LABEL, F.col(LABEL).cast("double"))
    train = df.filter(F.col("split") == "train").cache()
    test = df.filter(F.col("split") == "test").cache()
    print(f"train {train.count():,}  test {test.count():,}")

    auc = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    results, preds = {"library": "Spark MLlib (pyspark.ml)", "models": {}}, []

    # 1. Logistic Regression: one-hot untuk kategori, skala standar, regulasi dipilih di validasi
    lr_pipe = Pipeline(stages=indexers() + [
        OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORICAL], outputCols=[f"{c}_oh" for c in CATEGORICAL]),
        VectorAssembler(inputCols=NUMERIC + [f"{c}_oh" for c in CATEGORICAL], outputCol="raw"),
        StandardScaler(inputCol="raw", outputCol="features"),
        LogisticRegression(labelCol=LABEL, maxIter=200, elasticNetParam=0.0)])
    lr = lr_pipe.getStages()[-1]
    candidates = [
        ("logistic_regression", lr_pipe, ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.01, 0.1]).build()),
    ]
    rf_pipe = Pipeline(stages=tree_stages(NUMERIC) + [RandomForestClassifier(labelCol=LABEL, numTrees=200, maxDepth=10, maxBins=64, seed=SEED)])
    candidates.append(("random_forest", rf_pipe, ParamGridBuilder().build()))
    gbt = GBTClassifier(labelCol=LABEL, maxBins=64, stepSize=0.1, seed=SEED)
    gbt_pipe = Pipeline(stages=tree_stages(NUMERIC) + [gbt])
    candidates.append(("gbt", gbt_pipe, ParamGridBuilder().addGrid(gbt.maxDepth, [4, 6]).addGrid(gbt.maxIter, [60, 120]).build()))

    best_gbt = None
    for name, pipe, grid in candidates:
        t = time.time()
        if grid:
            tvs = TrainValidationSplit(estimator=pipe, estimatorParamMaps=grid, evaluator=auc, trainRatio=0.8, seed=SEED, parallelism=2)
            fitted = tvs.fit(train)
            model, chosen = fitted.bestModel, grid[int(pd.Series(fitted.validationMetrics).idxmax())]
            params = {p.name: v for p, v in chosen.items()}
        else:
            model, params = pipe.fit(train), {}
        secs = time.time() - t
        pr = predict(model, test, name)
        preds.append(pr)
        m = evaluate_binary(pr.label, pr.p)
        m.update({"train_seconds": round(secs, 1), "params": params, "spark_auc_check": round(auc.evaluate(model.transform(test)), 4)})
        results["models"][name] = m
        print(name, {k: m[k] for k in ("roc_auc", "pr_auc", "f1", "precision_at_100", "train_seconds")})
        if name == "gbt":
            best_gbt, best_params = model, params

    # Kepentingan fitur GBT
    names = NUMERIC + [f"{c}_idx" for c in CATEGORICAL]
    imp = best_gbt.stages[-1].featureImportances.toArray()
    pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).to_csv(
        RESULTS / "spark_feature_importance.csv", index=False)

    # Ablasi: GBT tanpa fitur graf, hiperparameter sama
    no_graph = [c for c in NUMERIC if c not in GRAPH_FEATURES]
    abl = GBTClassifier(labelCol=LABEL, maxBins=64, stepSize=0.1, seed=SEED, maxDepth=best_params["maxDepth"], maxIter=best_params["maxIter"])
    t = time.time()
    abl_model = Pipeline(stages=tree_stages(no_graph) + [abl]).fit(train)
    pr = predict(abl_model, test, "gbt_no_graph")
    results["ablation_no_graph"] = evaluate_binary(pr.label, pr.p) | {"train_seconds": round(time.time() - t, 1)}

    # Regresi laju minyak setelah hidup lagi (sumur minyak yang hidup lagi)
    reg_df = df.filter((F.col(LABEL) == 1) & (F.col("is_oil") == 1) & F.col(REG_TARGET).isNotNull())
    rtrain, rtest = reg_df.filter(F.col("split") == "train"), reg_df.filter(F.col("split") == "test")
    t = time.time()
    reg = Pipeline(stages=tree_stages(NUMERIC) + [GBTRegressor(labelCol=REG_TARGET, maxDepth=5, maxIter=100, maxBins=64, seed=SEED)]).fit(rtrain)
    secs = time.time() - t
    rp = reg.transform(rtest).select(REG_TARGET, "prediction", "log_oil_3").toPandas()
    results["regression_gbt"] = evaluate_regression(rp[REG_TARGET], rp.prediction, rp.log_oil_3) | {
        "train_rows": rtrain.count(), "train_seconds": round(secs, 1)}
    rp.to_parquet(RESULTS / "spark_regression_predictions.parquet")

    pd.concat(preds).to_parquet(RESULTS / "spark_predictions.parquet")
    (RESULTS / "spark_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "models"}, indent=1))
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
