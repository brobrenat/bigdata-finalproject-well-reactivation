#!/usr/bin/env python3
"""Latih dan evaluasi model tanpa Spark, dengan scikit-learn di satu proses.

Data, fitur, pembagian waktu, dan fungsi evaluasi sama persis dengan
m1_train_spark.py, sehingga hasilnya bisa dibandingkan langsung. Bedanya,
seluruh data dimuat ke memori sebagai pandas DataFrame dan dilatih di satu
mesin.

Algoritma: Logistic Regression, Random Forest, dan HistGradientBoosting
(padanan GBT). Regresi: HistGradientBoostingRegressor.

Keluaran: results/sklearn_metrics.json, sklearn_predictions.parquet

Pakai:
    python3 pipeline/m2_train_sklearn.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # noqa: E402

from config import paths  # noqa: E402
from pipeline.ml_common import CATEGORICAL, LABEL, NUMERIC, REG_TARGET, evaluate_binary, evaluate_regression  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED = 42


def ordinal():
    return ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
                             remainder="passthrough", verbose_feature_names_out=False)


def tune(make, grid, X, y):
    """Pilih hiperparameter dengan satu holdout acak 20% dari data latih, sama seperti TrainValidationSplit."""
    if not grid:
        return {}
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scores = []
    for params in grid:
        p = make(**params).fit(Xa, ya).predict_proba(Xb)[:, 1]
        scores.append(evaluate_binary(yb, p)["roc_auc"])
    return grid[int(np.argmax(scores))]


def main() -> int:
    df = pd.read_parquet(paths.local_dir("gold") / "ml_features")
    cols = CATEGORICAL + NUMERIC
    train, test = df[df.split == "train"], df[df.split == "test"]
    X, y, Xt, yt = train[cols], train[LABEL].astype(int), test[cols], test[LABEL].astype(int)
    print(f"train {len(X):,}  test {len(Xt):,}  memori fitur {X.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    cat_idx = list(range(len(CATEGORICAL)))
    builders = {
        "logistic_regression": (lambda C=1.0: make_pipeline(
            ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL), ("num", StandardScaler(), NUMERIC)]),
            LogisticRegression(C=C, max_iter=2000)), [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]),
        "random_forest": (lambda: make_pipeline(ordinal(), RandomForestClassifier(
            n_estimators=200, max_depth=10, n_jobs=-1, random_state=SEED)), []),
        "hist_gbt": (lambda max_depth=6, max_iter=120: make_pipeline(ordinal(), HistGradientBoostingClassifier(
            max_depth=max_depth, max_iter=max_iter, learning_rate=0.1, categorical_features=cat_idx, random_state=SEED)),
                     [{"max_depth": d, "max_iter": i} for d in (4, 6) for i in (60, 120)]),
    }
    results, preds = {"library": "scikit-learn (tanpa Spark)", "models": {}}, []
    for name, (make, grid) in builders.items():
        t = time.time()
        params = tune(make, grid, X, y)
        model = make(**params).fit(X, y)
        secs = time.time() - t
        p = model.predict_proba(Xt)[:, 1]
        m = evaluate_binary(yt, p) | {"train_seconds": round(secs, 1), "params": params}
        results["models"][name] = m
        preds.append(pd.DataFrame({"uwi": test.uwi.values, "L": test.L.values, "label": yt.values, "p": p, "model": name}))
        print(name, {k: m[k] for k in ("roc_auc", "pr_auc", "f1", "precision_at_100", "train_seconds")})

    # Regresi
    r = df[(df[LABEL] == 1) & (df.is_oil == 1) & df[REG_TARGET].notna()]
    rtr, rte = r[r.split == "train"], r[r.split == "test"]
    t = time.time()
    reg = make_pipeline(ordinal(), HistGradientBoostingRegressor(max_depth=5, max_iter=100, categorical_features=cat_idx,
                                                                  random_state=SEED)).fit(rtr[cols], rtr[REG_TARGET])
    secs = time.time() - t
    pred = reg.predict(rte[cols])
    results["regression_hgb"] = evaluate_regression(rte[REG_TARGET], pred, rte.log_oil_3) | {
        "train_rows": len(rtr), "train_seconds": round(secs, 1)}

    pd.concat(preds).to_parquet(RESULTS / "sklearn_predictions.parquet")
    (RESULTS / "sklearn_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["regression_hgb"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
