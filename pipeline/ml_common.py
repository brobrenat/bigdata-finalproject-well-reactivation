"""Daftar fitur dan fungsi evaluasi yang dipakai bersama oleh m1 (Spark) dan m2 (scikit-learn).

Evaluasi kedua pustaka dihitung dengan fungsi yang sama, supaya perbandingannya adil.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics

LABEL = "y"
REG_TARGET = "log_oil_after"

# Riwayat produksi sumur sampai bulan aktif terakhir.
WELL_FEATURES = [
    "log_oil_last", "log_gas_last", "log_water_last", "log_cond_last", "water_cut_last", "log_gor_last", "hours_last",
    "log_oil_3", "log_oil_12", "log_gas_3", "log_gas_12", "log_water_12", "oil_decline", "gas_decline",
    "active_12", "hours_12", "log_cum_oil", "log_cum_gas", "log_cum_water", "months_observed", "prior_spells", "is_oil",
]
# Lokasi, waktu, dan harga minyak.
CONTEXT_FEATURES = ["latitude", "longitude", "township", "range", "dec_month", "wti_dec", "wti_change"]
# Turunan graf: jaringan sumur-fasilitas per bulan dan graf aliran antar-fasilitas (n1_graph.py).
GRAPH_FEATURES = [
    "log_fac_wells_L", "fac_trend", "fac_share_dec", "log_op_wells_L", "op_share_dec",
    "fac_pagerank", "fac_downstream_pr", "fac_out_degree", "log_community_size",
]
NUMERIC = WELL_FEATURES + CONTEXT_FEATURES + GRAPH_FEATURES
CATEGORICAL = ["jurisdiction", "facility_subtype", "operator_top", "meridian"]


def evaluate_binary(y_true, p, k_values=(100, 500)) -> dict:
    """Metrik klasifikasi. precision@k mengukur ketepatan daftar kandidat teratas, yang dipakai DSS."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    prec, rec, thr = metrics.precision_recall_curve(y_true, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    order = np.argsort(-p)
    base = y_true.mean()
    out = {
        "n": int(len(y_true)), "positive_rate": round(float(base), 4),
        "roc_auc": round(metrics.roc_auc_score(y_true, p), 4),
        "pr_auc": round(metrics.average_precision_score(y_true, p), 4),
        "log_loss": round(metrics.log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6)), 4),
        "brier": round(metrics.brier_score_loss(y_true, p), 4),
        "accuracy": round(metrics.accuracy_score(y_true, pred), 4),
        "precision": round(metrics.precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(metrics.recall_score(y_true, pred), 4),
        "f1": round(metrics.f1_score(y_true, pred), 4),
        "best_f1": round(float(f1[:-1].max()), 4), "best_f1_threshold": round(float(thr[f1[:-1].argmax()]), 4),
        "confusion": metrics.confusion_matrix(y_true, pred).tolist(),
    }
    for k in k_values:
        out[f"precision_at_{k}"] = round(float(y_true[order[:k]].mean()), 4)
    top = order[: max(1, len(order) // 10)]
    out["precision_top10pct"] = round(float(y_true[top].mean()), 4)
    out["lift_top10pct"] = round(float(y_true[top].mean() / base), 3)
    return out


def evaluate_regression(y_log, pred_log, base_log=None) -> dict:
    """Metrik regresi pada skala log dan skala asli (m3 per bulan)."""
    y_log, pred_log = np.asarray(y_log, float), np.asarray(pred_log, float)
    y, pred = np.expm1(y_log), np.expm1(pred_log)
    out = {
        "n": int(len(y_log)),
        "rmse_log": round(float(np.sqrt(metrics.mean_squared_error(y_log, pred_log))), 4),
        "mae_log": round(float(metrics.mean_absolute_error(y_log, pred_log)), 4),
        "r2_log": round(float(metrics.r2_score(y_log, pred_log)), 4),
        "mae_m3": round(float(metrics.mean_absolute_error(y, pred)), 2),
        "median_ae_m3": round(float(metrics.median_absolute_error(y, pred)), 2),
    }
    if base_log is not None:
        b = np.asarray(base_log, float)
        out["baseline_rmse_log"] = round(float(np.sqrt(metrics.mean_squared_error(y_log, b))), 4)
        out["baseline_r2_log"] = round(float(metrics.r2_score(y_log, b)), 4)
        out["baseline_mae_m3"] = round(float(metrics.mean_absolute_error(y, np.expm1(b))), 2)
    return out
