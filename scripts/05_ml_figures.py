#!/usr/bin/env python3
"""Gambar untuk Bab 5 (machine learning) dan Bab 6 (graph analytics).

Membaca keluaran m1, m2, dan n1 di folder results, menulis PNG ke
laporan/gambar.

Pakai:
    python3 scripts/05_ml_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
import matplotlib.patches  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn import metrics  # noqa: E402

from pipeline.ml_common import CONTEXT_FEATURES, GRAPH_FEATURES, WELL_FEATURES  # noqa: E402

R, OUT = ROOT / "results", ROOT / "laporan" / "gambar"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLOR = {"logistic_regression": BLUE, "random_forest": ORANGE, "gbt": AQUA, "hist_gbt": AQUA}
MODEL_NAME = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "gbt": "GBT", "hist_gbt": "HistGradientBoosting"}

plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print(name)


def curves(kind):
    sp, sk = pd.read_parquet(R / "spark_predictions.parquet"), pd.read_parquet(R / "sklearn_predictions.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, preds, title in [(axes[0], sp, "Spark MLlib"), (axes[1], sk, "scikit-learn")]:
        preds = preds[preds.model != "gbt_no_graph"]
        for name, g in preds.groupby("model", sort=False):
            if kind == "roc":
                x, y, _ = metrics.roc_curve(g.label, g.p)
                score = metrics.roc_auc_score(g.label, g.p)
            else:
                y, x, _ = metrics.precision_recall_curve(g.label, g.p)
                score = metrics.average_precision_score(g.label, g.p)
            ax.plot(x, y, color=MODEL_COLOR[name], lw=2, label=f"{MODEL_NAME[name]} ({score:.3f})")
        base = preds.label.mean()
        ax.plot([0, 1], [0, 1] if kind == "roc" else [base, base], color=INK2, lw=1, ls="--", label="acak")
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("False positive rate" if kind == "roc" else "Recall")
        ax.legend(loc="lower right" if kind == "roc" else "upper right", fontsize=9.5)
    axes[0].set_ylabel("True positive rate" if kind == "roc" else "Precision")
    save(fig, f"gambar_5_{'1_roc' if kind == 'roc' else '2_pr'}.png")


def model_compare():
    sp, sk = json.loads((R / "spark_metrics.json").read_text()), json.loads((R / "sklearn_metrics.json").read_text())
    pairs = [("logistic_regression", "logistic_regression"), ("random_forest", "random_forest"), ("gbt", "hist_gbt")]
    labels = ["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(3)
    for ax, key, title in [(axes[0], "roc_auc", "ROC-AUC (data uji)"), (axes[1], "train_seconds", "Waktu latih, detik")]:
        a = [sp["models"][s][key] for s, _ in pairs]
        b = [sk["models"][k][key] for _, k in pairs]
        ax.bar(x - 0.19, a, 0.36, color=BLUE, label="Spark MLlib")
        ax.bar(x + 0.19, b, 0.36, color=ORANGE, label="scikit-learn")
        for xi, v in zip(x - 0.19, a):
            ax.text(xi, v, f"{v:.3f}" if key == "roc_auc" else f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
        for xi, v in zip(x + 0.19, b):
            ax.text(xi, v, f"{v:.3f}" if key == "roc_auc" else f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(x, labels)
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        if key == "roc_auc":
            lo = min(a + b)
            ax.set_ylim(max(0.5, lo - 0.05), 1.0)
    axes[0].legend(loc="upper left")
    save(fig, "gambar_5_3_perbandingan_model.png")


def importance():
    imp = pd.read_csv(R / "spark_feature_importance.csv").head(15).iloc[::-1]
    group = lambda f: "graf" if f in GRAPH_FEATURES else "konteks" if f in CONTEXT_FEATURES else "riwayat sumur" if f in WELL_FEATURES else "kategori"
    colors = {"riwayat sumur": BLUE, "graf": ORANGE, "konteks": AQUA, "kategori": VIOLET}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(imp.feature, imp.importance, color=[colors[group(f)] for f in imp.feature])
    ax.set_xlabel("Kepentingan fitur (GBT, Spark MLlib)")
    ax.set_xlim(0, imp.importance.max() * 1.12)
    ax.legend(handles=[matplotlib.patches.Patch(color=c, label=g) for g, c in colors.items()], loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "gambar_5_4_kepentingan_fitur.png")


def precision_at_k():
    sp, sk = pd.read_parquet(R / "spark_predictions.parquet"), pd.read_parquet(R / "sklearn_predictions.parquet")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for preds, name, color, lab in [(sp, "gbt", AQUA, "GBT (Spark)"), (sk, "hist_gbt", ORANGE, "HistGradientBoosting"),
                                    (sp, "logistic_regression", BLUE, "Logistic Regression (Spark)")]:
        g = preds[preds.model == name].sort_values("p", ascending=False)
        k = np.arange(1, len(g) + 1)
        ax.plot(k, g.label.cumsum().values / k, color=color, lw=2, label=lab)
    base = sp[sp.model == "gbt"].label.mean()
    ax.axhline(base, color=INK2, ls="--", lw=1, label=f"tanpa model ({base:.2f})")
    ax.set_xscale("log")
    ax.set_xlim(10, None)
    ax.set_xlabel("k kandidat teratas (skala log)")
    ax.set_ylabel("Precision@k")
    ax.legend()
    save(fig, "gambar_5_5_precision_at_k.png")


def confusion():
    sp = json.loads((R / "spark_metrics.json").read_text())["models"]["gbt"]
    cm = np.array(sp["confusion"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center", fontsize=13,
                color="white" if v > cm.max() / 2 else INK, fontweight="bold")
    ax.set_xticks([0, 1], ["prediksi 0", "prediksi 1"])
    ax.set_yticks([0, 1], ["aktual 0", "aktual 1"])
    ax.set_title("GBT Spark, ambang 0,5", loc="left", fontweight="bold", color=INK)
    save(fig, "gambar_5_6_confusion_matrix.png")


def regression():
    rp = pd.read_parquet(R / "spark_regression_predictions.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, col, title, c in [(axes[0], "log_oil_3", "Baseline: laju sebelum berhenti", INK2), (axes[1], "prediction", "GBT Regressor (Spark)", AQUA)]:
        ax.scatter(rp.log_oil_after, rp[col], s=8, alpha=0.35, color=c, edgecolors="none")
        lim = [0, max(rp.log_oil_after.max(), rp[col].max())]
        ax.plot(lim, lim, color=INK, lw=1, ls="--")
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("aktual, log(1 + m³/bulan)")
    axes[0].set_ylabel("prediksi, log(1 + m³/bulan)")
    save(fig, "gambar_5_7_regresi.png")


def graph_figs():
    g = json.loads((R / "graph_summary.json").read_text())
    deg = pd.read_csv(R / "graph_degree_dist.csv")
    wpf = pd.read_csv(R / "graph_wells_per_facility.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(deg.degree, deg.nodes, s=12, color=BLUE)
    axes[0].set(xscale="log", yscale="log", xlabel="derajat fasilitas", ylabel="jumlah fasilitas")
    axes[0].set_title("Graf aliran antar-fasilitas", loc="left", fontweight="bold", color=INK)
    axes[1].scatter(wpf.wells, wpf.facilities, s=12, color=ORANGE)
    axes[1].set(xscale="log", yscale="log", xlabel="jumlah sumur per fasilitas", ylabel="jumlah fasilitas")
    axes[1].set_title("Graf sumur ke fasilitas", loc="left", fontweight="bold", color=INK)
    save(fig, "gambar_6_1_distribusi_derajat.png")

    s = json.loads((R / "graph_community_sample.json").read_text())
    G = nx.DiGraph()
    G.add_nodes_from((n["id"], n) for n in s["nodes"])
    G.add_weighted_edges_from(s["edges"], weight="months")
    types = sorted({n["type"] for n in s["nodes"]})
    palette = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, "#008300", "#e34948"]
    tcol = {t: palette[i % len(palette)] for i, t in enumerate(types)}
    pr = np.array([G.nodes[n]["pagerank"] for n in G.nodes])
    pos = nx.spring_layout(G.to_undirected(), seed=7, k=0.6)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axis("off")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#b8bec6", arrows=True, arrowsize=8, width=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[tcol[G.nodes[n]["type"]] for n in G.nodes],
                           node_size=60 + 4000 * pr / pr.max(), edgecolors="white", linewidths=1)
    for t in types:
        ax.scatter([], [], color=tcol[t], s=60, label=t)
    ax.legend(title="tipe fasilitas", loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "gambar_6_2_komunitas.png")

    size = pd.DataFrame(g["restart_by_battery_size"])
    down = pd.DataFrame(g["restart_by_downstream_pr"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, d, col, title, c in [(axes[0], size, "battery_size", "Menurut jumlah sumur di baterai", BLUE),
                                 (axes[1], down, "downstream_pr_q", "Menurut PageRank hub hilir", ORANGE)]:
        ax.bar(d[col].astype(str), d["mean"], color=c)
        for i, (v, n) in enumerate(zip(d["mean"], d["count"])):
            ax.text(i, v, f"{v:.0%}\nn={n:,}".replace(",", "."), ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.set_ylim(0, max(size["mean"].max(), down["mean"].max()) * 1.3)
    axes[0].set_ylabel("tingkat hidup kembali ≤ 12 bulan")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save(fig, "gambar_6_3_restart_jaringan.png")


def roc_pr():
    """ROC dan precision-recall dalam satu gambar: warna = algoritma, garis putus = scikit-learn."""
    sp, sk = pd.read_parquet(R / "spark_predictions.parquet"), pd.read_parquet(R / "sklearn_predictions.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for preds, lib, ls in [(sp, "Spark", "-"), (sk, "sklearn", "--")]:
        for name, g in preds[preds.model != "gbt_no_graph"].groupby("model", sort=False):
            fpr, tpr, _ = metrics.roc_curve(g.label, g.p)
            prec, rec, _ = metrics.precision_recall_curve(g.label, g.p)
            lab = f"{MODEL_NAME[name]}, {lib}"
            axes[0].plot(fpr, tpr, color=MODEL_COLOR[name], ls=ls, lw=1.8, label=f"{lab} ({metrics.roc_auc_score(g.label, g.p):.3f})")
            axes[1].plot(rec, prec, color=MODEL_COLOR[name], ls=ls, lw=1.8, label=f"{lab} ({metrics.average_precision_score(g.label, g.p):.3f})")
    base = sp[sp.model == "gbt"].label.mean()
    axes[0].plot([0, 1], [0, 1], color=INK2, lw=1, ls=":", label="acak (0.500)")
    axes[1].axhline(base, color=INK2, lw=1, ls=":", label=f"acak ({base:.3f})")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    axes[0].set_title("ROC (AUC)", loc="left", fontweight="bold", color=INK)
    axes[1].set_title("Precision-recall (PR-AUC)", loc="left", fontweight="bold", color=INK)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].legend(loc="upper right", fontsize=8)
    save(fig, "gambar_5_1_roc_pr.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    roc_pr()
    model_compare()
    importance()
    precision_at_k()
    confusion()
    regression()
    graph_figs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
