#!/usr/bin/env python3
"""Graph analytics jaringan fasilitas produksi.

Graf dibangun dari layer silver:

    sumur -> fasilitas      baris PROD dengan FromToIDType = WI (wells_monthly)
    fasilitas -> fasilitas  baris DISP (pelapor ke lawan aliran) dan REC
                            (lawan aliran ke pelapor), selain sumur

Satu aliran sering dilaporkan dari dua sisi (DISP di pengirim, REC di
penerima), jadi sisi digabung per pasangan berarah. Bobot sisi adalah jumlah
bulan aliran itu muncul, karena volume minyak, gas, dan air tidak satu satuan.

Agregasi atas 42 juta baris dikerjakan Spark. Graf hasilnya hanya puluhan ribu
simpul, jadi analisisnya memakai NetworkX: komponen, PageRank, komunitas
Louvain, dan betweenness aproksimasi. PageRank juga dihitung dengan Spark
memakai iterasi DataFrame, sebagai pembanding untuk graf yang tidak muat di
satu mesin.

Keluaran:
    gold/graph_edges     sisi fasilitas ke fasilitas
    gold/graph_facility  metrik per fasilitas, dipakai f1_features.py
    results/graph_*      ringkasan untuk laporan

Pakai:
    python3 pipeline/n1_graph.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from config import paths, settings  # noqa: E402
from pipeline.spark_session import build_spark  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
HORIZON = settings.IDLE_THRESHOLD_MONTHS + settings.CENSOR_WINDOW_MONTHS  # 18 bulan


def month_index(column: str):
    return F.substring(F.col(column), 1, 4).cast("int") * 12 + F.substring(F.col(column), 6, 2).cast("int")


def spark_pagerank(spark, edges, iterations: int = 30, damping: float = 0.85):
    """PageRank berbobot dengan iterasi DataFrame, perlakuan dangling sama seperti NetworkX."""
    nodes = edges.select(F.col("src").alias("id")).union(edges.select(F.col("dst").alias("id"))).distinct().cache()
    n = nodes.count()
    out_w = edges.groupBy("src").agg(F.sum("months").alias("out_w"))
    links = edges.join(out_w, "src").select("src", "dst", (F.col("months") / F.col("out_w")).alias("p")).cache()
    dangling = nodes.join(out_w.withColumnRenamed("src", "id"), "id", "left_anti").cache()
    rank = nodes.withColumn("rank", F.lit(1.0 / n))
    for i in range(iterations):
        dangling_mass = rank.join(dangling, "id").agg(F.sum("rank")).collect()[0][0] or 0.0
        contrib = links.join(rank.withColumnRenamed("id", "src"), "src").groupBy("dst").agg(F.sum(F.col("rank") * F.col("p")).alias("c"))
        rank = (nodes.join(contrib.withColumnRenamed("dst", "id"), "id", "left")
                .select("id", (F.lit((1 - damping) / n) + damping * (F.coalesce("c", F.lit(0.0)) + F.lit(dangling_mass / n))).alias("rank")))
        if i % 5 == 4:
            rank = rank.localCheckpoint()
    return rank


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    spark = build_spark("n1_graph")
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.time()

    vol = spark.read.parquet(paths.spark_path("silver", "volumetrics"))
    monthly = spark.read.parquet(paths.spark_path("silver", "wells_monthly"))
    spells = spark.read.parquet(paths.spark_path("gold", "idle_spells"))

    # --- Sisi fasilitas ke fasilitas -------------------------------------------
    flows = vol.filter(
        F.col("ActivityID").isin("DISP", "REC")
        & F.col("FromToID").isNotNull()
        & (F.col("FromToIDType") != "WI")
        & (F.col("FromToID") != F.col("ReportingFacilityID"))
    )
    is_disp = F.col("ActivityID") == "DISP"
    edges = (
        flows.withColumn("src", F.when(is_disp, F.col("ReportingFacilityID")).otherwise(F.col("FromToID")))
        .withColumn("dst", F.when(is_disp, F.col("FromToID")).otherwise(F.col("ReportingFacilityID")))
        .groupBy("src", "dst")
        .agg(F.countDistinct("ProductionMonth").alias("months"), F.collect_set("ProductID").alias("products"))
        .cache()
    )
    edges.write.mode("overwrite").parquet(paths.spark_path("gold", "graph_edges"))

    # --- Atribut fasilitas ---------------------------------------------------
    attrs = vol.groupBy(F.col("ReportingFacilityID").alias("facility_id")).agg(
        F.max_by("ReportingFacilityName", "ProductionMonth").alias("name"),
        F.max_by("ReportingFacilitySubTypeDesc", "ProductionMonth").alias("subtype"),
        F.max_by("OperatorName", "ProductionMonth").alias("operator"),
    )

    # --- Status sumur per fasilitas ------------------------------------------
    m = monthly.withColumn("m_idx", month_index("production_month")).withColumn(
        "total", F.col("oil_m3") + F.col("gas_e3m3") + F.col("water_m3") + F.col("cond_m3"))
    max_idx = m.agg(F.max("m_idx")).collect()[0][0]
    last_fac = m.groupBy("uwi").agg(F.max_by("facility_id", "m_idx").alias("facility_id"))
    active_last = m.filter((F.col("m_idx") == max_idx) & (F.col("total") > 0)).select("uwi").distinct().withColumn("active_last", F.lit(1))
    idle_end = spells.filter(~F.col("resumed")).select("uwi").distinct().withColumn("idle_end", F.lit(1))
    # label horizon tetap 18 bulan, sama dengan f1_features.py
    labeled = (spells.withColumn("y", F.when(F.col("last_active_idx") + HORIZON <= F.lit(max_idx),
                                             F.when(F.col("resumed") & (F.col("idle_months") < HORIZON), 1).otherwise(0)))
               .filter(F.col("y").isNotNull()).join(last_fac, "uwi"))
    restart = labeled.groupBy("facility_id").agg(F.count("*").alias("spells"), F.avg("y").alias("restart_rate"))
    well_stats = (last_fac.join(active_last, "uwi", "left").join(idle_end, "uwi", "left")
                  .groupBy("facility_id").agg(F.count("*").alias("wells"),
                                              F.sum(F.coalesce("active_last", F.lit(0))).alias("wells_active_last"),
                                              F.sum(F.coalesce("idle_end", F.lit(0))).alias("wells_idle_end")))

    e = edges.select("src", "dst", "months").toPandas()
    fac = (attrs.join(well_stats, "facility_id", "full").join(restart, "facility_id", "left")).toPandas()
    spells_fac = labeled.select("uwi", "last_active_idx", "y", "facility_id").toPandas()
    t_spark = time.time() - t0
    print(f"Agregasi Spark selesai: {len(e):,} sisi, {len(fac):,} fasilitas, {t_spark:.0f} detik")

    # --- Analisis NetworkX ---------------------------------------------------
    t1 = time.time()
    G = nx.DiGraph()
    G.add_weighted_edges_from(e[["src", "dst", "months"]].itertuples(index=False), weight="months")
    U = G.to_undirected()
    wcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    pr = nx.pagerank(G, alpha=0.85, weight="months")
    comms = nx.community.louvain_communities(U, weight="months", seed=42)
    comm_of = {node: i for i, c in enumerate(sorted(comms, key=len, reverse=True)) for node in c}
    lcc = U.subgraph(wcc[0]).copy()
    btw = nx.betweenness_centrality(lcc, k=min(300, len(lcc)), seed=42, normalized=True)
    # hub hilir: PageRank tertinggi yang dapat dicapai dalam dua langkah
    down_pr = {}
    for u in G.nodes:
        reach = set(G.successors(u))
        for v in list(reach):
            reach.update(G.successors(v))
        down_pr[u] = max((pr[v] for v in reach), default=0.0)
    t_nx = time.time() - t1

    nodes = pd.DataFrame({"facility_id": list(G.nodes)})
    nodes["pagerank"] = nodes.facility_id.map(pr)
    nodes["in_degree"] = nodes.facility_id.map(dict(G.in_degree()))
    nodes["out_degree"] = nodes.facility_id.map(dict(G.out_degree()))
    nodes["community"] = nodes.facility_id.map(comm_of)
    nodes["betweenness"] = nodes.facility_id.map(btw).fillna(0.0)
    nodes["downstream_pr"] = nodes.facility_id.map(down_pr)
    sizes = nodes.community.value_counts()
    nodes["community_size"] = nodes.community.map(sizes)
    fac = fac.merge(nodes, on="facility_id", how="outer")
    fac["type"] = fac.facility_id.str[2:4]
    for c in ["wells", "wells_active_last", "wells_idle_end", "spells"]:
        fac[c] = fac[c].fillna(0).astype(int)
    fac["idle_share"] = (fac.wells_idle_end / fac.wells.where(fac.wells > 0)).round(4)

    # --- PageRank Spark sebagai pembanding -----------------------------------
    t2 = time.time()
    spr = spark_pagerank(spark, edges.select("src", "dst", "months")).toPandas().set_index("id")["rank"]
    t_spr = time.time() - t2
    both = nodes.set_index("facility_id")[["pagerank"]].join(spr.rename("spark_pagerank"), how="inner")
    top_nx = set(both.pagerank.nlargest(20).index)
    top_sp = set(both.spark_pagerank.nlargest(20).index)

    spark.createDataFrame(fac[["facility_id", "type", "pagerank", "downstream_pr", "in_degree", "out_degree",
                               "community", "community_size", "betweenness", "wells"]]
                          .fillna({"community": -1, "pagerank": 0.0, "downstream_pr": 0.0, "in_degree": 0, "out_degree": 0,
                                   "community_size": 0, "betweenness": 0.0})
                          ).write.mode("overwrite").parquet(paths.spark_path("gold", "graph_facility"))

    # --- Insight: tingkat hidup kembali menurut posisi di jaringan ----------
    sp = spells_fac.merge(fac[["facility_id", "wells", "downstream_pr", "community"]], on="facility_id", how="left")
    sp["battery_size"] = pd.cut(sp.wells, [0, 5, 20, 100, 500, 10_000], labels=["1-5", "6-20", "21-100", "101-500", ">500"])
    sp["has_downstream"] = sp.downstream_pr.fillna(0) > 0
    sp["downstream_pr_q"] = pd.qcut(sp.downstream_pr.rank(method="first"), 5, labels=["Q1 rendah", "Q2", "Q3", "Q4", "Q5 tinggi"])
    by_size = sp.groupby("battery_size", observed=True).y.agg(["count", "mean"]).reset_index()
    by_down = sp.groupby("downstream_pr_q", observed=True).y.agg(["count", "mean"]).reset_index()
    by_conn = sp.groupby("has_downstream").y.agg(["count", "mean"]).reset_index()

    comm = (fac[fac.community.notna()].groupby("community")
            .agg(facilities=("facility_id", "count"), wells=("wells", "sum"), wells_idle_end=("wells_idle_end", "sum"),
                 spells=("spells", "sum"), top_hub=("pagerank", "idxmax")).reset_index())
    comm["top_hub"] = fac.loc[comm.top_hub, "facility_id"].values
    comm_rate = sp.groupby("community").y.mean().rename("restart_rate")
    comm = comm.merge(comm_rate, left_on="community", right_index=True, how="left")
    comm["idle_share"] = (comm.wells_idle_end / comm.wells.where(comm.wells > 0)).round(4)
    comm = comm.sort_values("wells_idle_end", ascending=False)

    top = fac.sort_values("pagerank", ascending=False).head(15)[
        ["facility_id", "type", "name", "operator", "pagerank", "in_degree", "out_degree", "betweenness", "community"]]

    deg = pd.Series(dict(G.degree())).value_counts().sort_index()
    wells_per_bt = fac.loc[fac.wells > 0, "wells"].value_counts().sort_index()
    big = sorted(comms, key=len, reverse=True)
    show = min(big, key=lambda c: abs(len(c) - 50))  # komunitas berukuran sedang untuk digambar
    sub = G.subgraph(show)

    summary = {
        "edges_facility": int(G.number_of_edges()), "nodes_facility": int(G.number_of_nodes()),
        "facilities_with_wells": int((fac.wells > 0).sum()), "wells": int(fac.wells.sum()),
        "wcc": len(wcc), "largest_wcc_nodes": len(wcc[0]), "largest_wcc_share": round(len(wcc[0]) / G.number_of_nodes(), 4),
        "communities": len(comms), "modularity": round(nx.community.modularity(U, comms, weight="months"), 4),
        "density": nx.density(G), "avg_degree": round(2 * G.number_of_edges() / G.number_of_nodes(), 3),
        "type_counts": fac.type.value_counts().head(12).to_dict(),
        "pagerank_compare": {"spearman": round(both.pagerank.corr(both.spark_pagerank, method="spearman"), 6),
                             "max_abs_diff": float((both.pagerank - both.spark_pagerank).abs().max()),
                             "top20_overlap": len(top_nx & top_sp)},
        "seconds": {"spark_aggregation": round(t_spark, 1), "networkx_analysis": round(t_nx, 1), "spark_pagerank": round(t_spr, 1)},
        "restart_by_battery_size": by_size.to_dict("records"),
        "restart_by_downstream_pr": by_down.astype({"downstream_pr_q": str}).to_dict("records"),
        "restart_by_has_downstream": by_conn.to_dict("records"),
    }
    (RESULTS / "graph_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    top.to_csv(RESULTS / "graph_top_facilities.csv", index=False)
    comm.head(20).to_csv(RESULTS / "graph_top_communities.csv", index=False)
    deg.rename_axis("degree").rename("nodes").to_csv(RESULTS / "graph_degree_dist.csv")
    wells_per_bt.rename_axis("wells").rename("facilities").to_csv(RESULTS / "graph_wells_per_facility.csv")
    (RESULTS / "graph_community_sample.json").write_text(json.dumps({
        "nodes": [{"id": n, "type": n[2:4], "pagerank": pr[n]} for n in sub.nodes],
        "edges": [[u, v, d["months"]] for u, v, d in sub.edges(data=True)]}))

    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("restart")}, indent=1, default=str))
    print(by_size, by_down, by_conn, sep="\n")
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
