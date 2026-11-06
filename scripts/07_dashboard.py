"""Dashboard eksekutif DSS sumur idle, dari prediksi m1_train_spark.py (uji balik 2025).

Pakai (dari root proyek, supaya .streamlit/config.toml terbaca):
    ~/bigdata-venv/bin/python -m streamlit run scripts/07_dashboard.py
Lalu buka http://localhost:8501
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydeck as pdk  # noqa: E402
import streamlit as st  # noqa: E402

from config import paths  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODEL = "gbt"

# Palet referensi skill dataviz, mode terang.
INK, INK_2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
# Ramp biru sekuensial, langkah 250 sampai 700, untuk peluang di peta.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Fitur dikelompokkan ke bahasa bisnis untuk grafik pendorong.
GROUPS = {
    "Perusahaan pengelola": ("operator_top", "op_share", "log_op_wells"),
    "Fasilitas penampung hasil": ("facility_subtype", "fac_", "log_fac_wells", "log_community"),
    "Catatan produksi sumur": ("log_oil", "log_gas", "log_water", "log_cond", "log_gor", "log_cum", "water_cut",
                               "hours", "oil_decline", "gas_decline", "active_12", "prior_spells", "months_observed",
                               "is_oil"),
    "Lokasi": ("latitude", "longitude", "township", "range", "meridian", "jurisdiction"),
    "Harga minyak & waktu": ("wti", "dec_month"),
}


@st.cache_data
def load():
    pred = pd.read_parquet(RESULTS / "spark_predictions.parquet").query("model == @MODEL")
    # Salinan kecil kolom gold untuk deploy cloud, yang tidak punya data lake.
    # Dibuat otomatis saat dashboard pertama dijalankan di mesin yang punya gold/ml_features.
    small = RESULTS / "dashboard_features.parquet"
    if small.exists():
        feat = pd.read_parquet(small)
    else:
        feat = pd.read_parquet(
            paths.local_dir("gold") / "ml_features",
            columns=["uwi", "L", "jurisdiction", "operator_top", "facility_subtype", "oil_last", "gas_last",
                     "decision_month", "latitude", "longitude"],
        ).merge(pred[["uwi", "L"]].drop_duplicates(), on=["uwi", "L"])
        feat.to_parquet(small, index=False)
    df = pred.merge(feat, on=["uwi", "L"]).sort_values("p", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    imp = pd.read_csv(RESULTS / "spark_feature_importance.csv")
    imp["group"] = imp["feature"].map(lambda f: next((g for g, ps in GROUPS.items() if f.startswith(ps)), "Lainnya"))
    drivers = imp.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    metrics = json.loads((RESULTS / "spark_metrics.json").read_text())["models"][MODEL]
    return df, drivers, metrics


def theme(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(font=FONT)
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=INK_2, gridColor=GRID, domainColor=BASE, tickColor=BASE,
                            labelFontSize=12, titleFontSize=12, titleFontWeight="normal"))


st.set_page_config(page_title="DSS Sumur Idle", layout="wide")
st.markdown(f"""<style>
html, body, [class*="css"] {{ font-family: {FONT}; }}
.block-container {{ padding-top: 2rem; max-width: 1400px; }}
h1 {{ font-weight: 650; letter-spacing: -0.02em; }}
.kpi {{ background:#fcfcfb; border:1px solid rgba(11,11,11,.10); border-radius:12px; padding:18px 20px; height:100%; }}
.kpi .lbl {{ color:{INK_2}; font-size:13px; }}
.kpi .val {{ color:{INK}; font-size:34px; font-weight:650; line-height:1.2; margin-top:4px; }}
.kpi .sub {{ color:{MUTED}; font-size:13px; margin-top:2px; }}
.kpi .up {{ color:#006300; font-weight:600; }}
.card-title {{ color:{INK}; font-size:16px; font-weight:600; margin:0 0 2px; }}
.legend {{ display:flex; gap:16px; align-items:center; color:{INK_2}; font-size:12px; margin-top:6px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
</style>""", unsafe_allow_html=True)


def kpi(col, label, value, sub):
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


df_all, drivers, metrics = load()
ORANGE = "#eb6834"  # slot kategorikal 2, berdampingan dengan BLUE lolos uji CVD


def fluid(subtype: str) -> str:
    s = str(subtype).upper()
    return "Pasir minyak" if "OIL SANDS" in s else "Minyak" if "OIL" in s else "Gas" if "GAS" in s else "Lainnya"


def title(text: str):
    st.markdown(f"<p class='card-title'>{text}</p>", unsafe_allow_html=True)


def chart(c, height):
    st.altair_chart(theme(c.properties(height=height)), width="stretch")


h, f1, f2 = st.columns([3, 2, 1.4], vertical_alignment="bottom")
h.title("DSS Reaktivasi Sumur Idle")
n = f1.select_slider("Jumlah kandidat", options=[50, 100, 200, 300, 500, 750, 1000, 1500], value=500)
juris = f2.multiselect("Provinsi", ["AB", "SK"], default=["AB", "SK"],
                       format_func={"AB": "Alberta", "SK": "Saskatchewan"}.get)

df = df_all[df_all["jurisdiction"].isin(juris)].reset_index(drop=True)
if df.empty:
    st.warning("Pilih minimal satu provinsi.")
    st.stop()
df["rank"] = df.index + 1
df["jenis"] = df["facility_subtype"].map(fluid)
n = min(n, len(df))
top = df.head(n)
base = df["label"].mean()
hit = top["label"].mean()
found = int(top["label"].sum())

k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, "Sumur idle dinilai", f"{len(df):,}".replace(",", "."), "Jan–Jul 2025")
kpi(k2, "Tingkat hidup lagi, acak", f"{base:.0%}", "seluruh sumur idle")
kpi(k3, f"Ketepatan {n} teratas", f"{hit:.0%}", f"<span class='up'>▲ {hit / base:.1f}× lift</span>")
kpi(k4, "Kandidat yang hidup lagi", f"{found:,}".replace(",", "."), f"acak: ±{round(n * base)}")
kpi(k5, "ROC-AUC model", f"{metrics['roc_auc']:.3f}", f"P@100 {metrics['precision_at_100']:.0%}")

st.write("")
left, right = st.columns([3, 2], gap="large")

with left:
    title(f"Peta {n} kandidat teratas")
    rest = df.iloc[n:].assign(color=[[195, 194, 183, 90]] * (len(df) - n))
    bins = np.minimum((top["p"].rank(pct=True) * len(RAMP)).astype(int), len(RAMP) - 1)
    top_map = top.assign(color=[list(int(RAMP[b][i:i + 2], 16) for i in (1, 3, 5)) + [230] for b in bins],
                         pct=(top["p"] * 100).round(0).astype(int),
                         status=np.where(top["label"] == 1, "✓ hidup lagi", "✗ tetap idle"))
    st.pydeck_chart(pdk.Deck(
        map_style=pdk.map_styles.CARTO_LIGHT,
        initial_view_state=pdk.ViewState(latitude=52.6, longitude=-108.5, zoom=4.4),
        height=520,
        layers=[
            pdk.Layer("ScatterplotLayer", rest, get_position=["longitude", "latitude"], get_fill_color="color",
                      get_radius=900, radius_min_pixels=1.5),
            pdk.Layer("ScatterplotLayer", top_map, get_position=["longitude", "latitude"], get_fill_color="color",
                      get_line_color=[252, 252, 251], line_width_min_pixels=1, stroked=True,
                      get_radius=2500, radius_min_pixels=4, pickable=True),
        ],
        tooltip={"html": "<b>#{rank}</b> · {uwi}<br/>Skor: <b>{pct}%</b><br/>{jenis} · {operator_top}<br/>{status}",
                 "style": {"background": "#fcfcfb", "color": INK, "font-family": FONT, "font-size": "12px",
                           "border": "1px solid rgba(11,11,11,.10)", "border-radius": "8px"}},
    ))
    st.markdown("<div class='legend'>" + "".join(f"<span><i style='background:{c}'></i></span>" for c in RAMP)
                + "<span>skor rendah → tinggi</span>"
                f"<span><i style='background:{BASE}'></i>sumur idle lain</span></div>", unsafe_allow_html=True)

with right:
    title("Ketepatan menurut jumlah sumur dikerjakan")
    k = np.arange(1, min(len(df), 2000) + 1)
    curve = pd.DataFrame({"k": k, "rate": df["label"].cumsum().iloc[: len(k)].to_numpy() / k}).query("k >= 25")
    x = alt.X("k:Q", title="Kandidat teratas", scale=alt.Scale(domain=[0, int(k[-1])], nice=False),
              axis=alt.Axis(grid=False))
    y = alt.Y("rate:Q", title=None, axis=alt.Axis(format=".0%", tickCount=5), scale=alt.Scale(domain=[0, 1]))
    line = alt.Chart(curve).mark_line(color=BLUE, strokeWidth=2).encode(x, y)
    hover = alt.selection_point(fields=["k"], nearest=True, on="pointerover", empty=False)
    pts = alt.Chart(curve).mark_circle(size=70, color=BLUE).encode(
        x, y, opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("k:Q", title="Kandidat"), alt.Tooltip("rate:Q", title="Ketepatan", format=".0%")],
    ).add_params(hover)
    base_rule = alt.Chart(pd.DataFrame({"y": [base]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    base_lbl = alt.Chart(pd.DataFrame({"y": [base], "k": [k[-1]]})).mark_text(
        align="right", dy=-8, color=INK_2, fontSize=12, text=f"acak {base:.0%}").encode(x="k:Q", y="y:Q")
    sel = pd.DataFrame({"k": [n], "rate": [hit]})
    sel_rule = alt.Chart(sel).mark_rule(color=INK_2, strokeWidth=1).encode(x="k:Q")
    sel_pt = alt.Chart(sel).mark_circle(size=90, color=BLUE, stroke="#fcfcfb", strokeWidth=2).encode(x="k:Q", y="rate:Q")
    sel_lbl = alt.Chart(sel).mark_text(align="left", dx=8, dy=-12, color=INK, fontSize=13, fontWeight=600).encode(
        x="k:Q", y="rate:Q", text=alt.value(f"{n} → {hit:.0%}"))
    chart(line + base_rule + base_lbl + sel_rule + sel_pt + sel_lbl + pts, 230)

    title("Kelompok fitur paling berpengaruh")
    bars = alt.Chart(drivers).mark_bar(color=BLUE, cornerRadiusEnd=4, height=18).encode(
        x=alt.X("importance:Q", title=None, axis=alt.Axis(format=".0%", tickCount=4, grid=True),
                scale=alt.Scale(domain=[0, float(drivers["importance"].max()) * 1.15])),
        y=alt.Y("group:N", sort="-x", title=None, axis=alt.Axis(labelColor=INK_2, labelLimit=220, ticks=False,
                                                               domain=False)),
        tooltip=[alt.Tooltip("group:N", title="Kelompok"), alt.Tooltip("importance:Q", title="Pengaruh", format=".0%")],
    )
    vals = bars.mark_text(align="left", dx=6, color=INK_2, fontSize=12).encode(text=alt.Text("importance:Q", format=".0%"))
    chart(bars + vals, 200)

st.write("")
c1, c2, c3 = st.columns(3, gap="large")
OUTCOME = alt.Color("hasil:N", title=None, scale=alt.Scale(domain=["Hidup lagi", "Tetap idle"], range=[BLUE, ORANGE]),
                    legend=alt.Legend(orient="top", labelColor=INK_2, symbolType="circle"))

with c1:
    title("Sebaran skor menurut hasil nyata")
    d = df.assign(bin=(np.minimum(df["p"], 0.999) * 10).astype(int) / 10,
                  hasil=np.where(df["label"] == 1, "Hidup lagi", "Tetap idle"))
    hist = d.groupby(["bin", "hasil"], as_index=False).size()
    chart(alt.Chart(hist).mark_bar(cornerRadiusEnd=4, width=alt.RelativeBandSize(0.9)).encode(
        x=alt.X("bin:O", title="Skor model", axis=alt.Axis(labelExpr="format(datum.value, '.0%')", labelAngle=0)),
        xOffset="hasil:N",
        y=alt.Y("size:Q", title=None, axis=alt.Axis(tickCount=4)),
        color=OUTCOME,
        tooltip=[alt.Tooltip("bin:O", title="Skor ≥", format=".0%"), alt.Tooltip("hasil:N", title="Hasil"),
                 alt.Tooltip("size:Q", title="Sumur", format=",")],
    ), 260)

with c2:
    title(f"10 operator dengan kandidat terbanyak ({n} teratas)")
    named = top[top["operator_top"] != "OTHER"]  # operator kecil digabung OTHER, tidak informatif
    ops = (named.assign(op=named["operator_top"], hasil=np.where(named["label"] == 1, "Hidup lagi", "Tetap idle"))
           .groupby(["op", "hasil"], as_index=False).size())
    order = ops.groupby("op")["size"].sum().sort_values(ascending=False).head(10).index.tolist()
    chart(alt.Chart(ops[ops["op"].isin(order)]).mark_bar(stroke="#f9f9f7", strokeWidth=2).encode(
        y=alt.Y("op:N", sort=order, title=None,
                axis=alt.Axis(ticks=False, domain=False, labelColor=INK_2, labelOverlap=False)),
        x=alt.X("size:Q", title=None, stack="zero", axis=alt.Axis(tickCount=4)),
        color=OUTCOME, order=alt.Order("hasil:N"),
        tooltip=[alt.Tooltip("op:N", title="Operator"), alt.Tooltip("hasil:N", title="Hasil"),
                 alt.Tooltip("size:Q", title="Sumur")],
    ), 260)

with c3:
    title(f"Ketepatan per jenis sumur ({n} teratas)")
    jen = top.groupby("jenis", as_index=False).agg(rate=("label", "mean"), count=("label", "size"))
    jen["txt"] = [f"{r:.0%} (n={c})" for r, c in zip(jen["rate"], jen["count"])]
    jb = alt.Chart(jen).mark_bar(color=BLUE, cornerRadiusEnd=4, height=22).encode(
        y=alt.Y("jenis:N", sort="-x", title=None, axis=alt.Axis(ticks=False, domain=False, labelColor=INK_2)),
        x=alt.X("rate:Q", title=None, scale=alt.Scale(domain=[0, 1.35]),
                axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1])),
        tooltip=[alt.Tooltip("jenis:N", title="Jenis"), alt.Tooltip("rate:Q", title="Ketepatan", format=".0%"),
                 alt.Tooltip("count:Q", title="Kandidat")],
    )
    jt = jb.mark_text(align="left", dx=6, color=INK_2, fontSize=12).encode(text="txt:N")
    rule = alt.Chart(pd.DataFrame({"x": [base]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(x="x:Q")
    chart(jb + jt + rule, 260)

st.write("")
title(f"{n} kandidat teratas")
BULAN = "Jan Feb Mar Apr Mei Jun Jul Agu Sep Okt Nov Des".split()
last_prod = pd.PeriodIndex(top["decision_month"], freq="M") - 6
table = pd.DataFrame({
    "#": top["rank"],
    "UWI": top["uwi"],
    "Provinsi": top["jurisdiction"],
    "Jenis": top["jenis"],
    "Operator": top["operator_top"].replace({"OTHER": "Lainnya"}),
    "Produksi terakhir": [f"{BULAN[m.month - 1]} {m.year}" for m in last_prod],
    "Minyak terakhir (m³)": top["oil_last"].round(1),
    "Gas terakhir (e³m³)": top["gas_last"].round(1),
    "Skor": (top["p"] * 100).round(0),
    "Hasil 12 bulan": np.where(top["label"] == 1, "✓ Hidup lagi", "✗ Tetap idle"),
})
st.dataframe(table, hide_index=True, width="stretch", height=420, column_config={
    "Skor": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
})
