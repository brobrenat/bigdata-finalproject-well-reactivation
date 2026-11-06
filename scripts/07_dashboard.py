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
.card-sub {{ color:{INK_2}; font-size:13px; margin:0 0 10px; }}
.legend {{ display:flex; gap:16px; align-items:center; color:{INK_2}; font-size:12px; margin-top:6px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
</style>""", unsafe_allow_html=True)


def kpi(col, label, value, sub):
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


df_all, drivers, metrics = load()

st.title("Sumur mati mana yang masih bisa berproduksi lagi?")
st.markdown("<p class='card-sub'>Sumur minyak & gas di Alberta dan Saskatchewan yang berhenti berproduksi "
            "Januari–Juli 2025</p>", unsafe_allow_html=True)
st.info("**Cara membaca dashboard ini.** Sistem hanya diberi data sampai akhir 2024, lalu diminta menebak sumur mati "
        "mana yang akan berproduksi lagi. Setahun kemudian tebakannya dicocokkan dengan kenyataan. Jadi semua angka di "
        "bawah ini adalah **hasil ujian**, bukan janji.")

# Satu baris filter di atas semua grafik.
f1, f2, _ = st.columns([2, 2, 3])
n = f1.select_slider("Berapa sumur yang mau dihidupkan kembali?",
                     options=[50, 100, 200, 300, 500, 750, 1000, 1500], value=500)
juris = f2.multiselect("Provinsi", ["AB", "SK"], default=["AB", "SK"],
                       format_func={"AB": "Alberta", "SK": "Saskatchewan"}.get)

df = df_all[df_all["jurisdiction"].isin(juris)].reset_index(drop=True)
if df.empty:
    st.info("Pilih minimal satu provinsi.")
    st.stop()
df["rank"] = df.index + 1
n = min(n, len(df))
top = df.head(n)
base = df["label"].mean()
hit = top["label"].mean()
found = int(top["label"].sum())


def per100(x: float) -> str:
    return f"{round(x * 100)} dari 100"


st.write("")
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Sumur mati yang dinilai", f"{len(df):,}".replace(",", "."), "berhenti berproduksi minimal 6 bulan")
kpi(k2, "Kalau dipilih sembarang", per100(base), "yang ternyata bisa berproduksi lagi")
kpi(k3, f"Kalau ikut {n} rekomendasi sistem", per100(hit),
    f"<span class='up'>▲ {hit / base:.1f}× lebih sering tepat</span>")
kpi(k4, "Sumur yang berhasil hidup lagi", f"{found:,}".replace(",", "."),
    f"dari {n} · pilih sembarang hanya ±{round(n * base)}")

st.write("")
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown(f"<p class='card-title'>Di mana {n} sumur yang direkomendasikan?</p>"
                "<p class='card-sub'>Makin pekat birunya, makin yakin sistem bahwa sumur itu masih bisa berproduksi. "
                "Titik abu-abu: sumur mati lainnya.</p>", unsafe_allow_html=True)
    rest = df.iloc[n:].assign(color=[[195, 194, 183, 90]] * (len(df) - n))
    bins = np.minimum((top["p"].rank(pct=True) * len(RAMP)).astype(int), len(RAMP) - 1)
    top_map = top.assign(color=[list(int(RAMP[b][i:i + 2], 16) for i in (1, 3, 5)) + [230] for b in bins],
                         pct=(top["p"] * 100).round(0).astype(int),
                         status=np.where(top["label"] == 1, "✓ berproduksi lagi", "✗ tetap mati"))
    st.pydeck_chart(pdk.Deck(
        map_style=pdk.map_styles.CARTO_LIGHT,
        initial_view_state=pdk.ViewState(latitude=52.6, longitude=-108.5, zoom=4.4),
        height=470,
        layers=[
            pdk.Layer("ScatterplotLayer", rest, get_position=["longitude", "latitude"], get_fill_color="color",
                      get_radius=900, radius_min_pixels=1.5),
            pdk.Layer("ScatterplotLayer", top_map, get_position=["longitude", "latitude"], get_fill_color="color",
                      get_line_color=[252, 252, 251], line_width_min_pixels=1, stroked=True,
                      get_radius=2500, radius_min_pixels=4, pickable=True),
        ],
        tooltip={"html": "<b>Urutan #{rank}</b> · sumur {uwi}<br/>Keyakinan sistem: <b>{pct}%</b><br/>"
                         "Kenyataannya: {status}",
                 "style": {"background": "#fcfcfb", "color": INK, "font-family": FONT, "font-size": "12px",
                           "border": "1px solid rgba(11,11,11,.10)", "border-radius": "8px"}},
    ))
    st.markdown("<div class='legend'>" + "".join(f"<span><i style='background:{c}'></i></span>" for c in RAMP)
                + "<span>kurang yakin → sangat yakin</span>"
                f"<span><i style='background:{BASE}'></i>sumur mati lainnya</span></div>", unsafe_allow_html=True)

with right:
    st.markdown("<p class='card-title'>Makin selektif, makin sering tepat</p>"
                "<p class='card-sub'>Dari sumur yang dipilih sesuai urutan sistem, berapa persen yang benar-benar "
                "berproduksi lagi</p>", unsafe_allow_html=True)
    k = np.arange(1, min(len(df), 2000) + 1)
    curve = pd.DataFrame({"k": k, "rate": df["label"].cumsum().iloc[: len(k)].to_numpy() / k}).query("k >= 25")
    x = alt.X("k:Q", title="Jumlah sumur yang dikerjakan", scale=alt.Scale(domain=[0, int(k[-1])], nice=False),
              axis=alt.Axis(grid=False))
    y = alt.Y("rate:Q", title=None, axis=alt.Axis(format=".0%", tickCount=5), scale=alt.Scale(domain=[0, 1]))
    line = alt.Chart(curve).mark_line(color=BLUE, strokeWidth=2).encode(x, y)
    hover = alt.selection_point(fields=["k"], nearest=True, on="pointerover", empty=False)
    pts = alt.Chart(curve).mark_circle(size=70, color=BLUE).encode(
        x, y, opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("k:Q", title="Sumur dikerjakan"),
                 alt.Tooltip("rate:Q", title="Yang berproduksi lagi", format=".0%")],
    ).add_params(hover)
    base_rule = alt.Chart(pd.DataFrame({"y": [base]})).mark_rule(color=MUTED, strokeWidth=1).encode(y="y:Q")
    base_lbl = alt.Chart(pd.DataFrame({"y": [base], "k": [k[-1]]})).mark_text(
        align="right", dy=-8, color=INK_2, fontSize=12, text=f"Pilih sembarang {base:.0%}").encode(x="k:Q", y="y:Q")
    sel = pd.DataFrame({"k": [n], "rate": [hit]})
    sel_rule = alt.Chart(sel).mark_rule(color=INK_2, strokeWidth=1).encode(x="k:Q")
    sel_pt = alt.Chart(sel).mark_circle(size=90, color=BLUE, stroke="#fcfcfb", strokeWidth=2).encode(x="k:Q", y="rate:Q")
    sel_lbl = alt.Chart(sel).mark_text(align="left", dx=8, dy=-12, color=INK, fontSize=13, fontWeight=600).encode(
        x="k:Q", y="rate:Q", text=alt.value(f"{n} sumur → {hit:.0%} tepat"))
    st.altair_chart(theme((line + base_rule + base_lbl + sel_rule + sel_pt + sel_lbl + pts).properties(height=215)),
                    use_container_width=True)

    st.markdown("<p class='card-title'>Apa yang dilihat sistem saat menilai sumur?</p>"
                "<p class='card-sub'>Seberapa besar pengaruh tiap hal terhadap penilaian</p>",
                unsafe_allow_html=True)
    bars = alt.Chart(drivers).mark_bar(color=BLUE, cornerRadiusEnd=4, height=18).encode(
        x=alt.X("importance:Q", title=None, axis=alt.Axis(format=".0%", tickCount=4, grid=True),
                scale=alt.Scale(domain=[0, float(drivers["importance"].max()) * 1.15])),
        y=alt.Y("group:N", sort="-x", title=None, axis=alt.Axis(labelColor=INK_2, labelLimit=220, ticks=False,
                                                               domain=False)),
        tooltip=[alt.Tooltip("group:N", title="Hal yang dilihat"),
                 alt.Tooltip("importance:Q", title="Pengaruh", format=".0%")],
    )
    vals = bars.mark_text(align="left", dx=6, color=INK_2, fontSize=12).encode(text=alt.Text("importance:Q", format=".0%"))
    st.altair_chart(theme((bars + vals).properties(height=200)), use_container_width=True)


def fluid(subtype: str) -> str:
    s = str(subtype).upper()
    return "Pasir minyak" if "OIL SANDS" in s else "Minyak" if "OIL" in s else "Gas" if "GAS" in s else "Lainnya"


st.write("")
st.markdown(f"<p class='card-title'>Daftar {n} sumur yang disarankan untuk dihidupkan kembali</p>"
            "<p class='card-sub'>Diurutkan dari yang paling menjanjikan. Kolom terakhir adalah apa yang benar-benar "
            "terjadi dalam 12 bulan berikutnya, dipakai untuk membuktikan apakah tebakan sistem tepat.</p>",
            unsafe_allow_html=True)
BULAN = "Jan Feb Mar Apr Mei Jun Jul Agu Sep Okt Nov Des".split()
last_prod = pd.PeriodIndex(top["decision_month"], freq="M") - 6
table = pd.DataFrame({
    "Urutan": top["rank"],
    "Kode sumur": top["uwi"],
    "Provinsi": top["jurisdiction"].map({"AB": "Alberta", "SK": "Saskatchewan"}),
    "Jenis": top["facility_subtype"].map(fluid),
    "Terakhir berproduksi": [f"{BULAN[m.month - 1]} {m.year}" for m in last_prod],
    "Keyakinan sistem": (top["p"] * 100).round(0),
    "Kenyataan 12 bulan kemudian": np.where(top["label"] == 1, "✓ Berproduksi lagi", "✗ Tetap mati"),
})
st.dataframe(table, hide_index=True, use_container_width=True, height=420, column_config={
    "Keyakinan sistem": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
})

with st.expander("Catatan teknis (untuk tim analis)"):
    st.markdown(
        f"- Data: Petrinex volumetrik bulanan 2022–2026, 212.953 sumur. Diproses dengan Spark (bronze → silver → gold).\n"
        f"- Sumur dianggap mati setelah 6 bulan tanpa produksi. \"Berproduksi lagi\" berarti ada produksi dalam "
        f"12 bulan setelah titik itu.\n"
        f"- Uji buta menurut waktu: model Gradient-Boosted Trees (Spark MLlib) dilatih dengan keputusan sampai "
        f"Des 2024, diuji pada {metrics['n']:,} keputusan Jan–Jul 2025 yang tidak pernah dilihat saat pelatihan.\n"
        f"- ROC-AUC {metrics['roc_auc']:.2f} · precision@100 {metrics['precision_at_100']:.0%} · "
        f"precision@500 {metrics['precision_at_500']:.0%}.\n"
        f"- \"Keyakinan sistem\" dipakai untuk **mengurutkan**, bukan probabilitas absolut; model belum dikalibrasi.\n"
        f"- Fitur graf fasilitas (PageRank, komunitas) dihitung dari seluruh periode data, jadi ada sedikit informasi "
        f"masa depan di fitur tersebut."
    )
