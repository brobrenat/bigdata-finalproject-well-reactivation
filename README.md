# DSS Potensi Sumur Idle

Pipeline big data untuk memeringkat sumur idle berdasarkan peluang masih
adanya sisa minyak. Tujuannya membantu memilih exploitation well yang murah
daripada exploration well yang mahal.

Data publik dari Petrinex, yaitu data volumetrik bulanan Alberta dan
Saskatchewan.

## Alur

```
unduh (zip)  ->  bronze (CSV)  ->  silver (Parquet)  ->  gold (periode idle berlabel)
```

Kode ada di folder ini. Data TIDAK disimpan di sini, melainkan di
`~/bigdata-data` pada filesystem Linux. Folder proyek berada di drive
Windows yang ruangnya tipis dan lambat dibaca dari WSL.

## Menjalankan

Tahap pengambilan data hanya butuh Python standar. Tahap Spark dijalankan di
dalam Docker, sehingga Java dan PySpark tidak perlu dipasang di host.

### Tahap 1, pengambilan data. Tanpa Docker, tanpa Java.

```bash
# Unduh data. Idempoten, aman dijalankan ulang kalau koneksi putus.
python3 scripts/01_download_petrinex.py

# Ekstrak zip bersarang ke CSV
python3 pipeline/b1_extract.py

# Lihat isi datanya sebelum diproses
python3 scripts/02_probe_schema.py

# Hitung apakah jumlah label idle memadai, tanpa Spark
python3 scripts/03_label_feasibility.py

# Uji
python3 -m unittest discover -s tests -v
```

### Tahap 2, Spark di Docker

Prasyarat: Docker Desktop berjalan, dan integrasi WSL untuk distro ini
dinyalakan di Settings, Resources, WSL Integration.

```bash
# Nyalakan cluster
docker compose -f docker/docker-compose.yml up -d

# Periksa kesiapannya sebelum memproses apa pun
bash docker/check.sh

# Jalankan tahap pipeline
bash docker/run.sh pipeline/s1_to_silver.py
bash docker/run.sh pipeline/s2_wells_monthly.py
bash docker/run.sh pipeline/g1_idle_spells.py
```

Antarmuka Spark master ada di http://localhost:8080 dan worker di
http://localhost:8081. Tangkapan layar halaman master saat job berjalan
adalah bukti paling langsung bahwa Spark benar-benar dipakai.

Skrip dipanggil dengan `bash` di depannya karena folder proyek berada di
drive Windows, dan bit executable tidak selalu tersimpan di sana.

Mode lokal di dalam satu container, tanpa worker terpisah:

```bash
SPARK_MASTER='local[*]' bash docker/run.sh pipeline/s1_to_silver.py
```

### Tahap 3, kueri SQL

```bash
python3 -m pip install --break-system-packages duckdb
python3 scripts/04_query.py
python3 scripts/04_query.py --sql "SELECT count(*) FROM spells"
```

DuckDB membaca Parquet layer gold langsung. Tidak ada server, tidak ada
proses muat.

## Yang sudah diverifikasi dari datanya

Bukan asumsi. Angka di bawah diambil dari berkas yang benar-benar diunduh
dan dibuka.

| Aspek | Nilai |
|---|---|
| Endpoint | `.../publicdata/API/Files/{AB\|SK}/Vol/{YYYY-MM}/CSV` |
| Bulan tersedia | 2022-01 sampai 2026-07 |
| Satu bulan Alberta, zip | sekitar 7,7 MB |
| CSV di dalamnya | sekitar 121 MB |
| Baris per bulan | sekitar 562.000 |
| Rasio ekspansi | 15,7 kali |
| Jumlah kolom | 30 |
| Sumur unik berproduksi per bulan | sekitar 139.000 |
| UWI berhasil diurai | 100 persen |

Setelah 110 berkas diunduh dan diekstrak, seluruh dataset berukuran seperti
ini.

| Aspek | Nilai |
|---|---|
| Zip diunduh | 574 MB |
| CSV bronze | 8,8 GB |
| Baris CSV | 42.147.422 |
| Baris produksi tingkat sumur | 18.666.850 |
| Sumur unik | 212.953 |
| Bulan tercakup | 55 |

Hasil pelabelan periode idle, dihitung oleh
`scripts/03_label_feasibility.py`.

| Label | Jumlah |
|---|---|
| positive, sumur hidup lagi | 13.885 |
| negative, tetap mati | 23.158 |
| censored, belum bisa dinilai | 8.381 |
| bisa dilatih | 37.043 |

Proporsi positive 37,5 persen, jadi tidak ada masalah ketimpangan kelas.
Median durasi idle 21 bulan, terpanjang 54 bulan.

## Tiga hal yang menentukan bentuk pipeline

**Zip bersarang.** Endpoint mengembalikan zip yang berisi zip lagi, baru
kemudian CSV. Ditangani di `pipeline/b1_extract.py` dan dikunci oleh
`tests/test_extract.py`.

**Sumur bukan fasilitas pelapor.** Kolom `ReportingFacilityType` hanya
berisi BT, GS, IF, dan GP. Tidak ada tipe sumur di sana. Sumur muncul
sebagai lawan aliran, yaitu `FromToIDType` bernilai `WI`. Baris produksi
sumur adalah `FromToIDType = WI` dan `ActivityID = PROD`.

**Format panjang.** Minyak, gas, air, dan kondensat menempati baris
terpisah untuk bulan yang sama, jadi perlu pivot.

Satu hal yang ternyata TIDAK perlu: dedupe amendemen. Tiap berkas bulanan
hanya memuat satu ProductionMonth, dan baris volumetriknya tidak
berduplikat. Duplikat yang ada hanyalah baris faktor prorasi yang
ActivityID dan ProductID-nya kosong.

## Koordinat peta

Data volumetrik tidak memuat lintang bujur. Koordinat diturunkan dari
komponen Dominion Land Survey yang terkandung di dalam UWI, dikerjakan oleh
`pipeline/dls.py`.

Hasilnya perkiraan, yaitu titik tengah legal subdivision. Road allowance
dan koreksi survei tidak dimodelkan. Untuk peta sebaran sumur ini memadai,
tetapi jangan dipakai untuk keperluan survei.

## Kenapa tanpa HDFS

Tugasnya meminta teknologi penyimpanan, bukan HDFS secara khusus. Yang
dipakai di sini data lake berlapis, yaitu Parquet berpartisi di layer
bronze, silver, dan gold. Kueri analitiknya lewat DuckDB, yang berperan
sebagai lapisan warehouse.

Alasan HDFS dibuang: datanya 8,8 GB dan jalan di satu mesin. HDFS butuh
namenode, datanode, dan formatting, tetapi tidak memberi apa pun yang
Parquet berpartisi belum berikan pada skala ini. Kalau suatu saat datanya
melewati satu mesin, penggantinya object storage dengan API S3, bukan HDFS.

Pemetaan ke soal:

| Diminta | Dipakai |
|---|---|
| Data ingestion | Unduhan HTTP bulanan, idempoten, bermanifes |
| Data lake | Parquet berpartisi per yurisdiksi dan tahun |
| Data warehouse | DuckDB di atas Parquet layer gold |
| Processing framework | Spark di Docker |

## Tiga label, bukan dua

`pipeline/g1_idle_spells.py` menghasilkan `positive`, `negative`, dan
`censored`.

Baris `censored` adalah sumur yang masih idle saat data berakhir tetapi
belum cukup lama diamati untuk disimpulkan. Baris ini wajib dikeluarkan
dari data latih. Kalau dianggap negatif, model akan belajar bahwa sumur
yang baru berhenti selalu gagal hidup lagi, padahal itu hanya akibat data
yang belum cukup panjang.

## Gerbang keputusan

Riwayat data hanya sekitar 4,6 tahun, jadi jumlah sumur yang sempat idle
lalu hidup lagi bisa sedikit. Setelah `g1_idle_spells.py` selesai, lihat
jumlah label `positive` yang dicetaknya.

Kalau memadai, lanjut ke pemodelan dan dashboard. Kalau terlalu sedikit,
ubah sasaran prediksi menjadi memperkirakan sumur mana yang akan menjadi
idle, atau memperkirakan sisa cadangan dari decline curve. Keduanya
menjawab rumusan masalah yang sama dan memakai pipeline yang persis sama.

## Assignment II: machine learning dan graph analytics

Tahap ini dijalankan dalam mode Spark lokal, tanpa Docker. Prasyarat: JRE 17
di `~/opt` dan virtualenv `~/bigdata-venv` berisi paket di `requirements.txt`.

```bash
bash scripts/run_local.sh \
  pipeline/s1_to_silver.py pipeline/s2_wells_monthly.py pipeline/g1_idle_spells.py \
  pipeline/n1_graph.py pipeline/f1_features.py \
  pipeline/m1_train_spark.py pipeline/m2_train_sklearn.py
~/bigdata-venv/bin/python scripts/05_ml_figures.py
```

| Skrip | Isi | Waktu di laptop 8 core |
|---|---|---|
| `n1_graph.py` | Graf aliran fasilitas: PageRank (NetworkX dan Spark), Louvain, komponen | ±3 menit |
| `f1_features.py` | Label horizon 12 bulan, 42 fitur, pembagian menurut waktu | ±2 menit |
| `m1_train_spark.py` | LR, Random Forest, GBT di Spark MLlib, ablasi fitur graf, regresi | ±7 menit |
| `m2_train_sklearn.py` | Model yang sama dengan scikit-learn, tanpa Spark | ±1 menit |

Hasil evaluasi ditulis ke folder `results`. Data tambahan: harga minyak WTI
bulanan dari FRED, seri MCOILWTICO, disimpan di `raw/wti/wti_monthly.csv`.
