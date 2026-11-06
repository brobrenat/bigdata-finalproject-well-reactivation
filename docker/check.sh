#!/usr/bin/env bash
# Periksa kesiapan cluster Spark di Docker sebelum pipeline dijalankan.
#
# Memeriksa daemon Docker, container yang jalan, Python dan PySpark di dalam
# image, akses baca tulis ke /data, lalu mencoba satu job Spark kecil.
#
# Pakai:
#   ./docker/check.sh

set -uo pipefail

COMPOSE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker-compose.yml"
FAILED=0

step() { printf '%-42s' "$1"; }
pass() { echo "OK"; }
fail() { echo "GAGAL  $1"; FAILED=1; }

step "daemon Docker"
if docker info >/dev/null 2>&1; then
  pass
else
  fail "aktifkan integrasi WSL di Docker Desktop, lalu pastikan Docker Desktop berjalan"
  exit 1
fi

step "container spark-master jalan"
if docker compose -f "$COMPOSE_FILE" ps --status running --services 2>/dev/null | grep -q spark-master; then
  pass
else
  fail "jalankan: docker compose -f docker/docker-compose.yml up -d"
  exit 1
fi

step "container spark-worker jalan"
if docker compose -f "$COMPOSE_FILE" ps --status running --services 2>/dev/null | grep -q spark-worker; then
  pass
else
  fail "worker tidak jalan, cek: docker compose -f docker/docker-compose.yml logs spark-worker"
fi

in_master() { docker compose -f "$COMPOSE_FILE" exec -T spark-master "$@" 2>&1; }

step "python3 di dalam image"
PY=$(in_master python3 --version)
if [[ "$PY" == Python* ]]; then
  echo "OK   $PY"
else
  fail "image tanpa Python. Ganti image ke apache/spark-py:3.4.1 di docker-compose.yml"
fi

step "pyspark di dalam image"
SPARK_VER=$(in_master python3 -c "import pyspark; print(pyspark.__version__)")
if [[ "$SPARK_VER" =~ ^3\. ]]; then
  echo "OK   $SPARK_VER"
else
  fail "pyspark tidak terimpor: $SPARK_VER"
fi

step "kode proyek termount di /app"
if in_master test -f /app/config/settings.py; then
  pass
else
  fail "mount ../:/app tidak terbaca"
fi

step "data termount dan bisa ditulis"
COUNT=$(in_master bash -c 'ls /data/bronze/petrinex 2>/dev/null | wc -l')
if in_master bash -c 'touch /data/.write_probe && rm -f /data/.write_probe'; then
  echo "OK   yurisdiksi di bronze: ${COUNT//[!0-9]/}"
else
  fail "tidak bisa menulis ke /data. Cek kepemilikan berkas, container jalan sebagai uid 1001"
fi

step "job Spark kecil lewat master"
OUT=$(docker compose -f "$COMPOSE_FILE" exec -T \
  -e SPARK_MASTER="spark://spark-master:7077" \
  -e SPARK_DRIVER_MEMORY=1g \
  -e PYTHONPATH=/app \
  spark-master python3 -c "
from pipeline.spark_session import build_spark
s = build_spark('docker_check', driver_memory='1g')
print('ROWS', s.range(1000).count())
print('PARALLELISM', s.sparkContext.defaultParallelism)
s.stop()
" 2>&1)
if grep -q "ROWS 1000" <<<"$OUT"; then
  echo "OK   $(grep PARALLELISM <<<"$OUT" | head -1)"
else
  fail "job gagal"
  echo "$OUT" | tail -20
fi

echo
if [ "$FAILED" -eq 0 ]; then
  echo "Semua siap. Lanjut:  ./docker/run.sh pipeline/s1_to_silver.py"
  echo "Antarmuka master  :  http://localhost:8080"
else
  echo "Ada yang belum siap. Perbaiki yang bertanda GAGAL di atas."
fi
exit "$FAILED"
