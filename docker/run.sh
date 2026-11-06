#!/usr/bin/env bash
# Jalankan satu tahap pipeline di dalam cluster Spark pada Docker.
#
# Driver dijalankan di dalam container spark-master, bukan di container
# sekali pakai. Alasannya jaringan: worker perlu menghubungi balik driver,
# dan nama spark-master sudah terdaftar di DNS jaringan compose. Container
# sekali pakai tidak punya nama yang bisa dihubungi worker.
#
# Pakai:
#   ./docker/run.sh pipeline/s1_to_silver.py
#   ./docker/run.sh pipeline/s2_wells_monthly.py
#   ./docker/run.sh pipeline/g1_idle_spells.py
#
#   # Mode lokal di dalam satu container, tanpa worker terpisah
#   SPARK_MASTER='local[*]' ./docker/run.sh pipeline/s1_to_silver.py

set -euo pipefail

if [ $# -eq 0 ]; then
  echo "Pakai: $0 <skrip python> [argumen...]" >&2
  echo "Contoh: $0 pipeline/s1_to_silver.py" >&2
  exit 2
fi

COMPOSE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker-compose.yml"

SPARK_MASTER="${SPARK_MASTER:-spark://spark-master:7077}"
SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-2g}"
SPARK_SHUFFLE_PARTITIONS="${SPARK_SHUFFLE_PARTITIONS:-16}"

if ! docker compose -f "$COMPOSE_FILE" ps --status running --services 2>/dev/null | grep -q spark-master; then
  echo "spark-master belum jalan. Nyalakan dulu:" >&2
  echo "  docker compose -f docker/docker-compose.yml up -d" >&2
  exit 1
fi

echo "Master  : $SPARK_MASTER"
echo "Skrip   : $*"
echo

exec docker compose -f "$COMPOSE_FILE" exec -T \
  -e SPARK_MASTER="$SPARK_MASTER" \
  -e SPARK_DRIVER_MEMORY="$SPARK_DRIVER_MEMORY" \
  -e SPARK_SHUFFLE_PARTITIONS="$SPARK_SHUFFLE_PARTITIONS" \
  -e BIGDATA_DATA_ROOT=/data \
  -e PYTHONPATH=/app \
  spark-master python3 "$@"
