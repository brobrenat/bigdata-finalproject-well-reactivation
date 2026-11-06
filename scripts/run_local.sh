#!/usr/bin/env bash
# Jalankan tahap pipeline dalam mode Spark lokal, tanpa Docker.
# Prasyarat: JRE 17 di ~/opt dan virtualenv ~/bigdata-venv (lihat README).
#   bash scripts/run_local.sh pipeline/s1_to_silver.py pipeline/s2_wells_monthly.py ...
set -euo pipefail
export JAVA_HOME="${JAVA_HOME:-$(ls -d "$HOME"/opt/jdk-17*-jre | head -1)}"
export PATH="$JAVA_HOME/bin:$PATH"
export SPARK_MASTER="${SPARK_MASTER:-local[*]}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-4g}"
PY="${PY:-$HOME/bigdata-venv/bin/python}"
cd "$(dirname "$0")/.."
for step in "$@"; do
  echo "== $step  ($(date +%H:%M:%S))"
  /usr/bin/time -f "   selesai dalam %e detik, memori maks %M KB" "$PY" "$step"
done
