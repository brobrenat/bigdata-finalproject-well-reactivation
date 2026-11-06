#!/usr/bin/env python3
"""Lapisan kueri di atas data lake, memakai DuckDB.

DuckDB membaca Parquet langsung, jadi tidak ada proses muat terpisah dan
tidak ada server yang perlu dijalankan. Ini yang menggantikan peran HDFS
plus warehouse pada rancangan awal.

Pakai:
    python3 scripts/04_query.py
    python3 scripts/04_query.py --sql "SELECT count(*) FROM spells"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402

QUERIES = {
    "sebaran label": """
        SELECT label, count(*) AS n, round(median(idle_months), 1) AS median_idle
        FROM spells GROUP BY label ORDER BY n DESC
    """,
    "operator dengan sumur idle terbanyak": """
        SELECT operator_name, count(*) AS idle_spells,
               round(avg(idle_months), 1) AS avg_idle
        FROM spells WHERE label <> 'censored'
        GROUP BY operator_name ORDER BY idle_spells DESC LIMIT 10
    """,
    "peluang hidup lagi menurut durasi idle": """
        SELECT idle_months / 6 * 6 AS bucket_bulan,
               count(*) AS n,
               round(100.0 * sum(label = 'positive') / count(*), 1) AS persen_hidup_lagi
        FROM spells WHERE label <> 'censored'
        GROUP BY bucket_bulan ORDER BY bucket_bulan
    """,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql", default=None, help="Kueri sendiri atas view spells")
    args = parser.parse_args()

    try:
        import duckdb
    except ImportError:
        print("duckdb belum terpasang: python3 -m pip install duckdb")
        return 1

    spells = settings.GOLD_DIR / "idle_spells"
    if not spells.exists():
        print(f"Belum ada {spells}. Jalankan pipeline/g1_idle_spells.py dulu.")
        return 1

    con = duckdb.connect()
    con.execute(
        "CREATE VIEW spells AS SELECT * FROM read_parquet(?, hive_partitioning=1)",
        [f"{spells}/**/*.parquet"],
    )

    if args.sql:
        print(con.sql(args.sql))
        return 0

    for title, sql in QUERIES.items():
        print(f"--- {title} ---")
        print(con.sql(sql))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
