"""Load the raw CSV into a typed Parquet file. Nothing is removed or changed here."""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb

# Identifier columns that look numeric but are codes: read as text so leading zeros survive.
TEXT_COLUMNS = ("hospital_pk", "ccn", "zip", "fips_code")


def ingest(csv_path: Path, out_path: Path) -> int:
    """Write every row and column of `csv_path` to `out_path`. Returns the row count."""
    with csv_path.open(newline="") as f:
        header = next(csv.reader(f))
    types = ", ".join(f"'{c}': 'VARCHAR'" for c in TEXT_COLUMNS if c in header)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_csv('{csv_path}', header = true, sample_size = -1,
                                   types = {{{types}}})
        ) TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    n_rows = con.execute(f"SELECT count(*) FROM '{out_path}'").fetchone()[0]
    n_cols = len(con.execute(f"SELECT * FROM '{out_path}' LIMIT 0").description)

    # Every data line in the CSV must become exactly one row (no silently skipped lines).
    with csv_path.open(newline="") as f:
        n_lines = sum(1 for _ in csv.reader(f)) - 1
    if n_rows != n_lines:
        raise ValueError(f"CSV has {n_lines} records but Parquet has {n_rows} rows")
    if n_cols != len(header):
        raise ValueError(f"CSV has {len(header)} columns but Parquet has {n_cols}")
    print(f"ingested {n_rows:,} rows x {n_cols} columns -> {out_path}")
    return n_rows
