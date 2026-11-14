"""Data-stage tests on a tiny hand-built CSV, so CI never downloads the 481 MB file.

Each row of the fixture exercises one case the cleaning code must get right.
"""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from smartops.data.clean import MEASURES, SENTINEL, DataValidationError, clean
from smartops.data.download import read_sums, sha256_of, verify, write_sums
from smartops.data.ingest import ingest

S = SENTINEL
W0 = date(2020, 8, 2)  # first training week, a Sunday


def week(i: int) -> str:
    return (W0 + timedelta(weeks=i)).isoformat()


def row(pk: str, wk: str, **overrides) -> dict:
    r = {
        "hospital_pk": pk,
        "collection_week": wk,
        "state": "PA",
        "hospital_name": f"HOSPITAL {pk}",
        "city": "PITTSBURGH",
        "hospital_subtype": "Short Term",
        "is_metro_micro": "true",
        "is_corrected": "false",
        "zip": "01234",
    }
    for stem in MEASURES.values():
        r[f"{stem}_7_day_avg"], r[f"{stem}_7_day_sum"], r[f"{stem}_7_day_coverage"] = 10.0, 70, 7
    r["inpatient_beds_7_day_avg"], r["inpatient_beds_7_day_sum"] = 60.0, 420
    r.update(overrides)
    return r


def target(avg, total, cov) -> dict:
    return {
        "inpatient_beds_used_7_day_avg": avg,
        "inpatient_beds_used_7_day_sum": total,
        "inpatient_beds_used_7_day_coverage": cov,
    }


ROWS = [
    row("010001", week(0), **target(50.0, 350, 7)),  # measured
    row("010001", week(1), **target(S, 14, 7)),  # hidden average, sum known -> derived 2.0
    # week(2) not reported -> inserted as a gap row
    row("010001", week(3), **target(S, S, 7)),  # sum hidden too -> suppressed, filled 2/7
    row("020002", week(-1), **target(30.0, 210, 7)),  # before the training window
    row("020002", week(0), **target(70.0, 490, 7)),  # more beds used than capacity (60)
    row("020002", week(1), **target("", "", 0)),  # not reported -> missing
    row("020002", week(2), **target(12.0, 60, 5)),  # only 5 of 7 days reported
]


def write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.fixture
def cleaned(tmp_path):
    csv_path = write_csv(tmp_path / "raw.csv", ROWS)
    ingest(csv_path, tmp_path / "raw.parquet")
    counts = clean(tmp_path / "raw.parquet", tmp_path / "clean")
    df = duckdb.sql(
        f"SELECT * FROM '{tmp_path}/clean/beds_weekly.parquet' ORDER BY hospital_pk, week"
    )
    rows = {(r["hospital_pk"], str(r["week"])): r for r in _records(df)}
    return rows, counts, tmp_path


def _records(rel) -> list[dict]:
    cols = rel.columns
    return [dict(zip(cols, t, strict=True)) for t in rel.fetchall()]


def test_ingest_keeps_leading_zeros_and_all_rows(tmp_path):
    csv_path = write_csv(tmp_path / "raw.csv", ROWS)
    assert ingest(csv_path, tmp_path / "raw.parquet") == len(ROWS)
    pks = {r[0] for r in duckdb.sql(f"SELECT hospital_pk FROM '{tmp_path}/raw.parquet'").fetchall()}
    assert pks == {"010001", "020002"}


def test_measured_value_is_kept(cleaned):
    rows, _, _ = cleaned
    r = rows[("010001", week(0))]
    assert (r["beds_used"], r["beds_used_filled"], r["beds_used_status"]) == (
        50.0,
        50.0,
        "measured",
    )


def test_hidden_average_is_recovered_from_sum(cleaned):
    rows, _, _ = cleaned
    r = rows[("010001", week(1))]
    assert r["beds_used_status"] == "derived"
    assert r["beds_used"] == pytest.approx(2.0)


def test_suppressed_value_is_filled_only_in_filled_column(cleaned):
    rows, _, _ = cleaned
    r = rows[("010001", week(3))]
    assert r["beds_used_status"] == "suppressed"
    assert r["beds_used"] is None
    assert r["beds_used_filled"] == pytest.approx(2 / 7)


def test_missing_week_is_inserted_into_grid(cleaned):
    rows, counts, _ = cleaned
    gap = rows[("010001", week(2))]
    assert gap["reported"] is False
    assert gap["beds_used_status"] == "not_reported"
    assert counts["grid"]["unreported_weeks_inserted"] == 1


def test_null_target_is_missing(cleaned):
    rows, _, _ = cleaned
    r = rows[("020002", week(1))]
    assert r["reported"] is True
    assert r["beds_used_status"] == "missing"
    assert r["beds_used"] is None and r["beds_used_filled"] is None


def test_flags(cleaned):
    rows, _, _ = cleaned
    assert rows[("020002", week(0))]["flag_over_capacity"] is True
    assert rows[("010001", week(0))]["flag_over_capacity"] is False
    assert rows[("020002", week(2))]["flag_low_coverage"] is True


def test_periods(cleaned):
    rows, _, _ = cleaned
    assert rows[("020002", week(-1))]["period"] == "pre"
    assert rows[("020002", week(0))]["period"] == "train"


def test_counts_waterfall_adds_up(cleaned):
    _, counts, tmp_path = cleaned
    steps = counts["waterfall"]
    assert steps[0]["rows"] == len(ROWS)
    assert sum(s["rows"] for s in steps[:-1]) == steps[-1]["rows"]
    # 7 rows: 1 before the window, 1 missing, 1 suppressed -> 4 usable (3 measured, 1 derived)
    assert steps[-1]["rows"] == 4
    assert counts["target_usable_in_window"] == {
        "measured": 3,
        "derived": 1,
        "filled_only_in_beds_used_filled": 1,
    }
    assert json.loads((tmp_path / "clean" / "counts.json").read_text()) == counts


def test_duplicate_report_is_rejected(tmp_path):
    csv_path = write_csv(tmp_path / "raw.csv", [ROWS[0], ROWS[0]])
    ingest(csv_path, tmp_path / "raw.parquet")
    with pytest.raises(DataValidationError, match="more than once"):
        clean(tmp_path / "raw.parquet", tmp_path / "clean")


def test_non_sunday_week_is_rejected(tmp_path):
    csv_path = write_csv(tmp_path / "raw.csv", [row("010001", "2020-08-03")])
    ingest(csv_path, tmp_path / "raw.parquet")
    with pytest.raises(DataValidationError, match="not a Sunday"):
        clean(tmp_path / "raw.parquet", tmp_path / "clean")


def test_hospital_holdout_is_20_percent_and_stable(tmp_path):
    rows = [row(f"{i:06d}", week(0)) for i in range(50)]
    ingest(write_csv(tmp_path / "raw.csv", rows), tmp_path / "raw.parquet")
    first = clean(tmp_path / "raw.parquet", tmp_path / "a")["splits"]["hospitals"]
    clean(tmp_path / "raw.parquet", tmp_path / "b")
    assert first == {"holdout": 10, "train": 40}
    q = "SELECT hospital_pk, hospital_split FROM '{}/hospitals.parquet' ORDER BY 1"
    a = duckdb.sql(q.format(tmp_path / "a")).fetchall()
    b = duckdb.sql(q.format(tmp_path / "b")).fetchall()
    assert a == b


def test_checksum_roundtrip(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n")
    write_sums(tmp_path, {"x.csv": sha256_of(f)})
    assert read_sums(tmp_path) == {"x.csv": sha256_of(f)}
    assert verify(tmp_path, "x.csv") == sha256_of(f)
    f.write_text("a,b\n1,3\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(tmp_path, "x.csv")
