"""Download the raw HHS file and check it against the committed SHA256SUMS.

The dataset stopped updating in May 2024, so its checksum is fixed. Everyone on the team, and
the grader, should end up with byte-identical raw data; `verify` proves it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

DATASET_ID = "anag-cw7u"
URL = f"https://healthdata.gov/api/views/{DATASET_ID}/rows.csv?accessType=DOWNLOAD"
FILENAME = "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility.csv"
SUMS_FILE = "SHA256SUMS"
_CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def read_sums(raw_dir: Path) -> dict[str, str]:
    """Parse a `sha256sum`-style file: '<hex>  <filename>' per line."""
    sums_path = raw_dir / SUMS_FILE
    if not sums_path.exists():
        return {}
    out = {}
    for line in sums_path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip().lstrip("*")] = digest
    return out


def write_sums(raw_dir: Path, sums: dict[str, str]) -> None:
    lines = [f"{digest}  {name}" for name, digest in sorted(sums.items())]
    (raw_dir / SUMS_FILE).write_text("\n".join(lines) + "\n")


def verify(raw_dir: Path, filename: str = FILENAME) -> str:
    """Return the file's checksum, or raise if it does not match SHA256SUMS."""
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `smartops download`")
    expected = read_sums(raw_dir).get(filename)
    actual = sha256_of(path)
    if expected is None:
        raise ValueError(f"{filename} is not listed in {raw_dir / SUMS_FILE}")
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {filename}:\n  expected {expected}\n  actual   {actual}\n"
            "The source may have changed. Do not overwrite SHA256SUMS without telling the team."
        )
    return actual


def download(raw_dir: Path, url: str = URL, filename: str = FILENAME, force: bool = False) -> Path:
    """Stream the file to disk, then verify it.

    If SHA256SUMS has no entry yet (first download ever), the new checksum is recorded.
    Writes to a .part file first so an interrupted download never looks complete.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / filename
    if dest.exists() and not force:
        print(f"{dest.name} already present; verifying")
        verify(raw_dir, filename)
        print("checksum OK")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    size = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for chunk in r.iter_bytes(_CHUNK):
                f.write(chunk)
                h.update(chunk)
                size += len(chunk)
                print(f"\r{size / 1e6:8.1f} MB", end="", flush=True)
    print()
    part.replace(dest)

    sums = read_sums(raw_dir)
    if filename not in sums:
        sums[filename] = h.hexdigest()
        write_sums(raw_dir, sums)
        print(f"recorded new checksum in {SUMS_FILE}")
    verify(raw_dir, filename)
    print(f"checksum OK: {dest} ({size / 1e6:.1f} MB)")
    return dest
