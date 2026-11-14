"""Project paths and global settings.

Everything that touches the filesystem resolves paths through here, so notebooks,
the CLI, tests, and the dashboard all agree on where data lives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data: Path = field(
        default_factory=lambda: Path(os.getenv("SMARTOPS_DATA_DIR", PROJECT_ROOT / "data"))
    )

    @property
    def raw(self) -> Path:
        """Untouched downloads. Never edited by hand."""
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        """Cleaned / typed data, one file per source."""
        return self.data / "interim"

    @property
    def clean(self) -> Path:
        """Cleaned, validated tables the model trains on. Shipped in the zip as data/clean/."""
        return self.data / "clean"

    @property
    def processed(self) -> Path:
        """Model-ready tables (features, census series)."""
        return self.data / "processed"

    def ensure(self) -> None:
        for p in (self.raw, self.interim, self.clean, self.processed):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

# One seed for every stochastic component (splits, model initialization)
# so results in the README are reproducible.
RANDOM_SEED = 42

# Evaluation design from the proposal. Weeks are HHS collection weeks (Sundays).
# Periods are inclusive of their start date; each ends the day before the next starts.
TRAIN_START = date(2020, 8, 2)
VAL_START = date(2023, 1, 1)
TEST_START = date(2023, 7, 2)
DATA_END = date(2024, 4, 21)

# Share of hospitals held out entirely (never seen in training), chosen by hash, not at random,
# so the same hospitals are held out on every machine.
HOLDOUT_FRACTION = 0.20
