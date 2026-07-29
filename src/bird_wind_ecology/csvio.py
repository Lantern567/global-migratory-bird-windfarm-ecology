"""Small CSV readers for the stable cross-repository contracts."""

from __future__ import annotations

import csv
from pathlib import Path

from .models import AEPOrientationPoint, EcologyCurvePoint


def read_aep_curve(path: str | Path) -> list[AEPOrientationPoint]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            AEPOrientationPoint(
                farm_id=row["farm_id"],
                theta_deg=float(row["theta_deg"]),
                aep_gwh=float(row["aep_gwh"]),
            )
            for row in csv.DictReader(handle)
        ]


def read_ecology_curve(path: str | Path) -> list[EcologyCurvePoint]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            EcologyCurvePoint(
                farm_id=row["farm_id"],
                theta_deg=float(row["theta_deg"]),
                risk_score=float(row["risk_score"]),
            )
            for row in csv.DictReader(handle)
        ]

