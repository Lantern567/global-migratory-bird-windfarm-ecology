"""Validated in-memory records for repository data contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .angles import normalize_axis_deg, normalize_direction_deg


@dataclass(frozen=True)
class BirdDirectionSignature:
    farm_id: str
    receptor_id: str
    season: str
    direction_deg: float
    concentration: float
    flux: float
    rotor_height_fraction: float
    evidence_level: str
    conservation_weight: float = 1.0
    n_observations: int = 1
    source: str = ""

    def __post_init__(self) -> None:
        if not self.farm_id or not self.receptor_id or not self.season:
            raise ValueError("farm_id, receptor_id, and season are required")
        object.__setattr__(self, "direction_deg", normalize_direction_deg(self.direction_deg))
        if not 0 <= self.concentration <= 1:
            raise ValueError("concentration must be in [0, 1]")
        if self.flux < 0 or not math.isfinite(self.flux):
            raise ValueError("flux must be finite and non-negative")
        if not 0 <= self.rotor_height_fraction <= 1:
            raise ValueError("rotor_height_fraction must be in [0, 1]")
        if self.conservation_weight <= 0 or not math.isfinite(self.conservation_weight):
            raise ValueError("conservation_weight must be finite and positive")
        if self.n_observations < 1:
            raise ValueError("n_observations must be positive")


@dataclass(frozen=True)
class EcologyCurvePoint:
    farm_id: str
    theta_deg: float
    risk_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta_deg", normalize_axis_deg(self.theta_deg))
        if self.risk_score < 0 or not math.isfinite(self.risk_score):
            raise ValueError("risk_score must be finite and non-negative")


@dataclass(frozen=True)
class AEPOrientationPoint:
    """Read-only record produced by the engineering repository."""

    farm_id: str
    theta_deg: float
    aep_gwh: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta_deg", normalize_axis_deg(self.theta_deg))
        if self.aep_gwh < 0 or not math.isfinite(self.aep_gwh):
            raise ValueError("aep_gwh must be finite and non-negative")


@dataclass(frozen=True)
class TradeoffResult:
    farm_id: str
    budget_fraction: float
    theta_econ_deg: float
    theta_eco_deg: float
    aep_econ_gwh: float
    aep_eco_gwh: float
    risk_econ: float
    risk_eco: float

    @property
    def aep_cost_gwh(self) -> float:
        return self.aep_econ_gwh - self.aep_eco_gwh

    @property
    def relative_risk_reduction(self) -> float:
        if self.risk_econ == 0:
            return 0.0
        return 1.0 - self.risk_eco / self.risk_econ

