"""Bird-only direction analysis. No AEP or wake logic belongs here."""

from __future__ import annotations

from collections.abc import Iterable

from ..angles import directed_circular_mean_deg
from ..models import BirdDirectionSignature


def build_direction_signature(
    *,
    farm_id: str,
    receptor_id: str,
    season: str,
    bearings_deg: Iterable[float],
    observation_weights: Iterable[float] | None = None,
    flux: float,
    rotor_height_fraction: float,
    evidence_level: str,
    conservation_weight: float = 1.0,
    source: str = "",
) -> BirdDirectionSignature:
    """Aggregate directed bearings for one receptor, farm, and season."""
    bearings = list(bearings_deg)
    weights = None if observation_weights is None else list(observation_weights)
    direction, concentration = directed_circular_mean_deg(bearings, weights)
    return BirdDirectionSignature(
        farm_id=farm_id,
        receptor_id=receptor_id,
        season=season,
        direction_deg=direction,
        concentration=concentration,
        flux=flux,
        rotor_height_fraction=rotor_height_fraction,
        conservation_weight=conservation_weight,
        evidence_level=evidence_level,
        n_observations=len(bearings),
        source=source,
    )

