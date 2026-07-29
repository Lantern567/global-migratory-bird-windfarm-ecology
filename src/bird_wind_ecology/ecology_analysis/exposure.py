"""Replaceable ecological response models for array orientation."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..angles import axis_distance_deg, normalize_axis_deg
from ..models import BirdDirectionSignature, EcologyCurvePoint


def parallel_corridor_proxy_curve(
    farm_id: str,
    theta_values: Iterable[float],
    signatures: Iterable[BirdDirectionSignature],
) -> list[EcologyCurvePoint]:
    """Create a transparent geometry-only exposure proxy.

    Risk increases as the row axis moves from parallel toward perpendicular
    to the bird heading. This is a screening hypothesis, not a validated
    collision model. Only signatures for the requested farm are used.
    """
    relevant = [signature for signature in signatures if signature.farm_id == farm_id]
    if not relevant:
        raise ValueError(f"no bird direction signatures found for farm {farm_id!r}")

    points: list[EcologyCurvePoint] = []
    for theta in theta_values:
        axis = normalize_axis_deg(float(theta))
        risk = 0.0
        for signature in relevant:
            distance = axis_distance_deg(axis, signature.direction_deg)
            directional_term = math.sin(math.radians(distance)) ** 2
            risk += (
                signature.conservation_weight
                * signature.flux
                * signature.rotor_height_fraction
                * signature.concentration
                * directional_term
            )
        points.append(EcologyCurvePoint(farm_id=farm_id, theta_deg=axis, risk_score=risk))
    return points

