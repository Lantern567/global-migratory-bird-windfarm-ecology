"""Circular and axial angle utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable


def normalize_direction_deg(angle: float) -> float:
    """Normalize a directed bearing to [0, 360)."""
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    normalized = angle % 360.0
    if math.isclose(normalized, 0.0, abs_tol=1e-12) or math.isclose(
        normalized, 360.0, abs_tol=1e-12
    ):
        return 0.0
    return normalized


def normalize_axis_deg(angle: float) -> float:
    """Normalize an undirected row axis to [0, 180)."""
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    normalized = angle % 180.0
    if math.isclose(normalized, 0.0, abs_tol=1e-12) or math.isclose(
        normalized, 180.0, abs_tol=1e-12
    ):
        return 0.0
    return normalized


def axis_distance_deg(axis_angle: float, direction_angle: float) -> float:
    """Return the acute distance between a row axis and a directed bearing.

    The output is in [0, 90]. A bird heading of 10 degrees and 190 degrees
    has the same relation to a row axis because the row is axial.
    """
    axis = normalize_axis_deg(axis_angle)
    direction_axis = normalize_axis_deg(direction_angle)
    delta = abs(axis - direction_axis)
    return min(delta, 180.0 - delta)


def directed_circular_mean_deg(
    angles: Iterable[float], weights: Iterable[float] | None = None
) -> tuple[float, float]:
    """Return directed circular mean and mean resultant concentration.

    The concentration is in [0, 1]. Callers must keep biologically distinct
    seasons separate; combining opposing spring and autumn headings can erase
    a real migration axis.
    """
    values = [normalize_direction_deg(float(value)) for value in angles]
    if not values:
        raise ValueError("at least one angle is required")

    if weights is None:
        weight_values = [1.0] * len(values)
    else:
        weight_values = [float(value) for value in weights]
        if len(weight_values) != len(values):
            raise ValueError("weights must have the same length as angles")
        if any((not math.isfinite(value)) or value < 0 for value in weight_values):
            raise ValueError("weights must be finite and non-negative")

    total_weight = sum(weight_values)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")

    x = sum(weight * math.cos(math.radians(angle)) for angle, weight in zip(values, weight_values))
    y = sum(weight * math.sin(math.radians(angle)) for angle, weight in zip(values, weight_values))
    concentration = math.hypot(x, y) / total_weight
    if concentration < 1e-12:
        raise ValueError("directed mean is undefined for a near-uniform or opposing sample")
    mean = normalize_direction_deg(math.degrees(math.atan2(y, x)))
    return mean, concentration
