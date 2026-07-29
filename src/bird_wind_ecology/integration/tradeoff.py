"""Join independent AEP and ecology curves without calculating AEP."""

from __future__ import annotations

from collections.abc import Iterable

from ..models import AEPOrientationPoint, EcologyCurvePoint, TradeoffResult


def optimize_under_aep_budget(
    aep_points: Iterable[AEPOrientationPoint],
    ecology_points: Iterable[EcologyCurvePoint],
    budget_fraction: float,
) -> TradeoffResult:
    """Choose the minimum-risk orientation within an AEP-loss budget."""
    if not 0 <= budget_fraction < 1:
        raise ValueError("budget_fraction must be in [0, 1)")

    aep_rows = list(aep_points)
    ecology_rows = list(ecology_points)
    if not aep_rows or not ecology_rows:
        raise ValueError("both AEP and ecology curves are required")

    farm_ids = {row.farm_id for row in aep_rows} | {row.farm_id for row in ecology_rows}
    if len(farm_ids) != 1:
        raise ValueError("curves must contain exactly one shared farm_id")
    farm_id = next(iter(farm_ids))

    aep_by_theta = {round(row.theta_deg, 9): row.aep_gwh for row in aep_rows}
    risk_by_theta = {round(row.theta_deg, 9): row.risk_score for row in ecology_rows}
    common_angles = sorted(set(aep_by_theta) & set(risk_by_theta))
    if not common_angles:
        raise ValueError("AEP and ecology curves have no common orientation angles")

    theta_econ = max(common_angles, key=lambda theta: aep_by_theta[theta])
    aep_econ = aep_by_theta[theta_econ]
    threshold = (1.0 - budget_fraction) * aep_econ
    feasible = [theta for theta in common_angles if aep_by_theta[theta] >= threshold]
    theta_eco = min(feasible, key=lambda theta: (risk_by_theta[theta], -aep_by_theta[theta], theta))

    return TradeoffResult(
        farm_id=farm_id,
        budget_fraction=budget_fraction,
        theta_econ_deg=theta_econ,
        theta_eco_deg=theta_eco,
        aep_econ_gwh=aep_econ,
        aep_eco_gwh=aep_by_theta[theta_eco],
        risk_econ=risk_by_theta[theta_econ],
        risk_eco=risk_by_theta[theta_eco],
    )

