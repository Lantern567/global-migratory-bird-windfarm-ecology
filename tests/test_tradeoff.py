import pytest

from bird_wind_ecology.integration import optimize_under_aep_budget
from bird_wind_ecology.models import AEPOrientationPoint, EcologyCurvePoint


def test_tradeoff_reads_aep_curve_and_respects_budget():
    aep = [
        AEPOrientationPoint("f1", 0, 100.0),
        AEPOrientationPoint("f1", 30, 99.5),
        AEPOrientationPoint("f1", 60, 99.1),
        AEPOrientationPoint("f1", 90, 97.0),
    ]
    ecology = [
        EcologyCurvePoint("f1", 0, 80),
        EcologyCurvePoint("f1", 30, 40),
        EcologyCurvePoint("f1", 60, 10),
        EcologyCurvePoint("f1", 90, 5),
    ]
    result = optimize_under_aep_budget(aep, ecology, budget_fraction=0.01)
    assert result.theta_econ_deg == 0
    assert result.theta_eco_deg == 60
    assert result.aep_cost_gwh == pytest.approx(0.9)
    assert result.relative_risk_reduction == pytest.approx(0.875)


def test_tradeoff_rejects_mixed_farms():
    with pytest.raises(ValueError, match="one shared farm_id"):
        optimize_under_aep_budget(
            [AEPOrientationPoint("f1", 0, 100)],
            [EcologyCurvePoint("f2", 0, 10)],
            budget_fraction=0.01,
        )
