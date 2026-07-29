import pytest

from bird_wind_ecology.bird_analysis import build_direction_signature
from bird_wind_ecology.ecology_analysis import parallel_corridor_proxy_curve


def test_bird_signature_and_parallel_proxy_are_independent_of_aep():
    signature = build_direction_signature(
        farm_id="f1",
        receptor_id="raptor",
        season="autumn",
        bearings_deg=[40, 45, 50],
        flux=100,
        rotor_height_fraction=0.25,
        evidence_level="local-track",
    )
    assert signature.direction_deg == pytest.approx(45)
    curve = parallel_corridor_proxy_curve("f1", [45, 90, 135], [signature])
    risk = {point.theta_deg: point.risk_score for point in curve}
    assert risk[45] == pytest.approx(0)
    assert risk[90] > risk[45]
    assert risk[135] > risk[90]

