import pytest

from bird_wind_ecology.angles import axis_distance_deg, directed_circular_mean_deg


def test_axis_distance_is_180_degree_periodic():
    assert axis_distance_deg(10, 190) == pytest.approx(0)
    assert axis_distance_deg(10, 100) == pytest.approx(90)
    assert axis_distance_deg(170, 10) == pytest.approx(20)


def test_directed_mean_wraps_across_north():
    mean, concentration = directed_circular_mean_deg([350, 10])
    assert mean == pytest.approx(0)
    assert concentration > 0.98


def test_opposing_directed_sample_is_not_silently_aggregated():
    with pytest.raises(ValueError, match="undefined"):
        directed_circular_mean_deg([0, 180])

