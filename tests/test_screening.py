from bird_wind_ecology.stage0_screening import classify_sample_level


def test_missing_bird_direction_stays_at_ecological_overlap_level():
    level = classify_sample_level(
        has_windfarm_location=True,
        has_ecological_overlap=True,
        has_direction=False,
        has_layout_and_aep_inputs=False,
        has_validation_data=False,
    )
    assert level == "U1"


def test_fully_observed_site_is_u4():
    level = classify_sample_level(
        has_windfarm_location=True,
        has_ecological_overlap=True,
        has_direction=True,
        has_layout_and_aep_inputs=True,
        has_validation_data=True,
    )
    assert level == "U4"

