"""Stage 0 sample-universe and data-readiness rules."""

from __future__ import annotations


def classify_sample_level(
    *,
    has_windfarm_location: bool,
    has_ecological_overlap: bool,
    has_direction: bool,
    has_layout_and_aep_inputs: bool,
    has_validation_data: bool,
) -> str:
    """Classify a project into the highest defensible U0-U4 level.

    Missing tracking data never means low ecological risk. A project with
    ecological overlap but no direction data remains U1.
    """
    if not has_windfarm_location:
        return "excluded"
    if not has_ecological_overlap:
        return "U0"
    if not has_direction:
        return "U1"
    if not has_layout_and_aep_inputs:
        return "U2"
    if not has_validation_data:
        return "U3"
    return "U4"

