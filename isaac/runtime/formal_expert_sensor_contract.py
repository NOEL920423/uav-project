"""Dependency-free image contract for formal and legacy expert sensors."""

from __future__ import annotations


FPV_RGB_WIDTH = 320
FPV_RGB_HEIGHT = 180

# Formal FPV and TOP RGB are intentionally published on the same runtime tick.
FORMAL_RGB_PUBLISH_PERIOD_S = 0.20
TOP_RGB_WIDTH = 640
TOP_RGB_HEIGHT = 360
TOP_RGB_PUBLISH_PERIOD_S = FORMAL_RGB_PUBLISH_PERIOD_S
TOP_RGB_MODE = "fixed_global_top"
TOP_RGB_ALIGNMENT_TOLERANCE_S = 0.001

# 上視圖大小，採樣頻率更改
LEGACY_OBSERVER_RGB_WIDTH = 320
LEGACY_OBSERVER_RGB_HEIGHT = 180
LEGACY_OBSERVER_RGB_PUBLISH_PERIOD_S = 0.50

SENSOR_RATE_RELATIVE_TOLERANCE = 0.40
LEGACY_SENSOR_RATE_RELATIVE_TOLERANCE = 0.50


def nominal_rate_hz(publish_period_s: float) -> float:
    """Return the nominal frequency represented by a positive period."""
    period = float(publish_period_s)
    if period <= 0.0:
        raise ValueError("publish period must be positive")
    return 1.0 / period


def expected_rate_range_hz(
    publish_period_s: float,
    relative_tolerance: float = SENSOR_RATE_RELATIVE_TOLERANCE,
) -> tuple[float, float]:
    """Return the shared validator range around one sensor cadence."""
    nominal = nominal_rate_hz(publish_period_s)
    tolerance = float(relative_tolerance)
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("relative rate tolerance must be in [0, 1)")
    margin = nominal * tolerance
    return nominal - margin, nominal + margin


FORMAL_RGB_NOMINAL_RATE_HZ = nominal_rate_hz(FORMAL_RGB_PUBLISH_PERIOD_S)
FORMAL_RGB_EXPECTED_RATE_RANGE_HZ = expected_rate_range_hz(
    FORMAL_RGB_PUBLISH_PERIOD_S
)
TOP_RGB_NOMINAL_RATE_HZ = FORMAL_RGB_NOMINAL_RATE_HZ
TOP_RGB_EXPECTED_RATE_RANGE_HZ = FORMAL_RGB_EXPECTED_RATE_RANGE_HZ
LEGACY_OBSERVER_RGB_NOMINAL_RATE_HZ = nominal_rate_hz(
    LEGACY_OBSERVER_RGB_PUBLISH_PERIOD_S
)
LEGACY_OBSERVER_RGB_EXPECTED_RATE_RANGE_HZ = expected_rate_range_hz(
    LEGACY_OBSERVER_RGB_PUBLISH_PERIOD_S,
    LEGACY_SENSOR_RATE_RELATIVE_TOLERANCE,
)
