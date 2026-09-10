"""Pure termination geometry for live BC evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass


TERMINAL_REASONS = ("collision", "out_of_bounds", "success", "timeout")


@dataclass(frozen=True, slots=True)
class TerminationConfig:
    """Geometric and timing limits used only by the evaluation monitor."""

    goal_tolerance_m: float = 0.35
    timeout_s: float = 45.0
    uav_radius_m: float = 0.25
    collision_clearance_threshold_m: float = 0.0
    east_min_m: float = -5.0
    east_max_m: float = 5.0
    north_min_m: float = -2.0
    north_max_m: float = 7.0

    def __post_init__(self) -> None:
        """Reject non-finite and contradictory evaluation settings."""
        positive = ("goal_tolerance_m", "timeout_s", "uav_radius_m")
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in (
            "collision_clearance_threshold_m",
            "east_min_m",
            "east_max_m",
            "north_min_m",
            "north_max_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.east_min_m >= self.east_max_m:
            raise ValueError("east bounds are invalid")
        if self.north_min_m >= self.north_max_m:
            raise ValueError("north bounds are invalid")


def cylinder_clearance_m(
    north_m: float,
    east_m: float,
    obstacle_north_m: float,
    obstacle_east_m: float,
    obstacle_radius_m: float,
    uav_radius_m: float,
) -> float:
    """Return horizontal surface clearance between UAV and one cylinder."""
    values = (
        north_m,
        east_m,
        obstacle_north_m,
        obstacle_east_m,
        obstacle_radius_m,
        uav_radius_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("clearance inputs must be finite")
    if obstacle_radius_m <= 0.0 or uav_radius_m <= 0.0:
        raise ValueError("collision radii must be positive")
    return math.hypot(
        north_m - obstacle_north_m,
        east_m - obstacle_east_m,
    ) - obstacle_radius_m - uav_radius_m


def select_terminal_reason(
    *,
    goal_distance_m: float,
    minimum_clearance_m: float,
    bc_duration_s: float,
    north_m: float,
    east_m: float,
    config: TerminationConfig,
) -> str | None:
    """Apply deterministic collision/bounds/success/timeout ordering."""
    values = (
        goal_distance_m,
        minimum_clearance_m,
        bc_duration_s,
        north_m,
        east_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("termination inputs must be finite")
    if minimum_clearance_m <= config.collision_clearance_threshold_m:
        return "collision"
    if not (
        config.north_min_m <= north_m <= config.north_max_m
        and config.east_min_m <= east_m <= config.east_max_m
    ):
        return "out_of_bounds"
    if goal_distance_m <= config.goal_tolerance_m:
        return "success"
    if bc_duration_s >= config.timeout_s:
        return "timeout"
    return None
