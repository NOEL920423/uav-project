"""Typed pure-Python data models for UAV navigation."""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class Point3D:
    """A finite three-dimensional point or free vector in metres."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        """Normalize numeric inputs and reject non-finite coordinates."""
        for field_name in ("x", "y", "z"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)

    def xy(self) -> tuple[float, float]:
        """Return the horizontal components."""
        return self.x, self.y

    def almost_equals(self, other: "Point3D", tolerance: float = 1e-9) -> bool:
        """Return whether coordinates match within an absolute tolerance."""
        if tolerance < 0.0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be finite and nonnegative")
        return (
            math.isclose(self.x, other.x, abs_tol=tolerance, rel_tol=0.0)
            and math.isclose(self.y, other.y, abs_tol=tolerance, rel_tol=0.0)
            and math.isclose(self.z, other.z, abs_tol=tolerance, rel_tol=0.0)
        )


@dataclass(frozen=True, slots=True)
class CircularObstacle:
    """A vertical circular obstacle represented in the planning frame."""

    name: str
    center: Point3D
    radius: float
    height: float

    def __post_init__(self) -> None:
        """Reject missing names and invalid geometry."""
        if not self.name.strip():
            raise ValueError("obstacle name must not be empty")
        for field_name in ("radius", "height"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"obstacle {field_name} must be finite and nonnegative"
                )
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Pure planner configuration in metres and radians."""

    grid_resolution_m: float = 0.05
    grid_margin_m: float = 2.0
    uav_physical_radius_m: float = 0.18
    static_safety_margin_m: float = 0.13
    minimum_segment_clearance_m: float = 0.07
    endpoint_search_radius_m: float = 1.0
    simplification_tolerance_m: float = 0.05
    maximum_waypoint_spacing_m: float = 1.30
    use_direct_path_bias: bool = True
    direct_path_bias_weight: float = 0.07
    use_clearance_aware_cost: bool = True
    soft_clearance_radius_m: float = 0.40
    clearance_cost_weight: float = 0.25
    flight_altitude_m: float = 2.0
    enable_overfly_short_obstacles: bool = True
    overfly_vertical_clearance_m: float = 0.35
    ned_origin_offset_z_m: float = 0.0
    retry_extra_inflation_m: float = 0.07
    maximum_grid_cells: int = 4_000_000
    numerical_tolerance: float = 1e-9
    planning_bounds: Optional[tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        """Validate planner invariants before any search is attempted."""
        positive = (
            "grid_resolution_m",
            "grid_margin_m",
            "endpoint_search_radius_m",
            "maximum_waypoint_spacing_m",
            "flight_altitude_m",
            "maximum_grid_cells",
            "numerical_tolerance",
        )
        nonnegative = (
            "uav_physical_radius_m",
            "static_safety_margin_m",
            "minimum_segment_clearance_m",
            "simplification_tolerance_m",
            "direct_path_bias_weight",
            "soft_clearance_radius_m",
            "clearance_cost_weight",
            "overfly_vertical_clearance_m",
            "retry_extra_inflation_m",
        )
        for field_name in positive:
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        for field_name in nonnegative:
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{field_name} must be finite and nonnegative"
                )
        if not math.isfinite(float(self.ned_origin_offset_z_m)):
            raise ValueError("ned_origin_offset_z_m must be finite")
        cell_limit = float(self.maximum_grid_cells)
        if not cell_limit.is_integer():
            raise ValueError("maximum_grid_cells must be an integer")
        if self.planning_bounds is not None:
            if len(self.planning_bounds) != 4:
                raise ValueError(
                    "planning_bounds must contain xmin, xmax, ymin, ymax"
                )
            xmin, xmax, ymin, ymax = map(float, self.planning_bounds)
            bounds_are_finite = all(
                math.isfinite(value) for value in (xmin, xmax, ymin, ymax)
            )
            if not bounds_are_finite:
                raise ValueError("planning_bounds values must be finite")
            if xmin >= xmax or ymin >= ymax:
                raise ValueError("planning_bounds minima must be below maxima")
            object.__setattr__(
                self,
                "planning_bounds",
                (xmin, xmax, ymin, ymax),
            )
        object.__setattr__(
            self,
            "maximum_grid_cells",
            int(cell_limit),
        )

    @property
    def planning_altitude_ned_m(self) -> float:
        """Return the fixed NED down coordinate used by this 2.5D planner."""
        return self.ned_origin_offset_z_m - self.flight_altitude_m


@dataclass(frozen=True, slots=True)
class BSplineConfig:
    """Pure geometric B-spline candidate and validation configuration."""

    enable_bspline: bool = True
    bspline_degree: int = 3
    bspline_sample_spacing_m: float = 0.08
    bspline_minimum_samples: int = 16
    bspline_maximum_samples: int = 1000
    bspline_maximum_curvature: float = 8.0
    bspline_minimum_clearance_m: float = 0.07
    bspline_preserve_endpoints: bool = True
    bspline_allowed_bounds_margin_m: float = 0.0
    bspline_reject_self_intersection: bool = True
    bspline_control_point_strategy: str = "validated_simplified_path"

    def __post_init__(self) -> None:
        """Reject unsafe or ambiguous Phase 3 configuration."""
        integer_fields = (
            "bspline_degree",
            "bspline_minimum_samples",
            "bspline_maximum_samples",
        )
        for field_name in integer_fields:
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError(f"{field_name} must be an integer")
            object.__setattr__(self, field_name, int(value))
        if self.bspline_degree < 1:
            raise ValueError("bspline_degree must be at least one")
        if self.bspline_minimum_samples < 2:
            raise ValueError("bspline_minimum_samples must be at least two")
        if self.bspline_maximum_samples < self.bspline_minimum_samples:
            raise ValueError(
                "bspline_maximum_samples must not be below the minimum"
            )
        positive = (
            "bspline_sample_spacing_m",
            "bspline_maximum_curvature",
        )
        nonnegative = (
            "bspline_minimum_clearance_m",
            "bspline_allowed_bounds_margin_m",
        )
        for field_name in positive:
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
            object.__setattr__(self, field_name, value)
        for field_name in nonnegative:
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{field_name} must be finite and nonnegative"
                )
            object.__setattr__(self, field_name, value)
        if not self.bspline_preserve_endpoints:
            raise ValueError(
                "bspline_preserve_endpoints must remain true in Phase 3"
            )
        if self.bspline_control_point_strategy != "validated_simplified_path":
            raise ValueError(
                "bspline_control_point_strategy must be "
                "validated_simplified_path"
            )


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Continuous geometry validation result."""

    valid: bool
    reason: str
    minimum_validation_clearance_m: float
    maximum_segment_length_m: float


@dataclass(frozen=True, slots=True)
class SimplificationResult:
    """First safe simplified path and deterministic selection reason."""

    path: tuple[Point3D, ...]
    method: str
    fallback_reason: str
    validation: ValidationResult


@dataclass(frozen=True, slots=True)
class PathMetrics:
    """Geometric path metrics; these are not vehicle-dynamics measurements."""

    point_count: int
    path_length_m: float
    minimum_physical_clearance_m: float
    mean_segment_length_m: float
    maximum_segment_length_m: float
    mean_absolute_heading_change_rad: float
    maximum_absolute_heading_change_rad: float
    heading_change_variance_rad2: float
    mean_curvature_inverse_m: float = 0.0
    maximum_curvature_inverse_m: float = 0.0
    curvature_variance_inverse_m2: float = 0.0


@dataclass(frozen=True, slots=True)
class BSplineResult:
    """Structured candidate-generation, validation, and selection result."""

    candidate_path: tuple[Point3D, ...] = ()
    valid: bool = False
    selected: bool = False
    status_message: str = "not attempted"
    rejection_reason: str = ""
    effective_degree: int = 0
    control_point_count: int = 0
    provisional_sample_count: int = 0
    final_sample_count: int = 0
    minimum_clearance_m: float = math.inf
    maximum_curvature_inverse_m: float = 0.0
    self_intersection: bool = False
    metrics: Optional[PathMetrics] = None


@dataclass(frozen=True, slots=True)
class PlannerResult:
    """Structured deterministic planner output."""

    success: bool
    status: str
    raw_path: tuple[Point3D, ...] = ()
    simplified_path: tuple[Point3D, ...] = ()
    final_path: tuple[Point3D, ...] = ()
    simplification_method: str = "none"
    fallback_reason: str = ""
    raw_metrics: Optional[PathMetrics] = None
    simplified_metrics: Optional[PathMetrics] = None
    final_metrics: Optional[PathMetrics] = None
    diagnostics: tuple[tuple[str, str], ...] = ()
    bspline_enabled: bool = False
    bspline_candidate: tuple[Point3D, ...] = ()
    bspline_valid: bool = False
    bspline_selected: bool = False
    bspline_rejection_reason: str = ""
    bspline_effective_degree: int = 0
    bspline_metrics: Optional[PathMetrics] = None
    final_path_source: str = "NONE"
