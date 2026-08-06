"""Typed pure-Python data models for UAV navigation."""

import math
from dataclasses import dataclass


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
