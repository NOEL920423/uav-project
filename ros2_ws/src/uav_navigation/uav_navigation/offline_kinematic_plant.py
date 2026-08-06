"""Deterministic fixed-step kinematic plant for offline tracking tests."""

import math
from dataclasses import dataclass

from uav_navigation.models import Point3D
from uav_navigation.tracking_models import (
    VALID_TRACKING_FRAME,
    VehicleState,
    VelocityCommand,
)


@dataclass(frozen=True, slots=True)
class KinematicPlantConfig:
    """First-order response and deterministic fixture configuration."""

    integration_timestep_s: float = 0.02
    velocity_time_constant_s: float = 0.20
    yaw_rate_time_constant_s: float = 0.15
    maximum_simulated_acceleration_mps2: float = 2.0
    maximum_simulated_yaw_acceleration_radps2: float = 3.0
    initial_position: Point3D = Point3D(0.0, 0.0, -2.0)
    initial_velocity: Point3D = Point3D(0.0, 0.0, 0.0)
    initial_yaw_ned: float = 0.0
    initial_yaw_rate_radps: float = 0.0
    disturbance_velocity: Point3D = Point3D(0.0, 0.0, 0.0)
    measurement_noise_amplitude_m: float = 0.0

    def __post_init__(self) -> None:
        """Reject malformed kinematic fixture values."""
        positive = (
            "integration_timestep_s",
            "velocity_time_constant_s",
            "yaw_rate_time_constant_s",
            "maximum_simulated_acceleration_mps2",
            "maximum_simulated_yaw_acceleration_radps2",
        )
        nonnegative = ("measurement_noise_amplitude_m",)
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        vectors = (self.initial_position, self.initial_velocity,
                   self.disturbance_velocity)
        scalars = (self.initial_yaw_ned, self.initial_yaw_rate_radps)
        if not all(math.isfinite(value) for vector in vectors
                   for value in (vector.x, vector.y, vector.z)):
            raise ValueError("plant vectors must be finite")
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("plant yaw state must be finite")


@dataclass(frozen=True, slots=True)
class KinematicPlantState:
    """Exact internal state of the offline kinematic fixture."""

    time_s: float
    position: Point3D
    velocity: Point3D
    yaw_ned: float
    yaw_rate_radps: float
    step_index: int = 0


def _norm(vector: Point3D) -> float:
    return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)


def _limit_vector(vector: Point3D, limit: float) -> Point3D:
    magnitude = _norm(vector)
    if magnitude <= limit:
        return vector
    scale = limit / magnitude
    return Point3D(vector.x * scale, vector.y * scale, vector.z * scale)


class OfflineKinematicPlant:
    """First-order kinematic response; explicitly not UAV dynamics."""

    def __init__(self, config: KinematicPlantConfig | None = None) -> None:
        """Initialize the plant exactly at its configured state."""
        self.config = config or KinematicPlantConfig()
        self.state = KinematicPlantState(
            time_s=0.0,
            position=self.config.initial_position,
            velocity=self.config.initial_velocity,
            yaw_ned=self.config.initial_yaw_ned,
            yaw_rate_radps=self.config.initial_yaw_rate_radps,
        )

    def step(self, command: VelocityCommand) -> KinematicPlantState:
        """Advance exactly one configured fixed integration step."""
        values = (
            command.linear.x, command.linear.y, command.linear.z,
            command.yaw_rate_radps,
        )
        if command.frame_id != VALID_TRACKING_FRAME:
            raise ValueError("plant command frame must be px4_ned")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("plant command must be finite")
        dt = self.config.integration_timestep_s
        desired_acceleration = Point3D(
            (command.linear.x - self.state.velocity.x)
            / self.config.velocity_time_constant_s,
            (command.linear.y - self.state.velocity.y)
            / self.config.velocity_time_constant_s,
            (command.linear.z - self.state.velocity.z)
            / self.config.velocity_time_constant_s,
        )
        acceleration = _limit_vector(
            desired_acceleration,
            self.config.maximum_simulated_acceleration_mps2,
        )
        velocity = Point3D(
            self.state.velocity.x + acceleration.x * dt,
            self.state.velocity.y + acceleration.y * dt,
            self.state.velocity.z + acceleration.z * dt,
        )
        desired_yaw_acceleration = (
            command.yaw_rate_radps - self.state.yaw_rate_radps
        ) / self.config.yaw_rate_time_constant_s
        yaw_limit = self.config.maximum_simulated_yaw_acceleration_radps2
        yaw_acceleration = max(
            -yaw_limit, min(yaw_limit, desired_yaw_acceleration)
        )
        yaw_rate = self.state.yaw_rate_radps + yaw_acceleration * dt
        disturbance = self.config.disturbance_velocity
        position = Point3D(
            self.state.position.x + (velocity.x + disturbance.x) * dt,
            self.state.position.y + (velocity.y + disturbance.y) * dt,
            self.state.position.z + (velocity.z + disturbance.z) * dt,
        )
        self.state = KinematicPlantState(
            time_s=self.state.time_s + dt,
            position=position,
            velocity=velocity,
            yaw_ned=self.state.yaw_ned + yaw_rate * dt,
            yaw_rate_radps=yaw_rate,
            step_index=self.state.step_index + 1,
        )
        return self.state

    def measurement(self, timestamp_s: float | None = None) -> VehicleState:
        """Return deterministic state, with optional analytic fixture noise."""
        amplitude = self.config.measurement_noise_amplitude_m
        index = self.state.step_index
        noise = Point3D(
            amplitude * math.sin(index * 0.7),
            amplitude * math.cos(index * 0.5),
            amplitude * math.sin(index * 0.3),
        )
        return VehicleState(
            timestamp_s=(
                self.state.time_s if timestamp_s is None else timestamp_s
            ),
            frame_id=VALID_TRACKING_FRAME,
            position=Point3D(
                self.state.position.x + noise.x,
                self.state.position.y + noise.y,
                self.state.position.z + noise.z,
            ),
            velocity=self.state.velocity,
            yaw_ned=self.state.yaw_ned,
            yaw_rate_radps=self.state.yaw_rate_radps,
        )
