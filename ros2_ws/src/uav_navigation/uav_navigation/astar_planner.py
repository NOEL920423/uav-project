"""Canonical deterministic 8-connected A* planner for fixed-altitude paths."""

import heapq
import itertools
import math
from dataclasses import dataclass

from uav_navigation.geometry import distance_2d, point_to_segment_distance_2d
from uav_navigation.models import (
    CircularObstacle,
    PlannerConfig,
    PlannerResult,
    Point3D,
)
from uav_navigation.path_metrics import calculate_path_metrics
from uav_navigation.path_simplifier import simplify_with_fallback
from uav_navigation.path_validator import (
    filter_overflyable_obstacles,
    planning_radius,
    validate_path,
    validation_radius,
)

GridCell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class _Grid:
    """A finite occupancy grid with deterministic coordinate conversion."""

    xmin: float
    ymin: float
    resolution: float
    width: int
    height: int
    occupied: frozenset[GridCell]

    def contains(self, cell: GridCell) -> bool:
        """Return whether a cell index lies in the grid."""
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def point_to_cell(self, point: Point3D) -> GridCell:
        """Round a continuous point to its nearest grid cell."""
        return (
            int(round((point.x - self.xmin) / self.resolution)),
            int(round((point.y - self.ymin) / self.resolution)),
        )

    def cell_to_point(self, cell: GridCell, altitude: float) -> Point3D:
        """Return the continuous cell-center position."""
        return Point3D(
            self.xmin + cell[0] * self.resolution,
            self.ymin + cell[1] * self.resolution,
            altitude,
        )


def _failed(status: str, diagnostics: dict[str, object]) -> PlannerResult:
    """Build an immutable structured failure result."""
    pairs = tuple(
        (key, str(value))
        for key, value in sorted(diagnostics.items())
    )
    return PlannerResult(False, status, diagnostics=pairs)


def _point_is_valid(
    point: Point3D,
    obstacles: tuple[CircularObstacle, ...],
    config: PlannerConfig,
) -> tuple[bool, str]:
    """Check altitude, bounds, and the validation envelope for one endpoint."""
    tolerance = config.numerical_tolerance
    if not math.isclose(
        point.z,
        config.planning_altitude_ned_m,
        abs_tol=tolerance,
        rel_tol=0.0,
    ):
        return False, "endpoint does not use configured fixed NED altitude"
    if config.planning_bounds is not None:
        xmin, xmax, ymin, ymax = config.planning_bounds
        if not (
            xmin - tolerance <= point.x <= xmax + tolerance
            and ymin - tolerance <= point.y <= ymax + tolerance
        ):
            return False, "endpoint lies outside planning bounds"
    for obstacle in obstacles:
        clearance = distance_2d(point, obstacle.center) - validation_radius(
            obstacle,
            config,
        )
        if clearance < -tolerance:
            return False, f"endpoint intersects obstacle {obstacle.name}"
    return True, "valid"


def _grid_bounds(
    start: Point3D,
    goal: Point3D,
    obstacles: tuple[CircularObstacle, ...],
    config: PlannerConfig,
) -> tuple[float, float, float, float]:
    """Return explicit bounds or deterministic dynamic bounds."""
    if config.planning_bounds is not None:
        return config.planning_bounds
    xs = [start.x, goal.x]
    ys = [start.y, goal.y]
    for obstacle in obstacles:
        envelope = planning_radius(obstacle, config)
        xs.extend((obstacle.center.x - envelope, obstacle.center.x + envelope))
        ys.extend((obstacle.center.y - envelope, obstacle.center.y + envelope))
    return (
        min(xs) - config.grid_margin_m,
        max(xs) + config.grid_margin_m,
        min(ys) - config.grid_margin_m,
        max(ys) + config.grid_margin_m,
    )


def _build_grid(
    start: Point3D,
    goal: Point3D,
    obstacles: tuple[CircularObstacle, ...],
    config: PlannerConfig,
    extra_inflation: float,
) -> _Grid:
    """Rasterize conservative occupancy while preserving physical formulas."""
    xmin, xmax, ymin, ymax = _grid_bounds(start, goal, obstacles, config)
    resolution = config.grid_resolution_m
    width = int(math.ceil((xmax - xmin) / resolution)) + 1
    height = int(math.ceil((ymax - ymin) / resolution)) + 1
    if width * height > config.maximum_grid_cells:
        raise ValueError(
            f"grid has {width * height} cells, exceeding maximum_grid_cells"
        )
    reserve = 0.5 * math.sqrt(2.0) * resolution
    occupied: set[GridCell] = set()
    for obstacle in sorted(
        obstacles,
        key=lambda item: (
            item.name,
            item.center.x,
            item.center.y,
            item.radius,
        ),
    ):
        radius = planning_radius(obstacle, config) + reserve + extra_inflation
        min_ix = max(
            0,
            int(math.floor((obstacle.center.x - radius - xmin) / resolution)),
        )
        max_ix = min(
            width - 1,
            int(math.ceil((obstacle.center.x + radius - xmin) / resolution)),
        )
        min_iy = max(
            0,
            int(math.floor((obstacle.center.y - radius - ymin) / resolution)),
        )
        max_iy = min(
            height - 1,
            int(math.ceil((obstacle.center.y + radius - ymin) / resolution)),
        )
        for ix in range(min_ix, max_ix + 1):
            x = xmin + ix * resolution
            for iy in range(min_iy, max_iy + 1):
                y = ymin + iy * resolution
                distance = math.hypot(
                    x - obstacle.center.x,
                    y - obstacle.center.y,
                )
                if distance <= radius:
                    occupied.add((ix, iy))
    return _Grid(xmin, ymin, resolution, width, height, frozenset(occupied))


def _nearest_free_cell(
    grid: _Grid,
    requested: GridCell,
    search_radius_m: float,
) -> GridCell | None:
    """Return the closest deterministic free cell within a bounded radius."""
    if grid.contains(requested) and requested not in grid.occupied:
        return requested
    limit = int(math.ceil(search_radius_m / grid.resolution))
    candidates = []
    for dx in range(-limit, limit + 1):
        for dy in range(-limit, limit + 1):
            squared = dx * dx + dy * dy
            if squared > limit * limit:
                continue
            cell = requested[0] + dx, requested[1] + dy
            if grid.contains(cell) and cell not in grid.occupied:
                candidates.append((squared, cell[0], cell[1], cell))
    return min(candidates)[-1] if candidates else None


def _clearance_penalty(
    point: Point3D,
    obstacles: tuple[CircularObstacle, ...],
    config: PlannerConfig,
) -> float:
    """Return the canonical soft planning-envelope proximity cost."""
    if not obstacles or not config.use_clearance_aware_cost:
        return 0.0
    clearance = min(
        distance_2d(point, obstacle.center) - planning_radius(obstacle, config)
        for obstacle in obstacles
    )
    if clearance >= config.soft_clearance_radius_m:
        return 0.0
    normalized = (
        config.soft_clearance_radius_m - max(0.0, clearance)
    ) / max(config.soft_clearance_radius_m, config.numerical_tolerance)
    return config.clearance_cost_weight * normalized


def _search(
    grid: _Grid,
    start_cell: GridCell,
    goal_cell: GridCell,
    start: Point3D,
    goal: Point3D,
    obstacles: tuple[CircularObstacle, ...],
    config: PlannerConfig,
) -> tuple[GridCell, ...]:
    """Run deterministic 8-connected Euclidean A*."""
    moves = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    counter = itertools.count()

    def heuristic(cell: GridCell) -> float:
        return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])

    frontier = [
        (
            heuristic(start_cell),
            heuristic(start_cell),
            next(counter),
            start_cell,
        )
    ]
    costs = {start_cell: 0.0}
    parents: dict[GridCell, GridCell] = {}
    closed: set[GridCell] = set()
    while frontier:
        _, _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        if current == goal_cell:
            cells = [current]
            while current in parents:
                current = parents[current]
                cells.append(current)
            cells.reverse()
            return tuple(cells)
        closed.add(current)
        for dx, dy in moves:
            neighbor = current[0] + dx, current[1] + dy
            if (
                not grid.contains(neighbor)
                or neighbor in grid.occupied
                or neighbor in closed
            ):
                continue
            step_cost = math.hypot(dx, dy)
            point = grid.cell_to_point(
                neighbor,
                config.planning_altitude_ned_m,
            )
            penalty = _clearance_penalty(point, obstacles, config)
            if config.use_direct_path_bias:
                direct_distance = point_to_segment_distance_2d(
                    point,
                    start,
                    goal,
                )
                penalty += config.direct_path_bias_weight * direct_distance
            candidate_cost = costs[current] + step_cost + penalty
            old_cost = costs.get(neighbor, math.inf)
            if candidate_cost + config.numerical_tolerance < old_cost:
                costs[neighbor] = candidate_cost
                parents[neighbor] = current
                estimate = heuristic(neighbor)
                heapq.heappush(
                    frontier,
                    (
                        candidate_cost + estimate,
                        estimate,
                        next(counter),
                        neighbor,
                    ),
                )
    return ()


def _restore_exact_endpoints(
    cells: tuple[GridCell, ...],
    grid: _Grid,
    start: Point3D,
    goal: Point3D,
) -> tuple[Point3D, ...]:
    """Convert cells to points and restore the exact requested endpoints."""
    points = [
        grid.cell_to_point(cell, start.z)
        for cell in cells
    ]
    if not points:
        return ()
    if len(points) == 1 and not start.almost_equals(goal):
        return start, goal
    points[0] = start
    points[-1] = goal
    deduplicated = [points[0]]
    for point in points[1:]:
        if not point.almost_equals(deduplicated[-1]):
            deduplicated.append(point)
    return tuple(deduplicated)


def plan_path(
    start: Point3D,
    goal: Point3D,
    obstacles: tuple[CircularObstacle, ...],
    config: PlannerConfig | None = None,
) -> PlannerResult:
    """Plan, continuously validate, simplify, and measure a safe path."""
    config = config or PlannerConfig()
    obstacles = tuple(obstacles)
    if start.almost_equals(goal, config.numerical_tolerance):
        return _failed("invalid request: start and goal are identical", {})
    active = filter_overflyable_obstacles(obstacles, config)
    start_valid, start_reason = _point_is_valid(start, active, config)
    if not start_valid:
        return _failed(
            f"invalid start: {start_reason}",
            {"active_obstacles": len(active)},
        )
    goal_valid, goal_reason = _point_is_valid(goal, active, config)
    if not goal_valid:
        return _failed(
            f"invalid goal: {goal_reason}",
            {"active_obstacles": len(active)},
        )

    attempts = (0.0, config.retry_extra_inflation_m)
    last_reason = "A* found no path"
    raw_path: tuple[Point3D, ...] = ()
    used_extra = 0.0
    for extra_inflation in attempts:
        try:
            grid = _build_grid(start, goal, active, config, extra_inflation)
        except ValueError as error:
            return _failed(
                f"invalid grid: {error}",
                {"active_obstacles": len(active)},
            )
        requested_start = grid.point_to_cell(start)
        requested_goal = grid.point_to_cell(goal)
        start_cell = _nearest_free_cell(
            grid,
            requested_start,
            config.endpoint_search_radius_m,
        )
        goal_cell = _nearest_free_cell(
            grid,
            requested_goal,
            config.endpoint_search_radius_m,
        )
        if start_cell is None or goal_cell is None:
            last_reason = "no free grid cell near start or goal"
            continue
        cells = _search(
            grid,
            start_cell,
            goal_cell,
            start,
            goal,
            active,
            config,
        )
        if not cells:
            last_reason = "A* found no path"
            continue
        candidate = _restore_exact_endpoints(cells, grid, start, goal)
        validation = validate_path(
            candidate,
            active,
            config,
            expected_start=start,
            expected_goal=goal,
        )
        if validation.valid:
            raw_path = candidate
            used_extra = extra_inflation
            break
        last_reason = (
            "raw path failed continuous validation: "
            f"{validation.reason}"
        )
    if not raw_path:
        return _failed(
            f"no path: {last_reason}",
            {
                "active_obstacles": len(active),
                "input_obstacles": len(obstacles),
                "attempts": len(attempts),
            },
        )

    simplified = simplify_with_fallback(raw_path, active, config)
    if not simplified.validation.valid:
        return _failed(
            f"no safe simplified path: {simplified.fallback_reason}",
            {"raw_points": len(raw_path)},
        )
    final_path = simplified.path
    raw_metrics = calculate_path_metrics(raw_path, active)
    final_metrics = calculate_path_metrics(final_path, active)
    diagnostics = {
        "active_obstacles": len(active),
        "input_obstacles": len(obstacles),
        "grid_retry_extra_inflation_m": used_extra,
        "raw_validation": "safe",
        "final_validation": simplified.validation.reason,
    }
    return PlannerResult(
        True,
        "success",
        raw_path=raw_path,
        simplified_path=final_path,
        final_path=final_path,
        simplification_method=simplified.method,
        fallback_reason=simplified.fallback_reason,
        raw_metrics=raw_metrics,
        simplified_metrics=final_metrics,
        final_metrics=final_metrics,
        diagnostics=tuple(
            (key, str(value)) for key, value in sorted(diagnostics.items())
        ),
    )
