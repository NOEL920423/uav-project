"""Static single-owner contracts for the explicitly authorized SITL flight."""

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "uav_px4_control"
SUPERVISOR = PACKAGE / "px4_sitl_flight_supervisor_node.py"
FLIGHT_CONFIG = PACKAGE.parent / "config" / "px4_sitl_flight.yaml"


def _publisher_topics(source: Path) -> list[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    constants = {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    topics = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_publisher"
            and len(node.args) >= 2
        ):
            continue
        topic = None
        if isinstance(node.args[1], ast.Constant):
            topic = node.args[1].value
        elif isinstance(node.args[1], ast.Name):
            topic = constants.get(node.args[1].id)
        if isinstance(topic, str) and topic.startswith("/fmu/in/"):
            topics.append(topic)
    return topics


def test_vehicle_command_has_exactly_one_tracked_owner():
    """Only the explicitly authorized supervisor may publish commands."""
    owners = []
    for source in sorted(PACKAGE.glob("*.py")):
        if "/fmu/in/vehicle_command" in _publisher_topics(source):
            owners.append(source.name)
    assert owners == [SUPERVISOR.name]


def test_flight_supervisor_does_not_duplicate_setpoint_publishers():
    """The supervisor leaves both Phase 8 setpoint topics to the streamer."""
    assert _publisher_topics(SUPERVISOR) == ["/fmu/in/vehicle_command"]


def test_supervisor_accepts_existing_phase4_trajectory_provenance():
    """Flight readiness must match the reused parameterizer contract."""
    parameterizer = (
        PACKAGE.parents[1]
        / "uav_navigation"
        / "uav_navigation"
        / "trajectory_parameterizer_node.py"
    )
    marker = '"PHASE4_TIME_PARAMETERIZED"'
    assert marker in parameterizer.read_text(encoding="utf-8")
    assert marker in SUPERVISOR.read_text(encoding="utf-8")


def test_takeoff_acceptance_matches_follower_terminal_tolerance():
    """The follower cannot settle below the supervisor takeoff boundary."""
    config = FLIGHT_CONFIG.read_text(encoding="utf-8")
    assert "goal_position_tolerance_m: 0.25" in config
    assert "takeoff_altitude_tolerance_m: 0.25" in config
