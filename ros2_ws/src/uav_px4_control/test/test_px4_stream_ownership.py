"""Static ownership and forbidden-output tests for Phase 8 runtime code."""

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "uav_px4_control"
STREAMER = PACKAGE / "px4_setpoint_streamer_node.py"
ALLOWED = {
    "/fmu/in/trajectory_setpoint",
    "/fmu/in/offboard_control_mode",
}


def runtime_sources():
    """Return tracked-style runtime Python sources, excluding tests."""
    return sorted(PACKAGE.glob("*.py"))


def test_exact_live_topic_allowlist_has_one_owner():
    """Only the streamer may create either allowed live input publisher."""
    owners = {topic: [] for topic in ALLOWED}
    discovered = set()
    for source in runtime_sources():
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
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_publisher"
            and len(node.args) >= 2
        ]
        for call in calls:
            topic = None
            if isinstance(call.args[1], ast.Constant):
                topic = call.args[1].value
            elif isinstance(call.args[1], ast.Name):
                topic = constants.get(call.args[1].id)
            if isinstance(topic, str) and topic.startswith("/fmu/in/"):
                discovered.add(topic)
                if topic in owners:
                    owners[topic].append(source.name)
    assert discovered == ALLOWED
    assert owners == {
        topic: [STREAMER.name]
        for topic in ALLOWED
    }


def test_no_forbidden_command_type_or_constant_in_runtime_ast():
    """Prohibit active PX4 command, arming, mode, takeoff, or land symbols."""
    forbidden = {
        "VehicleCommand",
        "VEHICLE_CMD_DO_SET_MODE",
        "VEHICLE_CMD_COMPONENT_ARM_DISARM",
        "VEHICLE_CMD_NAV_TAKEOFF",
        "VEHICLE_CMD_NAV_LAND",
    }
    names = set()
    for source in runtime_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names.update(
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        )
        names.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )
    assert names.isdisjoint(forbidden)


def test_only_streamer_creates_publishers_for_live_inputs():
    """Require both create_publisher calls to reside in the sole owner node."""
    tree = ast.parse(STREAMER.read_text(encoding="utf-8"))
    publisher_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_publisher"
    ]
    argument_names = {
        node.args[1].id
        for node in publisher_calls
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Name)
    }
    assert "TRAJECTORY_SETPOINT_TOPIC" in argument_names
    assert "OFFBOARD_CONTROL_MODE_TOPIC" in argument_names
