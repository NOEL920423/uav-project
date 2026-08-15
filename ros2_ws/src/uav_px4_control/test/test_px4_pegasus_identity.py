"""Pure tests for the strict Pegasus PX4 process identity guard."""

from uav_px4_control.px4_setpoint_streamer_node import (
    pegasus_sitl_identity_matches,
)


EXPECTED = "/PX4-Autopilot/build/px4_sitl_default/bin/px4"
EXECUTABLE = "/home/test/PX4-Autopilot" + EXPECTED.split(
    "/PX4-Autopilot", 1
)[1]
RC_SCRIPT = "/home/test/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/rcS"
COMMAND = (
    f"{EXECUTABLE} /home/test/PX4-Autopilot/ROMFS/px4fmu_common/ "
    f"-s {RC_SCRIPT} -i 0 -d "
)
ENVIRONMENT = {b"PX4_SIM_MODEL=gazebo-classic_iris"}


def test_accepts_exact_pegasus_instance_zero_launch_shape():
    """Binary, absolute rcS, instance zero, daemon flag, and model agree."""
    assert pegasus_sitl_identity_matches(
        EXPECTED, EXECUTABLE, COMMAND, ENVIRONMENT
    )


def test_rejects_wrong_binary_or_model_or_instance():
    """Adjacent PX4-like processes cannot satisfy the flight guard."""
    assert not pegasus_sitl_identity_matches(
        EXPECTED, "/tmp/px4", COMMAND, ENVIRONMENT
    )
    assert not pegasus_sitl_identity_matches(
        EXPECTED, EXECUTABLE, COMMAND, {b"PX4_SIM_MODEL=sihsim_quadx"}
    )
    assert not pegasus_sitl_identity_matches(
        EXPECTED, EXECUTABLE, COMMAND.replace("-i 0", "-i 1"), ENVIRONMENT
    )


def test_rejects_relative_or_unexpected_rc_script():
    """The guard requires the rcS belonging to the exact PX4 repository."""
    assert not pegasus_sitl_identity_matches(
        EXPECTED,
        EXECUTABLE,
        COMMAND.replace(RC_SCRIPT, "etc/init.d-posix/rcS"),
        ENVIRONMENT,
    )
