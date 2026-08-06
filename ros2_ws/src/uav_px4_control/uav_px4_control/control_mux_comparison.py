"""Render deterministic Offline Control-Source Arbitration Comparison."""

from uav_px4_control.control_mux_fixtures import run_control_mux_fixtures


def render_control_mux_comparison() -> str:
    """Return a Markdown table from all deterministic fixture observations."""
    lines = [
        "# Offline Control-Source Arbitration Comparison",
        "",
        "| Fixture | Requested | Sequence | Service | Switch HOLD (s) | "
        "Max age (s) | Max speed (m/s) | Max yaw (rad/s) | HOLD cycles | "
        "Transitions | Latched | Recovery | Expected | Observed |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for item in run_control_mux_fixtures():
        lines.append(
            f"| {item.fixture} | {item.requested_source} | "
            f"{' -> '.join(item.command_sequence)} | {item.service_result} | "
            f"{item.switch_hold_s:.3f} | "
            f"{item.maximum_candidate_age_s:.3f} | "
            f"{item.maximum_selected_speed_mps:.3f} | "
            f"{item.maximum_selected_yaw_rate_radps:.3f} | "
            f"{item.hold_cycles} | {item.transition_count} | "
            f"{'yes' if item.fault_latched else 'no'} | "
            f"{'explicit request' if item.recovery_required else 'n/a'} | "
            f"{item.expected_terminal} | {item.observed_terminal} |"
        )
    lines.extend((
        "",
        "These are deterministic offline ROS-independent arbitration "
        "measurements. They are not PX4 setpoint mapping, a hardware "
        "joystick, a NavRL policy/runtime/model, Isaac Sim, UAV dynamics, "
        "or flight validation.",
    ))
    return "\n".join(lines)


def main() -> int:
    """Print the reproducible fixture table."""
    print(render_control_mux_comparison())
    return 0


if __name__ == "__main__":
    main()
