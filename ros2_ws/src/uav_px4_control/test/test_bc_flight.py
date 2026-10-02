"""Pure regression tests for BC handoff and evaluation termination."""

from dataclasses import replace
import io
from types import MethodType, SimpleNamespace
import threading

import pytest

from uav_px4_control.flight.bc_episode_monitor import (
    TerminationConfig,
    select_terminal_reason,
)
from uav_px4_control.flight.bc_flight_models import (
    BcFlightController,
    BcFlightEvidence,
    BcFlightState,
)
from uav_px4_control.control.control_mux import ControlSourceMux, fixed_candidate
from uav_px4_control.control.control_source_models import BC_POLICY, ControlMuxConfig


def test_mux_selects_independent_bc_policy() -> None:
    """Select BC through its own source identity and topic contract."""
    mux = ControlSourceMux(ControlMuxConfig(switch_hold_duration_s=0.0))
    mux.accept_candidate(
        BC_POLICY,
        fixed_candidate(source=BC_POLICY, stamp_s=1.0),
        receipt_time_s=1.0,
    )
    response = mux.request_source(BC_POLICY, 1.0)
    assert response.accepted
    result = mux.step(1.01)
    assert result.active_source == BC_POLICY
    assert result.state.value == "ACTIVE_BC_POLICY"


def test_takeoff_hands_control_to_bc_without_astar_action() -> None:
    """Reach navigation using lifecycle and BC actions only."""
    controller = BcFlightController()
    decisions = []
    evidence = BcFlightEvidence(
        landed=True, recovery_vehicle_state_fresh=True,
        runtime_ready=True,
        observations_ready=True,
        telemetry_fresh=True,
    )
    decisions.append(controller.step(1.0, evidence))
    evidence = BcFlightEvidence(
        landed=True, recovery_vehicle_state_fresh=True,
        runtime_ready=True,
        observations_ready=True,
        telemetry_fresh=True,
        lifecycle_selected=True,
        source_valid=True,
    )
    decisions.append(controller.step(1.1, evidence))
    evidence = replace(evidence, output_ready=True)
    decisions.append(controller.step(1.15, evidence))
    assert decisions[-1].actions == ("ENABLE_OUTPUT",)
    evidence = replace(evidence, output_ready=False, output_safe=True)
    decisions.append(controller.step(1.2, evidence))
    evidence = replace(evidence, stream_stable=True)
    decisions.append(controller.step(1.3, evidence))
    evidence = replace(evidence, offboard_active=True)
    decisions.append(controller.step(1.4, evidence))
    evidence = replace(evidence, vehicle_armed=True, landed=False)
    decisions.append(controller.step(1.5, evidence))
    evidence = replace(evidence, altitude_m=1.5)
    decisions.append(controller.step(1.6, evidence))
    evidence = replace(evidence, bc_enabled=True, bc_ready=True)
    decisions.append(controller.step(1.7, evidence))
    evidence = replace(
        evidence, lifecycle_selected=False, bc_selected=True
    )
    decisions.append(controller.step(1.8, evidence))
    assert controller.state == BcFlightState.NAVIGATING
    actions = {action for item in decisions for action in item.actions}
    assert "SELECT_BC" in actions
    assert "SELECT_ASTAR" not in actions


def test_output_readiness_timeout_does_not_enable_gate() -> None:
    """Missing gate evidence must time out without attempting to enable."""
    controller = BcFlightController()
    evidence = BcFlightEvidence(
        landed=True, recovery_vehicle_state_fresh=True,
        runtime_ready=True,
        observations_ready=True,
        telemetry_fresh=True,
        lifecycle_selected=True,
        source_valid=True,
    )
    controller.step(1.0, evidence)
    assert controller.step(1.1, evidence).actions == ()
    assert controller.step(10.0, evidence).actions == ()
    decision = controller.step(16.2, evidence)
    assert decision.state == BcFlightState.HOLDING
    assert decision.actions == ("SELECT_HOLD",)
    assert decision.failure_reason == "PX4 output gate did not become safe"


def test_source_loss_during_prestream_returns_to_lifecycle_selection() -> None:
    """Never request OFFBOARD or ARM after lifecycle authority is lost."""
    controller = BcFlightController()
    healthy = BcFlightEvidence(
        landed=True, recovery_vehicle_state_fresh=True,
        runtime_ready=True, observations_ready=True, telemetry_fresh=True,
        lifecycle_selected=True, source_valid=True, output_safe=False,
    )
    assert controller.step(1.0, healthy).state == BcFlightState.SELECTING_LIFECYCLE
    assert controller.step(1.1, healthy).state == BcFlightState.ENABLING_OUTPUT
    assert controller.step(1.2, replace(healthy, output_ready=True)).actions == ("ENABLE_OUTPUT",)
    assert controller.step(1.3, replace(healthy, output_safe=True)).state == BcFlightState.ENABLING_STREAM
    lost = replace(healthy, lifecycle_selected=False, source_valid=False, output_safe=True)
    decision = controller.step(1.4, lost)
    assert decision.state == BcFlightState.SELECTING_LIFECYCLE
    assert "SEND_OFFBOARD" not in decision.actions
    assert "SEND_ARM" not in decision.actions
    assert "DISABLE_OUTPUT" not in decision.actions


def test_source_loss_while_requesting_arm_cannot_enter_takeoff() -> None:
    """Source loss after OFFBOARD must not reset safety latches."""
    controller = BcFlightController()
    controller.state = BcFlightState.REQUESTING_ARM
    lost = BcFlightEvidence(
        landed=True, recovery_vehicle_state_fresh=True,
        runtime_ready=True, observations_ready=True, telemetry_fresh=True,
        lifecycle_selected=False, source_valid=False, offboard_active=True,
        vehicle_armed=False,
    )
    decision = controller.step(1.0, lost)
    assert decision.state == BcFlightState.HOLDING
    assert "SEND_ARM" not in decision.actions
    assert "DISABLE_OUTPUT" not in decision.actions


def test_bc_handoff_requires_all_settled_output_evidence() -> None:
    """BC selection alone cannot announce navigation readiness."""
    controller = BcFlightController()
    controller.state = BcFlightState.SELECTING_BC
    evidence = BcFlightEvidence(
        bc_selected=True, source_valid=True, output_safe=True,
        stream_stable=False,
    )
    decision = controller.step(1.0, evidence)
    assert decision.state == BcFlightState.SELECTING_BC
    assert decision.actions == ("SELECT_BC",)
    decision = controller.step(1.1, replace(evidence, stream_stable=True))
    assert decision.state == BcFlightState.NAVIGATING


def test_termination_precedence_is_safety_first() -> None:
    """Prefer collision, bounds, and success over an elapsed timeout."""
    config = TerminationConfig(goal_tolerance_m=0.35, timeout_s=45.0)
    common = {
        "goal_distance_m": 0.1,
        "minimum_clearance_m": -0.01,
        "bc_duration_s": 50.0,
        "north_m": 8.0,
        "east_m": 0.0,
        "config": config,
    }
    assert select_terminal_reason(**common) == "collision"
    common["minimum_clearance_m"] = 1.0
    assert select_terminal_reason(**common) == "out_of_bounds"
    common["north_m"] = 3.0
    assert select_terminal_reason(**common) == "success"
    common["goal_distance_m"] = 2.0
    assert select_terminal_reason(**common) == "timeout"


def test_default_termination_allows_non_contact_close_pass() -> None:
    """A close pass with positive physical clearance is not a collision."""
    config = TerminationConfig()
    assert select_terminal_reason(
        goal_distance_m=1.0,
        minimum_clearance_m=0.030162698241804287,
        bc_duration_s=1.0,
        north_m=2.2,
        east_m=0.37,
        config=config,
    ) is None


class _PolicyTiming:
    def __init__(self):
        self.stream = None
        self.events = []
        self.messages = []

    def record(self, event, **fields):
        self.events.append((event, fields))

    def publish(self, topic, message, **fields):
        self.events.append(("command_publish", fields))

    def send(self, publisher, message):
        self.messages.append(message)

    def tick(self):
        self.events.append(("tick", {}))

    def receive(self, topic, message):
        self.events.append(("receive", {"topic": topic}))


def _policy_stub(monkeypatch, receipt_age_ms=20):
    policy = pytest.importorskip("uav_px4_control.flight.bc_policy_node")
    now_ns = 1_000_000_000
    monkeypatch.setattr(policy.time, "monotonic_ns", lambda: now_ns)
    timing = _PolicyTiming()
    node = SimpleNamespace(
        _timing=timing,
        _enabled=True,
        _enable_generation=1,
        _last_accepted_request_id=0,
        _last_command=None,
        _pending_action_body=None,
        _previous_action=(0.0, 0.0, 0.0),
        _inference_count=0,
        _inferred_image_sequence=-1,
        _inference_position=None,
        _action_origin={},
        _action_publish_count=0,
        _last_action_publish_monotonic_ns=None,
        _last_new_action_publish_monotonic_ns=None,
        _last_command_stamp_ros_ns=None,
        _action_freshness_timeout_s=0.25,
        _inference_worker_failed=False,
        _inference_condition=threading.Condition(),
        _completed_inference=None,
        _pending_inference=None,
        _requested_source="top_rgb",
        _requested_sources=("top_rgb",),
        _runtime_device={},
        _video_spool_dir=None,
        _image_sequence=2,
        _last_error="",
        _waiting_logged=False,
        _command_publisher=object(),
    )
    node.get_logger = lambda: SimpleNamespace(error=lambda *_: None, info=lambda *_: None)
    node._ros_now_ns = [5_020_000_000]
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=node._ros_now_ns[0])
    )
    node._now_seconds = lambda: 1.0
    node._observation_error = lambda _: None
    node._schedule_inference = lambda: None
    node._publish_status = lambda ready, reason: timing.events.append(
        ("status", {"ready": ready, "reason": reason})
    )
    for name in ("_consume_inference_completion", "_publish_command", "_update_policy"):
        setattr(node, name, MethodType(getattr(policy.BcPolicyNode, name), node))
    request = {
        "request_id": 1,
        "enable_generation": 1,
        "image_sequence": 1,
        "image_sequences": {"top_rgb": 1},
        "image_header_stamp": (0, 0),
        "image_header_stamps": {"top_rgb": (0, 0)},
        "image_receipt_monotonic_ns": now_ns - receipt_age_ms * 1_000_000,
        "oldest_image_receipt_monotonic_ns": now_ns - receipt_age_ms * 1_000_000,
        "image_receipts_monotonic_ns": {
            "top_rgb": now_ns - receipt_age_ms * 1_000_000,
        },
        "image_receipt_ros_ns": 5_000_000_000,
        "observation_inputs": {},
        "scheduled_monotonic_ns": now_ns - 15_000_000,
        "yaw": 0.0,
        "position": {"north_m": 0.0, "east_m": 0.0},
        "state8": [0.0] * 8,
    }
    node._completed_inference = {
        "request": request,
        "action": [0.2, 0.1, 0.0],
        "inference_started_ns": now_ns - 10_000_000,
        "inference_completed_ns": now_ns - 2_000_000,
    }
    return policy, node, timing


def test_completion_publishes_without_timer_and_uses_snapshot(monkeypatch):
    policy, node, timing = _policy_stub(monkeypatch)
    policy.BcPolicyNode._inference_ready_callback(node)
    assert len(timing.messages) == 1
    assert timing.messages[0].header.stamp.sec == 5
    assert node._previous_action == (0.2, 0.1, 0.0)
    assert node._action_origin["request_id"] == 1
    assert any(event == "inference_result_handled" for event, _ in timing.events)
    assert any(event == "previous_action_updated" for event, _ in timing.events)


@pytest.mark.parametrize("rejection", ["expired", "disabled", "generation", "older"])
def test_rejected_inference_does_not_publish_or_update_history(monkeypatch, rejection):
    policy, node, timing = _policy_stub(
        monkeypatch, receipt_age_ms=251 if rejection == "expired" else 20
    )
    if rejection == "disabled":
        node._enabled = False
    elif rejection == "generation":
        node._enable_generation = 2
    elif rejection == "older":
        node._last_accepted_request_id = 2
        node._last_command = (0.8, 0.0, 0.0, 0.0)
    node._consume_inference_completion()
    assert timing.messages == []
    assert node._previous_action == (0.0, 0.0, 0.0)
    assert any(event == "inference_discarded" for event, _ in timing.events)
    if rejection == "older":
        assert node._last_command == (0.8, 0.0, 0.0, 0.0)


def test_new_image_does_not_starve_fresh_inflight_result(monkeypatch):
    policy, node, timing = _policy_stub(monkeypatch)
    node._image_sequence = 2
    policy.BcPolicyNode._inference_ready_callback(node)
    assert len(timing.messages) == 1
    assert node._inferred_image_sequence == 1


def test_publish_failure_does_not_update_previous_action(monkeypatch):
    policy, node, timing = _policy_stub(monkeypatch)

    def fail_send(*_):
        raise RuntimeError("publisher unavailable")

    timing.send = fail_send
    with pytest.raises(RuntimeError, match="publisher unavailable"):
        node._consume_inference_completion()
        node._publish_command()
    assert node._previous_action == (0.0, 0.0, 0.0)


def test_action_expiring_during_diagnostics_is_not_sent(monkeypatch):
    policy, node, timing = _policy_stub(monkeypatch)
    node._consume_inference_completion()
    times = iter((1_200_000_000, 1_260_000_000))
    monkeypatch.setattr(policy.time, "monotonic_ns", lambda: next(times))
    assert not node._publish_command()
    assert timing.messages == []
    assert node._previous_action == (0.0, 0.0, 0.0)
    assert any(event == "action_publish_rejected" for event, _ in timing.events)


def test_republication_cannot_extend_observation_deadline(monkeypatch):
    policy, node, timing = _policy_stub(monkeypatch)
    node._consume_inference_completion()
    node._publish_command()
    node._ros_now_ns[0] = 5_120_000_000
    monkeypatch.setattr(policy.time, "monotonic_ns", lambda: 1_100_000_000)
    node._publish_command()
    assert len(timing.messages) == 2
    assert timing.messages[1].header.stamp.nanosec == 120_000_000
    assert node._previous_action == (0.2, 0.1, 0.0)
    monkeypatch.setattr(policy.time, "monotonic_ns", lambda: 1_300_000_000)
    node._ros_now_ns[0] = 5_320_000_000
    node._publish_command()
    assert len(timing.messages) == 2
    assert node._action_origin["image_receipt_ros_ns"] == 5_000_000_000


def test_repeated_command_stamps_satisfy_mux_contract(monkeypatch):
    policy, node, timing = _policy_stub(monkeypatch)
    node._consume_inference_completion()
    node._publish_command()
    node._ros_now_ns[0] = 5_120_000_000
    monkeypatch.setattr(policy.time, "monotonic_ns", lambda: 1_100_000_000)
    node._publish_command()
    mux = ControlSourceMux(ControlMuxConfig(switch_hold_duration_s=0.0))
    for index, message in enumerate(timing.messages):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        record = mux.accept_candidate(
            BC_POLICY,
            fixed_candidate(source=BC_POLICY, stamp_s=stamp),
            receipt_time_s=5.02 + index * 0.1,
        )
        assert record.valid


def test_worker_error_stays_visible_and_fails_closed(monkeypatch):
    policy, node, timing = _policy_stub(monkeypatch)
    node._completed_inference["error"] = "inference worker response timed out"
    policy.BcPolicyNode._inference_ready_callback(node)
    assert node._inference_worker_failed
    assert timing.messages == []
    assert timing.events[-1][1]["reason"] == "inference_worker_failed"


def test_missing_observations_keep_status_unready(monkeypatch):
    policy, node, timing = _policy_stub(monkeypatch)
    node._completed_inference = None
    node._observation_error = lambda _: "waiting_for_top_rgb"
    policy.BcPolicyNode._tick(node)
    assert timing.events[-1] == (
        "status", {"ready": False, "reason": "waiting_for_top_rgb"}
    )


def test_missing_observations_trigger_existing_flight_hold():
    controller = BcFlightController()
    controller.state = BcFlightState.NAVIGATING
    evidence = BcFlightEvidence(
        runtime_ready=True,
        telemetry_fresh=True,
        bc_ready=False,
        bc_selected=True,
        source_valid=True,
        output_safe=True,
        stream_stable=True,
    )
    decision = controller.step(1.0, evidence)
    assert decision.state == BcFlightState.HOLDING
    assert "SELECT_HOLD" in decision.actions


def test_pending_slot_stays_bounded_and_keeps_inflight_result(monkeypatch):
    policy, node, timing = _policy_stub(monkeypatch)
    node._source_sequence = {"top_rgb": 2}
    node._requested_image_sequence = (1,)
    node._next_request_id = 1
    node._inference_snapshot = lambda: {
        "image_sequence": node._source_sequence["top_rgb"],
        "enable_generation": 1,
        "observation_inputs": {},
    }
    node._pending_inference = {"request_id": 99}
    policy.BcPolicyNode._schedule_inference(node)
    assert node._pending_inference["request_id"] == 2
    assert node._pending_inference["image_sequence"] == 2
    assert any(event == "inference_pending_replaced" for event, _ in timing.events)
    policy.BcPolicyNode._inference_ready_callback(node)
    assert len(timing.messages) == 1
    assert node._inferred_image_sequence == 1


def test_worker_completion_wakes_executor_without_timer(monkeypatch):
    policy, node, _ = _policy_stub(monkeypatch)
    started = threading.Event()
    finish = threading.Event()
    triggered = threading.Event()

    def read_worker(_):
        started.set()
        assert finish.wait(1.0)
        return {"action": [0.2, 0.1, 0.0]}

    node._read_worker = read_worker
    node._worker = SimpleNamespace(stdin=io.StringIO())
    node._inference_timeout_s = 0.5
    node._inference_guard = SimpleNamespace(trigger=triggered.set)
    node._inference_stop = False
    node._completed_inference = None
    node._pending_inference = {
        "request_id": 1,
        "images": {"top_rgb": b"image"},
        "state8": [0.0] * 8,
    }
    worker = threading.Thread(target=policy.BcPolicyNode._inference_loop, args=(node,))
    worker.start()
    try:
        assert started.wait(1.0)
        assert not triggered.is_set()
        finish.set()
        assert triggered.wait(1.0)
        assert node._completed_inference["request"]["request_id"] == 1
    finally:
        finish.set()
        with node._inference_condition:
            node._inference_stop = True
            node._inference_condition.notify_all()
        worker.join(timeout=1.0)
        assert not worker.is_alive()


def test_slow_inference_does_not_block_image_or_enable_callbacks(monkeypatch):
    policy, node, timing = _policy_stub(monkeypatch)
    monkeypatch.setattr(policy, "validate_live_image", lambda *_: None)
    node._inference_busy = True
    node._source_images = {"top_rgb": None}
    node._source_receipt_s = {"top_rgb": None}
    node._source_receipt_monotonic_ns = {"top_rgb": None}
    node._source_receipt_ros_ns = {"top_rgb": None}
    node._source_header_stamp = {"top_rgb": None}
    node._source_sequence = {"top_rgb": 0}
    node._source_contract_errors = {}
    node._image = None
    node._image_receipt_s = None
    node._requested_image_sequence = None
    node._next_request_id = 0
    node._inference_snapshot = lambda: {
        "image_sequence": node._image_sequence,
        "enable_generation": node._enable_generation,
        "observation_inputs": {},
    }
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=5_000_000_000)
    )
    node._schedule_inference = MethodType(policy.BcPolicyNode._schedule_inference, node)
    image = policy.CompressedImage()
    image.data = b"image"
    image.header.stamp.sec = 4
    policy.BcPolicyNode._image_callback(node, image)
    assert node._pending_inference["image_sequence"] == 1
    response = SimpleNamespace(success=False, message="")
    policy.BcPolicyNode._enable_callback(node, SimpleNamespace(data=False), response)
    assert response.success
    assert node._pending_inference is None
    assert node._enable_generation == 2
    assert timing.messages == []
