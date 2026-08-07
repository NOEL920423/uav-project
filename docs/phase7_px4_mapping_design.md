# Phase 7 PX4 Mapping Design

## Boundary

```text
/uav/control/selected_command
        ↓
PX4 candidate mapper
        ↓
independent candidate validator
        ↓
PX4 safety gate
        ↓
STOP

NO REAL PX4 PUBLISHER EXISTS IN PHASE 7
```

Candidate commands belong to individual controllers. The Phase 6 mux produces
the one selected command. Phase 7 identity-maps that already-`px4_ned` command
to a diagnostic PX4 setpoint candidate. An actual PX4 output command is a
fourth, future-only concept.

## Mapping contract

North, east, down, and NED yaw-rate signs are preserved exactly. There is no
ENU conversion. Absolute yaw is not integrated or fabricated. Position,
acceleration, jerk, and yaw use the locally verified NaN disable convention.

The mapper performs structural mapping and records a basic rejection on wrong
frame or nonzero angular X/Y. A separate validator rechecks mapping validity,
frame, canonical source, uint64/monotonic timestamp, receipt-time freshness,
mux validity/source consistency, finite commanded fields, NaN unused fields,
velocity-only flags, component/horizontal/total velocity limits, and yaw rate.

`Px4MappingConfig` defaults match Phase 6: north/east/horizontal/total velocity
at most 2.0 m/s, down magnitude at most 1.0 m/s, and yaw rate at most 1.5 rad/s.
The model rejects overrides that expand those boundary limits.

ROS seconds/nanoseconds convert to uint64 microseconds by integer truncation:

```text
timestamp_us = seconds * 1_000_000 + nanoseconds // 1_000
```

Sub-microsecond remainder is discarded. Normalization, nonnegative values,
overflow, equal timestamps, and backward timestamps are explicit failures.
