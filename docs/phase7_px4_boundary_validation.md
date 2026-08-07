# Offline PX4 Output-Boundary Validation

Phase 7 validates only the boundary from the Phase 6 selected command to a
diagnostic PX4 velocity candidate and a fail-closed permission bit. It does not
publish a PX4 input, start PX4, enter OFFBOARD, arm, or fly.

## Quantitative fixture matrix

The ages below are deterministic fixture inputs or configured thresholds. Live
ROS receipt ages vary with scheduler latency and are checked against the same
`0.25 s` command and `0.50 s` telemetry limits.

| Fixture | Active source | Command age (s) | Telemetry age (s) | NED velocity (m/s) | Yaw rate (rad/s) | Candidate valid | Enable requested | Safe | Gate state | Fault reason | Recovery |
|---|---|---:|---:|---|---:|---|---|---|---|---|---|
| startup | HOLD | inf | inf | none | none | false | false | false | `WAITING_SELECTED_COMMAND` | no command | wait for inputs |
| healthy-disabled | ASTAR_EXPERT | 0.00 | 0.00 | (0.40, 0.00, 0.00) | 0.10 | true | false | false | `READY_DISABLED` | explicit enable required | enable |
| healthy-enabled | ASTAR_EXPERT | 0.01 | 0.01 | (0.40, 0.00, 0.00) | 0.10 | true | true | true | `SAFE_TO_FORWARD` | none | none |
| stale-command | ASTAR_EXPERT | 0.30 | 0.00 | (0.40, -0.20, 0.10) | 0.20 | true | true | false | `DISABLED_STALE_COMMAND` | age > 0.25 | reset/re-enable |
| stale-telemetry | ASTAR_EXPERT | 0.00 | 0.60 | (0.40, -0.20, 0.10) | 0.20 | true | true | false | `DISABLED_STALE_TELEMETRY` | age > 0.50 | reset/re-enable |
| mux-HOLD | HOLD | 0.00 | 0.00 | zero | 0.00 | true | true | false | `DISABLED_MUX_HOLD` | mux HOLD | reset/re-enable |
| failsafe | ASTAR_EXPERT | <0.25 | <0.50 | bounded live candidate | bounded | true | true | false | `DISABLED_FAILSAFE` | failsafe active | reset/re-enable |
| recovered-data-only | ASTAR_EXPERT | <0.25 | <0.50 | bounded live candidate | bounded | true | false | false | `LATCHED_FAULT` | explicit reset required | still blocked |
| explicit recovery | ASTAR_EXPERT | <0.25 | <0.50 | bounded live candidate | bounded | true | true | true | `SAFE_TO_FORWARD` | none | recovered |

Pure regressions cover 16 mapping conditions, including exact NED signs,
positive/negative down velocity, horizontal and yaw limits, non-finite values,
wrong frame, unknown source, age, and timestamp regression. Gate regressions
cover all explicit state families and 12 named synthetic telemetry fixtures.

## ROS graph evidence

- `px4-map-check` verifies the pure timestamp/mapping suites and live candidate,
  status, and disabled permission topics.
- `px4-gate-check` obtains `SAFE_TO_FORWARD`, injects synthetic failsafe,
  observes immediate false plus a latch, proves healthy telemetry cannot clear
  it, then performs disable/reset/re-enable.
- `px4-boundary-check` runs scene → A* → accepted B-spline/fallback → timed
  trajectory → follower → mux → candidate mapping → synthetic gate. The plant
  remains the pre-existing direct offline odometry fixture; there is no fake
  PX4 layer between the diagnostic candidate and plant.

Every live monitor checks that selected-command has exactly one publisher and
that no `/fmu/in/*` topic exists. Static AST tests additionally prove that the
mapping node consumes the mux-owned output and no Phase 7 publisher call targets
a live PX4 input topic.

## Remaining boundary risks

No evidence here validates DDS bridge QoS against a running PX4, boot-relative
timestamp behavior outside the audited local UXRCE contract, setpoint prestream
sequencing, OFFBOARD transition, arming, vehicle dynamics, estimator behavior,
or real failsafe recovery. Those remain deliberately outside Phase 7.
