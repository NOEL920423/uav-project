# Active Isaac runtime

`runtime/bootstrap.py` and `runtime/runtime_bridge.py` are the active Phase 9
Isaac/Pegasus integration. The bootstrap loads the bridge from its own
directory, so the pair must remain together.

Historical Isaac Script Editor pipelines are retained under `legacy/` and are
not part of the verified flight command.
