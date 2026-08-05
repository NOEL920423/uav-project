# uav_interfaces

Owns the typed ROS 2 message, service, and action contracts shared by the UAV
integration packages. Phase 1 establishes only the package boundary; runtime
behavior belongs to later phases.

- Inputs: interface definitions maintained in this package.
- Outputs: generated language bindings after a workspace build.
- Must not: contain planners, controllers, PX4 command logic, or simulator I/O.
- Future phase: consumers and producers will adopt the declared contracts.
