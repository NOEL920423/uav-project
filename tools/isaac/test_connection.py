import omni.usd

stage = omni.usd.get_context().get_stage()

if stage is None:
    print("[VSCode Test] No active USD stage.")
else:
    print("[VSCode Test] Isaac Sim connection works.")
    print("[VSCode Test] Stage root layer:", stage.GetRootLayer().identifier)