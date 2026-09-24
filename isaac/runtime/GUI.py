"""Isaac GUI for preview-only environment presets."""
from __future__ import annotations
import json
import re
from pathlib import Path

_WINDOW = None

def show_environment_gui(bridge) -> None:
    import omni.ui as ui
    root = Path(__file__).resolve().parents[1] / "env"
    root.mkdir(parents=True, exist_ok=True)
    name, brightness = ui.SimpleStringModel("untitled"), ui.SimpleFloatModel(800.0)
    color, blocker = ui.SimpleStringModel("0.03, 0.08, 0.18"), ui.SimpleFloatModel(3.2)
    random_enabled, status = ui.SimpleBoolModel(True), ui.SimpleStringModel("Adjust values, then Apply or Save.")
    def values():
        rgb = [float(item.strip()) for item in color.get_value_as_string().split(",")]
        if len(rgb) != 3 or not all(0 <= item <= 1 for item in rgb): raise ValueError("RGB must have three values from 0 to 1")
        return {"dome_intensity": brightness.get_value_as_float(), "obstacle_color": rgb, "blocker_height": blocker.get_value_as_float(), "random_obstacles_enabled": random_enabled.get_value_as_bool()}
    def apply():
        try: bridge.apply_gui_preview(values()); status.set_value("Applied to preview scene.")
        except Exception as error: status.set_value(f"Error: {error}")
    def save():
        try:
            preset = name.get_value_as_string().strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]+", preset): raise ValueError("Name may use letters, numbers, _ and - only")
            (root / f"{preset}.json").write_text(json.dumps({"schema":"uav_environment_preset/v1", "preview":values()}, indent=2) + "\n", encoding="utf-8")
            status.set_value(f"Saved isaac/env/{preset}.json")
        except Exception as error: status.set_value(f"Error: {error}")
    global _WINDOW
    _WINDOW = ui.Window("Environment Preset", width=420, height=300)
    with _WINDOW.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("Preset name"); ui.StringField(model=name)
            ui.Label("Dome brightness"); ui.FloatField(model=brightness)
            ui.Label("Obstacle RGB (comma separated)"); ui.StringField(model=color)
            ui.Label("Fixed path blocker height (m)"); ui.FloatField(model=blocker)
            ui.Label("Show random obstacles")
            ui.CheckBox(model=random_enabled)
            with ui.HStack(): ui.Button("Apply", clicked_fn=apply); ui.Button("Save As Preset", clicked_fn=save)
            ui.Label("Status")
            ui.StringField(model=status, read_only=True)
