from pathlib import Path

script_path = Path(__file__).resolve().parents[1] / "isaac_direct_pipeline" / "3.front_camera_png_recorder.py"

code = script_path.read_text(encoding="utf-8")

namespace = {
    "__file__": str(script_path),
    "__name__": "__main__",
}

exec(compile(code, str(script_path), "exec"), namespace)
