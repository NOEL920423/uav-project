from pathlib import Path

script_path = Path("/home/noel_614420090/uav-project/3.front_camera_png_recorder.py")

code = script_path.read_text(encoding="utf-8")

namespace = {
    "__file__": str(script_path),
    "__name__": "__main__",
}

exec(compile(code, str(script_path), "exec"), namespace)