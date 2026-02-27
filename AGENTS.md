# AGENTS.md

## Cursor Cloud specific instructions

This is a Python webcam-based gaze/face tracking application using OpenCV and MediaPipe. See `README.md` for feature overview and usage.

### Key constraints

- **Webcam required**: All entry points (`main.py`, `example.py`, `debug_gaze.py`, `run_calibration.py`, `run_prosaccade.py`, `run_antisaccade.py`) use `cv2.VideoCapture(0)`. Without a physical camera, the app initializes then exits gracefully.
- **GUI display required**: Scripts use `cv2.imshow()`. Start Xvfb before running: `Xvfb :99 -screen 0 1280x720x24 &` then `export DISPLAY=:99`.
- **MediaPipe version**: The `requirements.txt` is unpinned. `main.py` uses the legacy `mp.solutions.face_mesh` API, while `gaze_adapter.py` uses the newer `mediapipe.tasks` API. Use `mediapipe==0.10.21` which has both APIs. Versions >= 0.10.31 removed `solutions`.

### Running and testing

- **Install**: `pip install -r requirements.txt` (then `pip install mediapipe==0.10.21` to get a compatible version).
- **Verify**: `python3 test.py` prints the OpenCV version.
- **No automated test suite**: There are no unit/integration tests. `test.py` only prints the OpenCV version.
- **No linter config**: No `.flake8`, `.pylintrc`, or similar. You can run `python3 -m py_compile <file>` to syntax-check individual files.
- **Lint all files**: `for f in *.py; do python3 -m py_compile "$f" && echo "$f: OK"; done`

### Project structure

All Python files live in the repository root. Key modules:
- `main.py` — Primary real-time tracking application
- `gaze_adapter.py` — MediaPipe Tasks-based adapter for calibration/experiment scripts
- `filters.py` — One Euro Filter for signal smoothing
- `AngleBuffer.py` — Moving average buffer for head pose angles
- `task_utils.py` — Shared utilities for experimental task scripts
- `face_landmarker.task` — MediaPipe model file (auto-downloaded by `gaze_adapter.py` if missing)
