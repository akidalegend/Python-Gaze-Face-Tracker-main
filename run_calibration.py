import argparse
import json
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# IMPORT CHANGE: Use the adapter instead of the old library
from gaze_adapter import GazeAdapter as GazeTracking
from task_utils import ensure_directories, prompt_label
from filters import OneEuroFilter

def _wait_for_click(
    window_name: str = "Calibration Start",
    width: int = 600,
    height: int = 400,
    text: str = "Click to start",
) -> bool:
    """Display a simple start screen and wait for a left-click or 'q'/ESC."""
    clicked = False

    def _on_mouse(event, _x, _y, _flags, _param):
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.setMouseCallback(window_name, _on_mouse)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(canvas, text, (text_x, text_y), font, 1.0, (0, 255, 255), 2)
        cv2.rectangle(
            canvas,
            (text_x - 20, text_y - text_size[1] - 20),
            (text_x + text_size[0] + 20, text_y + 20),
            (0, 255, 0),
            2,
        )
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10) & 0xFF
        if clicked:
            cv2.destroyWindow(window_name)
            return True
        if key in (ord("q"), 27):  # q or ESC
            cv2.destroyWindow(window_name)
            return False


def _countdown(
    window_name: str,
    width: int,
    height: int,
    win_x: int,
    win_y: int,
    seconds: int = 3,
    message: str = "Starting calibration",
) -> bool:
    """Show a brief countdown before calibration begins."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, win_x, win_y)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for remaining in range(seconds, 0, -1):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        label_text = f"{message} in {remaining}"
        text_size = cv2.getTextSize(label_text, font, 1.2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(canvas, label_text, (text_x, text_y), font, 1.2, (0, 255, 0), 3)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(1000) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            cv2.destroyWindow(window_name)
            return False

    cv2.destroyWindow(window_name)
    return True


def get_primary_display_size(fallback_w: int = 1400, fallback_h: int = 900) -> Tuple[int, int]:
    """Return primary display size in logical points (what OpenCV uses on macOS).

    Tries Quartz (current Python), then system Python with Quartz,
    then falls back to the provided defaults.
    """
    # Method 1: Quartz in current Python
    try:
        import Quartz
        bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        w, h = int(bounds.size.width), int(bounds.size.height)
        if w > 0 and h > 0:
            print(f"Detected display: {w}x{h}")
            return w, h
    except Exception:
        pass

    # Method 2: System Python (always has PyObjC on macOS)
    try:
        import subprocess
        result = subprocess.run(
            ["/usr/bin/python3", "-c",
             "import Quartz; d = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID()); "
             "print(int(d.size.width), int(d.size.height))"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            w, h = int(parts[0]), int(parts[1])
            if w > 0 and h > 0:
                print(f"Detected display: {w}x{h}")
                return w, h
    except Exception:
        pass

    return fallback_w, fallback_h


def get_screen_resolution(default_w: int = 1920, default_h: int = 1080) -> Tuple[int, int]:
    """Return a safe window size; caller can override via CLI."""
    return default_w, default_h


def draw_calibration_target(frame: np.ndarray, x: int, y: int, radius: int = 15, color=(0, 0, 255)) -> None:
    """Draw a target (dot with center)."""
    cv2.circle(frame, (x, y), radius, color, -1)
    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)


# --- 2D polynomial calibration helpers ---
def _poly2_features(h: float, v: float) -> np.ndarray:
    # [1, h, v, h*v, h^2, v^2]
    return np.array([1.0, float(h), float(v), float(h) * float(v), float(h) ** 2, float(v) ** 2], dtype=float)


def _poly2_design_matrix(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    ones = np.ones_like(h, dtype=float)
    return np.column_stack([ones, h, v, h * v, h * h, v * v]).astype(float)


def _fit_poly2(h: np.ndarray, v: np.ndarray, target: np.ndarray) -> List[float]:
    a = _poly2_design_matrix(h, v)
    coef, _, _, _ = np.linalg.lstsq(a, target, rcond=None)
    return [float(c) for c in coef.tolist()]


def _apply_model(model: dict, h: float, v: float) -> Tuple[float, float]:
    """Apply either poly2 model (preferred) or linear fallback."""
    if model and model.get("type") == "poly2" and "x_coef" in model and "y_coef" in model:
        feats = _poly2_features(h, v)
        x = float(np.dot(np.array(model["x_coef"], dtype=float), feats))
        y = float(np.dot(np.array(model["y_coef"], dtype=float), feats))
        return x, y

    # linear fallback
    x = float(model.get("x_slope", 1.0)) * float(h) + float(model.get("x_intercept", 0.0))
    y = float(model.get("y_slope", 1.0)) * float(v) + float(model.get("y_intercept", 0.0))
    return x, y


def collect_calibration_points(
    session_label: str,
    calib_w: int,
    calib_h: int,
    win_x: int,
    win_y: int,
    fullscreen: bool = False,
) -> Optional[dict]:
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Unable to access camera.")
        return None

    # Points for calibration (13 points: extra coverage in top half for better accuracy)
    points_norm = [
        (0.05, 0.05), (0.50, 0.05), (0.95, 0.05),
        (0.25, 0.05), (0.75, 0.05),                   # extra: top row quarter marks
        (0.25, 0.25), (0.75, 0.25),                   # extra: upper-mid row
        (0.05, 0.50), (0.50, 0.50), (0.95, 0.50),
        (0.05, 0.95), (0.50, 0.95), (0.95, 0.95),
    ]

    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, calib_w, calib_h)
    cv2.moveWindow(window_name, win_x, win_y)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calibration_data = {
        "label": session_label,
        "timestamp": time.time(),
        "screen_width": calib_w,
        "screen_height": calib_h,
        "points": [],
    }

    print(f"Starting calibration for {session_label}. Look at the red dots.")

    # Warmup: feed a few frames so the gaze tracker stabilizes before point 1
    print("Warming up tracker (2s)...")
    warmup_end = time.time() + 2.0
    while time.time() < warmup_end:
        ret, frame = webcam.read()
        if ret:
            gaze.refresh(frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            webcam.release()
            cv2.destroyAllWindows()
            return None

    for i, (px, py) in enumerate(points_norm):
        target_x = int(px * calib_w)
        target_y = int(py * calib_h)

        # 5 seconds total (1.5 settle, 3.5 record) — longer recording helps get valid gaze samples
        start_time = time.time()
        samples_h: List[float] = []
        samples_v: List[float] = []
        samples_total = 0
        samples_blink = 0

        while True:
            ret, frame = webcam.read()
            if not ret:
                print("Error: Failed to read from camera.")
                webcam.release()
                cv2.destroyAllWindows()
                return None

            stimulus = np.zeros((calib_h, calib_w, 3), dtype=np.uint8)
            draw_calibration_target(stimulus, target_x, target_y)
            cv2.putText(
                stimulus,
                f"Point {i + 1}/{len(points_norm)}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            gaze.refresh(frame)
            elapsed = time.time() - start_time

            if elapsed > 1.5:
                samples_total += 1
                if gaze.pupils_located and not gaze.is_blinking():
                    h_ratio = gaze.horizontal_ratio()
                    v_ratio = gaze.vertical_ratio()
                    if h_ratio is not None and v_ratio is not None:
                        samples_h.append(float(h_ratio))
                        samples_v.append(float(v_ratio))
                else:
                    if gaze.is_blinking():
                        samples_blink += 1

            cv2.imshow(window_name, stimulus)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                webcam.release()
                cv2.destroyAllWindows()
                return None

            if elapsed > 5.0:
                break

        if samples_h and samples_v:
            avg_h = float(np.median(samples_h))
            avg_v = float(np.median(samples_v))
            std_h = float(np.std(samples_h))
            std_v = float(np.std(samples_v))
            calibration_data["points"].append(
                {
                    "target_x": int(target_x),
                    "target_y": int(target_y),
                    "gaze_h": avg_h,
                    "gaze_v": avg_v,
                    "std_h": std_h,
                    "std_v": std_v,
                    "samples": int(len(samples_h)),
                    "samples_total": int(samples_total),
                    "samples_blink": int(samples_blink),
                }
            )
        else:
            print(f"Warning: No valid gaze data for point {i + 1}")
            calibration_data["points"].append(
                {
                    "target_x": int(target_x),
                    "target_y": int(target_y),
                    "gaze_h": None,
                    "gaze_v": None,
                    "std_h": None,
                    "std_v": None,
                    "samples": 0,
                    "samples_total": int(samples_total),
                    "samples_blink": int(samples_blink),
                }
            )

    webcam.release()
    cv2.destroyAllWindows()
    return calibration_data


def compute_calibration_model(data: dict) -> Optional[dict]:
    """
    Fit a 2D (degree-2) polynomial mapping from (gaze_h, gaze_v) -> (screen_x, screen_y).
    Also saves a linear fallback.
    """
    points = data.get("points", [])

    valid = [p for p in points if p.get("gaze_h") is not None and p.get("gaze_v") is not None]
    if len(valid) < 4:
        print(f"Not enough points for calibration ({len(valid)} valid, need at least 4).")
        print("Tips: Face the camera directly, ensure good lighting, and look at each red dot until it moves.")
        return None

    tx = np.array([p["target_x"] for p in valid], dtype=float)
    ty = np.array([p["target_y"] for p in valid], dtype=float)
    gh = np.array([p["gaze_h"] for p in valid], dtype=float)
    gv = np.array([p["gaze_v"] for p in valid], dtype=float)

    # Linear fallback
    poly_x = np.polyfit(gh, tx, 1)
    poly_y = np.polyfit(gv, ty, 1)
    model = {
        "x_slope": float(poly_x[0]),
        "x_intercept": float(poly_x[1]),
        "y_slope": float(poly_y[0]),
        "y_intercept": float(poly_y[1]),
        "type": "linear",
    }

    # RESTORED: Poly2 fit
    # Because your eye data is clean now, this will properly map the spherical
    # movement of your eye to the flat plane of your monitor.
    if len(valid) >= 6:
        try:
            x_coef = _fit_poly2(gh, gv, tx)
            y_coef = _fit_poly2(gh, gv, ty)
            model.update(
                {
                    "type": "poly2",
                    "x_coef": x_coef,
                    "y_coef": y_coef,
                }
            )
        except Exception as e:
            print(f"Warning: poly2 fit failed, falling back to linear: {e}")

    return model


def _compute_fit_metrics(points: Sequence[dict], model: Optional[dict]) -> Optional[dict]:
    """Compute in-sample pixel error metrics."""
    valid = [p for p in points if p.get("gaze_h") is not None and p.get("gaze_v") is not None]
    if not valid or not model:
        return None

    preds = []
    targets = []
    for p in valid:
        x, y = _apply_model(model, float(p["gaze_h"]), float(p["gaze_v"]))
        preds.append((x, y))
        targets.append((float(p["target_x"]), float(p["target_y"])))

    preds_arr = np.array(preds, dtype=float)
    targets_arr = np.array(targets, dtype=float)
    d = preds_arr - targets_arr
    dist = np.sqrt(np.sum(d * d, axis=1))

    return {
        "model_type": str(model.get("type", "linear")),
        "n_points_valid": int(len(valid)),
        "rmse_px": float(np.sqrt(np.mean(dist * dist))) if dist.size else None,
        "mae_px": float(np.mean(np.abs(dist))) if dist.size else None,
        "p95_px": float(np.percentile(dist, 95)) if dist.size else None,
    }


def verify_calibration(model: dict, screen_w: int, screen_h: int, win_x: int, win_y: int, fullscreen: bool = False, grid_cols: int = 1, grid_rows: int = 1):
    """Verification loop: show estimated gaze point with Grid."""
    print("\n--- VERIFICATION MODE ---")
    print(f"Testing Grid: {grid_cols}x{grid_rows}")
    print("Look at different boxes. The active box will light up.")
    print("Press 'q' or ESC to finish.")

    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    
    window_name = "Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, screen_w, screen_h)
    cv2.moveWindow(window_name, win_x, win_y)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    now = time.time()
    h_filt = OneEuroFilter(now, 0.5, min_cutoff=0.1, beta=0.01, d_cutoff=1.0)
    v_filt = OneEuroFilter(now, 0.5, min_cutoff=0.1, beta=0.01, d_cutoff=1.0)
    was_blinking = False

    BLINK_SKIP_FRAMES = 2
    blink_skip_remaining = 0

    # Asymmetric hysteresis
    HYST_LEAVE = 7
    HYST_RETURN = 3
    display_c, display_r = 0, 0
    prev_display_c, prev_display_r = 0, 0
    candidate_c, candidate_r = -1, -1
    candidate_count = 0

    # Calculate cell dimensions
    cell_w = screen_w / grid_cols
    cell_h = screen_h / grid_rows

    while True:
        ret, frame = webcam.read()
        if not ret: break
        gaze.refresh(frame)

        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

        # --- Draw Grid Lines ---
        for i in range(1, grid_cols):
            x = int(i * cell_w)
            cv2.line(canvas, (x, 0), (x, screen_h), (50, 50, 50), 1)
        for j in range(1, grid_rows):
            y = int(j * cell_h)
            cv2.line(canvas, (0, y), (screen_w, y), (50, 50, 50), 1)

        if gaze.pupils_located:
            is_blinking = gaze.is_blinking()
            if is_blinking:
                was_blinking = True
            elif not is_blinking:
                if was_blinking:
                    was_blinking = False
                    blink_skip_remaining = BLINK_SKIP_FRAMES

                if blink_skip_remaining > 0:
                    blink_skip_remaining -= 1
                else:
                    h_ratio = gaze.horizontal_ratio()
                    v_ratio = gaze.vertical_ratio()
                    if h_ratio is not None and v_ratio is not None:
                        current_time = time.time()
                        smooth_h = float(h_ratio)
                        smooth_v = float(v_ratio)
                        smooth_x, smooth_y = _apply_model(model, smooth_h, smooth_v)
                        smooth_x, smooth_y = int(smooth_x), int(smooth_y)

                        c_idx = max(0, min(grid_cols - 1, int(smooth_x / cell_w)))
                        r_idx = max(0, min(grid_rows - 1, int(smooth_y / cell_h)))

                        if (c_idx, r_idx) == (display_c, display_r):
                            candidate_c, candidate_r = -1, -1
                            candidate_count = 0
                        elif (c_idx, r_idx) == (candidate_c, candidate_r):
                            candidate_count += 1
                            threshold = HYST_RETURN if (c_idx, r_idx) == (prev_display_c, prev_display_r) else HYST_LEAVE
                            if candidate_count >= threshold:
                                prev_display_c, prev_display_r = display_c, display_r
                                display_c, display_r = candidate_c, candidate_r
                                candidate_count = 0
                        else:
                            candidate_c, candidate_r = c_idx, r_idx
                            candidate_count = 1

                        tl = (int(display_c * cell_w), int(display_r * cell_h))
                        br = (int((display_c + 1) * cell_w), int((display_r + 1) * cell_h))
                        cv2.rectangle(canvas, tl, br, (0, 100, 0), -1)

                        cv2.circle(canvas, (smooth_x, smooth_y), 15, (0, 255, 0), -1)

        cv2.imshow(window_name, canvas)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")): break

    webcam.release()
    cv2.destroyAllWindows()

def print_resolution_metrics(calibration_data, model, screen_w, screen_h):
    points = calibration_data.get("points", [])
    valid_points = [p for p in points if "std_h" in p]
    
    if not valid_points:
        return

    # 1. Calculate Average Jitter in Pixels
    # We estimate pixel jitter by multiplying the normalized std_dev (0-1) by screen size
    avg_jitter_h_norm = np.mean([p["std_h"] for p in valid_points])
    avg_jitter_v_norm = np.mean([p["std_v"] for p in valid_points])
    
    jitter_px_h = avg_jitter_h_norm * screen_w
    jitter_px_v = avg_jitter_v_norm * screen_h
    
    # 2. Get Accuracy (RMSE) from existing metrics
    fit_metrics = calibration_data.get("fit_metrics", {})
    rmse_px = fit_metrics.get("rmse_px", 50.0) # Default to 50px if missing
    
    # 3. Calculate Safe Zone Size (6-sigma rule + RMSE)
    # This represents the minimum width a button/quadrant needs to be 
    # for the user to hit it reliably 99.7% of the time.
    safe_width_px = rmse_px + (6 * jitter_px_h)
    safe_height_px = rmse_px + (6 * jitter_px_v)
    
    # 4. Calculate Max Quadrants
    max_cols = int(screen_w / safe_width_px)
    max_rows = int(screen_h / safe_height_px)
    
    print("\n--- QUANTIFIABLE METRICS ---")
    print(f"Average Jitter (Noise): {jitter_px_h:.1f}px Horiz, {jitter_px_v:.1f}px Vert")
    print(f"Calibration Error (RMSE): {rmse_px:.1f}px")
    print(f"Min Reliable Target Size: {safe_width_px:.0f} x {safe_height_px:.0f} pixels")
    print("-" * 30)
    print(f"MAX RELIABLE GRID: {max_cols} Columns x {max_rows} Rows")
    print("-" * 30)

    return max_cols, max_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", help="Participant label")
    parser.add_argument("--stimulus-width", type=int, default=1400, help="Stimulus window width")
    parser.add_argument("--stimulus-height", type=int, default=900, help="Stimulus window height")
    parser.add_argument("--window-x", type=int, default=0, help="Top-left X position")
    parser.add_argument("--window-y", type=int, default=0, help="Top-left Y position")
    parser.add_argument("--fullscreen", action="store_true", help="Force fullscreen")
    args = parser.parse_args()

    label = args.label if args.label else prompt_label()

    ensure_directories(["sessions/calibration"])

    print("Loading gaze tracking model...")
    _ = GazeTracking()  # Preload MediaPipe tracker
    print("Gaze tracking model loaded successfully!")

    print("\nCalibration instructions:")
    print("- Sit ~60cm from the screen with steady lighting (50-60cm works well for 14\" MacBook Pro).")
    print("- Click to start, then look at each red dot.")
    print("- On Retina Macs, if fullscreen clips the right side, run without --fullscreen and use:")
    print("  --stimulus-width 1512 --stimulus-height 982 to match scaled resolution.")
    
    if not _wait_for_click():
        print("Calibration cancelled.")
        raise SystemExit(0)

    if args.fullscreen:
        screen_w, screen_h = get_primary_display_size()
    else:
        screen_w, screen_h = get_screen_resolution(args.stimulus_width, args.stimulus_height)

    if not _countdown(
        window_name="Calibration Countdown",
        width=screen_w,
        height=screen_h,
        win_x=args.window_x,
        win_y=args.window_y,
        seconds=3,
    ):
        print("Calibration cancelled.")
        raise SystemExit(0)

    cal_data = collect_calibration_points(label, screen_w, screen_h, args.window_x, args.window_y, args.fullscreen)

    if not cal_data:
        print("Calibration aborted.")
        raise SystemExit(0)

    # Diagnostic: show why some points had no valid gaze
    print("\nPer-point summary (frames in record window | valid gaze samples | blinks):")
    for j, pt in enumerate(cal_data["points"]):
        valid = pt.get("samples", 0)
        total = pt.get("samples_total", 0)
        blinks = pt.get("samples_blink", 0)
        status = "OK" if pt.get("gaze_h") is not None else "NO DATA"
        n_pts = len(cal_data["points"])
        print(f"  Point {j + 1}/{n_pts}: total_frames={total}, valid_samples={valid}, blinks={blinks} -> {status}")

    model = compute_calibration_model(cal_data)
    if not model:
        print("Calibration failed.")
        raise SystemExit(1)

    cal_data["model"] = model
    fit_metrics = _compute_fit_metrics(cal_data.get("points", []), model)
    cal_data["fit_metrics"] = fit_metrics

    print("\nCalibration Successful!")
    if fit_metrics:
        print(f"Fit error: RMSE={fit_metrics['rmse_px']:.1f}px")

    filename = f"sessions/calibration/{label}_calibration.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(cal_data, f, indent=4)
    print(f"Saved to {filename}")

    print(f"Saved to {filename}")

    # 1. Calculate and Get Grid Size
    rec_cols, rec_rows = print_resolution_metrics(cal_data, model, screen_w, screen_h)
    
    # 2. Launch Verification with that Grid
    verify_calibration(
        model, 
        screen_w, 
        screen_h, 
        args.window_x, 
        args.window_y, 
        args.fullscreen,
        grid_cols=rec_cols,
        grid_rows=rec_rows
    )