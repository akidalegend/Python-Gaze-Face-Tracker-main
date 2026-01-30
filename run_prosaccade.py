import argparse
import csv
import datetime
import random
import time
import cv2
import numpy as np
import os

# --- UPDATED IMPORTS ---
from gaze_adapter import GazeAdapter as GazeTracking  # Use the new Adapter
from filters import OneEuroFilter                     # Use the new Filter
from task_utils import ensure_directories, prompt_label

def run_prosaccade_trial(
    writer, 
    trial_num, 
    target_side, 
    window_name, 
    screen_w, 
    screen_h, 
    gaze, 
    webcam, 
    filter_x
):
    # 1. Fixation (Center)
    fixation_duration = random.uniform(1.0, 1.5)  # Randomize to prevent anticipation
    start_fix = time.time()
    
    while (time.time() - start_fix) < fixation_duration:
        ret, frame = webcam.read()
        if not ret: break
        gaze.refresh(frame)
        
        # Draw Center Cross
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cx, cy = screen_w // 2, screen_h // 2
        cv2.line(canvas, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 2)
        cv2.line(canvas, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 2)
        cv2.imshow(window_name, canvas)
        if cv2.waitKey(1) & 0xFF == 27: return False

    # 2. Target Appearance
    target_x = int(screen_w * 0.85) if target_side == "RIGHT" else int(screen_w * 0.15)
    target_y = screen_h // 2
    
    start_stimulus = time.time()
    saccade_detected = False
    reaction_time = None
    
    # We record for 1.0 second after target appears
    while (time.time() - start_stimulus) < 1.0:
        current_t = time.time()
        ret, frame = webcam.read()
        if not ret: break
        gaze.refresh(frame)

        # Get Gaze & Smooth it
        raw_ratio = gaze.horizontal_ratio()
        if raw_ratio is not None:
            # Apply One Euro Filter
            smooth_ratio = filter_x(current_t, raw_ratio)
            
            # Simple Threshold Logic for Real-time Feedback (optional)
            # 0.0=Right, 1.0=Left (approx)
            # NOTE: You will analyze the RAW data in the CSV later for precision
            pass

        # Draw Target (Green Circle)
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(canvas, (target_x, target_y), 20, (0, 255, 0), -1)
        cv2.imshow(window_name, canvas)
        
        # Log Data
        writer.writerow([
            trial_num,
            "PROSACCADE",
            target_side,
            current_t,
            raw_ratio if raw_ratio else "",
            smooth_ratio if raw_ratio else ""
        ])

        if cv2.waitKey(1) & 0xFF == 27: return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", help="Participant Label")
    args = parser.parse_args()
    label = args.label if args.label else prompt_label()

    ensure_directories(["sessions/raw"])
    
    # Initialize Tracker
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    
    # Screen Setup (Full HD Default)
    screen_w, screen_h = 1920, 1080
    cv2.namedWindow("Prosaccade Task", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Prosaccade Task", screen_w, screen_h)

    # Initialize Filter
    filter_x = OneEuroFilter(time.time(), 0.5, min_cutoff=0.1, beta=0.05)

    # Prepare CSV
    filename = f"sessions/raw/{label}_prosaccade_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "task_type", "target_side", "timestamp", "raw_gaze_x", "smooth_gaze_x"])

        print("Press any key to start...")
        cv2.waitKey(0)

        # Run 20 Trials
        for i in range(1, 21):
            side = random.choice(["LEFT", "RIGHT"])
            if not run_prosaccade_trial(writer, i, side, "Prosaccade Task", screen_w, screen_h, gaze, webcam, filter_x):
                break
            
            # Inter-trial Interval (Blank Screen)
            blank = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.imshow("Prosaccade Task", blank)
            cv2.waitKey(random.randint(500, 1000))

    webcam.release()
    cv2.destroyAllWindows()