import argparse
import csv
import datetime
import random
import time
import cv2
import numpy as np
import os
import mediapipe as mp

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
    fixation_duration = random.uniform(1.0, 1.5)
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
        cv2.putText(canvas, "FIXATE ON CENTER", (cx - 200, cy - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(window_name, canvas)
        if cv2.waitKey(1) & 0xFF == 27: return False

    # 2. Target Appearance
    target_x = int(screen_w * 0.85) if target_side == "RIGHT" else int(screen_w * 0.15)
    target_y = screen_h // 2
    
    start_stimulus = time.time()
    
    # We record for 1.0 second after target appears
    while (time.time() - start_stimulus) < 1.0:
        current_t = time.time()
        ret, frame = webcam.read()
        if not ret: break
        gaze.refresh(frame)

        # Get Gaze & Smooth it
        raw_ratio = gaze.horizontal_ratio()
        smooth_ratio = ""
        gaze_direction = "UNKNOWN"
        
        if raw_ratio is not None:
            smooth_ratio = filter_x(current_t, raw_ratio)
            
            # Determine gaze direction (same thresholds as example.py)
            if smooth_ratio < 0.35:
                gaze_direction = "LEFT"
            elif smooth_ratio > 0.65:
                gaze_direction = "RIGHT"
            else:
                gaze_direction = "CENTER"

        # Draw Target (Green Circle)
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(canvas, (target_x, target_y), 20, (0, 255, 0), -1)
        
        # Show current gaze direction to user
        if raw_ratio is not None:
            color = (0, 255, 0) if gaze_direction == target_side else (0, 0, 255)
            cv2.putText(canvas, f"Looking: {gaze_direction}", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(canvas, f"Ratio: {smooth_ratio:.2f}", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        
        cv2.imshow(window_name, canvas)
        
        # Log Data
        writer.writerow([
            trial_num,
            "PROSACCADE",
            target_side,
            gaze_direction,
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