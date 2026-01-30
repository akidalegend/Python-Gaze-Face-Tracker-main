import argparse
import csv
import datetime
import random
import time
import cv2
import numpy as np
import os

from gaze_adapter import GazeAdapter as GazeTracking
from filters import OneEuroFilter
from task_utils import ensure_directories, prompt_label

def run_antisaccade_trial(writer, trial_num, target_side, window_name, screen_w, screen_h, gaze, webcam, filter_x):
    # 1. Fixation
    fixation_duration = random.uniform(1.0, 1.5)
    start_fix = time.time()
    
    while (time.time() - start_fix) < fixation_duration:
        ret, frame = webcam.read()
        if not ret: break
        gaze.refresh(frame)
        
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        # Red Cross for Antisaccade to distinguish it
        cx, cy = screen_w // 2, screen_h // 2
        cv2.line(canvas, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
        cv2.line(canvas, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
        cv2.imshow(window_name, canvas)
        if cv2.waitKey(1) & 0xFF == 27: return False

    # 2. Target Appearance (But look AWAY)
    # If target is RIGHT, user must look LEFT.
    target_x = int(screen_w * 0.85) if target_side == "RIGHT" else int(screen_w * 0.15)
    target_y = screen_h // 2
    
    correct_side = "LEFT" if target_side == "RIGHT" else "RIGHT"

    start_stimulus = time.time()
    
    while (time.time() - start_stimulus) < 1.0:
        current_t = time.time()
        ret, frame = webcam.read()
        if not ret: break
        gaze.refresh(frame)

        raw_ratio = gaze.horizontal_ratio()
        smooth_ratio = ""
        if raw_ratio is not None:
            smooth_ratio = filter_x(current_t, raw_ratio)

        # Draw Target (Red Circle)
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(canvas, (target_x, target_y), 20, (0, 0, 255), -1)
        cv2.imshow(window_name, canvas)
        
        writer.writerow([
            trial_num,
            "ANTISACCADE",
            target_side,
            correct_side,
            current_t,
            raw_ratio if raw_ratio else "",
            smooth_ratio if smooth_ratio != "" else ""
        ])

        if cv2.waitKey(1) & 0xFF == 27: return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", help="Participant Label")
    args = parser.parse_args()
    label = args.label if args.label else prompt_label()

    ensure_directories(["sessions/raw"])
    
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    screen_w, screen_h = 1920, 1080
    cv2.namedWindow("Antisaccade Task", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Antisaccade Task", screen_w, screen_h)

    filter_x = OneEuroFilter(time.time(), 0.5, min_cutoff=0.1, beta=0.05)

    filename = f"sessions/raw/{label}_antisaccade_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "task_type", "target_side", "correct_side", "timestamp", "raw_gaze_x", "smooth_gaze_x"])

        print("Press any key to start...")
        cv2.waitKey(0)

        for i in range(1, 21):
            side = random.choice(["LEFT", "RIGHT"])
            if not run_antisaccade_trial(writer, i, side, "Antisaccade Task", screen_w, screen_h, gaze, webcam, filter_x):
                break
            
            blank = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.imshow("Antisaccade Task", blank)
            cv2.waitKey(random.randint(500, 1000))

    webcam.release()
    cv2.destroyAllWindows()