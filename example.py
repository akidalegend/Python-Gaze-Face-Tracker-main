"""
Simple example to test left/right gaze detection in real-time.
Shows on screen which direction you're looking.
"""

import cv2
import numpy as np
import time
from gaze_adapter import GazeAdapter as GazeTracking
from filters import OneEuroFilter

def main():
    # Initialize gaze tracker
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    
    if not webcam.isOpened():
        print("Error: Cannot open webcam")
        return
    
    # Get frame dimensions
    ret, frame = webcam.read()
    if not ret:
        print("Error: Cannot read from webcam")
        return
    
    screen_h, screen_w = frame.shape[:2]
    
    # Create window
    cv2.namedWindow("Gaze Direction Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Direction Test", 1200, 800)
    
    # Initialize filter for smoothing
    filter_x = OneEuroFilter(time.time(), 0.5, min_cutoff=0.1, beta=0.05)
    
    print("=" * 60)
    print("GAZE DIRECTION TEST")
    print("=" * 60)
    print("\nLook LEFT, CENTER, or RIGHT")
    print("Watch the screen to see if it detects your direction correctly")
    print("\nPress 'q' or ESC to exit")
    print("=" * 60)
    
    frame_count = 0
    
    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        
        current_t = time.time()
        frame_count += 1
        
        # Process gaze
        gaze.refresh(frame)
        
        # Create display canvas
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        
        # Get gaze data
        h_ratio = gaze.horizontal_ratio()
        v_ratio = gaze.vertical_ratio()
        
        # Debug: Print if no gaze detected
        if h_ratio is None and frame_count % 30 == 0:
            print(f"Frame {frame_count}: No gaze detected - landmarks={gaze.landmarks is not None}, pupils_located={gaze.pupils_located}")
        
        if h_ratio is not None:
            # Apply smoothing
            smooth_h = filter_x(current_t, h_ratio)
            
            # Determine direction
            if smooth_h < 0.35:
                direction = "LOOKING LEFT"
                color = (0, 255, 0)  # Green
                bar_width = int(smooth_h * screen_w)
            elif smooth_h > 0.65:
                direction = "LOOKING RIGHT"
                color = (255, 0, 0)  # Blue
                bar_width = int(smooth_h * screen_w)
            else:
                direction = "LOOKING CENTER"
                color = (0, 255, 255)  # Yellow
                bar_width = int(0.5 * screen_w)
            
            # Draw background
            canvas[:] = (30, 30, 30)
            
            # Draw title
            cv2.putText(canvas, "GAZE DIRECTION TEST", (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 3)
            
            # Draw direction indicator
            cv2.putText(canvas, direction, (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)
            
            # Draw a horizontal bar showing where you're looking
            bar_height = 300
            cv2.rectangle(canvas, (0, bar_height), (screen_w, bar_height + 100), (100, 100, 100), -1)
            cv2.rectangle(canvas, (0, bar_height), (bar_width, bar_height + 100), color, -1)
            
            # Add markers
            center_x = int(0.5 * screen_w)
            cv2.line(canvas, (center_x, bar_height - 20), (center_x, bar_height + 120), (150, 150, 150), 2)
            cv2.putText(canvas, "CENTER", (center_x - 60, bar_height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            
            left_x = int(0.2 * screen_w)
            cv2.line(canvas, (left_x, bar_height - 20), (left_x, bar_height + 120), (150, 150, 150), 2)
            cv2.putText(canvas, "LEFT", (left_x - 40, bar_height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            
            right_x = int(0.8 * screen_w)
            cv2.line(canvas, (right_x, bar_height - 20), (right_x, bar_height + 120), (150, 150, 150), 2)
            cv2.putText(canvas, "RIGHT", (right_x - 50, bar_height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            
            # Draw numerical data
            y_offset = 500
            cv2.putText(canvas, f"Raw Horizontal Ratio: {h_ratio:.3f}", (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 255), 2)
            cv2.putText(canvas, f"Smoothed Horizontal Ratio: {smooth_h:.3f}", (50, y_offset + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 255), 2)
            cv2.putText(canvas, f"Vertical Ratio: {v_ratio:.3f}", (50, y_offset + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 255), 2)
            
            # Blinking status
            blinking = gaze.is_blinking()
            blink_text = "BLINKING" if blinking else "Eyes Open"
            blink_color = (0, 0, 255) if blinking else (0, 255, 0)
            cv2.putText(canvas, blink_text, (50, y_offset + 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, blink_color, 2)
            
            # Frame counter
            cv2.putText(canvas, f"Frame: {frame_count}", (screen_w - 300, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 1)
            
            # Instructions
            cv2.putText(canvas, "Press 'q' or ESC to exit", (50, screen_h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
        
        else:
            # No face detected
            canvas[:] = (30, 30, 30)
            cv2.putText(canvas, "NO FACE DETECTED", (100, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(canvas, "Please face the camera", (100, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
        # Display
        cv2.imshow("Gaze Direction Test", canvas)
        
        # Exit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
    
    webcam.release()
    cv2.destroyAllWindows()
    print("\nTest ended. Gaze tracking working!")

if __name__ == "__main__":
    main()
