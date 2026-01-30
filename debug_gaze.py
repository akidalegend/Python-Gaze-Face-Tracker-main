"""
Debug version - shows detailed gaze metrics to troubleshoot tracking issues.
Fixed: All y coordinates converted to int() for cv2.putText
"""

import cv2
import numpy as np
import time
from gaze_adapter import GazeAdapter as GazeTracking
from filters import OneEuroFilter

def main():
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    
    if not webcam.isOpened():
        print("Error: Cannot open webcam")
        return
    
    ret, frame = webcam.read()
    if not ret:
        print("Error: Cannot read from webcam")
        return
    
    screen_h, screen_w = frame.shape[:2]
    cv2.namedWindow("DEBUG: Gaze Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DEBUG: Gaze Tracking", 1400, 900)
    
    filter_x = OneEuroFilter(time.time(), 0.5, min_cutoff=0.1, beta=0.05)
    
    print("\n" + "="*60)
    print("DEBUG MODE - Check these values when looking LEFT/CENTER/RIGHT:")
    print("="*60)
    print("\nWatch the 'raw_gaze_x' values:")
    print("  - Looking LEFT  should show values ~0.0-0.3")
    print("  - Looking CENTER should show values ~0.4-0.6")
    print("  - Looking RIGHT should show values ~0.7-1.0")
    print("\nIf values don't match, we need to adjust thresholds or check iris tracking")
    print("="*60 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        
        current_t = time.time()
        frame_count += 1
        
        gaze.refresh(frame)
        
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)
        
        y = 50
        line_height = 60
        
        # Title
        cv2.putText(canvas, "DEBUG: GAZE TRACKING", (50, int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
        y += line_height * 1.5
        
        # Face detection status
        if gaze.pupils_located:
            face_color = (0, 255, 0)
            face_status = "FACE DETECTED"
        else:
            face_color = (0, 0, 255)
            face_status = "NO FACE DETECTED"
        
        cv2.putText(canvas, f"Status: {face_status}", (50, int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, face_color, 2)
        y += line_height
        
        if not gaze.pupils_located:
            cv2.putText(canvas, "Please face camera directly", (50, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("DEBUG: Gaze Tracking", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            continue
        
        # Get all the raw data
        h_ratio = gaze.horizontal_ratio()
        v_ratio = gaze.vertical_ratio()
        is_blinking = gaze.is_blinking()
        
        # Apply smoothing
        smooth_h = filter_x(current_t, h_ratio) if h_ratio else None
        
        # HORIZONTAL RATIO ANALYSIS
        cv2.putText(canvas, "=== HORIZONTAL GAZE ===", (50, int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        y += line_height
        
        if h_ratio is not None:
            # Raw value
            cv2.putText(canvas, f"Raw Horizontal Ratio:      {h_ratio:.4f}", (50, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 1)
            y += line_height
            
            # Smoothed value
            cv2.putText(canvas, f"Smoothed Horizontal Ratio: {smooth_h:.4f}", (50, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 1)
            y += line_height
            
            # Direction prediction
            if smooth_h < 0.40:
                direction = "LOOKING LEFT"
                direction_color = (0, 255, 0)
            elif smooth_h > 0.60:
                direction = "LOOKING RIGHT"
                direction_color = (255, 0, 0)
            else:
                direction = "LOOKING CENTER"
                direction_color = (0, 255, 255)
            
            cv2.putText(canvas, f"Direction: {direction}", (50, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, direction_color, 2)
            y += line_height
            
            # Thresholds used
            cv2.putText(canvas, "Thresholds: LEFT < 0.40 | CENTER 0.40-0.60 | RIGHT > 0.60", (50, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
            y += line_height * 1.5
            
            # Visual bar
            cv2.putText(canvas, "Visual Gauge:", (50, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 1)
            y += 40
            
            bar_y = int(y)
            bar_width = screen_w - 100
            cv2.rectangle(canvas, (50, bar_y), (50 + bar_width, bar_y + 60), (100, 100, 100), -1)
            
            # Draw filled portion
            filled = int(smooth_h * bar_width)
            cv2.rectangle(canvas, (50, bar_y), (50 + filled, bar_y + 60), direction_color, -1)
            
            # Draw reference lines
            left_line = int(0.40 * bar_width)
            right_line = int(0.60 * bar_width)
            center_line = int(0.50 * bar_width)
            
            cv2.line(canvas, (50 + left_line, bar_y - 10), (50 + left_line, bar_y + 70), (150, 150, 150), 2)
            cv2.line(canvas, (50 + right_line, bar_y - 10), (50 + right_line, bar_y + 70), (150, 150, 150), 2)
            cv2.line(canvas, (50 + center_line, bar_y - 10), (50 + center_line, bar_y + 70), (100, 100, 200), 2)
            
            cv2.putText(canvas, "L", (50 + left_line - 10, bar_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            cv2.putText(canvas, "C", (50 + center_line - 10, bar_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            cv2.putText(canvas, "R", (50 + right_line - 10, bar_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            
            y = bar_y + 100
        else:
            cv2.putText(canvas, "h_ratio = None (eyes not tracked)", (50, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
            y += line_height
        
        # VERTICAL RATIO
        y += 20
        cv2.putText(canvas, "=== VERTICAL GAZE ===", (50, int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        y += line_height
        
        if v_ratio is not None:
            cv2.putText(canvas, f"Vertical Ratio: {v_ratio:.4f}", (50, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 1)
            y += line_height
            
            if v_ratio < 0.40:
                v_direction = "LOOKING UP"
                v_color = (0, 255, 0)
            elif v_ratio > 0.60:
                v_direction = "LOOKING DOWN"
                v_color = (255, 0, 0)
            else:
                v_direction = "LOOKING STRAIGHT"
                v_color = (0, 255, 255)
            
            cv2.putText(canvas, f"Direction: {v_direction}", (50, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, v_color, 2)
        else:
            cv2.putText(canvas, "v_ratio = None", (50, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        
        y += line_height * 1.5
        
        # BLINK DETECTION
        cv2.putText(canvas, "=== BLINK DETECTION ===", (50, int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        y += line_height
        
        blink_color = (0, 0, 255) if is_blinking else (0, 255, 0)
        blink_text = "BLINKING" if is_blinking else "Eyes Open"
        cv2.putText(canvas, f"Status: {blink_text}", (50, int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, blink_color, 2)
        
        # Footer
        cv2.putText(canvas, f"Frame: {frame_count} | Press 'q' to exit", 
                   (50, screen_h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        cv2.imshow("DEBUG: Gaze Tracking", canvas)
        
        # Print to console for easy reference
        if frame_count % 30 == 0:  # Every 30 frames (~1 second at 30fps)
            print(f"Frame {frame_count}: h_ratio={h_ratio:.3f} smooth={smooth_h:.3f} v_ratio={v_ratio:.3f} blink={is_blinking}")
        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
    
    webcam.release()
    cv2.destroyAllWindows()
    print("\nDebug mode ended.")

if __name__ == "__main__":
    main()
