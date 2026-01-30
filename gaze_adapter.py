import cv2
import mediapipe as mp
import numpy as np

class GazeAdapter:
    """
    Adapts the MediaPipe Face Mesh to the API expected by run_calibration.py.
    Based on the original Python-Gaze-Face-Tracker approach.
    """
    
    # Landmark indices (from original main.py)
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    L_H_LEFT = 33   # Left eye left corner
    L_H_RIGHT = 133  # Left eye right corner
    R_H_LEFT = 362   # Right eye left corner
    R_H_RIGHT = 263  # Right eye right corner
    
    # Blink detection landmarks
    RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
    LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
    
    BLINK_THRESHOLD = 0.51
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarks = None
        self.frame = None
        self.frame_shape = (480, 640)
        self._blinking = False
        
        # Store calculated values
        self._left_iris_pos = None
        self._right_iris_pos = None
        self._left_eye_center = None
        self._right_eye_center = None

    def refresh(self, frame):
        """Processes the frame to find face landmarks."""
        self.frame = frame
        self.frame_shape = frame.shape[:2]  # h, w
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0]
            self._calculate_iris_positions()
            self._detect_blink()
        else:
            self.landmarks = None
            self._blinking = False
            self._left_iris_pos = None
            self._right_iris_pos = None

    @property
    def pupils_located(self):
        return self.landmarks is not None and self._left_iris_pos is not None

    def is_blinking(self):
        return self._blinking

    def _get_landmark_coords(self, index):
        """Get (x, y) pixel coordinates for a landmark."""
        if not self.landmarks:
            return None
        h, w = self.frame_shape
        lm = self.landmarks.landmark[index]
        return np.array([lm.x * w, lm.y * h])

    def _get_iris_center(self, iris_indices):
        """Calculate iris center from 4 iris landmarks (original approach)."""
        if not self.landmarks:
            return None
        
        h, w = self.frame_shape
        points = []
        for idx in iris_indices:
            lm = self.landmarks.landmark[idx]
            points.append([lm.x * w, lm.y * h])
        
        # Average of all 4 iris points = center
        return np.mean(points, axis=0)

    def _calculate_iris_positions(self):
        """Calculate iris positions relative to eye corners (original approach)."""
        if not self.landmarks:
            return
        
        h, w = self.frame_shape
        
        # Get eye corners
        l_left = self._get_landmark_coords(self.L_H_LEFT)
        l_right = self._get_landmark_coords(self.L_H_RIGHT)
        r_left = self._get_landmark_coords(self.R_H_LEFT)
        r_right = self._get_landmark_coords(self.R_H_RIGHT)
        
        # Get iris centers (using all 4 iris points like original)
        left_iris = self._get_iris_center(self.LEFT_IRIS)
        right_iris = self._get_iris_center(self.RIGHT_IRIS)
        
        if any(x is None for x in [l_left, l_right, r_left, r_right, left_iris, right_iris]):
            return
        
        # Calculate eye centers
        self._left_eye_center = (l_left + l_right) / 2
        self._right_eye_center = (r_left + r_right) / 2
        
        # Calculate relative iris position (dx, dy) from eye center
        # This is similar to the original vector_position approach
        self._left_iris_pos = left_iris - self._left_eye_center
        self._right_iris_pos = right_iris - self._right_eye_center
        
        # Calculate eye widths for normalization
        self._left_eye_width = np.linalg.norm(l_right - l_left)
        self._right_eye_width = np.linalg.norm(r_right - r_left)

    def _euclidean_distance_3D(self, eye_points_indices):
        """Calculate eye aspect ratio for blink detection (from original)."""
        if not self.landmarks:
            return 1.0
        
        points = []
        for idx in eye_points_indices:
            lm = self.landmarks.landmark[idx]
            points.append(np.array([lm.x, lm.y, lm.z]))
        
        P0, P3, P4, P5, P8, P11, P12, P13 = points
        
        numerator = (
            np.linalg.norm(P3 - P13) ** 3
            + np.linalg.norm(P4 - P12) ** 3
            + np.linalg.norm(P5 - P11) ** 3
        )
        denominator = 3 * np.linalg.norm(P0 - P8) ** 3
        
        if denominator < 1e-6:
            return 1.0
        
        return numerator / denominator

    def _detect_blink(self):
        """Blink detection using EAR (Eye Aspect Ratio) from original."""
        if not self.landmarks:
            self._blinking = False
            return
        
        left_ear = self._euclidean_distance_3D(self.LEFT_EYE_POINTS)
        right_ear = self._euclidean_distance_3D(self.RIGHT_EYE_POINTS)
        avg_ear = (left_ear + right_ear) / 2
        
        self._blinking = avg_ear < self.BLINK_THRESHOLD

    def horizontal_ratio(self):
        """
        Returns horizontal gaze ratio (0.0 = looking left, 1.0 = looking right).
        Uses iris position relative to eye center, normalized by eye width.
        """
        if not self.pupils_located:
            return None
        
        # Average the normalized horizontal offset from both eyes
        left_ratio = 0.5 + (self._left_iris_pos[0] / self._left_eye_width) if self._left_eye_width > 0 else 0.5
        right_ratio = 0.5 + (self._right_iris_pos[0] / self._right_eye_width) if self._right_eye_width > 0 else 0.5
        
        ratio = (left_ratio + right_ratio) / 2
        return max(0.0, min(1.0, ratio))

    def vertical_ratio(self):
        """
        Returns vertical gaze ratio (0.0 = looking up, 1.0 = looking down).
        Uses iris position relative to eye center.
        """
        if not self.pupils_located:
            return None
        
        # Use eye width as approximate scale for vertical too
        # (vertical range is smaller than horizontal)
        left_ratio = 0.5 + (self._left_iris_pos[1] / (self._left_eye_width * 0.5)) if self._left_eye_width > 0 else 0.5
        right_ratio = 0.5 + (self._right_iris_pos[1] / (self._right_eye_width * 0.5)) if self._right_eye_width > 0 else 0.5
        
        ratio = (left_ratio + right_ratio) / 2
        return max(0.0, min(1.0, ratio))

    def annotated_frame(self):
        """Returns frame with eye tracking visualization."""
        if self.frame is None:
            return np.zeros((self.frame_shape[0], self.frame_shape[1], 3), dtype=np.uint8)
        
        annotated = self.frame.copy()
        
        if self.landmarks and self._left_eye_center is not None:
            # Draw eye centers
            cv2.circle(annotated, tuple(self._left_eye_center.astype(int)), 3, (255, 0, 0), -1)
            cv2.circle(annotated, tuple(self._right_eye_center.astype(int)), 3, (255, 0, 0), -1)
            
            # Draw iris centers
            left_iris = self._left_eye_center + self._left_iris_pos
            right_iris = self._right_eye_center + self._right_iris_pos
            cv2.circle(annotated, tuple(left_iris.astype(int)), 3, (0, 255, 0), -1)
            cv2.circle(annotated, tuple(right_iris.astype(int)), 3, (0, 255, 0), -1)
            
            # Draw eye corners
            for idx in [self.L_H_LEFT, self.L_H_RIGHT, self.R_H_LEFT, self.R_H_RIGHT]:
                pt = self._get_landmark_coords(idx)
                if pt is not None:
                    cv2.circle(annotated, tuple(pt.astype(int)), 2, (0, 0, 255), -1)
        
        return annotated