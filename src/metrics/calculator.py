"""
Module for calculating metrics and analyzing player poses
"""

import numpy as np
from typing import Dict, Optional, Tuple


class PlayerMetricsCalculator:
    """
    Class for calculating various player metrics based on keypoints
    """

    def __init__(self, frame_rate: int = 30):
        """
        Initialize metrics calculator

        Parameters:
        - frame_rate: video frame rate (for speed calculation)
        """
        self.frame_rate = frame_rate

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculates Euclidean distance between two points

        Parameters:
        - point1: coordinates of first point (x, y)
        - point2: coordinates of second point (x, y)

        Returns:
        - distance: distance between points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """
        Calculates angle between three points (point2 - angle vertex)

        Parameters:
        - point1: first point
        - point2: angle vertex
        - point3: third point

        Returns:
        - angle: angle in degrees
        """
        v1 = np.array(point1) - np.array(point2)
        v2 = np.array(point3) - np.array(point2)

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
        return np.degrees(angle_rad)

    def calculate_player_height(self, keypoints: Dict) -> Optional[float]:
        """
        Calculates player height (from nose to ankle)

        Parameters:
        - keypoints: dictionary with player keypoints

        Returns:
        - height: player height in pixels, or None if data unavailable
        """
        nose = keypoints.get('nose', {})
        left_ankle = keypoints.get('left_ankle', {})
        right_ankle = keypoints.get('right_ankle', {})

        if nose and (left_ankle or right_ankle):
            ankle = left_ankle if left_ankle else right_ankle
            height = self.calculate_distance(
                (nose['x'], nose['y']),
                (ankle['x'], ankle['y'])
            )
            return height
        return None

    def calculate_step_length(self, keypoints: Dict) -> Optional[float]:
        """
        Calculates step length (distance between ankles)

        Parameters:
        - keypoints: dictionary with player keypoints

        Returns:
        - step_length: step length in pixels, or None
        """
        left_ankle = keypoints.get('left_ankle', {})
        right_ankle = keypoints.get('right_ankle', {})

        if left_ankle and right_ankle:
            step = self.calculate_distance(
                (left_ankle['x'], left_ankle['y']),
                (right_ankle['x'], right_ankle['y'])
            )
            return step
        return None

    def calculate_stance_width(self, keypoints: Dict) -> Optional[float]:
        """
        Calculates stance width (distance between hips)

        Parameters:
        - keypoints: dictionary with player keypoints

        Returns:
        - width: stance width in pixels, or None
        """
        left_hip = keypoints.get('left_hip', {})
        right_hip = keypoints.get('right_hip', {})

        if left_hip and right_hip:
            width = self.calculate_distance(
                (left_hip['x'], left_hip['y']),
                (right_hip['x'], right_hip['y'])
            )
            return width
        return None

    def calculate_trunk_angle(self, keypoints: Dict) -> Optional[float]:
        """
        Calculates trunk tilt angle

        Parameters:
        - keypoints: dictionary with player keypoints

        Returns:
        - angle: trunk tilt angle in degrees, or None
        """
        left_shoulder = keypoints.get('left_shoulder', {})
        right_shoulder = keypoints.get('right_shoulder', {})
        left_hip = keypoints.get('left_hip', {})
        right_hip = keypoints.get('right_hip', {})

        if left_shoulder and left_hip:
            shoulder_mid = (
                (left_shoulder['x'] + right_shoulder.get('x', left_shoulder['x'])) / 2,
                (left_shoulder['y'] + right_shoulder.get('y', left_shoulder['y'])) / 2
            )
            hip_mid = (
                (left_hip['x'] + right_hip.get('x', left_hip['x'])) / 2,
                (left_hip['y'] + right_hip.get('y', left_hip['y'])) / 2
            )

            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            angle = np.degrees(np.arctan2(dx, dy))
            return abs(angle)
        return None

    def calculate_knee_flex(self, keypoints: Dict) -> Optional[float]:
        """
        Calculates knee flexion angle

        Parameters:
        - keypoints: dictionary with player keypoints

        Returns:
        - angle: average knee flexion angle in degrees, or None
        """
        left_hip = keypoints.get('left_hip', {})
        left_knee = keypoints.get('left_knee', {})
        left_ankle = keypoints.get('left_ankle', {})

        angles = []

        if left_hip and left_knee and left_ankle:
            angle = self.calculate_angle(
                (left_hip['x'], left_hip['y']),
                (left_knee['x'], left_knee['y']),
                (left_ankle['x'], left_ankle['y'])
            )
            angles.append(angle)

        right_hip = keypoints.get('right_hip', {})
        right_knee = keypoints.get('right_knee', {})
        right_ankle = keypoints.get('right_ankle', {})

        if right_hip and right_knee and right_ankle:
            angle = self.calculate_angle(
                (right_hip['x'], right_hip['y']),
                (right_knee['x'], right_knee['y']),
                (right_ankle['x'], right_ankle['y'])
            )
            angles.append(angle)

        return np.mean(angles) if angles else None

    def detect_shooting_stance(self, keypoints: Dict) -> Tuple[bool, float]:
        """
        Detects shooting stance

        Parameters:
        - keypoints: dictionary with player keypoints

        Returns:
        - is_shooting_stance: whether this is a shooting stance
        - score: confidence score (0.0 - 1.0)
        """
        knee_flex = self.calculate_knee_flex(keypoints)
        trunk_angle = self.calculate_trunk_angle(keypoints)

        if knee_flex is None or trunk_angle is None:
            return False, 0.0

        score = 0.0

        if 90 < knee_flex < 160:
            score += 0.4

        if trunk_angle > 10:
            score += 0.3

        left_elbow = keypoints.get('left_elbow', {})
        right_elbow = keypoints.get('right_elbow', {})
        nose = keypoints.get('nose', {})

        if left_elbow and nose:
            elbow_to_nose = self.calculate_distance(
                (left_elbow['x'], left_elbow['y']),
                (nose['x'], nose['y'])
            )
            if elbow_to_nose > 20:
                score += 0.3

        is_shooting_stance = score > 0.6
        return is_shooting_stance, score

    def detect_defensive_stance(self, keypoints: Dict) -> Tuple[bool, float]:
        """
        Detects defensive stance

        Parameters:
        - keypoints: dictionary with player keypoints

        Returns:
        - is_defensive: whether this is a defensive stance
        - score: confidence score (0.0 - 1.0)
        """
        stance_width = self.calculate_stance_width(keypoints)
        trunk_angle = self.calculate_trunk_angle(keypoints)

        if stance_width is None:
            return False, 0.0

        score = 0.0

        if stance_width > 100:
            score += 0.5

        if trunk_angle is not None and trunk_angle < 20:
            score += 0.5

        is_defensive = score > 0.7
        return is_defensive, score
