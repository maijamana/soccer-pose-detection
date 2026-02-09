"""
Module for detecting player poses using YOLOv8-Pose
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional


def load_pose_model(model_name: str = "yolov8l-pose.pt") -> YOLO:
    """
    Loads YOLOv8-Pose model

    Parameters:
    - model_name: model name (nano, small, medium, large, xlarge)
                  Available options: yolov8n-pose.pt, yolov8s-pose.pt,
                  yolov8m-pose.pt, yolov8l-pose.pt, yolov8x-pose.pt

    Returns:
    - model: loaded YOLO model
    """
    model = YOLO(model_name)
    return model


def detect_poses_in_frame(frame: np.ndarray, model: YOLO, conf: float = 0.5) -> Tuple[np.ndarray, List]:
    """
    Detects poses in a single frame

    Parameters:
    - frame: input frame (numpy array)
    - model: loaded YOLO model
    - conf: confidence threshold for detection (default: 0.5)

    Returns:
    - annotated_frame: frame with visualized joint points
    - poses: list of joint coordinates for each detected player
    """
    results = model(frame, conf=conf)

    # Extract joint coordinates
    poses = []

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy  # Coordinates [x, y]

        for pose in keypoints:
            poses.append(pose)

    # Visualization
    annotated_frame = results[0].plot()

    return annotated_frame, poses


def extract_player_keypoints(frame: np.ndarray, model: YOLO, conf: float = 0.5) -> List[Dict]:
    """
    Extracts coordinates of player's main joints

    Parameters:
    - frame: input frame
    - model: loaded YOLO model
    - conf: confidence threshold for detection

    Returns:
    - all_players_kpts: list of dictionaries with data for each player
      Each dictionary contains:
      - player_id: player index
      - keypoints: dictionary with coordinates and confidence for each joint
    """
    results = model(frame, conf=conf)

    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    all_players_kpts = []

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy
        confidences = results[0].keypoints.conf

        for player_idx, (kpt, conf) in enumerate(zip(keypoints, confidences)):
            player_data = {
                'player_id': player_idx,
                'keypoints': {}
            }

            for joint_idx, (point, confidence) in enumerate(zip(kpt, conf)):
                if joint_idx < len(keypoint_names):
                    player_data['keypoints'][keypoint_names[joint_idx]] = {
                        'x': float(point[0]),
                        'y': float(point[1]),
                        'confidence': float(confidence)
                    }

            all_players_kpts.append(player_data)

    return all_players_kpts


def analyze_player_pose(keypoints_dict: Dict) -> str:
    """
    Analyzes player pose based on joints

    Parameters:
    - keypoints_dict: dictionary with player data (result of extract_player_keypoints)

    Returns:
    - pose_type: pose type ('standing', 'kneeling', 'jumping', 'falling', 'unknown')
    """
    kpts = keypoints_dict['keypoints']

    if not kpts:
        return 'unknown'

    # Extract key points
    left_hip = kpts.get('left_hip', {})
    right_hip = kpts.get('right_hip', {})
    left_knee = kpts.get('left_knee', {})
    right_knee = kpts.get('right_knee', {})
    nose = kpts.get('nose', {})

    if not all([left_hip.get('y'), right_hip.get('y'), left_knee.get('y'),
                right_knee.get('y'), nose.get('y')]):
        return 'unknown'

    # Calculate player height (from head to foot)
    hip_y = (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2
    knee_y = (left_knee.get('y', 0) + right_knee.get('y', 0)) / 2
    head_y = nose.get('y', 0)

    # Determine pose type
    if head_y < hip_y:  # Head above hips - standing
        if abs(knee_y - hip_y) > 50:  # Knees below
            return 'standing'
        else:
            return 'kneeling'
    elif hip_y - head_y > 100:  # Head significantly lower
        return 'falling'
    else:
        return 'jumping'
