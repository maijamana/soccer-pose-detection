"""
Module for video processing with pose detection and metrics calculation
"""

import cv2
import json
import numpy as np
from collections import deque
from typing import List, Dict, Optional
from ultralytics import YOLO

from ..models.pose_detector import extract_player_keypoints, analyze_player_pose
from ..metrics.calculator import PlayerMetricsCalculator


def detect_poses_in_video(video_path: str, 
                         model: YOLO, 
                         output_path: str = "output_poses.mp4",
                         conf: float = 0.5,
                         verbose: bool = True) -> None:
    """
    Detects player poses in video

    Parameters:
    - video_path: path to input video
    - model: loaded YOLO model
    - output_path: path to save result
    - conf: confidence threshold for detection
    - verbose: whether to print processing progress
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Video parameters
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writer for saving
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, conf=conf, verbose=False)

        # Visualize results
        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        frame_count += 1

        if verbose and frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    
    if verbose:
        print(f"Results saved to {output_path}")


def batch_processing_optimized(video_path: str,
                               model: YOLO,
                               output_path: str = "output_optimized.mp4",
                               batch_size: int = 4,
                               conf: float = 0.5,
                               verbose: bool = True) -> None:
    """
    Optimized batch processing for increased speed

    Parameters:
    - video_path: path to input video
    - model: loaded YOLO model
    - output_path: path to save result
    - batch_size: batch size for processing (recommended 4-8)
    - conf: confidence threshold for detection
    - verbose: whether to print processing progress
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_queue = deque(maxlen=batch_size)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_queue.append(frame)

        if len(frame_queue) == batch_size:
            results = model(list(frame_queue), conf=conf, verbose=False)

            for result in results:
                annotated_frame = result.plot()
                out.write(annotated_frame)
                frame_count += 1

                if verbose and frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")

    # Process remaining frames
    if len(frame_queue) > 0:
        results = model(list(frame_queue), conf=conf, verbose=False)
        for result in results:
            annotated_frame = result.plot()
            out.write(annotated_frame)
            frame_count += 1

    cap.release()
    out.release()
    
    if verbose:
        print(f"✅ Optimized processing completed: {frame_count} frames processed")
        print(f"   Results saved to {output_path}")


def detect_with_metrics(video_path: str,
                        model: YOLO,
                        output_path: str = "output_with_metrics.mp4",
                        metrics_output_path: Optional[str] = "metrics_log.json",
                        conf: float = 0.5,
                        frame_rate: int = 30,
                        save_metrics: bool = True,
                        verbose: bool = True) -> List[Dict]:
    """
    Detection with simultaneous metrics calculation

    Parameters:
    - video_path: path to input video
    - model: loaded YOLO model
    - output_path: path to save annotated video
    - metrics_output_path: path to save metrics (JSON)
    - conf: confidence threshold for detection
    - frame_rate: video frame rate
    - save_metrics: whether to save metrics to file
    - verbose: whether to print processing progress

    Returns:
    - metrics_log: list of dictionaries with metrics for each frame
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    calculator = PlayerMetricsCalculator(frame_rate=frame_rate)
    metrics_log = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pose detection
        results = model(frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot()

        # Calculate metrics
        keypoints_list = extract_player_keypoints(frame, model, conf=conf)
        frame_metrics = {
            'frame': frame_count,
            'players': []
        }

        for kpt_data in keypoints_list:
            player_id = kpt_data['player_id']
            kpts = kpt_data['keypoints']

            player_metrics = {
                'player_id': player_id,
                'pose_type': analyze_player_pose(kpt_data),
                'height': calculator.calculate_player_height(kpts),
                'step_length': calculator.calculate_step_length(kpts),
                'stance_width': calculator.calculate_stance_width(kpts),
                'trunk_angle': calculator.calculate_trunk_angle(kpts),
                'knee_flex': calculator.calculate_knee_flex(kpts),
            }

            is_shooting, shooting_score = calculator.detect_shooting_stance(kpts)
            player_metrics['shooting_stance'] = is_shooting
            player_metrics['shooting_score'] = shooting_score

            is_defensive, defensive_score = calculator.detect_defensive_stance(kpts)
            player_metrics['defensive_stance'] = is_defensive
            player_metrics['defensive_score'] = defensive_score

            frame_metrics['players'].append(player_metrics)

        metrics_log.append(frame_metrics)

        out.write(annotated_frame)
        frame_count += 1

        if verbose and frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()

    if save_metrics and metrics_output_path:
        with open(metrics_output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_log, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"✅ Metrics saved to {metrics_output_path}")

    if verbose:
        print(f"Results saved to {output_path}")
    
    return metrics_log
