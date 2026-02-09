"""
Main module for running player pose analysis
"""

import argparse
import cv2
import sys
from pathlib import Path

from .models.pose_detector import (
    load_pose_model,
    detect_poses_in_frame,
    extract_player_keypoints,
    analyze_player_pose
)
from .utils.video_processor import (
    detect_poses_in_video,
    batch_processing_optimized,
    detect_with_metrics
)


def process_image(input_path: str, output_path: str, model_name: str = "yolov8l-pose.pt", conf: float = 0.5):
    """
    Process a single image
    
    Parameters:
    - input_path: path to input image
    - output_path: path to save result
    - model_name: model name
    - conf: confidence threshold
    """
    print(f"Loading model {model_name}...")
    model = load_pose_model(model_name)
    print("✅ Model loaded!")
    
    print(f"\nProcessing image: {input_path}")
    frame = cv2.imread(input_path)
    
    if frame is None:
        print(f"❌ Error: failed to load image {input_path}")
        return
    
    annotated_frame, poses = detect_poses_in_frame(frame, model, conf=conf)
    cv2.imwrite(output_path, annotated_frame)
    print(f"✅ Found poses: {len(poses)}")
    print(f"✅ Result saved to {output_path}")
    
    # Analyze poses for each player
    keypoints_list = extract_player_keypoints(frame, model, conf=conf)
    for kpt_data in keypoints_list:
        pose_type = analyze_player_pose(kpt_data)
        print(f"Player {kpt_data['player_id']}: {pose_type}")


def process_video(input_path: str, 
                 output_path: str, 
                 model_name: str = "yolov8l-pose.pt",
                 conf: float = 0.5,
                 mode: str = "standard",
                 batch_size: int = 4,
                 save_metrics: bool = False,
                 metrics_path: str = "metrics_log.json"):
    """
    Process video
    
    Parameters:
    - input_path: path to input video
    - output_path: path to save result
    - model_name: model name
    - conf: confidence threshold
    - mode: processing mode ('standard', 'batch', 'metrics')
    - batch_size: batch size (for 'batch' mode)
    - save_metrics: whether to save metrics (for 'metrics' mode)
    - metrics_path: path to save metrics
    """
    print(f"Loading model {model_name}...")
    model = load_pose_model(model_name)
    print("✅ Model loaded!")
    
    print(f"\nProcessing video: {input_path}")
    print(f"Mode: {mode}")
    
    if mode == "standard":
        detect_poses_in_video(input_path, model, output_path, conf=conf)
    elif mode == "batch":
        batch_processing_optimized(input_path, model, output_path, batch_size, conf=conf)
    elif mode == "metrics":
        detect_with_metrics(
            input_path, 
            model, 
            output_path, 
            metrics_path if save_metrics else None,
            conf=conf,
            save_metrics=save_metrics
        )
    else:
        print(f"❌ Unknown mode: {mode}")
        print("Available modes: standard, batch, metrics")


def main():
    parser = argparse.ArgumentParser(
        description="Soccer Player Pose Detection and Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Process image
  python main.py image input.jpg output.jpg
  
  # Process video (standard mode)
  python main.py video input.mp4 output.mp4
  
  # Process video with batch optimization
  python main.py video input.mp4 output.mp4 --mode batch --batch-size 8
  
  # Process video with metrics
  python main.py video input.mp4 output.mp4 --mode metrics --save-metrics
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Parser for image processing
    img_parser = subparsers.add_parser('image', help='Process image')
    img_parser.add_argument('input', help='Path to input image')
    img_parser.add_argument('output', help='Path to save result')
    img_parser.add_argument('--model', default='yolov8l-pose.pt', help='Model name (default: yolov8l-pose.pt)')
    img_parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    
    # Parser for video processing
    vid_parser = subparsers.add_parser('video', help='Process video')
    vid_parser.add_argument('input', help='Path to input video')
    vid_parser.add_argument('output', help='Path to save result')
    vid_parser.add_argument('--model', default='yolov8l-pose.pt', help='Model name (default: yolov8l-pose.pt)')
    vid_parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    vid_parser.add_argument('--mode', choices=['standard', 'batch', 'metrics'], default='standard',
                          help='Processing mode (default: standard)')
    vid_parser.add_argument('--batch-size', type=int, default=4, help='Batch size for batch mode (default: 4)')
    vid_parser.add_argument('--save-metrics', action='store_true', help='Save metrics (for metrics mode)')
    vid_parser.add_argument('--metrics-path', default='metrics_log.json', help='Path to save metrics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'image':
            process_image(args.input, args.output, args.model, args.conf)
        elif args.command == 'video':
            process_video(
                args.input, 
                args.output, 
                args.model, 
                args.conf,
                args.mode,
                args.batch_size,
                args.save_metrics,
                args.metrics_path
            )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
