"""
Examples of using modules for player pose detection
"""

import cv2
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pose_detector import (
    load_pose_model,
    detect_poses_in_frame,
    extract_player_keypoints,
    analyze_player_pose
)
from src.utils.video_processor import (
    detect_poses_in_video,
    batch_processing_optimized,
    detect_with_metrics
)
from src.metrics.calculator import PlayerMetricsCalculator


def example_single_frame():
    """Example of processing a single frame"""
    print("=" * 60)
    print("EXAMPLE 1: Processing a single frame")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    model = load_pose_model("yolov8l-pose.pt")
    print("✅ Model loaded!")
    
    # Load image (replace with your path)
    image_path = "path/to/your/image.jpg"
    
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        print("   Skipping example...")
        return
    
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    # Detect poses
    annotated_frame, poses = detect_poses_in_frame(frame, model)
    print(f"✅ Found poses: {len(poses)}")
    
    # Save result
    output_path = "output_frame_poses.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"✅ Result saved to {output_path}")
    
    # Analyze poses for each player
    keypoints_list = extract_player_keypoints(frame, model)
    for kpt_data in keypoints_list:
        pose_type = analyze_player_pose(kpt_data)
        print(f"Player {kpt_data['player_id']}: {pose_type}")


def example_video_standard():
    """Example of standard video processing"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Video processing (standard mode)")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    model = load_pose_model("yolov8l-pose.pt")
    print("✅ Model loaded!")
    
    # Process video (replace with your path)
    video_path = "path/to/your/video.mp4"
    
    if not Path(video_path).exists():
        print(f"⚠️  Video not found: {video_path}")
        print("   Skipping example...")
        return
    
    output_path = "output_standard.mp4"
    detect_poses_in_video(video_path, model, output_path)
    print(f"✅ Result saved to {output_path}")


def example_video_batch():
    """Example of optimized batch video processing"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Video processing (optimized with batch)")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    model = load_pose_model("yolov8l-pose.pt")
    print("✅ Model loaded!")
    
    # Process video (replace with your path)
    video_path = "path/to/your/video.mp4"
    
    if not Path(video_path).exists():
        print(f"⚠️  Video not found: {video_path}")
        print("   Skipping example...")
        return
    
    output_path = "output_batch.mp4"
    batch_processing_optimized(video_path, model, output_path, batch_size=4)
    print(f"✅ Result saved to {output_path}")


def example_video_with_metrics():
    """Example of video processing with metrics calculation"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Video processing with metrics calculation")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    model = load_pose_model("yolov8l-pose.pt")
    print("✅ Model loaded!")
    
    # Process video (replace with your path)
    video_path = "path/to/your/video.mp4"
    
    if not Path(video_path).exists():
        print(f"⚠️  Video not found: {video_path}")
        print("   Skipping example...")
        return
    
    output_path = "output_with_metrics.mp4"
    metrics_path = "metrics_log.json"
    
    metrics = detect_with_metrics(
        video_path, 
        model, 
        output_path, 
        metrics_path,
        save_metrics=True
    )
    
    print(f"✅ Result saved to {output_path}")
    print(f"✅ Metrics saved to {metrics_path}")
    print(f"✅ Processed {len(metrics)} frames")


def example_metrics_calculation():
    """Example of calculating metrics for a single player"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Player metrics calculation")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    model = load_pose_model("yolov8l-pose.pt")
    print("✅ Model loaded!")
    
    # Load image (replace with your path)
    image_path = "path/to/your/image.jpg"
    
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        print("   Skipping example...")
        return
    
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    # Get keypoints
    keypoints_list = extract_player_keypoints(frame, model)
    
    # Create metrics calculator
    calculator = PlayerMetricsCalculator(frame_rate=30)
    
    # Calculate metrics for each player
    for kpt_data in keypoints_list:
        player_id = kpt_data['player_id']
        kpts = kpt_data['keypoints']
        
        print(f"\n--- Player {player_id} ---")
        print(f"Height: {calculator.calculate_player_height(kpts):.2f} pixels")
        print(f"Step length: {calculator.calculate_step_length(kpts):.2f} pixels")
        print(f"Stance width: {calculator.calculate_stance_width(kpts):.2f} pixels")
        print(f"Trunk angle: {calculator.calculate_trunk_angle(kpts):.2f} degrees")
        print(f"Knee flexion: {calculator.calculate_knee_flex(kpts):.2f} degrees")
        
        is_shooting, shooting_score = calculator.detect_shooting_stance(kpts)
        print(f"Shooting stance: {is_shooting} (score: {shooting_score:.2f})")
        
        is_defensive, defensive_score = calculator.detect_defensive_stance(kpts)
        print(f"Defensive stance: {is_defensive} (score: {defensive_score:.2f})")


if __name__ == "__main__":
    print("Soccer Player Pose Detection - Usage Examples\n")
    
    # Run examples
    example_single_frame()
    example_video_standard()
    example_video_batch()
    example_video_with_metrics()
    example_metrics_calculation()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
