# Soccer Player Pose Detection and Analytics

Professional project for detecting poses and analyzing players in soccer videos using YOLOv8-Pose.

## Description

This project provides tools for:
- Detecting player poses in videos and images
- Extracting keypoints (joints) of players
- Analyzing pose types (standing, kneeling, jumping, falling)
- Calculating player metrics (height, step length, trunk angle, etc.)
- Detecting special poses (shooting, defensive stance)
- Optimized video processing with batch mode

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for fast processing)

### Step 1: Navigate to project directory

```bash
cd "/Users/mac/Desktop/soccer-pose-detection"
```

### Step 2: Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

#### Process image

```bash
python run.py image input.jpg output.jpg
```

Or from root directory:

```bash
python -m src.main image input.jpg output.jpg
```

With additional parameters:

```bash
python run.py image input.jpg output.jpg --model yolov8n-pose.pt --conf 0.6
```

**Important:** If your file paths contain spaces, wrap them in quotes:
```bash
python run.py image "/path/to/your image.jpg" output.jpg
```

#### Process video (standard mode)

```bash
python run.py video input.mp4 output.mp4
```

#### Process video with batch optimization

```bash
python run.py video input.mp4 output.mp4 --mode batch --batch-size 8
```

#### Process video with metrics

```bash
python run.py video input.mp4 output.mp4 --mode metrics --save-metrics --metrics-path metrics.json
```

### Programmatic Interface

#### Example 1: Process single frame

```python
import cv2
from src.models.pose_detector import load_pose_model, detect_poses_in_frame

# Load model
model = load_pose_model("yolov8l-pose.pt")

# Load image
frame = cv2.imread("input.jpg")

# Detect poses
annotated_frame, poses = detect_poses_in_frame(frame, model)

# Save result
cv2.imwrite("output.jpg", annotated_frame)
print(f"Found poses: {len(poses)}")
```

#### Example 2: Process video

```python
from src.models.pose_detector import load_pose_model
from src.utils.video_processor import detect_poses_in_video

# Load model
model = load_pose_model("yolov8l-pose.pt")

# Process video
detect_poses_in_video("input.mp4", model, "output.mp4")
```

#### Example 3: Calculate metrics

```python
from src.models.pose_detector import load_pose_model, extract_player_keypoints
from src.metrics.calculator import PlayerMetricsCalculator

# Load model
model = load_pose_model("yolov8l-pose.pt")

# Get keypoints
keypoints_list = extract_player_keypoints(frame, model)

# Calculate metrics
calculator = PlayerMetricsCalculator(frame_rate=30)

for kpt_data in keypoints_list:
    kpts = kpt_data['keypoints']
    
    height = calculator.calculate_player_height(kpts)
    step_length = calculator.calculate_step_length(kpts)
    trunk_angle = calculator.calculate_trunk_angle(kpts)
    
    is_shooting, score = calculator.detect_shooting_stance(kpts)
    
    print(f"Height: {height}, Trunk angle: {trunk_angle}")
    print(f"Shooting stance: {is_shooting} (score: {score})")
```

More examples can be found in `examples/example_usage.py`.

## Project Structure

```
soccer-pose-detection/
├── README.md                 # Documentation
├── QUICKSTART.md            # Quick start guide
├── requirements.txt         # Python dependencies
├── run.py                   # Entry point (run from root)
├── .gitignore               # Git ignore file
├── src/                     # Main code
│   ├── __init__.py
│   ├── main.py             # CLI module
│   ├── models/             # Model modules
│   │   ├── __init__.py
│   │   └── pose_detector.py # Pose detection
│   ├── metrics/            # Metrics modules
│   │   ├── __init__.py
│   │   └── calculator.py   # Metrics calculation
│   └── utils/              # Utilities
│       ├── __init__.py
│       └── video_processor.py # Video processing
├── examples/               # Usage examples
│   └── example_usage.py
└── config/                 # Configuration files (optional)
```

## Parameters and Settings

### YOLOv8-Pose Models

Available models (from fastest to most accurate):
- `yolov8n-pose.pt` - Nano (fastest)
- `yolov8s-pose.pt` - Small
- `yolov8m-pose.pt` - Medium
- `yolov8l-pose.pt` - Large (default)
- `yolov8x-pose.pt` - XLarge (most accurate)

### Confidence Threshold (conf)

- **0.3-0.4**: More detections, but possible errors
- **0.5**: Balanced option (default)
- **0.6-0.7**: Fewer detections, but higher accuracy

### Batch Size

For `batch` mode:
- **2-4**: For weaker GPUs
- **4-8**: Recommended for most systems
- **8-16**: For powerful GPUs

## Metrics

The project calculates the following metrics:

- **Player Height**: Distance from nose to ankle
- **Step Length**: Distance between ankles
- **Stance Width**: Distance between hips
- **Trunk Angle**: Trunk tilt relative to vertical
- **Knee Flexion Angle**: Angle between thigh, knee, and ankle
- **Pose Type**: standing, kneeling, jumping, falling
- **Shooting Stance**: Detection of shooting pose
- **Defensive Stance**: Detection of defensive stance

## Performance Optimization

### For faster processing:

1. **Use GPU**: Model automatically uses CUDA if available
2. **Batch processing**: Use `--mode batch` for videos
3. **Smaller model**: Use `yolov8n-pose.pt` or `yolov8s-pose.pt`
4. **Reduce resolution**: Crop or reduce video before processing

### Expected performance (on GPU):

- **yolov8n-pose**: ~100-150 FPS
- **yolov8s-pose**: ~60-80 FPS
- **yolov8l-pose**: ~30-40 FPS (default)
- **yolov8x-pose**: ~20-30 FPS

## Metrics Format (JSON)

When saving metrics to a file, the structure looks like this:

```json
[
  {
    "frame": 0,
    "players": [
      {
        "player_id": 0,
        "pose_type": "standing",
        "height": 450.5,
        "step_length": 120.3,
        "stance_width": 80.2,
        "trunk_angle": 5.4,
        "knee_flex": 165.2,
        "shooting_stance": false,
        "shooting_score": 0.3,
        "defensive_stance": true,
        "defensive_score": 0.8
      }
    ]
  }
]
```

## Troubleshooting

### Error: "CUDA out of memory"

- Reduce `batch_size`
- Use smaller model (yolov8n or yolov8s)
- Process video in parts

### Slow processing

- Make sure GPU is being used
- Use batch mode
- Reduce video resolution

### Low detection accuracy

- Increase `conf` to 0.6-0.7
- Use larger model (yolov8l or yolov8x)
- Make sure video has sufficient quality

## License

This project uses YOLOv8 from Ultralytics, which has its own license. Check [Ultralytics License](https://github.com/ultralytics/ultralytics) for details.

## Contributing

Pull requests and issues are welcome. For major changes, please open an issue first to discuss.

## Contact

For questions and suggestions, create an issue in the repository.

---

**Note**: The first run will automatically download the YOLOv8-Pose model. The model will be saved locally for future use.
