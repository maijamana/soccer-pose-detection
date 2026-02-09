# Soccer Player Pose Detection and Analytics
Professional project for detecting poses and analyzing players in soccer videos using YOLOv8-Pose.

## Demo
![Pose detection demo](data/photo.jpeg)
![Pose Detection GIF](data/video.gif)

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

## License

This project uses YOLOv8 from Ultralytics, which has its own license. Check [Ultralytics License](https://github.com/ultralytics/ultralytics) for details.

---

**Note**: The first run will automatically download the YOLOv8-Pose model. The model will be saved locally for future use.
