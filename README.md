# Landmine Detection Final

A deep learning project for detecting landmines in thermal imagery using YOLO (You Only Look Once) object detection models.

## Purpose/Description

This project implements computer vision techniques to detect landmines from thermal drone imagery. It uses the YOLO architecture to identify and localize landmines in images captured by drones equipped with thermal cameras. The system supports multiple YOLO versions (v5, v8, v11) and includes training, validation, and ensemble evaluation capabilities.

## Tech Stack

- **Python** - Programming language
- **Ultralytics YOLO** - Object detection framework
- **PyTorch** - Deep learning framework
- **OpenCV** - Image processing library
- **PyCocoTools** - COCO evaluation metrics
- **Jupyter Notebook** - Interactive development environment
- **NumPy/Pandas** - Data manipulation

## Project Structure

```
landmine_detection_final/
├── landmine_detection(1).ipynb    # Main training notebook
├── kaggle.json                     # Kaggle API credentials
├── .gitignore
├── models/                          # Trained model weights
│   ├── yoloV5_best.pt              # Best YOLOv5 model
│   ├── yoloV8_best.pt              # Best YOLOv8 model
│   ├── yoloV11_best.pt             # Best YOLOv11 model
│   └── augmented_yolo_v11.pt       # Augmented YOLOv11 model
├── results/                         # Training and validation results
│   ├── yolov5/
│   │   └── test/
│   │       └── labels/             # YOLO format predictions
│   ├── yolov8/
│   │   ├── train/                  # Training metrics and visualizations
│   │   └── val/                    # Validation metrics
│   └── yolov11/
│       ├── train/                  # Training metrics and visualizations
│       └── val/                    # Validation metrics
├── sample_Data/                    # Sample test images
│   ├── DM-11_19_png_jpg.rf.*.jpg
│   ├── DM-11_43_png.rf.*.jpg
│   └── ...
└── scripts/                        # Utility scripts
    ├── download_data.py            # Download dataset from Roboflow
    ├── test_yolov5.py              # YOLOv5 evaluation script
    ├── test_yolov8.py              # YOLOv8 evaluation script
    ├── test_yolov11.py             # YOLOv11 evaluation script
    └── evaluate_ensemble.py        # Ensemble model evaluation
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Kaggle account (for dataset download)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd landmine_detection_final

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install ultralytics
pip install pycocotools
pip install opencv-python
pip install numpy pandas matplotlib

# For Kaggle dataset download
pip install kaggle
# Place kaggle.json in ~/.kaggle/ or project root
```

### Dataset Download

```bash
# Using the provided script
python scripts/download_data.py

# Or using Kaggle API directly
kaggle datasets download -d <dataset-name>
unzip <dataset-name>.zip -d data/
```

## Usage

### Training

Open and run the Jupyter notebook:

```bash
jupyter notebook landmine_detection\(1\).ipynb
```

The notebook includes:
- Dataset loading and preprocessing
- Model initialization (YOLOv5, v8, v11)
- Training configuration
- Training execution
- Model evaluation

### Testing Models

```bash
# Test YOLOv5
python scripts/test_yolov5.py

# Test YOLOv8
python scripts/test_yolov8.py

# Test YOLOv11
python scripts/test_yolov11.py

# Evaluate ensemble
python scripts/evaluate_ensemble.py
```

### Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('models/yoloV11_best.pt')

# Run inference on an image
results = model.predict(source='path/to/image.jpg', conf=0.25)

# Access results
for result in results:
    boxes = result.boxes
    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, BBox: {box.xyxy}")
```

## Model Performance

Results are stored in the `results/` directory with:
- **Confusion matrices** (`confusion_matrix.png`, `confusion_matrix_normalized.png`)
- **Precision-Recall curves** (`PR_curve.png`, `P_curve.png`, `R_curve.png`)
- **F1 curves** (`F1_curve.png`)
- **Training metrics** (`results.csv`, `results.png`)
- **Sample predictions** (`train_batch*.jpg`, `val_batch*.jpg`)

### Evaluation Metrics

The evaluation includes:
- **AP** - Average Precision
- **AP50/AP75** - AP at IoU thresholds
- **AR** - Average Recall
- **AP/AR for small, medium, large objects**

## Data Format

### Dataset Structure
```
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### YOLO Label Format
Each `.txt` file corresponds to an image with format:
```
class_id x_center y_center width height
```
Values are normalized (0-1).

## Features

- Multi-version YOLO support (v5, v8, v11)
- Thermal imagery landmine detection
- Training visualization and metrics tracking
- COCO evaluation metrics integration
- Ensemble model evaluation
- Support for custom datasets from Roboflow

## Use Cases

- Humanitarian demining operations
- Military landmine detection
- Post-conflict area surveying
- Drone-based surveillance
- Safety assessment for construction/excavation projects