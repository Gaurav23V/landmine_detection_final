import sys
import os
from pathlib import Path
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob

import torch

# Clone YOLOv5 repo if not already present
YOLOV5_PATH = "yolov5"
if not os.path.exists(YOLOV5_PATH):
    os.system("git clone https://github.com/ultralytics/yolov5.git")
sys.path.append(YOLOV5_PATH)

def predict_and_save_labels(model_path, test_images_path, output_dir, conf_threshold=0.25):
    """
    Function to predict and save bounding box coordinates in txt files using YOLOv5.

    Args:
        model_path (str): Path to the YOLOv5 model weights.
        test_images_path (str): Path to test images directory.
        output_dir (str): Path to save prediction files.
        conf_threshold (float): Confidence threshold for predictions.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv5 model
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local', device='cpu')
    model.conf = conf_threshold  # Set confidence threshold

    # Collect all image paths
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_images_path, ext)))

    if not image_paths:
        print("No images found in the specified test_images_path.")
        return

    # Run inference
    results = model(image_paths, size=640)  # Adjust size as needed

    # Save predictions
    results.save()  # Saves to 'runs/detect/exp' by default
    # Move saved txt files to desired output_dir
    save_txt_dir = os.path.join('runs', 'detect', 'exp', 'labels')
    if os.path.exists(save_txt_dir):
        for txt_file in glob.glob(os.path.join(save_txt_dir, '*.txt')):
            Path(txt_file).rename(os.path.join(output_dir, Path(txt_file).name))
    else:
        print("No prediction txt files found to move.")

def evaluate_metrics(ground_truth_dir, predicted_dir):
    """
    Function to evaluate object detection metrics and print them to the terminal.

    Args:
        ground_truth_dir (str): Path to ground truth labels.
        predicted_dir (str): Path to predicted labels.
    """
    # Convert YOLO format to COCO format for ground truth
    def convert_yolo_to_coco_gt(label_dir, images_dir):
        coco = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        category_set = set()
        annotation_id = 1

        for img_id, image_path in enumerate(glob.glob(os.path.join(images_dir, '*.jpg')), 1):
            image_name = Path(image_path).stem
            coco["images"].append({
                "id": img_id,
                "file_name": os.path.basename(image_path)
            })

            label_path = os.path.join(label_dir, f"{image_name}.txt")
            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id, x_center, y_center, width, height = map(float, parts)
                    category_set.add(int(class_id))
                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": int(class_id),
                        "bbox": [
                            x_center - width / 2,
                            y_center - height / 2,
                            width,
                            height
                        ],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

        for category_id in sorted(category_set):
            coco["categories"].append({
                "id": category_id,
                "name": f"class_{category_id}",
                "supercategory": "none"
            })

        return coco

    # Convert YOLO format to COCO predictions format
    def convert_yolo_to_coco_pred(label_dir, images_dir, image_id_map):
        predictions = []

        for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
            image_name = Path(label_file).stem
            img_id = image_id_map.get(image_name)
            if img_id is None:
                continue

            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id, x_center, y_center, width, height = map(float, parts)
                    # In YOLOv5, confidence scores are not saved in txt files, set to 1.0
                    score = 1.0
                    bbox = [
                        x_center - width / 2,
                        y_center - height / 2,
                        width,
                        height
                    ]
                    predictions.append({
                        "image_id": img_id,
                        "category_id": int(class_id),
                        "bbox": bbox,
                        "score": score
                    })

        return predictions

    # Paths
    images_dir = "data/test/images"
    ground_truth_coco = convert_yolo_to_coco_gt(ground_truth_dir, images_dir)

    # Create a mapping from image name to image_id
    image_id_map = {Path(img["file_name"]).stem: img["id"] for img in ground_truth_coco["images"]}

    predictions = convert_yolo_to_coco_pred(predicted_dir, images_dir, image_id_map)

    if not predictions:
        print("No predictions found. Please ensure that the predicted labels are correctly generated.")
        return

    # Save ground truth COCO annotations to a temporary JSON file
    gt_coco_path = "temp_ground_truth.json"
    with open(gt_coco_path, 'w') as f:
        json.dump(ground_truth_coco, f)

    # Save predictions to a temporary JSON file
    pred_coco_path = "temp_predictions.json"
    with open(pred_coco_path, 'w') as f:
        json.dump(predictions, f)

    try:
        # Load COCO annotations
        coco_gt = COCO(gt_coco_path)
        coco_pred = coco_gt.loadRes(pred_coco_path)

        # Initialize COCOeval
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract and print metrics
        metrics = {
            "AP": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
            "AP_S": coco_eval.stats[3],
            "AP_M": coco_eval.stats[4],
            "AP_L": coco_eval.stats[5],
            "AR1": coco_eval.stats[6],
            "AR10": coco_eval.stats[7],
            "AR100": coco_eval.stats[8],
            "AR_S": coco_eval.stats[9],
            "AR_M": coco_eval.stats[10],
            "AR_L": coco_eval.stats[11]
        }

        print("\n=== Evaluation Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    except IndexError as e:
        print("Error during evaluation:", e)
        print("Please ensure that there are predictions to evaluate.")

    finally:
        # Clean up temporary files
        if os.path.exists(gt_coco_path):
            os.remove(gt_coco_path)
        if os.path.exists(pred_coco_path):
            os.remove(pred_coco_path)

# Example usage
if __name__ == "__main__":
    MODEL_PATH = "models/yoloV5_best.pt"
    TEST_IMAGES = "data/test/images"
    OUTPUT_DIR = "results/yolov5/test/predict"

    # Run predictions
    # predict_and_save_labels(MODEL_PATH, TEST_IMAGES, OUTPUT_DIR)

    # Evaluate metrics
    evaluate_metrics("data/test/labels", os.path.join(OUTPUT_DIR, "labels"))