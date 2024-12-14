import numpy as np
from pathlib import Path
import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob
# ...existing code...

def evaluate_metrics(ground_truth_dir, predicted_dirs, ensemble_output_dir):
    """
    Function to evaluate object detection metrics and print them to the terminal.
    Also performs ensemble on multiple predicted labels and saves the result.

    Args:
        ground_truth_dir (str): Path to ground truth labels.
        predicted_dirs (list): List of paths to predicted labels from different models.
        ensemble_output_dir (str): Path to save ensemble predictions.
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

    # Paths
    images_dir = "data/test/images"
    ground_truth_coco = convert_yolo_to_coco_gt(ground_truth_dir, images_dir)

    # Create a mapping from image name to image_id
    image_id_map = {Path(img["file_name"]).stem: img["id"] for img in ground_truth_coco["images"]}

    # Initialize an empty list for ensemble predictions
    ensemble_predictions = []

    for image in ground_truth_coco["images"]:
        image_id = image["id"]
        image_name = Path(image["file_name"]).stem
        # Collect predictions from each model for this image
        model_detections = []
        for pred_dir in predicted_dirs:
            label_path = os.path.join(pred_dir, f"{image_name}.txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        if len(parts) >= 6:
                            class_id, x_center, y_center, width, height, score = map(float, parts[:6])
                        else:
                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                            score = 1.0  # Default score
                        x = x_center - width / 2
                        y = y_center - height / 2
                        w = width
                        h = height
                        x1 = x
                        y1 = y
                        x2 = x + w
                        y2 = y + h
                        model_detections.append([
                            x1, y1, x2, y2, score, int(class_id)
                        ])
        # Now perform ensemble on model_detections
        if model_detections:
            detections = np.array(model_detections)
            keep = nms(detections, iou_threshold=0.5)
            for idx in keep:
                x1, y1, x2, y2, score, category_id = detections[idx]
                bbox = [
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                ]
                ensemble_predictions.append({
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": bbox,
                    "score": float(score)
                })
    if not ensemble_predictions:
        print("No predictions found. Please ensure that the predicted labels are correctly generated.")
        return

    # Save ground truth COCO annotations to a temporary JSON file
    gt_coco_path = "temp_ground_truth.json"
    with open(gt_coco_path, 'w') as f:
        json.dump(ground_truth_coco, f)

    # Save ensemble predictions to a temporary JSON file
    ensemble_pred_coco_path = "temp_ensemble_predictions.json"
    with open(ensemble_pred_coco_path, 'w') as f:
        json.dump(ensemble_predictions, f)

    # Save ensemble predictions in YOLO format to ensemble_output_dir
    if not os.path.exists(ensemble_output_dir):
        os.makedirs(ensemble_output_dir)

    # Clear existing files in ensemble_output_dir
    for f in glob.glob(os.path.join(ensemble_output_dir, '*.txt')):
        os.remove(f)

    # Write predictions
    for pred in ensemble_predictions:
        image_id = pred["image_id"]
        category_id = pred["category_id"]
        bbox = pred["bbox"]
        score = pred["score"]

        image_info = next((img for img in ground_truth_coco["images"] if img["id"] == image_id), None)
        if image_info is None:
            continue
        image_name = Path(image_info["file_name"]).stem
        label_file = os.path.join(ensemble_output_dir, f"{image_name}.txt")
        # Convert bbox back to YOLO format
        x, y, w, h = bbox
        x_center = x + w / 2
        y_center = y + h / 2
        with open(label_file, 'a') as f:
            f.write(f"{category_id} {x_center} {y_center} {w} {h} {score}\n")

    try:
        # Load COCO annotations
        coco_gt = COCO(gt_coco_path)
        coco_pred = coco_gt.loadRes(ensemble_pred_coco_path)

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
        if os.path.exists(ensemble_pred_coco_path):
            os.remove(ensemble_pred_coco_path)

def nms(detections, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression on detections.

    Args:
        detections (numpy.ndarray): Detections in format [[x1, y1, x2, y2, score, class_id], ...]
        iou_threshold (float): IoU threshold for NMS.

    Returns:
        List of indices of detections to keep.
    """
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]
    categories = detections[:, 5]

    indices = np.argsort(-scores)

    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        if len(indices) == 1:
            break
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_rest = (x2[indices[1:]] - x1[indices[1:]]) * (y2[indices[1:]] - y1[indices[1:]])
        union = area_i + area_rest - inter
        iou = inter / union

        # Keep boxes with IoU less than threshold or different categories
        mask = (iou <= iou_threshold) | (categories[indices[1:]] != categories[i])
        indices = indices[1:][mask]

    return np.array(keep)

# ...existing code...

if __name__ == "__main__":
    YOLOV5_LABELS = 'results/yolov5/test/labels'
    YOLOV8_LABELS = 'results/yolov8/yolov/predict/labels'
    YOLOV11_LABELS = 'results/yolov11/test/predict/labels'
    GROUND_TRUTH_LABELS = 'data/test/labels'
    ENSEMBLE_OUTPUT_DIR = 'results/ensemble_result'

    predicted_dirs = [YOLOV8_LABELS, YOLOV11_LABELS]

    # Evaluate metrics and print them
    evaluate_metrics(GROUND_TRUTH_LABELS, predicted_dirs, ENSEMBLE_OUTPUT_DIR)