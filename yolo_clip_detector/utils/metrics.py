# yolo_clip_detector/utils/metrics.py 파일 내용
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

def bbox_iou(box1: np.ndarray, box2: np.ndarray, format: str = 'xyxy') -> np.ndarray:
    """
    Calculate IoU between boxes
    
    Args:
        box1: Boxes of shape (..., 4)
        box2: Boxes of shape (..., 4)
        format: Box format ('xyxy' or 'xywh')
        
    Returns:
        IoU of shape (...)
    """
    # Convert boxes to xyxy format if needed
    if format == 'xywh':
        x1_1, y1_1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
        x2_1, y2_1 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
        x1_2, y1_2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
        x2_2, y2_2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        x1_1, y1_1, x2_1, y2_1 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        x1_2, y1_2, x2_2, y2_2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Calculate intersection area
    x1 = np.maximum(x1_1, x1_2)
    y1 = np.maximum(y1_1, y1_2)
    x2 = np.minimum(x2_1, x2_2)
    y2 = np.minimum(y2_1, y2_2)
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-7)
    
    return iou

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from xywh to xyxy format
    
    Args:
        boxes: Boxes of shape (..., 4) in xywh format
        
    Returns:
        Boxes of shape (..., 4) in xyxy format
    """
    xyxy = np.zeros_like(boxes)
    xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return xyxy

def xyxy2xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from xyxy to xywh format
    
    Args:
        boxes: Boxes of shape (..., 4) in xyxy format
        
    Returns:
        Boxes of shape (..., 4) in xywh format
    """
    xywh = np.zeros_like(boxes)
    xywh[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
    xywh[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
    xywh[..., 2] = boxes[..., 2] - boxes[..., 0]
    xywh[..., 3] = boxes[..., 3] - boxes[..., 1]
    return xywh

def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) from precision-recall curve
    
    Args:
        recalls: Recall values
        precisions: Precision values
        
    Returns:
        AP value
    """
    # Sort by recall
    i = np.argsort(recalls)
    recalls = recalls[i]
    precisions = precisions[i]
    
    # Append sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute piecewise constant precision envelope
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # Compute area under PR curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap

def calculate_map(preds: List[Dict], targets: List[Dict], iou_threshold: float = 0.5) -> Tuple[float, float]:
    """
    Calculate mAP@50 and mAP@50-95 for a set of predictions and targets
    
    Args:
        preds: List of prediction dictionaries with 'boxes', 'scores', 'class_ids'
        targets: List of target dictionaries with 'boxes', 'class_ids'
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (mAP@50, mAP@50-95)
    """
    # Get unique class IDs
    class_ids = set()
    for target in targets:
        class_ids.update(np.unique(target['class_ids']))
    class_ids = sorted(list(class_ids))
    
    # Calculate AP for each class and IoU threshold
    ap50 = np.zeros(len(class_ids))
    ap = np.zeros(len(class_ids))
    
    for i, class_id in enumerate(class_ids):
        # Calculate AP@50
        ap50[i] = calculate_ap_at_iou(preds, targets, class_id, 0.5)
        
        # Calculate AP@50-95
        ap_iou = np.zeros(10)
        for j, iou_t in enumerate(np.linspace(0.5, 0.95, 10)):
            ap_iou[j] = calculate_ap_at_iou(preds, targets, class_id, iou_t)
        ap[i] = ap_iou.mean()
    
    # Calculate mAP
    map50 = ap50.mean()
    map = ap.mean()
    
    return map50, map

def calculate_ap_at_iou(preds: List[Dict], targets: List[Dict], class_id: int, iou_threshold: float) -> float:
    """
    Calculate AP for a specific class and IoU threshold
    
    Args:
        preds: List of prediction dictionaries
        targets: List of target dictionaries
        class_id: Class ID to calculate AP for
        iou_threshold: IoU threshold for matching
        
    Returns:
        AP value
    """
    # Collect all predictions and targets for this class
    all_preds = []
    all_targets = []
    
    for batch_idx in range(len(preds)):
        pred = preds[batch_idx]
        target = targets[batch_idx]
        
        # Get predictions for this class
        mask = pred['class_ids'] == class_id
        boxes = pred['boxes'][mask]
        scores = pred['scores'][mask]
        
        # Get targets for this class
        mask = target['class_ids'] == class_id
        target_boxes = target['boxes'][mask]
        
        # Add batch index to identify predictions
        batch_indices = np.full(len(boxes), batch_idx)
        
        all_preds.append(np.column_stack((batch_indices, boxes, scores)))
        all_targets.append(np.column_stack((np.full(len(target_boxes), batch_idx), target_boxes)))
    
    if not all_preds or not all_targets:
        return 0.0
    
    # Concatenate predictions and targets
    all_preds = np.vstack(all_preds) if all_preds else np.zeros((0, 6))
    all_targets = np.vstack(all_targets) if all_targets else np.zeros((0, 5))
    
    # Sort predictions by score
    all_preds = all_preds[all_preds[:, -1].argsort()[::-1]]
    
    # Initialize true positives and false positives
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    
    # Assign predictions to targets
    for i, pred in enumerate(all_preds):
        batch_idx = int(pred[0])
        pred_box = pred[1:5]
        score = pred[5]
        
        # Get targets for this batch
        batch_targets = all_targets[all_targets[:, 0] == batch_idx]
        target_boxes = batch_targets[:, 1:5]
        
        if len(target_boxes) == 0:
            fp[i] = 1
            continue
        
        # Calculate IoU with all targets
        ious = bbox_iou(pred_box, target_boxes)
        
        # Assign prediction to target with highest IoU
        max_iou_idx = ious.argmax()
        max_iou = ious[max_iou_idx]
        
        if max_iou >= iou_threshold:
            # Remove assigned target
            all_targets = np.delete(all_targets, batch_targets[max_iou_idx], axis=0)
            tp[i] = 1
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / max(len(all_targets), 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
    
    # Calculate AP
    ap = calculate_ap(recalls, precisions)
    
    return ap