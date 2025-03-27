# yolo_clip_detector/utils/visualize.py 파일 내용
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import logging
import colorsys

logger = logging.getLogger(__name__)

def generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for visualization
    
    Args:
        num_classes: Number of classes
        
    Returns:
        List of RGB color tuples
    """
    colors = []
    for i in range(num_classes):
        # Generate colors based on HSV color space and then convert to RGB
        h = i / num_classes
        s = 0.8
        v = 0.8
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    return colors

def draw_detections(image: np.ndarray, 
                   detections: List[Dict], 
                   class_names: Optional[List[str]] = None,
                   thickness: int = 2) -> np.ndarray:
    """
    Draw detection results on an image
    
    Args:
        image: Input image
        detections: List of detections (each with 'box', 'score', 'class_id')
        class_names: List of class names
        thickness: Line thickness for bounding boxes
        
    Returns:
        Image with detections drawn
    """
    # Make a copy to avoid modifying the original
    drawn_image = image.copy()
    
    # Generate colors for classes
    num_classes = max([det['class_id'] for det in detections]) + 1 if detections else 80
    colors = generate_colors(num_classes)
    
    # Draw each detection
    for det in detections:
        box = det['box']
        score = det['score']
        class_id = det['class_id']
        
        # Get class name
        if class_names is not None and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class {class_id}"
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        x1, y1, x2, y2 = box
        cv2.rectangle(drawn_image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw class name and score
        label = f"{class_name}: {score:.2f}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(drawn_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(drawn_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return drawn_image

def plot_precision_recall_curve(recalls: np.ndarray, 
                               precisions: np.ndarray, 
                               class_name: str = '',
                               save_path: Optional[str] = None) -> None:
    """
    Plot precision-recall curve
    
    Args:
        recalls: Recall values
        precisions: Precision values
        class_name: Name of the class
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 7))
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({class_name})' if class_name else 'Precision-Recall Curve')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    
    # Calculate AP
    ap = np.trapz(precisions, recalls)
    plt.text(0.5, 0.5, f'AP = {ap:.4f}', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()