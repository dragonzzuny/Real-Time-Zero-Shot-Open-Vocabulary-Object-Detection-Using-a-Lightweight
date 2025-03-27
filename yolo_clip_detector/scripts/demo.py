# yolo_clip_detector/scripts/demo.py 파일 내용
#!/usr/bin/env python
"""
Demo script for YOLO-CLIP detector

This script demonstrates how to use YOLO-CLIP detector for object detection
with either predefined classes or custom text prompts.
"""

import os
import sys
import argparse
import logging
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.yolo_clip import YOLOCLIP
from inference.detector import YOLOCLIPDetector
from utils.visualize import draw_detections

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLO-CLIP detector demo')
    parser.add_argument('--model', type=str, default='outputs/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--backbone', type=str, default='n', help='Backbone variant (n, s, m, l)')
    parser.add_argument('--mode', type=str, default='offline', choices=['offline', 'online'], help='Detection mode')
    parser.add_argument('--classes', type=str, default=None, help='Classes to detect (comma-separated)')
    parser.add_argument('--text', type=str, default=None, help='Text prompts (comma-separated)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None, help='Path to output image')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set up classes and text prompts
    class_names = None
    text_prompts = None
    
    if args.mode == 'offline':
        # Use offline mode with predefined classes
        if args.classes is not None:
            class_names = args.classes.split(',')
        else:
            # Default COCO classes
            class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
        logger.info(f"Using offline mode with {len(class_names)} classes")
    else:
        # Use online mode with text prompts
        if args.text is not None:
            text_prompts = args.text.split(',')
        else:
            text_prompts = ["a photo of a cat", "a photo of a dog", "a photo of a person"]
        logger.info(f"Using online mode with text prompts: {text_prompts}")
    
    # Create detector
    detector = YOLOCLIPDetector(
        model_path=args.model,
        class_names=class_names,
        vocab_path=None,
        device=device,
        image_size=(640, 640),
        conf_threshold=args.conf,
        iou_threshold=0.45,
        backbone_variant=args.backbone,
        clip_model='ViT-B/32',
        embed_dim=512
    )
    
    # Run detection
    logger.info(f"Running detection on {args.image}")
    detections = detector.detect(args.image, text_prompts=text_prompts)
    
    # Log detection results
    logger.info(f"Found {len(detections)} objects:")
    for det in detections:
        logger.info(f"  {det['class_name']}: {det['score']:.3f} at {det['box']}")
    
    # Draw detections
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    drawn_img = detector.draw_detections(img, detections)
    
    # Show or save image
    if args.output:
        # Save result
        drawn_img_bgr = cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, drawn_img_bgr)
        logger.info(f"Result saved to {args.output}")
    else:
        # Show result
        plt.figure(figsize=(12, 8))
        plt.imshow(drawn_img)
        plt.axis('off')
        plt.title(f"YOLO-CLIP Detection: {len(detections)} objects")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()