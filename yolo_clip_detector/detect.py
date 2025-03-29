# yolo_clip_detector/detect.py 파일 내용
import os
import torch
import cv2
import numpy as np
import argparse
import logging
import yaml
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union

from model.yolo_clip import YOLOCLIP
from inference.detector import YOLOCLIPDetector
from config.default_config import InferenceConfig

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
    parser = argparse.ArgumentParser(description='Run object detection with YOLO-CLIP model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default=None, help='Path to vocabulary file')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None, help='Path to output directory')
    parser.add_argument('--conf', type=float, default=None, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=None, help='IoU threshold for NMS')
    parser.add_argument('--text_prompts', type=str, default=None, help='Text prompts (comma-separated)')
    parser.add_argument('--classes', type=str, default=None, help='Classes to detect (comma-separated)')
    parser.add_argument('--backbone', type=str, default=None, help='Backbone variant')
    parser.add_argument('--device', type=str, default='0', help='Device to use')
    return parser.parse_args()

def main():
    """Main inference function"""
    args = parse_args()
    
    # Load config
    config = InferenceConfig()
    
    if args.config is not None:
        logger.info(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            for k, v in config_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)
    
    # Override config with command line arguments
    if args.model is not None:
        config.model_path = args.model
    if args.vocab is not None:
        config.vocab_path = args.vocab
    if args.output is not None:
        config.output_dir = args.output
    if args.conf is not None:
        config.conf_threshold = args.conf
    if args.iou is not None:
        config.iou_threshold = args.iou
    if args.backbone is not None:
        config.backbone_variant = args.backbone
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parse classes
    class_names = config.class_names
    if args.classes is not None:
        class_names = args.classes.split(',')
    
    # Parse text prompts
    text_prompts = None
    if args.text_prompts is not None:
        text_prompts = args.text_prompts.split(',')
        config.use_offline_vocab = False
    
    # Create output directory
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    # Create detector
    detector = YOLOCLIPDetector(
        model_path=config.model_path,
        class_names=class_names,
        vocab_path=config.vocab_path if config.use_offline_vocab else None,
        device=device,
        image_size=config.img_size,
        conf_threshold=config.conf_threshold,
        iou_threshold=config.iou_threshold,
        backbone_variant=config.backbone_variant,
        clip_model=config.clip_model,
        embed_dim=config.embed_dim
    )
    
    # Get input paths
    input_path = args.input
    input_paths = []
    
    if os.path.isdir(input_path):
        # Process all images in directory
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for filename in os.listdir(input_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                input_paths.append(os.path.join(input_path, filename))
    else:
        # Process single image
        input_paths = [input_path]
    
    # Run detection on each image
    for img_path in input_paths:
        logger.info(f"Processing {img_path}")
        
        # Run detection
        detections = detector.detect(img_path, text_prompts=text_prompts)
        
        # Draw detections
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        drawn_img = detector.draw_detections(img, detections)
        
        # Save result
        filename = os.path.basename(img_path)
        output_path = os.path.join(config.output_dir, f"det_{filename}")
        drawn_img_bgr = cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, drawn_img_bgr)
        
        # Log detection results
        logger.info(f"Found {len(detections)} objects:")
        for det in detections:
            logger.info(f"  {det['class_name']}: {det['score']:.3f} at {det['box']}")
        
        logger.info(f"Result saved to {output_path}")
    
    logger.info("Detection completed")

if __name__ == "__main__":
    main()