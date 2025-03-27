# yolo_clip_detector/inference/detector.py 파일 내용
import torch
import torch.nn as nn
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
import os
import time
from ..model.yolo_clip import YOLOCLIP

logger = logging.getLogger(__name__)

class YOLOCLIPDetector:
    """
    YOLO-CLIP Detector for inference
    
    This class handles loading a trained model and performing object detection on images.
    
    Attributes:
        model (YOLOCLIP): The YOLO-CLIP model
        device (torch.device): Device to run inference on
        image_size (Tuple[int, int]): Input image size (height, width)
        conf_threshold (float): Confidence threshold for detections
        iou_threshold (float): IoU threshold for NMS
        class_names (List[str]): List of class names
        use_offline_vocab (bool): Whether to use offline vocabulary
    """
    
    def __init__(self, 
                 model_path: str,
                 class_names: Optional[List[str]] = None,
                 vocab_path: Optional[str] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 image_size: Tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 backbone_variant: str = 'n',
                 clip_model: str = 'ViT-B/32',
                 embed_dim: int = 512):
        """
        Initialize the YOLOCLIPDetector.
        
        Args:
            model_path: Path to the model checkpoint
            class_names: List of class names
            vocab_path: Path to the vocabulary file (optional)
            device: Device to run inference on
            image_size: Input image size (height, width)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            backbone_variant: YOLOv8 backbone variant
            clip_model: CLIP model variant
            embed_dim: Embedding dimension
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Set input image size
        self.image_size = image_size
        
        # Set detection thresholds
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Set class names
        self.class_names = class_names
        
        # Create model
        self.model = YOLOCLIP(
            backbone_variant=backbone_variant,
            clip_model=clip_model,
            embed_dim=embed_dim,
            num_classes=len(class_names) if class_names is not None else 80,
            offline_mode=vocab_path is not None or class_names is not None
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Load model weights
        self._load_model(model_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Set up offline vocabulary if provided
        self.use_offline_vocab = False
        
        if vocab_path is not None:
            self.model.load_offline_vocabulary(vocab_path)
            self.use_offline_vocab = True
        elif class_names is not None:
            self.model.set_offline_vocabulary(class_names)
            self.use_offline_vocab = True
        
        logger.info(f"YOLOCLIPDetector initialized with model from {model_path}, "
                  f"device={self.device}, image_size={image_size}")
    
    def _load_model(self, model_path: str) -> None:
        """
        Load model weights from a checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        logger.info(f"Model loaded from {model_path}")
    
    def preprocess_image(self, 
                        image: Union[str, np.ndarray]) -> Tuple[torch.Tensor, np.ndarray, float]:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image path or numpy array
            
        Returns:
            Tuple of (preprocessed image tensor, original image, scale factor)
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original image and its dimensions
        orig_image = image.copy()
        orig_h, orig_w = orig_image.shape[:2]
        
        # Resize image
        input_h, input_w = self.image_size
        scale_factor = min(input_h / orig_h, input_w / orig_w)
        resized_h, resized_w = int(orig_h * scale_factor), int(orig_w * scale_factor)
        
        resized_image = cv2.resize(image, (resized_w, resized_h))
        
        # Create a black canvas of input size
        canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        
        # Paste the resized image on the canvas
        canvas[:resized_h, :resized_w, :] = resized_image
        
        # Normalize the image
        canvas = canvas.astype(np.float32) / 255.0
        
        # Transpose the image to (C, H, W) format
        canvas = canvas.transpose(2, 0, 1)
        
        # Convert to tensor
        tensor = torch.from_numpy(canvas).unsqueeze(0).to(self.device)
        
        return tensor, orig_image, scale_factor
    
    def postprocess_detections(self, 
                              outputs: Dict[str, torch.Tensor], 
                              orig_size: Tuple[int, int],
                              scale_factor: float) -> List[Dict]:
        """
        Postprocess raw model outputs into usable detections.
        
        Args:
            outputs: Model outputs
            orig_size: Original image size (height, width)
            scale_factor: Scale factor used for resizing
            
        Returns:
            List of detections, each containing box, score, class_id, and class_name
        """
        # Extract outputs
        boxes = outputs['boxes'][0].cpu().numpy()  # (num_boxes, 4)
        scores = outputs['scores'][0].cpu().numpy()  # (num_boxes)
        class_ids = outputs['class_ids'][0].cpu().numpy()  # (num_boxes)
        
        # Apply confidence threshold
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # Rescale boxes to original image size
        orig_h, orig_w = orig_size
        input_h, input_w = self.image_size
        
        boxes[:, 0] = boxes[:, 0] / scale_factor
        boxes[:, 1] = boxes[:, 1] / scale_factor
        boxes[:, 2] = boxes[:, 2] / scale_factor
        boxes[:, 3] = boxes[:, 3] / scale_factor
        
        # Clip boxes to image boundaries
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
        
        # Apply NMS
        keep_indices = self._nms(boxes, scores, self.iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]
        
        # Create detections list
        detections = []
        
        for i in range(len(boxes)):
            detection = {
                'box': boxes[i].astype(int).tolist(),
                'score': float(scores[i]),
                'class_id': int(class_ids[i]),
                'class_name': self.class_names[class_ids[i]] if self.class_names is not None else f"Class {class_ids[i]}"
            }
            
            detections.append(detection)
        
        return detections
    
    def _nms(self, 
            boxes: np.ndarray, 
            scores: np.ndarray, 
            iou_threshold: float) -> List[int]:
        """
        Apply Non-Maximum Suppression to boxes.
        
        Args:
            boxes: Bounding boxes of shape (num_boxes, 4)
            scores: Confidence scores of shape (num_boxes)
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of indices to keep
        """
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        keep_indices = []
        
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_indices.append(box_id)
            
            # Compute IoU of the picked box with the rest
            ious = self._compute_iou(boxes[box_id], boxes[sorted_indices[1:]])
            
            # Remove boxes with IoU over threshold
            mask = ious <= iou_threshold
            sorted_indices = sorted_indices[1:][mask]
        
        return keep_indices
    
    def _compute_iou(self, 
                    box: np.ndarray, 
                    boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU between a box and an array of boxes.
        
        Args:
            box: A single box of shape (4)
            boxes: An array of boxes of shape (num_boxes, 4)
            
        Returns:
            IoUs of shape (num_boxes)
        """
        # Calculate intersection
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-7)
        
        return iou
    
    def detect(self, 
              image: Union[str, np.ndarray], 
              text_prompts: Optional[List[str]] = None) -> List[Dict]:
        """
        Perform object detection on an image.
        
        Args:
            image: Input image path or numpy array
            text_prompts: Text prompts for online vocabulary (ignored in offline mode)
            
        Returns:
            List of detections
        """
        # Start timing
        start_time = time.time()
        
        # Preprocess image
        tensor, orig_image, scale_factor = self.preprocess_image(image)
        orig_h, orig_w = orig_image.shape[:2]
        
        # Run inference
        with torch.no_grad():
            if self.use_offline_vocab:
                outputs = self.model(tensor)
            else:
                if text_prompts is None:
                    raise ValueError("Text prompts must be provided in online mode")
                outputs = self.model(tensor, text_prompts=text_prompts)
        
        # Postprocess detections
        detections = self.postprocess_detections(outputs, (orig_h, orig_w), scale_factor)
        
        # End timing
        inference_time = time.time() - start_time
        logger.info(f"Detection completed in {inference_time:.3f} seconds with {len(detections)} objects")
        
        return detections
    
    def draw_detections(self, 
                       image: Union[str, np.ndarray], 
                       detections: List[Dict]) -> np.ndarray:
        """
        Draw detection results on an image.
        
        Args:
            image: Input image path or numpy array
            detections: List of detections from detect method
            
        Returns:
            Image with detections drawn
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Make a copy to avoid modifying the original
        drawn_image = image.copy()
        
        # Define colors for different classes
        colors = self._generate_colors(len(self.class_names) if self.class_names is not None else 80)
        
        # Draw each detection
        for det in detections:
            box = det['box']
            score = det['score']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = box
            cv2.rectangle(drawn_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw class name and score
            label = f"{class_name}: {score:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(drawn_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(drawn_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return drawn_image
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """
        Generate distinct colors for visualization.
        
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


# Add missing import at the top of the file
import colorsys