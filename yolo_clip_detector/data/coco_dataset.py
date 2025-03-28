# yolo_clip_detector/data/coco_dataset.py 파일 내용
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import json
from typing import List, Dict, Tuple, Optional, Union, Callable
import logging
from pycocotools.coco import COCO
import random

logger = logging.getLogger(__name__)

class COCODataset(Dataset):
    """
    COCO Dataset for YOLO-CLIP training
    
    This dataset loads images and annotations from the COCO dataset format.
    
    Attributes:
        coco (COCO): COCO API object
        image_ids (List[int]): List of image ids
        image_infos (List[Dict]): List of image info dictionaries
        class_names (List[str]): List of class names
        transform (Callable): Transform function for data augmentation
        img_size (Tuple[int, int]): Input image size
        mode (str): Dataset mode ('train', 'val', or 'test')
        mosaic_prob (float): Probability of applying mosaic augmentation
    """
    
    def __init__(self, 
                 anno_path: str,
                 img_dir: str,
                 class_names: List[str],
                 img_size: Tuple[int, int] = (640, 640),
                 transform: Optional[Callable] = None,
                 mode: str = 'train',
                 mosaic_prob: float = 0.5,
                 max_objects: int = 100):
        """
        Initialize the COCODataset.
        
        Args:
            anno_path: Path to the COCO annotation file
            img_dir: Path to the image directory
            class_names: List of class names
            img_size: Input image size (height, width)
            transform: Transform function for data augmentation
            mode: Dataset mode ('train', 'val', or 'test')
            mosaic_prob: Probability of applying mosaic augmentation
            max_objects: Maximum number of objects per image
        """
        self.img_dir = img_dir
        self.class_names = class_names
        self.img_size = img_size
        self.transform = transform
        self.mode = mode
        self.mosaic_prob = mosaic_prob if mode == 'train' else 0.0
        self.max_objects = max_objects
        
        # Load COCO annotations
        self.coco = COCO(anno_path)
        
        # Get all category IDs
        cat_ids = self.coco.getCatIds()
        
        # Create a mapping from COCO category ID to our class index
        self.cat_id_to_class_id = {}
        for i, cat_id in enumerate(cat_ids):
            cat_name = self.coco.loadCats(cat_id)[0]['name']
            if cat_name in self.class_names:
                class_id = self.class_names.index(cat_name)
                self.cat_id_to_class_id[cat_id] = class_id
        
        # Get image IDs that have annotations with valid categories
        self.image_ids = []
        self.image_infos = []
        
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=list(self.cat_id_to_class_id.keys()))
            if len(ann_ids) > 0:
                self.image_ids.append(img_id)
                self.image_infos.append(self.coco.loadImgs(img_id)[0])
        
        logger.info(f"COCODataset initialized with {len(self.image_ids)} images, "
                  f"{len(self.class_names)} classes, mode={mode}")
    
    def __len__(self) -> int:
        """Get the length of the dataset"""
        return len(self.image_ids)
    
    
    
    def _resize_image_and_boxes(self, 
                               img: np.ndarray, 
                               boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize image and adjust boxes accordingly.
        
        Args:
            img: Input image
            boxes: Bounding boxes
            
        Returns:
            Tuple of (resized image, adjusted boxes)
        """
        # Get original size
        orig_h, orig_w = img.shape[:2]
        
        # Get target size
        target_h, target_w = self.img_size
        
        # Calculate resize ratio
        ratio = min(target_h / orig_h, target_w / orig_w)
        
        # Resize image
        new_h, new_w = int(orig_h * ratio), int(orig_w * ratio)
        resized_img = cv2.resize(img, (new_w, new_h))
        
        # Create padded image
        padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded_img[:new_h, :new_w, :] = resized_img
        
        # Adjust boxes
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio
        
        return padded_img, boxes
    
    def _to_tensor(self, img) -> torch.Tensor:
        """
        Convert image to tensor.
        
        Args:
            img: Input image (numpy array or torch.Tensor)
            
        Returns:
            Image tensor in (C, H, W) format normalized to [0, 1]
        """
        # 만약 img가 이미 torch.Tensor라면, 그대로 반환하거나 정규화만 진행
        if isinstance(img, torch.Tensor):
            # 만약 0~255 범위라면 255로 나누어 정규화할 수 있지만,
            # 이미 정규화된 상태라면 그냥 반환합니다.
            # 여기서는 이미 정규화되어 있다고 가정하고 그대로 반환합니다.
            return img
        else:
            # NumPy 배열인 경우, float32로 변환 후 정규화
            img = img.astype(np.float32) / 255.0
            # (H, W, C) -> (C, H, W)로 차원 변경
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img



    
    def __getitem__(self, index: int) -> Dict:
        """
        Get a single data item.
        
        Args:
            index: Index of the data item
            
        Returns:
            Dictionary containing the processed data item
        """
        # Apply mosaic augmentation with probability mosaic_prob
        if random.random() < self.mosaic_prob:
            return self._get_mosaic_item(index)
        
        # Get image info
        img_id = self.image_ids[index]
        img_info = self.image_infos[index]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Initialize boxes and class_ids
        boxes = []
        class_ids = []
        
        for ann in anns:
            if ann['category_id'] not in self.cat_id_to_class_id:
                continue
            
            # Skip annotations with no area or no bbox
            if ann['area'] <= 0 or not ann['bbox']:
                continue
            
            # COCO bbox format is [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Convert to [x1, y1, x2, y2] format
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_info['width'], x + w)
            y2 = min(img_info['height'], y + h)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Add box and class ID
            boxes.append([x1, y1, x2, y2])
            class_ids.append(self.cat_id_to_class_id[ann['category_id']])
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int64)
        
        # Resize and preprocess
        img, boxes = self._resize_image_and_boxes(img, boxes)
        
        # Apply transform if provided
        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=boxes, class_ids=class_ids)
            img = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            class_ids = np.array(transformed['class_ids'], dtype=np.int64)
        
        # Convert to tensors
        img = self._to_tensor(img)
        
        # Create valid mask
        valid_mask = np.ones(len(boxes), dtype=np.bool_)
        
        # Pad boxes and class_ids to fixed size
        boxes_padded = np.zeros((self.max_objects, 4), dtype=np.float32)
        class_ids_padded = np.zeros(self.max_objects, dtype=np.int64)
        valid_mask_padded = np.zeros(self.max_objects, dtype=np.bool_)
        
        if len(boxes) > 0:
            boxes_padded[:len(boxes)] = boxes
            class_ids_padded[:len(class_ids)] = class_ids
            valid_mask_padded[:len(valid_mask)] = valid_mask
        
        # Create text prompts for each unique class in the image
        unique_classes = np.unique(class_ids)
        text_prompts = [f"a photo of a {self.class_names[i]}" for i in unique_classes]
        
        # Ensure there's at least one prompt for images with no objects
        if not text_prompts:
            text_prompts = [f"a photo of a {self.class_names[0]}"]
        
        return {
            'images': img,                             # (3, H, W) tensor
            'boxes': torch.from_numpy(boxes_padded),   # (max_objects, 4) tensor
            'class_ids': torch.from_numpy(class_ids_padded),  # (max_objects,) tensor
            'valid_mask': torch.from_numpy(valid_mask_padded),  # (max_objects,) tensor
            'text_prompts': text_prompts,              # List of text prompts
            'image_id': img_id,                        # Original image ID
            'orig_size': (img_info['height'], img_info['width'])  # Original image size
        }

    def _get_mosaic_item(self, index: int) -> Dict:
        """
        Get a mosaic-augmented item.
        
        Args:
            index: Index of the center image
            
        Returns:
            Dictionary containing the mosaic-augmented item
        """
        # Get target size
        target_h, target_w = self.img_size
        
        # Initialize mosaic image
        mosaic_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Initialize boxes and class IDs lists
        mosaic_boxes = []
        mosaic_class_ids = []
        
        # Get indices of 4 images
        indices = [index] + [random.randint(0, len(self.image_ids) - 1) for _ in range(3)]
        
        # Random center point of mosaic
        cx, cy = target_w // 2 + random.randint(-target_w // 4, target_w // 4), \
                target_h // 2 + random.randint(-target_h // 4, target_h // 4)
        
        # Placing 4 tiles
        for i, idx in enumerate(indices):
            # Get image info
            img_id = self.image_ids[idx]
            img_info = self.image_infos[idx]
            
            # Load image
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get annotations
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Get boxes and class IDs
            boxes = []
            class_ids = []
            
            for ann in anns:
                if ann['category_id'] not in self.cat_id_to_class_id:
                    continue
                
                # Skip annotations with no area or no bbox
                if ann['area'] <= 0 or not ann['bbox']:
                    continue
                
                # COCO bbox format is [x, y, width, height]
                x, y, w, h = ann['bbox']
                
                # Convert to [x1, y1, x2, y2] format
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_info['width'], x + w)
                y2 = min(img_info['height'], y + h)
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Add box and class ID
                boxes.append([x1, y1, x2, y2])
                class_ids.append(self.cat_id_to_class_id[ann['category_id']])
            
            # Convert to numpy arrays
            boxes = np.array(boxes, dtype=np.float32)
            class_ids = np.array(class_ids, dtype=np.int64)
            
            # Resize image and adjust boxes
            img, boxes = self._resize_image_and_boxes(img, boxes)
            
            # Get placement coordinates
            x1_place, y1_place, x2_place, y2_place = 0, 0, 0, 0
            
            if i == 0:  # Top-left
                x1_place, y1_place, x2_place, y2_place = 0, 0, cx, cy
            elif i == 1:  # Top-right
                x1_place, y1_place, x2_place, y2_place = cx, 0, target_w, cy
            elif i == 2:  # Bottom-left
                x1_place, y1_place, x2_place, y2_place = 0, cy, cx, target_h
            elif i == 3:  # Bottom-right
                x1_place, y1_place, x2_place, y2_place = cx, cy, target_w, target_h
            
            # Place image on mosaic
            mosaic_img[y1_place:y2_place, x1_place:x2_place] = \
                cv2.resize(img, (x2_place - x1_place, y2_place - y1_place))
            
            # Adjust boxes to new position
            if len(boxes) > 0:
                # Scale to placement size
                h_scale = (y2_place - y1_place) / target_h
                w_scale = (x2_place - x1_place) / target_w
                
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * w_scale + x1_place
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * h_scale + y1_place
                
                # Clip to mosaic boundaries
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, target_w - 1)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, target_h - 1)
                
                # Filter out invalid boxes
                valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes = boxes[valid_boxes]
                class_ids = class_ids[valid_boxes]
                
                # Add to mosaic list
                if len(boxes) > 0:
                    mosaic_boxes.append(boxes)
                    mosaic_class_ids.append(class_ids)
        
        # Concatenate boxes and class_ids
        if mosaic_boxes:
            mosaic_boxes = np.vstack(mosaic_boxes)
            mosaic_class_ids = np.hstack(mosaic_class_ids)
        else:
            mosaic_boxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_class_ids = np.zeros(0, dtype=np.int64)
        
        # Apply transform if provided
        if self.transform is not None:
            transformed = self.transform(image=mosaic_img, bboxes=mosaic_boxes, class_ids=mosaic_class_ids)
            mosaic_img = transformed['image']
            mosaic_boxes = np.array(transformed['bboxes'], dtype=np.float32)
            mosaic_class_ids = np.array(transformed['class_ids'], dtype=np.int64)
        
        # Convert to tensors
        mosaic_img = self._to_tensor(mosaic_img)
        
        # Create valid mask
        valid_mask = np.ones(len(mosaic_boxes), dtype=np.bool_)
        
        # Pad boxes and class_ids to fixed size
        boxes_padded = np.zeros((self.max_objects, 4), dtype=np.float32)
        class_ids_padded = np.zeros(self.max_objects, dtype=np.int64)
        valid_mask_padded = np.zeros(self.max_objects, dtype=np.bool_)
        
        if len(mosaic_boxes) > 0:
            n = min(len(mosaic_boxes), self.max_objects)
            boxes_padded[:n] = mosaic_boxes[:n]
            class_ids_padded[:n] = mosaic_class_ids[:n]
            valid_mask_padded[:n] = valid_mask[:n]
        
        # Create text prompts for unique classes in the mosaic
        unique_classes = np.unique(mosaic_class_ids)
        text_prompts = [f"a photo of a {self.class_names[i]}" for i in unique_classes if i < len(self.class_names)]
        
        # Ensure there's at least one prompt for mosaic with no objects
        if not text_prompts:
            text_prompts = [f"a photo of a {self.class_names[0]}"]
        
        return {
            'images': mosaic_img,                        # (3, H, W) tensor
            'boxes': torch.from_numpy(boxes_padded),     # (max_objects, 4) tensor
            'class_ids': torch.from_numpy(class_ids_padded),  # (max_objects,) tensor
            'valid_mask': torch.from_numpy(valid_mask_padded),  # (max_objects,) tensor
            'text_prompts': text_prompts,                # List of text prompts
            'image_id': -1,                              # Mosaic has no image ID
            'orig_size': self.img_size                   # Original size is the same as target size
        }