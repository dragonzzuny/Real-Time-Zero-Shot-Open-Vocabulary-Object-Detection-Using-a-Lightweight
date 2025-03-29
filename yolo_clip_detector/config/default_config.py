# yolo_clip_detector/config/default_config.py 파일 내용
from typing import Dict, List, Tuple, Any
import os

# 프로젝트 루트 디렉토리 찾기
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class Config:
    """Base configuration class"""
    
    def __init__(self):
        pass
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class TrainingConfig(Config):
    """Configuration for model training"""
    
    def __init__(self):
        super().__init__()
        
        # Model settings
        self.backbone_variant = 'n'  # YOLOv8 backbone variant ('n', 's', 'm', 'l')
        self.clip_model = 'ViT-B/32'  # CLIP model variant
        self.embed_dim = 512  # Embedding dimension
        self.reg_max = 16  # Maximum value for DFL (Distributed Focal Loss)
        
        # Dataset settings
        self.train_anno_path = 'yolo_clip_detector/data/coco/annotations/instances_train2017.json'
        self.train_img_dir = 'yolo_clip_detector/data/coco/train2017'
        self.val_anno_path = 'yolo_clip_detector/data/coco/annotations/instances_val2017.json'
        self.val_img_dir = 'yolo_clip_detector/data/coco/val2017'
        self.class_names = [  # COCO 80 classes
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
        self.img_size = (640, 640)  # Input image size
        self.max_objects = 100  # Maximum number of objects per image
        self.mosaic_prob = 0.5  # Probability of applying mosaic augmentation
        
        # Training settings
        self.batch_size = 16  # Batch size
        self.num_workers = 8  # Number of dataloader workers
        self.learning_rate = 1e-4  # Learning rate
        self.weight_decay = 1e-4  # Weight decay
        self.max_epochs = 100  # Maximum number of epochs
        self.warmup_epochs = 5  # Number of warmup epochs
        self.save_interval = 10  # Epoch interval to save checkpoints
        self.eval_interval = 5  # Epoch interval to run evaluation
        
        # Loss settings
        self.temperature = 0.1  # Temperature parameter for contrastive loss
        self.iou_type = 'ciou'  # Type of IoU loss ('iou', 'giou', 'diou', 'ciou')
        self.label_smoothing = 0.1  # Label smoothing parameter
        self.loss_weights = {  # Weights for different loss components
            'contrastive': 1.0,
            'iou': 5.0,
            'dfl': 1.0
        }
        
        # Optimizer settings
        self.optimizer_type = 'AdamW'  # Optimizer type
        self.lr_scheduler_type = 'OneCycleLR'  # Learning rate scheduler type
        
        # Output settings
        self.output_dir = 'outputs/'  # Output directory


class InferenceConfig(Config):
    """Configuration for model inference"""
    
    def __init__(self):
        super().__init__()
        
        # Model settings
        self.backbone_variant = 'n'  # YOLOv8 backbone variant ('n', 's', 'm', 'l')
        self.clip_model = 'ViT-B/32'  # CLIP model variant
        self.embed_dim = 512  # Embedding dimension
        
        # Inference settings
        self.model_path = 'outputs/best_model.pth'  # Path to model checkpoint
        self.vocab_path = None  # Path to vocabulary file (optional)
        self.img_size = (640, 640)  # Input image size
        self.conf_threshold = 0.25  # Confidence threshold for detections
        self.iou_threshold = 0.45  # IoU threshold for NMS
        self.class_names = [  # COCO 80 classes
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
        self.use_offline_vocab = True  # Whether to use offline vocabulary
        self.output_dir = 'outputs/detections/'  # Output directory for detection results