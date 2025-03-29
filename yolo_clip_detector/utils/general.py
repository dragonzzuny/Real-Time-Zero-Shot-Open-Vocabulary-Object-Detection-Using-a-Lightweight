# yolo_clip_detector/utils/general.py 파일 내용
import os
import random
import numpy as np
import torch
import yaml
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import time
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def load_yaml(file_path: str) -> Dict:
    """
    Load YAML file
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary containing YAML content
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_yaml(data: Dict, file_path: str) -> None:
    """
    Save dictionary to YAML file
    
    Args:
        data: Dictionary to save
        file_path: Path to save to
    """
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class Timer:
    """Simple timer class for measuring execution time"""
    
    def __init__(self, name: str = ''):
        """
        Initialize timer
        
        Args:
            name: Timer name
        """
        self.name = name
        self.start_time = None
        self.total_time = 0.0
        self.calls = 0
    
    def __enter__(self):
        """Start timer"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop timer"""
        self.total_time += time.time() - self.start_time
        self.calls += 1
        self.start_time = None
    
    def reset(self):
        """Reset timer"""
        self.start_time = None
        self.total_time = 0.0
        self.calls = 0
    
    @property
    def avg_time(self) -> float:
        """Get average time"""
        return self.total_time / max(self.calls, 1)
    
    def __str__(self) -> str:
        """Get string representation"""
        return f"{self.name}: {self.total_time:.4f}s total, {self.avg_time:.4f}s avg ({self.calls} calls)"

def create_unique_output_dir(base_dir: str, prefix: str = '') -> str:
    """
    Create a unique output directory
    
    Args:
        base_dir: Base directory
        prefix: Directory name prefix
        
    Returns:
        Path to created directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory name
    dir_name = f"{prefix}_{timestamp}" if prefix else timestamp
    
    # Create full path
    output_dir = os.path.join(base_dir, dir_name)
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def copy_code_to_dir(output_dir: str, ignore_patterns: List[str] = None) -> None:
    """
    Copy current codebase to output directory
    
    Args:
        output_dir: Output directory
        ignore_patterns: List of patterns to ignore
    """
    if ignore_patterns is None:
        ignore_patterns = ['.git', '__pycache__', '*.pyc', 'outputs', 'logs', '*.pt', '*.pth']
    
    code_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(code_dir)  # Go up one level
    
    # Create code directory in output directory
    dest_dir = os.path.join(output_dir, 'code')
    os.makedirs(dest_dir, exist_ok=True)
    
    # Walk through code directory and copy files
    for root, dirs, files in os.walk(code_dir):
        # Check if root should be ignored
        if any(pattern in root for pattern in ignore_patterns):
            continue
        
        # Create corresponding directory in dest_dir
        rel_path = os.path.relpath(root, code_dir)
        dest_path = os.path.join(dest_dir, rel_path)
        os.makedirs(dest_path, exist_ok=True)
        
        # Copy files
        for file in files:
            # Check if file should be ignored
            if any(pattern in file for pattern in ignore_patterns):
                continue
            
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            shutil.copy2(src_file, dest_file)
    
    logger.info(f"Code copied to {dest_dir}")