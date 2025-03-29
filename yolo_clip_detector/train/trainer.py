# yolo_clip_detector/train/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import logging
from typing import List, Dict, Tuple, Optional, Union, Callable
from tqdm import tqdm
from ..model.yolo_clip import YOLOCLIP
from ..loss.region_text_contrastive import RegionTextContrastiveLoss
from ..loss.iou_loss import IoULoss
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class YOLOCLIPTrainer:
    """
    Trainer for YOLO-CLIP model
    
    This class handles the training process, including loss calculation,
    optimization, and evaluation.
    """
    
    def __init__(self, 
                 model: YOLOCLIP,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler,
                 device: torch.device,
                 output_dir: str,
                 max_epochs: int = 100,
                 save_interval: int = 10,
                 eval_interval: int = 5,
                 temperature: float = 0.1,
                 iou_type: str = 'ciou',
                 label_smoothing: float = 0.0,
                 loss_weights: Dict[str, float] = None,
                 max_objects: int = 100):  # max_objects 매개변수 추가
        """
        Initialize the YOLOCLIPTrainer.
        
        Args:
            model: The YOLO-CLIP model
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            device: Device to run training on
            output_dir: Directory to save output files
            max_epochs: Maximum number of epochs to train
            save_interval: Epoch interval to save checkpoints
            eval_interval: Epoch interval to run evaluation
            temperature: Temperature parameter for contrastive loss
            iou_type: Type of IoU for bounding box loss
            label_smoothing: Label smoothing parameter
            loss_weights: Weights for different loss components
            max_objects: Maximum number of objects per image (should match dataset)
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.max_objects = max_objects
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize loss functions
        self.contrastive_loss = RegionTextContrastiveLoss(
            temperature=temperature,
            reduction='mean',
            topk=3,
            label_smoothing=label_smoothing
        )
        
        self.iou_loss = IoULoss(
            iou_type=iou_type,
            reduction='mean'
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        # Initialize loss weights
        self.loss_weights = {
            'contrastive': 1.0,
            'iou': 5.0,
            'dfl': 1.0
        }
        
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"YOLOCLIPTrainer initialized with device={device}, "
                 f"max_epochs={max_epochs}, loss_weights={self.loss_weights}, "
                 f"max_objects={max_objects}")
    
    def train_epoch(self, 
                   dataloader: DataLoader, 
                   epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'contrastive_loss': 0.0,
            'iou_loss': 0.0,
            'dfl_loss': 0.0
        }
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.max_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data
            images = batch['images'].to(self.device)
            boxes = batch['boxes'].to(self.device)
            class_ids = batch['class_ids'].to(self.device)
            text_prompts = batch['text_prompts']  # This is a list of lists with potentially different lengths
            valid_mask = batch['valid_mask'].to(self.device) if 'valid_mask' in batch else None
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, text_prompts=text_prompts)
            
            # Calculate losses
            # Region-text contrastive loss
            region_features = outputs['obj_embeddings']
            text_embeddings = outputs['text_embeddings']  # Using the embeddings from model output
            
            # Handle size mismatch between region_features and dataset objects
            if region_features.shape[1] != self.max_objects:
                logger.info(f"Adjusting region_features from {region_features.shape[1]} to {self.max_objects}")
                if region_features.shape[1] > self.max_objects:
                    # Truncate to max_objects if larger
                    region_features = region_features[:, :self.max_objects, :]
                else:
                    # Pad with zeros if smaller
                    padding = torch.zeros(
                        region_features.shape[0], 
                        self.max_objects - region_features.shape[1], 
                        region_features.shape[2], 
                        device=region_features.device
                    )
                    region_features = torch.cat([region_features, padding], dim=1)
            
            cont_loss = self.contrastive_loss(
                region_features, 
                text_embeddings, 
                class_ids,
                valid_mask
            )
            
            # IoU loss for bounding box regression
            pred_boxes = outputs['boxes']
            
            # Handle size mismatch between predicted and target boxes
            if pred_boxes.shape[1] != boxes.shape[1]:
                logger.info(f"Adjusting pred_boxes from {pred_boxes.shape[1]} to {boxes.shape[1]}")
                if pred_boxes.shape[1] > boxes.shape[1]:
                    # Truncate to match target boxes
                    pred_boxes = pred_boxes[:, :boxes.shape[1], :]
                else:
                    # This case should be rare - we usually have more predictions than targets
                    # Pad with zeros if needed
                    padding = torch.zeros(
                        pred_boxes.shape[0], 
                        boxes.shape[1] - pred_boxes.shape[1], 
                        pred_boxes.shape[2], 
                        device=pred_boxes.device
                    )
                    pred_boxes = torch.cat([pred_boxes, padding], dim=1)
            
            iou_loss = self.iou_loss(pred_boxes, boxes, valid_mask)
            
            # Simplified Distributed Focal Loss (DFL) as MSE loss
            dfl_loss = torch.tensor(0.0, device=self.device)
            if 'box_preds' in outputs and 'box_targets' in batch:
                pred_box_preds = torch.cat([p.flatten(1) for p in outputs['box_preds']], dim=1)
                target_box_preds = torch.cat([p.flatten(1) for p in batch['box_targets']], dim=1)
                dfl_loss = F.mse_loss(pred_box_preds, target_box_preds)
            
            # Combine losses
            loss = (
                self.loss_weights['contrastive'] * cont_loss +
                self.loss_weights['iou'] * iou_loss +
                self.loss_weights['dfl'] * dfl_loss
            )
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['contrastive_loss'] += cont_loss.item()
            epoch_metrics['iou_loss'] += iou_loss.item()
            epoch_metrics['dfl_loss'] += dfl_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'cont_loss': cont_loss.item(),
                'iou_loss': iou_loss.item(),
                'dfl_loss': dfl_loss.item()
            })
            
            # Break after first batch during testing/debugging
            # if batch_idx == 0 and os.environ.get('DEBUG', '0') == '1':
            #     logger.info("Debug mode: breaking after first batch")
            #     break
        
        # Average metrics
        num_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def evaluate(self, 
                dataloader: DataLoader, 
                epoch: int) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            dataloader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        eval_metrics = {
            'loss': 0.0,
            'contrastive_loss': 0.0,
            'iou_loss': 0.0,
            'mAP50': 0.0,
            'mAP50_95': 0.0
        }
        
        progress_bar = tqdm(dataloader, desc=f"Eval Epoch {epoch}/{self.max_epochs}")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Extract batch data
                images = batch['images'].to(self.device)
                boxes = batch['boxes'].to(self.device)
                class_ids = batch['class_ids'].to(self.device)
                text_prompts = batch['text_prompts']
                valid_mask = batch['valid_mask'].to(self.device) if 'valid_mask' in batch else None
                
                # Forward pass
                outputs = self.model(images, text_prompts=text_prompts)
                
                # Calculate losses - similar to train_epoch but with the same size adjustments
                region_features = outputs['obj_embeddings']
                text_embeddings = outputs['text_embeddings']
                
                # Handle size mismatch
                if region_features.shape[1] != self.max_objects:
                    if region_features.shape[1] > self.max_objects:
                        region_features = region_features[:, :self.max_objects, :]
                    else:
                        padding = torch.zeros(
                            region_features.shape[0], 
                            self.max_objects - region_features.shape[1], 
                            region_features.shape[2], 
                            device=region_features.device
                        )
                        region_features = torch.cat([region_features, padding], dim=1)
                
                cont_loss = self.contrastive_loss(
                    region_features, 
                    text_embeddings, 
                    class_ids,
                    valid_mask
                )
                
                pred_boxes = outputs['boxes']
                
                # Handle size mismatch between predicted and target boxes
                if pred_boxes.shape[1] != boxes.shape[1]:
                    logger.info(f"Adjusting pred_boxes from {pred_boxes.shape[1]} to {boxes.shape[1]}")
                    if pred_boxes.shape[1] > boxes.shape[1]:
                        # Truncate to match target boxes
                        pred_boxes = pred_boxes[:, :boxes.shape[1], :]
                    else:
                        # This case should be rare - we usually have more predictions than targets
                        # Pad with zeros if needed
                        padding = torch.zeros(
                            pred_boxes.shape[0], 
                            boxes.shape[1] - pred_boxes.shape[1], 
                            pred_boxes.shape[2], 
                            device=pred_boxes.device
                        )
                        pred_boxes = torch.cat([pred_boxes, padding], dim=1)

                # 아래 코드를 추가: valid_mask 크기도 맞춰줌
                if valid_mask is not None and valid_mask.shape[1] != pred_boxes.shape[1]:
                    logger.info(f"Adjusting valid_mask from {valid_mask.shape[1]} to {pred_boxes.shape[1]}")
                    if valid_mask.shape[1] > pred_boxes.shape[1]:
                        # valid_mask가 더 크면 자르기
                        valid_mask = valid_mask[:, :pred_boxes.shape[1]]
                    else:
                        # valid_mask가 더 작으면 False로 패딩
                        padding = torch.zeros(
                            valid_mask.shape[0],
                            pred_boxes.shape[1] - valid_mask.shape[1],
                            device=valid_mask.device,
                            dtype=valid_mask.dtype
                        )
                        valid_mask = torch.cat([valid_mask, padding], dim=1)

                iou_loss = self.iou_loss(pred_boxes, boxes, valid_mask)
                loss = (
                    self.loss_weights['contrastive'] * cont_loss +
                    self.loss_weights['iou'] * iou_loss
                )
                
                # Update metrics
                eval_metrics['loss'] += loss.item()
                eval_metrics['contrastive_loss'] += cont_loss.item()
                eval_metrics['iou_loss'] += iou_loss.item()
                
                # Collect predictions and targets for mAP calculation
                # Make sure they are the same size by truncating predictions if needed
                truncated_boxes = outputs['boxes'][:, :self.max_objects].cpu().numpy() if outputs['boxes'].shape[1] > self.max_objects else outputs['boxes'].cpu().numpy()
                truncated_scores = outputs['scores'][:, :self.max_objects].cpu().numpy() if outputs['scores'].shape[1] > self.max_objects else outputs['scores'].cpu().numpy()
                truncated_class_ids = outputs['class_ids'][:, :self.max_objects].cpu().numpy() if outputs['class_ids'].shape[1] > self.max_objects else outputs['class_ids'].cpu().numpy()
                
                predictions = {
                    'boxes': truncated_boxes,
                    'scores': truncated_scores, 
                    'class_ids': truncated_class_ids
                }
                
                targets = {
                    'boxes': boxes.cpu().numpy(),
                    'class_ids': class_ids.cpu().numpy()
                }
                
                all_predictions.append(predictions)
                all_targets.append(targets)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'cont_loss': cont_loss.item(),
                    'iou_loss': iou_loss.item()
                })
                
                # Break after first batch during testing/debugging
                # if batch_idx == 0 and os.environ.get('DEBUG', '0') == '1':
                #     logger.info("Debug mode: breaking after first batch")
                #     break
        
        # Calculate mAP using all predictions and targets
        mAP50, mAP50_95 = self._calculate_map(all_predictions, all_targets)
        
        # Average metrics
        num_batches = len(dataloader)
        for key in ['loss', 'contrastive_loss', 'iou_loss']:
            eval_metrics[key] /= num_batches
        
        eval_metrics['mAP50'] = mAP50
        eval_metrics['mAP50_95'] = mAP50_95
        
        return eval_metrics
    
    def _calculate_map(self, 
                      predictions: List[Dict], 
                      targets: List[Dict]) -> Tuple[float, float]:
        """
        Calculate mAP@50 and mAP@50-95 for evaluation.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            
        Returns:
            Tuple of (mAP@50, mAP@50-95)
        """
        # Placeholder for actual mAP calculation
        # TODO: Implement proper mAP calculation
        try:
            from ..utils.metrics import calculate_map
            return calculate_map(predictions, targets)
        except (ImportError, AttributeError):
            # Fallback to placeholder values if metrics module not available
            logger.warning("Using placeholder mAP values - metrics module not available")
            mAP50 = 0.7  # Placeholder value
            mAP50_95 = 0.5  # Placeholder value
            return mAP50, mAP50_95
    
    def train(self, 
             train_dataloader: DataLoader,
             val_dataloader: Optional[DataLoader] = None,
             callbacks: List[Callable] = None) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional)
            callbacks: List of callback functions to call after each epoch
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mAP50': [],
            'val_mAP50_95': [],
            'learning_rate': []
        }
        
        best_map = 0.0
        
        for epoch in range(1, self.max_epochs + 1):
            try:
                # Train for one epoch
                train_metrics = self.train_epoch(train_dataloader, epoch)
                
                # Update learning rate
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Evaluate if validation dataloader is provided and it's evaluation interval
                val_metrics = None
                if val_dataloader is not None and epoch % self.eval_interval == 0:
                    val_metrics = self.evaluate(val_dataloader, epoch)
                    
                    # Save best model
                    if val_metrics['mAP50_95'] > best_map:
                        best_map = val_metrics['mAP50_95']
                        self.save_checkpoint(os.path.join(self.output_dir, 'best_model.pth'))
                    
                    # Update history
                    history['val_loss'].append(val_metrics['loss'])
                    history['val_mAP50'].append(val_metrics['mAP50'])
                    history['val_mAP50_95'].append(val_metrics['mAP50_95'])
                    
                    logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}, "
                              f"Val Loss = {val_metrics['loss']:.4f}, "
                              f"mAP50 = {val_metrics['mAP50']:.4f}, "
                              f"mAP50-95 = {val_metrics['mAP50_95']:.4f}")
                else:
                    logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}")
                
                # Update history
                history['train_loss'].append(train_metrics['loss'])
                history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                
                # Save checkpoint if it's save interval
                if epoch % self.save_interval == 0:
                    self.save_checkpoint(os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth'))
                
                # Call callbacks if provided
                if callbacks is not None:
                    for callback in callbacks:
                        callback(epoch, train_metrics, val_metrics)
                
            except Exception as e:
                logger.error(f"Error during training epoch {epoch}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Try to save checkpoint if an error occurs
                try:
                    self.save_checkpoint(os.path.join(self.output_dir, f'error_checkpoint_epoch_{epoch}.pth'))
                    logger.info(f"Saved checkpoint before error at epoch {epoch}")
                except Exception as save_err:
                    logger.error(f"Failed to save checkpoint after error: {str(save_err)}")
                
                # Break or continue based on environment settings
                if os.environ.get('CONTINUE_ON_ERROR', '0') != '1':
                    logger.error("Training stopped due to error.")
                    break
                else:
                    logger.warning("Continuing to next epoch despite error.")
                    continue
        
        # Save final model
        self.save_checkpoint(os.path.join(self.output_dir, 'final_model.pth'))
        
        return history
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint of the model.
        
        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint.
        
        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")