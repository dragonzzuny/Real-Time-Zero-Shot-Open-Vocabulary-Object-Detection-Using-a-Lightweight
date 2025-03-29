import os
import sys
import argparse
import chardet
import logging
import yaml
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 현재 디렉토리를 PYTHONPATH에 추가
sys.path.insert(0, os.path.abspath('.'))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='Train YOLO-CLIP model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--backbone', type=str, default=None, help='Backbone variant')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--no_eval', action='store_true', help='Disable evaluation during training')
    parser.add_argument('--devices', type=str, default='0', help='Devices to use (comma-separated)')
    return parser.parse_args()

def create_transforms(img_size, training=True):
    if training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), p=0.2),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids']))
    else:
        return A.Compose([
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids']))

def custom_collate_fn(batch):
    return {
        'images': torch.stack([b['images'] for b in batch]),
        'boxes': torch.stack([b['boxes'] for b in batch]),
        'class_ids': torch.stack([b['class_ids'] for b in batch]),
        'valid_mask': torch.stack([b['valid_mask'] for b in batch]),
        'text_prompts': [b['text_prompts'] for b in batch],
        'image_id': [b['image_id'] for b in batch],
        'orig_size': [b['orig_size'] for b in batch]
    }

def main():
    args = parse_args()
    from yolo_clip_detector.config.default_config import TrainingConfig
    config = TrainingConfig()

    if args.config is not None:
        logger.info(f"Loading config from {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            for k, v in config_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)

    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.backbone is not None:
        config.backbone_variant = args.backbone
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"train_{config.backbone_variant}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config.to_dict(), f)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = create_transforms(config.img_size, training=True)
    val_transform = create_transforms(config.img_size, training=False)

    from yolo_clip_detector.data.coco_dataset import COCODataset

    train_dataset = COCODataset(
        anno_path=config.train_anno_path,
        img_dir=config.train_img_dir,
        class_names=config.class_names,
        img_size=config.img_size,
        transform=train_transform,
        mode='train',
        mosaic_prob=config.mosaic_prob,
        max_objects=config.max_objects
    )

    val_dataset = None
    if not args.no_eval:
        val_dataset = COCODataset(
            anno_path=config.val_anno_path,
            img_dir=config.val_img_dir,
            class_names=config.class_names,
            img_size=config.img_size,
            transform=val_transform,
            mode='val',
            max_objects=config.max_objects
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

    from yolo_clip_detector.model.yolo_clip import YOLOCLIP

    model = YOLOCLIP(
        backbone_variant=config.backbone_variant,
        clip_model=config.clip_model,
        embed_dim=config.embed_dim,
        num_classes=len(config.class_names),
        strides=[8, 16, 32],
        reg_max=config.reg_max,
        offline_mode=False
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(train_dataloader) * config.max_epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        div_factor=25,
        final_div_factor=1e4
    )

    from yolo_clip_detector.train.trainer import YOLOCLIPTrainer

    trainer = YOLOCLIPTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        output_dir=output_dir,
        max_epochs=config.max_epochs,
        save_interval=config.save_interval,
        eval_interval=config.eval_interval,
        temperature=config.temperature,
        iou_type=config.iou_type,
        label_smoothing=config.label_smoothing,
        loss_weights=config.loss_weights
    )

    if args.resume is not None:
        logger.info(f"Resuming from checkpoint {args.resume}")
        trainer.load_checkpoint(args.resume)

    logger.info("Starting training...")
    trainer.train(train_dataloader, val_dataloader)
    logger.info(f"Training completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()