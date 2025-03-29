# yolo_clip_detector/utils/data_utils.py
import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def custom_collate_fn(batch):
    """
    커스텀 collate 함수 - 가변 길이 리스트(text_prompts)를 처리합니다.
    
    Args:
        batch: 데이터셋에서 가져온 샘플들의 리스트
        
    Returns:
        병합된 배치 데이터가 포함된 딕셔너리
    """
    if not batch:
        return {}
    
    elem = batch[0]
    result = {}
    
    for key in elem:
        if key == 'text_prompts':
            # text_prompts는 각 배치 항목의 리스트를 그대로 유지
            result[key] = [d[key] for d in batch]
        else:
            # 다른 필드는 기본 collate 함수 사용
            try:
                result[key] = torch.utils.data._utils.collate.default_collate([d[key] for d in batch])
            except (TypeError, RuntimeError) as e:
                logger.warning(f"Could not collate field {key}: {str(e)}")
                # 기본 collate가 실패하면 리스트로 유지
                result[key] = [d[key] for d in batch]
    
    return result


def compute_padding_size(original_size, target_size):
    """
    원본 이미지 크기를 기반으로 패딩 크기를 계산합니다.
    
    Args:
        original_size: 원본 이미지 크기 (높이, 너비)
        target_size: 목표 이미지 크기 (높이, 너비)
        
    Returns:
        패딩 크기 (위, 아래, 왼쪽, 오른쪽)
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size
    
    # 비율 계산
    ratio = min(target_h / orig_h, target_w / orig_w)
    
    # 새 크기 계산
    new_h, new_w = int(orig_h * ratio), int(orig_w * ratio)
    
    # 패딩 계산
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # 패딩 크기 (위, 아래, 왼쪽, 오른쪽)
    padding = (pad_h, target_h - new_h - pad_h, pad_w, target_w - new_w - pad_w)
    
    return padding


def adjust_box_coordinates(boxes, padding, scale_factor, original_size, target_size):
    """
    패딩과 스케일링을 고려하여 바운딩 박스 좌표를 조정합니다.
    
    Args:
        boxes: 바운딩 박스 좌표 (x1, y1, x2, y2) 형식
        padding: 패딩 값 (위, 아래, 왼쪽, 오른쪽)
        scale_factor: 스케일 팩터
        original_size: 원본 이미지 크기 (높이, 너비)
        target_size: 목표 이미지 크기 (높이, 너비)
        
    Returns:
        조정된 바운딩 박스 좌표
    """
    pad_top, _, pad_left, _ = padding
    
    # 박스 조정
    adjusted_boxes = boxes.clone()
    
    # 스케일 적용
    adjusted_boxes[:, [0, 2]] = adjusted_boxes[:, [0, 2]] * scale_factor
    adjusted_boxes[:, [1, 3]] = adjusted_boxes[:, [1, 3]] * scale_factor
    
    # 패딩 오프셋 적용
    adjusted_boxes[:, [0, 2]] += pad_left
    adjusted_boxes[:, [1, 3]] += pad_top
    
    return adjusted_boxes