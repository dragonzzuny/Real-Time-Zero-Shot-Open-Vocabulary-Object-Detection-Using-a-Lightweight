# debug_architecture.py
import torch
import logging
import sys
import os
from typing import List, Dict, Tuple, Optional
import yaml
import argparse

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """환경 설정 및 필요한 모듈 임포트"""
    # 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    try:
        from yolo_clip_detector.model.yolo_clip import YOLOCLIP
        from yolo_clip_detector.data.coco_dataset import COCODataset
        from yolo_clip_detector.train.trainer import YOLOCLIPTrainer
        from yolo_clip_detector.loss.region_text_contrastive import RegionTextContrastiveLoss
        from yolo_clip_detector.config.default_config import TrainingConfig
        
        logger.info("모듈 임포트 성공")
        return True
    except ImportError as e:
        logger.error(f"모듈 임포트 실패: {e}")
        return False

def inspect_model_architecture(model):
    """모델 아키텍처 검사"""
    logger.info("===== 모델 아키텍처 검사 =====")
    
    # 모델 구조 출력
    logger.info(f"모델 타입: {type(model)}")
    
    # 주요 컴포넌트 검사
    components = ['backbone', 'neck', 'text_encoder', 'contrastive_heads', 'box_head']
    for comp in components:
        if hasattr(model, comp):
            component = getattr(model, comp)
            logger.info(f"{comp} 타입: {type(component)}")
            
            # contrastive_heads는 리스트이므로 별도 처리
            if comp == 'contrastive_heads' and isinstance(component, torch.nn.ModuleList):
                logger.info(f"contrastive_heads 개수: {len(component)}")
                if len(component) > 0:
                    logger.info(f"첫 번째 head 타입: {type(component[0])}")
    
    # YOLOv8 백본 스트라이드 값 검사
    if hasattr(model, 'strides'):
        logger.info(f"모델 스트라이드: {model.strides}")
    
    # 오프라인 모드 검사
    if hasattr(model, 'offline_mode'):
        logger.info(f"오프라인 모드 설정: {model.offline_mode}")
    
    # 임베딩 차원 검사
    if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'embed_dim'):
        logger.info(f"텍스트 임베딩 차원: {model.text_encoder.embed_dim}")

def forward_dummy_data(model, batch_size=2, img_size=(640, 640), num_classes=16, max_objects=100):
    """더미 데이터로 모델 포워드 패스 테스트"""
    logger.info("===== 더미 데이터로 모델 포워드 패스 테스트 =====")
    
    # 더미 이미지 생성 (batch_size, 3, height, width)
    device = next(model.parameters()).device
    dummy_images = torch.randn(batch_size, 3, img_size[0], img_size[1], device=device)
    
    # 더미 텍스트 프롬프트 생성
    dummy_text_prompts = [f"a photo of a class_{i}" for i in range(num_classes)]
    
    # 포워드 패스 실행
    logger.info("포워드 패스 시작...")
    try:
        with torch.no_grad():
            outputs = model(dummy_images, text_prompts=dummy_text_prompts)
        
        # 출력 형태 및 크기 검사
        logger.info("포워드 패스 성공!")
        logger.info("출력 키:")
        tensor_shapes = {}
        
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                shape = value.shape
                tensor_shapes[key] = shape
                logger.info(f"  {key}: shape={shape}, type={value.dtype}")
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                shapes = [v.shape for v in value]
                tensor_shapes[key] = shapes
                logger.info(f"  {key}: list of tensors with shapes={shapes}")
            else:
                logger.info(f"  {key}: type={type(value)}")
        
        # 특별히 중요한 텐서 크기에 주목
        if 'obj_embeddings' in tensor_shapes and 'boxes' in tensor_shapes:
            obj_embed_regions = tensor_shapes['obj_embeddings'][1]  # 두 번째 차원 (region 수)
            num_boxes = tensor_shapes['boxes'][1]  # 두 번째 차원 (box 수)
            
            logger.info(f"\n중요 크기 비교:")
            logger.info(f"  obj_embeddings의 region 수: {obj_embed_regions}")
            logger.info(f"  boxes의 box 수: {num_boxes}")
            logger.info(f"  설정된 max_objects: {max_objects}")
            
            if obj_embed_regions != num_boxes:
                logger.warning(f"  [경고] region 수와 box 수가 일치하지 않습니다!")
            
            if obj_embed_regions != max_objects or num_boxes != max_objects:
                logger.warning(f"  [경고] 텐서 크기가 max_objects({max_objects})와 일치하지 않습니다!")
        
        return outputs
    except Exception as e:
        logger.error(f"포워드 패스 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def count_feature_map_regions(model, img_size=(640, 640), print_details=True):
    """각 피처맵 레벨에서 생성되는 region 수 계산"""
    logger.info("===== 피처맵 레벨별 region 수 계산 =====")
    
    if not hasattr(model, 'strides'):
        logger.error("모델에 strides 속성이 없습니다")
        return None
    
    strides = model.strides
    total_regions = 0
    level_details = []
    
    for i, stride in enumerate(strides):
        # 각 레벨의 피처맵 크기 계산
        h, w = img_size[0] // stride, img_size[1] // stride
        regions = h * w
        total_regions += regions
        level_details.append((i, stride, h, w, regions))
    
    if print_details:
        logger.info(f"이미지 크기: {img_size}")
        logger.info(f"스트라이드: {strides}")
        logger.info("\n피처맵 레벨별 상세 정보:")
        
        for level, stride, height, width, regions in level_details:
            logger.info(f"  레벨 {level} (stride={stride}): 피처맵 크기=({height}, {width}), regions={regions}")
        
        logger.info(f"\n총 region 수: {total_regions}")
    
    return total_regions, level_details

def analyze_loss_function(model, outputs, max_objects=100, num_classes=16):
    """손실 함수 분석"""
    logger.info("===== 손실 함수 분석 =====")
    
    batch_size = outputs['boxes'].shape[0]
    device = next(model.parameters()).device
    
    # 더미 레이블 데이터 생성
    dummy_boxes = torch.randn(batch_size, max_objects, 4, device=device)
    dummy_class_ids = torch.randint(0, num_classes-1, (batch_size, max_objects), device=device)
    dummy_valid_mask = torch.ones(batch_size, max_objects, dtype=torch.bool, device=device)
    
    # region_features 크기 검사
    obj_embeddings = outputs['obj_embeddings']
    logger.info(f"obj_embeddings shape: {obj_embeddings.shape}")
    
    # obj_embeddings의 region 수와 max_objects 비교
    num_regions = obj_embeddings.shape[1]
    logger.info(f"region 수: {num_regions}, max_objects: {max_objects}")
    
    if num_regions != max_objects:
        logger.warning(f"크기 불일치: region 수({num_regions})가 max_objects({max_objects})와 다릅니다")
    
    # 텍스트 임베딩 크기 검사
    text_embeddings = outputs.get('text_embeddings')
    if text_embeddings is not None:
        logger.info(f"text_embeddings shape: {text_embeddings.shape}")
    else:
        logger.warning("text_embeddings not found in outputs")
    
    # RegionTextContrastiveLoss 테스트
    try:
        from yolo_clip_detector.loss.region_text_contrastive import RegionTextContrastiveLoss
        loss_fn = RegionTextContrastiveLoss(temperature=0.1)
        
        logger.info("\n손실 함수 테스트:")
        logger.info(f"  입력 크기:")
        logger.info(f"  - obj_embeddings: {obj_embeddings.shape}")
        logger.info(f"  - text_embeddings: {text_embeddings.shape if text_embeddings is not None else 'None'}")
        logger.info(f"  - dummy_class_ids: {dummy_class_ids.shape}")
        logger.info(f"  - dummy_valid_mask: {dummy_valid_mask.shape}")
        
        # 크기 불일치 감지 및 조정
        if num_regions != max_objects:
            logger.info("\n크기 조정 시도:")
            
            if num_regions > max_objects:
                logger.info(f"  obj_embeddings를 {max_objects}개로 자릅니다")
                adjusted_embeddings = obj_embeddings[:, :max_objects, :]
            else:
                logger.info(f"  obj_embeddings를 {max_objects}개로 패딩합니다")
                padding = torch.zeros(batch_size, max_objects - num_regions, obj_embeddings.shape[-1], 
                                      device=obj_embeddings.device)
                adjusted_embeddings = torch.cat([obj_embeddings, padding], dim=1)
            
            logger.info(f"  조정 후 크기: {adjusted_embeddings.shape}")
        else:
            adjusted_embeddings = obj_embeddings
        
        # 손실 계산 시도
        if text_embeddings is not None:
            try:
                loss = loss_fn(adjusted_embeddings, text_embeddings, dummy_class_ids, dummy_valid_mask)
                logger.info(f"  손실 계산 성공: {loss.item()}")
            except Exception as e:
                logger.error(f"  조정 후에도 손실 계산 실패: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # 원본 크기로 다시 시도
                try:
                    logger.info("  원본 크기로 손실 계산 재시도...")
                    loss = loss_fn(obj_embeddings, text_embeddings, dummy_class_ids[:, :num_regions], dummy_valid_mask[:, :num_regions])
                    logger.info(f"  원본 크기로 손실 계산 성공: {loss.item()}")
                except Exception as e2:
                    logger.error(f"  원본 크기로도 손실 계산 실패: {e2}")
        else:
            logger.warning("  text_embeddings가 없어 손실 계산을 건너뜁니다")
    except Exception as e:
        logger.error(f"손실 함수 테스트 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())

def inspect_dataset(config_path=None):
    """데이터셋 검사"""
    logger.info("===== 데이터셋 검사 =====")
    
    try:
        from yolo_clip_detector.data.coco_dataset import COCODataset
        from yolo_clip_detector.config.default_config import TrainingConfig
        
        # 설정 로드
        config = TrainingConfig()
        if config_path is not None:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
                for k, v in config_dict.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
        
        # 데이터셋 정보 출력
        logger.info(f"img_size: {config.img_size}")
        logger.info(f"max_objects: {config.max_objects}")
        logger.info(f"클래스 수: {len(config.class_names)}")
        logger.info(f"변환(transform) 적용: {'있음' if hasattr(config, 'transform') and config.transform else '없음'}")
        
        # 주요 파일 경로 확인
        anno_path = config.train_anno_path
        img_dir = config.train_img_dir
        
        logger.info(f"annotation 파일 경로: {anno_path}")
        logger.info(f"이미지 디렉토리 경로: {img_dir}")
        
        # 파일 존재 여부 확인
        anno_exists = os.path.exists(anno_path)
        img_dir_exists = os.path.exists(img_dir)
        
        logger.info(f"annotation 파일 존재: {anno_exists}")
        logger.info(f"이미지 디렉토리 존재: {img_dir_exists}")
        
        if not anno_exists or not img_dir_exists:
            logger.error("필요한 데이터 파일이 없습니다. 경로를 확인하세요.")
            return None
        
        # 데이터셋 생성 시도
        try:
            dataset = COCODataset(
                anno_path=anno_path,
                img_dir=img_dir,
                class_names=config.class_names,
                img_size=config.img_size,
                max_objects=config.max_objects
            )
            logger.info(f"데이터셋 생성 성공: {len(dataset)} 이미지")
            
            # 첫 번째 항목 검사
            logger.info("\n첫 번째 데이터셋 항목 분석:")
            first_item = dataset[0]
            for key, value in first_item.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: shape={value.shape}, type={value.dtype}")
                elif isinstance(value, list):
                    logger.info(f"  {key}: list 길이={len(value)}, 내용 샘플={value[:3] if len(value) > 3 else value}")
                else:
                    logger.info(f"  {key}: type={type(value)}")
            
            # boxes와 class_ids 상세 분석
            if 'boxes' in first_item and 'class_ids' in first_item:
                boxes = first_item['boxes']
                class_ids = first_item['class_ids']
                valid_mask = first_item.get('valid_mask')
                
                # 실제 객체 수 계산 (valid_mask가 True인 항목 수)
                if valid_mask is not None:
                    actual_objects = valid_mask.sum().item()
                else:
                    # valid_mask가 없으면 0이 아닌 class_ids 수를 계산
                    actual_objects = (class_ids > 0).sum().item()
                
                logger.info(f"\n객체 통계:")
                logger.info(f"  max_objects 설정: {config.max_objects}")
                logger.info(f"  실제 객체 수: {actual_objects}")
                logger.info(f"  boxes 패딩 크기: {boxes.shape}")
                logger.info(f"  class_ids 패딩 크기: {class_ids.shape}")
                
                if actual_objects < config.max_objects:
                    logger.info(f"  패딩 영역: {config.max_objects - actual_objects}개 (전체의 {(config.max_objects - actual_objects) / config.max_objects * 100:.1f}%)")
            
            return dataset
        except Exception as e:
            logger.error(f"데이터셋 생성 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    except ImportError as e:
        logger.error(f"데이터셋 모듈 임포트 실패: {e}")
        return None

def analyze_model_and_dataset_compatibility(model, dataset, device, max_objects=100):
    """모델과 데이터셋 간의 호환성 분석"""
    logger.info("===== 모델-데이터셋 호환성 분석 =====")
    
    if model is None or dataset is None:
        logger.error("모델 또는 데이터셋이 None입니다")
        return
    
    # 모델의 region 수 계산
    total_regions, _ = count_feature_map_regions(model, dataset.img_size, print_details=False)
    logger.info(f"모델이 생성하는 총 region 수: {total_regions}")
    logger.info(f"데이터셋 max_objects: {dataset.max_objects}")
    
    if total_regions != dataset.max_objects:
        logger.warning(f"[경고] 모델 region 수({total_regions})와 데이터셋 max_objects({dataset.max_objects})가 일치하지 않습니다")
    
    # 첫 번째 배치 가져오기
    try:
        from torch.utils.data import DataLoader
        from yolo_clip_detector.utils.data_utils import custom_collate_fn
        
        logger.info("\n데이터로더 설정 및 배치 가져오기...")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
        batch = next(iter(dataloader))
        
        logger.info("배치 구조:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: shape={value.shape}, type={value.dtype}")
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], list):
                    logger.info(f"  {key}: list of lists, 첫 항목 길이={len(value[0])}")
                else:
                    logger.info(f"  {key}: list 길이={len(value)}")
            else:
                logger.info(f"  {key}: type={type(value)}")
        
        # 배치 데이터로 모델 포워드 패스 시도
        logger.info("\n배치 데이터로 포워드 패스 시도...")
        images = batch['images'].to(device)
        text_prompts = batch['text_prompts']
        
        try:
            with torch.no_grad():
                outputs = model(images, text_prompts=text_prompts)
            
            logger.info("배치 데이터로 포워드 패스 성공!")
            
            # 주요 텐서 크기 비교
            obj_embeddings = outputs['obj_embeddings']
            boxes = batch['boxes'].to(device)
            class_ids = batch['class_ids'].to(device)
            
            logger.info(f"\n주요 텐서 크기 비교:")
            logger.info(f"  obj_embeddings: {obj_embeddings.shape}")
            logger.info(f"  boxes: {boxes.shape}")
            logger.info(f"  class_ids: {class_ids.shape}")
            
            # region 수와 max_objects 비교
            obj_embed_regions = obj_embeddings.shape[1]
            
            if obj_embed_regions != dataset.max_objects:
                logger.warning(f"  [경고] region 수({obj_embed_regions})가 max_objects({dataset.max_objects})와 일치하지 않습니다")
                logger.info("  이 불일치가 손실 함수 계산 시 RuntimeError의 원인일 수 있습니다")
            
            # 손실 함수 계산 시도
            try:
                from yolo_clip_detector.loss.region_text_contrastive import RegionTextContrastiveLoss
                loss_fn = RegionTextContrastiveLoss(temperature=0.1)
                
                # 텍스트 임베딩 출력
                text_embeddings = outputs.get('text_embeddings')
                if text_embeddings is None:
                    logger.warning("  text_embeddings가 출력에 없어 모델의 text_encoder를 사용합니다")
                    text_embeddings = model.text_encoder(text_prompts).to(device)
                
                logger.info(f"  text_embeddings: {text_embeddings.shape}")
                
                # 크기 조정
                logger.info("\n손실 함수 계산 테스트:")
                logger.info(f"  1. 원본 크기로 손실 계산 시도...")
                
                try:
                    # 원본 크기로 먼저 시도
                    loss = loss_fn(obj_embeddings, text_embeddings, class_ids, batch['valid_mask'].to(device))
                    logger.info(f"  성공! 손실값: {loss.item()}")
                except Exception as e:
                    logger.error(f"  실패: {e}")
                    
                    # 크기 조정 후 시도
                    logger.info(f"\n  2. 크기 조정 후 손실 계산 시도...")
                    
                    num_regions = obj_embeddings.shape[1]
                    max_objects = class_ids.shape[1]
                    
                    if num_regions != max_objects:
                        logger.info(f"  obj_embeddings를 {num_regions}에서 {max_objects}로 조정...")
                        
                        if num_regions > max_objects:
                            adjusted_embeddings = obj_embeddings[:, :max_objects, :]
                        else:
                            padding = torch.zeros(
                                obj_embeddings.shape[0], 
                                max_objects - num_regions, 
                                obj_embeddings.shape[-1], 
                                device=device
                            )
                            adjusted_embeddings = torch.cat([obj_embeddings, padding], dim=1)
                        
                        try:
                            loss = loss_fn(adjusted_embeddings, text_embeddings, class_ids, batch['valid_mask'].to(device))
                            logger.info(f"  성공! 손실값: {loss.item()}")
                        except Exception as e2:
                            logger.error(f"  크기 조정 후에도 실패: {e2}")
            except Exception as e:
                logger.error(f"손실 계산 시도 중 오류 발생: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
        except Exception as e:
            logger.error(f"배치 데이터로 포워드 패스 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"데이터 로딩 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())

def analyze_region_text_contrastive_loss():
    """RegionTextContrastiveLoss 클래스 코드 분석"""
    logger.info("===== RegionTextContrastiveLoss 분석 =====")
    
    try:
        import inspect
        from yolo_clip_detector.loss.region_text_contrastive import RegionTextContrastiveLoss
        
        # 소스 코드 가져오기
        source = inspect.getsource(RegionTextContrastiveLoss.forward)
        
        # 크기 조정 로직 검색
        size_adjustment_code = None
        for line in source.splitlines():
            if 'shape' in line and ('adjust' in line.lower() or 'region_features' in line):
                size_adjustment_code = line.strip()
                break
        
        logger.info("RegionTextContrastiveLoss.forward 메서드 분석:")
        logger.info(f"  크기 조정 로직 존재 여부: {'있음' if size_adjustment_code else '없음'}")
        
        if not size_adjustment_code:
            logger.warning("  [경고] 손실 함수에 region_features와 labels 간 크기 조정 로직이 없습니다")
            logger.info("  이것이 크기 불일치 오류의 원인일 수 있습니다")
        
        # 문제 해결 제안
        logger.info("\n해결 방안:")
        logger.info("  RegionTextContrastiveLoss.forward 메서드에 다음과 같은 크기 조정 로직을 추가하세요:")
        
        solution_code = """
        def forward(self, 
                    region_features: torch.Tensor, 
                    text_embeddings: torch.Tensor, 
                    region_labels: torch.Tensor,
                    valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            
            batch_size, num_regions, embed_dim = region_features.shape
            num_classes = text_embeddings.shape[1]
            max_objects = region_labels.shape[1]
            
            # 크기 조정 로직 추가
            if num_regions != max_objects:
                logger.info(f"Adjusting region_features from {num_regions} to {max_objects}")
                if num_regions > max_objects:
                    region_features = region_features[:, :max_objects, :]
                else:
                    padding = torch.zeros(batch_size, max_objects - num_regions, embed_dim,
                                         device=region_features.device)
                    region_features = torch.cat([region_features, padding], dim=1)
                    
                    if valid_mask is None:
                        valid_mask = torch.ones(batch_size, num_regions, dtype=torch.bool, 
                                              device=region_features.device)
                    
                    padding_mask = torch.zeros(batch_size, max_objects - num_regions,
                                             dtype=torch.bool, device=region_features.device)
                    valid_mask = torch.cat([valid_mask, padding_mask], dim=1)
            
            # 기존 코드 계속...
        """
        logger.info(solution_code)
        
    except ImportError as e:
        logger.error(f"RegionTextContrastiveLoss 분석 중 오류 발생: {e}")
    except Exception as e:
        logger.error(f"RegionTextContrastiveLoss 분석 중 예상치 못한 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())

def summarize_issues_and_solutions():
    """발견된 문제점과 해결 방안 요약"""
    logger.info("\n===== 문제점 및 해결 방안 요약 =====")
    
    issues = [
        "1. 모델이 생성하는 region 수와 데이터셋의 max_objects 값 불일치",
        "2. RegionTextContrastiveLoss에 크기 조정 로직 부재",
        "3. 모델의 forward 메서드에서 obj_embeddings 개수 제한 미설정",
        "4. YOLOCLIPTrainer에서 모델 출력과 타겟 데이터 크기 불일치 처리 필요"
    ]
    
    solutions = [
        "1. RegionTextContrastiveLoss 클래스의 forward 메서드에 크기 조정 로직 추가",
        "2. YOLOCLIPTrainer 클래스의 train_epoch 메서드에서 크기 조정 로직 추가",
        "3. YOLOCLIP 모델 클래스에서 forward 메서드의 obj_embeddings 생성 부분에 max_objects 제한 추가",
        "4. 데이터셋 생성 시와 모델 초기화 시 동일한 max_objects 값을 사용하도록 코드 수정"
    ]
    
    code_changes = {
        "RegionTextContrastiveLoss.forward": """
    def forward(self, 
                region_features: torch.Tensor, 
                text_embeddings: torch.Tensor, 
                region_labels: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, num_regions, embed_dim = region_features.shape
        num_classes = text_embeddings.shape[1]
        max_objects = region_labels.shape[1]
        
        # 크기 조정 로직 추가
        if num_regions != max_objects:
            logger.info(f"Adjusting region_features from {num_regions} to {max_objects}")
            if num_regions > max_objects:
                region_features = region_features[:, :max_objects, :]
            else:
                padding = torch.zeros(batch_size, max_objects - num_regions, embed_dim,
                                     device=region_features.device)
                region_features = torch.cat([region_features, padding], dim=1)
                
                if valid_mask is None:
                    valid_mask = torch.ones(batch_size, num_regions, dtype=torch.bool, 
                                          device=region_features.device)
                
                padding_mask = torch.zeros(batch_size, max_objects - num_regions,
                                         dtype=torch.bool, device=region_features.device)
                valid_mask = torch.cat([valid_mask, padding_mask], dim=1)
        
        # 나머지 코드는 기존대로 유지...
        """,
        
        "YOLOCLIPTrainer.train_epoch": """
    # train_epoch 메서드 내에서 (region_features 처리 부분)
    region_features = outputs['obj_embeddings']
    
    # 크기 불일치 처리
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
        """,
        
        "YOLOCLIP.forward": """
    # obj_embeddings 생성 부분 수정
    obj_embeddings = []
    for i, embed in enumerate(all_obj_embeddings):
        B, C, H, W = embed.shape
        obj_embeddings.append(embed.permute(0, 2, 3, 1).reshape(B, H*W, C))
    
    obj_embeddings = torch.cat(obj_embeddings, dim=1)
    
    # max_objects 제한 추가
    if hasattr(self, 'max_objects') and obj_embeddings.shape[1] > self.max_objects:
        obj_embeddings = obj_embeddings[:, :self.max_objects, :]
        """
    }
    
    for issue in issues:
        logger.info(issue)
    
    logger.info("\n해결 방안:")
    for solution in solutions:
        logger.info(solution)
    
    logger.info("\n코드 수정 예시:")
    for location, code in code_changes.items():
        logger.info(f"\n{location}:")
        logger.info(code)
    
    # 추가적인 조치 안내
    logger.info("\n수정 후 테스트 방법:")
    logger.info("1. 각 파일에 위 코드 수정 사항을 적용")
    logger.info("2. 이 디버깅 스크립트를 다시 실행하여 문제 해결 여부 확인")
    logger.info("3. 문제가 계속되면 로그를 분석하여 추가 문제점 파악")
    logger.info("4. 학습 스크립트를 실행하여 실제 학습 가능 여부 확인")

def main(config_path=None):
    """메인 디버깅 함수"""
    parser = argparse.ArgumentParser(description='YOLO-CLIP 모델 디버깅 도구')
    parser.add_argument('--config', type=str, default=None, help='설정 파일 경로')
    parser.add_argument('--max_objects', type=int, default=100, help='최대 객체 수 (데이터셋 설정)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[640, 640], help='이미지 크기 (높이 너비)')
    parser.add_argument('--gpu', type=int, default=0, help='사용할 GPU 인덱스 (0, 1, ...)')
    args = parser.parse_args()
    
    if config_path is None and args.config is not None:
        config_path = args.config
    
    max_objects = args.max_objects
    img_size = tuple(args.img_size)
    
    logger.info("===== YOLO-CLIP 모델 디버깅 시작 =====")
    logger.info(f"설정 파일: {config_path if config_path else '기본값 사용'}")
    logger.info(f"최대 객체 수: {max_objects}")
    logger.info(f"이미지 크기: {img_size}")
    
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger.info(f"GPU {args.gpu}를 사용하도록 설정했습니다")
    
    # 환경 설정
    if not setup_environment():
        logger.error("환경 설정 실패, 종료합니다")
        return
    
    # 설정 로드
    try:
        from yolo_clip_detector.config.default_config import TrainingConfig
        config = TrainingConfig()
        if config_path is not None:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                for k, v in config_dict.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
        
        # 명령행 인수로 받은 값으로 덮어쓰기
        config.max_objects = max_objects
        config.img_size = img_size
    except Exception as e:
        logger.error(f"설정 로드 중 오류 발생: {e}")
        config = None
    
    # 모델 초기화
    try:
        from yolo_clip_detector.model.yolo_clip import YOLOCLIP
        
        # 모델 파라미터 (설정이 없으면 기본값 사용)
        backbone_variant = getattr(config, 'backbone_variant', 'n') if config else 'n'
        clip_model = getattr(config, 'clip_model', 'ViT-B/32') if config else 'ViT-B/32'
        embed_dim = getattr(config, 'embed_dim', 512) if config else 512
        num_classes = len(getattr(config, 'class_names', [])) if config else 80
        
        logger.info(f"모델 초기화: backbone={backbone_variant}, clip_model={clip_model}, embed_dim={embed_dim}, num_classes={num_classes}")
        
        # GPU 사용 가능 여부 확인
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"사용 중인 디바이스: {device}")
        
        # 모델 생성
        model = YOLOCLIP(
            backbone_variant=backbone_variant,
            clip_model=clip_model,
            embed_dim=embed_dim,
            num_classes=num_classes
        ).to(device)
        
        # max_objects 속성 추가 (수정사항 테스트용)
        model.max_objects = max_objects
        
        logger.info("모델 초기화 성공")
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        model = None
    
    # 모델 아키텍처 검사
    if model is not None:
        inspect_model_architecture(model)
        
        # 피처맵 레벨별 region 수 계산
        count_feature_map_regions(model, img_size)
        
        # 더미 데이터로 포워드 테스트
        outputs = forward_dummy_data(model, batch_size=2, img_size=img_size, num_classes=num_classes, max_objects=max_objects)
        
        # 손실 함수 분석
        if outputs is not None:
            analyze_loss_function(model, outputs, max_objects=max_objects, num_classes=num_classes)
    
    # 데이터셋 검사
    dataset = inspect_dataset(config_path)
    
    # 모델과 데이터셋 호환성 분석
    if model is not None and dataset is not None:
        analyze_model_and_dataset_compatibility(model, dataset, device, max_objects=max_objects)
    
    # RegionTextContrastiveLoss 분석
    analyze_region_text_contrastive_loss()
    
    # 문제점 및 해결 방안 요약
    summarize_issues_and_solutions()
    
    logger.info("===== 디버깅 완료 =====")
    
    return model, dataset

if __name__ == "__main__":
    # 설정 파일 경로를 명령행 인수로 받음
    main()