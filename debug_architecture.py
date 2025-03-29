# debug_architecture.py
import torch
import logging
import sys
import os
from typing import List, Dict, Tuple, Optional
import yaml

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
    logger.info(f"모델 속성: {dir(model)}")
    
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

def forward_dummy_data(model, batch_size=2, img_size=(640, 640), num_classes=16, max_objects=100):
    """더미 데이터로 모델 포워드 패스 테스트"""
    logger.info("===== 더미 데이터로 모델 포워드 패스 테스트 =====")
    
    # 더미 이미지 생성 (batch_size, 3, height, width)
    dummy_images = torch.randn(batch_size, 3, img_size[0], img_size[1], device=next(model.parameters()).device)
    
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
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: shape={value.shape}, type={value.dtype}")
            else:
                logger.info(f"  {key}: type={type(value)}")
        
        return outputs
    except Exception as e:
        logger.error(f"포워드 패스 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def analyze_loss_function(model, outputs, max_objects=100, num_classes=16):
    """손실 함수 분석"""
    logger.info("===== 손실 함수 분석 =====")
    
    batch_size = outputs['boxes'].shape[0]
    
    # 더미 레이블 데이터 생성
    dummy_boxes = torch.randn(batch_size, max_objects, 4, device=next(model.parameters()).device)
    dummy_class_ids = torch.randint(0, num_classes-1, (batch_size, max_objects), device=next(model.parameters()).device)
    dummy_valid_mask = torch.ones(batch_size, max_objects, dtype=torch.bool, device=next(model.parameters()).device)
    
    # region_features 크기 검사
    obj_embeddings = outputs['obj_embeddings']
    logger.info(f"obj_embeddings shape: {obj_embeddings.shape}")
    
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
        
        # 크기 불일치 감지 및 조정
        num_regions = obj_embeddings.shape[1]
        if num_regions != max_objects:
            logger.warning(f"크기 불일치 감지: obj_embeddings={num_regions}, max_objects={max_objects}")
            
            if num_regions > max_objects:
                logger.info(f"obj_embeddings를 {max_objects}개로 자릅니다")
                obj_embeddings = obj_embeddings[:, :max_objects, :]
            else:
                logger.info(f"obj_embeddings를 {max_objects}개로 패딩합니다")
                padding = torch.zeros(batch_size, max_objects - num_regions, obj_embeddings.shape[-1], 
                                      device=obj_embeddings.device)
                obj_embeddings = torch.cat([obj_embeddings, padding], dim=1)
        
        # 손실 계산 시도
        if text_embeddings is not None:
            loss = loss_fn(obj_embeddings, text_embeddings, dummy_class_ids, dummy_valid_mask)
            logger.info(f"손실 계산 성공: {loss.item()}")
        else:
            logger.warning("text_embeddings가 없어 손실 계산을 건너뜁니다")
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
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                for k, v in config_dict.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
        
        # 데이터셋 정보 출력
        logger.info(f"img_size: {config.img_size}")
        logger.info(f"max_objects: {config.max_objects}")
        logger.info(f"클래스 수: {len(config.class_names)}")
        
        # 데이터셋 생성 시도
        try:
            dataset = COCODataset(
                anno_path=config.train_anno_path,
                img_dir=config.train_img_dir,
                class_names=config.class_names,
                img_size=config.img_size,
                max_objects=config.max_objects
            )
            logger.info(f"데이터셋 생성 성공: {len(dataset)} 이미지")
            
            # 첫 번째 항목 검사
            first_item = dataset[0]
            for key, value in first_item.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: shape={value.shape}, type={value.dtype}")
                elif isinstance(value, list):
                    logger.info(f"  {key}: list 길이={len(value)}")
                else:
                    logger.info(f"  {key}: type={type(value)}")
            
            return dataset
        except Exception as e:
            logger.error(f"데이터셋 생성 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
    except ImportError as e:
        logger.error(f"데이터셋 모듈 임포트 실패: {e}")

def analyze_model_and_dataset_compatibility(model, dataset):
    """모델과 데이터셋 간의 호환성 분석"""
    logger.info("===== 모델-데이터셋 호환성 분석 =====")
    
    if model is None or dataset is None:
        logger.error("모델 또는 데이터셋이 None입니다")
        return
    
    # 첫 번째 배치 가져오기
    try:
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        
        logger.info("배치 구조:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: shape={value.shape}, type={value.dtype}")
            elif isinstance(value, list):
                logger.info(f"  {key}: list 길이={len(value)}")
            else:
                logger.info(f"  {key}: type={type(value)}")
        
        # 배치 데이터로 모델 포워드 패스 시도
        logger.info("배치 데이터로 포워드 패스 시도...")
        images = batch['images'].to(next(model.parameters()).device)
        text_prompts = batch['text_prompts']
        
        try:
            with torch.no_grad():
                outputs = model(images, text_prompts=text_prompts)
            
            logger.info("배치 데이터로 포워드 패스 성공!")
            
            # region_features와 class_ids 크기 비교
            obj_embeddings = outputs['obj_embeddings']
            class_ids = batch['class_ids']
            
            logger.info(f"obj_embeddings shape: {obj_embeddings.shape}")
            logger.info(f"class_ids shape: {class_ids.shape}")
            
            if obj_embeddings.shape[1] != class_ids.shape[1]:
                logger.warning(f"크기 불일치! obj_embeddings: {obj_embeddings.shape[1]}, class_ids: {class_ids.shape[1]}")
                logger.info("이 불일치가 손실 함수 계산 시 RuntimeError의 원인일 수 있습니다")
            
            # 손실 함수 계산 시도
            try:
                from yolo_clip_detector.loss.region_text_contrastive import RegionTextContrastiveLoss
                loss_fn = RegionTextContrastiveLoss(temperature=0.1)
                
                # 텍스트 임베딩 출력
                text_embeddings = outputs.get('text_embeddings')
                if text_embeddings is None:
                    logger.warning("text_embeddings가 출력에 없어 모델의 text_encoder를 사용합니다")
                    text_embeddings = model.text_encoder(text_prompts).to(next(model.parameters()).device)
                
                logger.info(f"text_embeddings shape: {text_embeddings.shape}")
                
                # 크기 조정
                num_regions = obj_embeddings.shape[1]
                max_objects = class_ids.shape[1]
                
                if num_regions != max_objects:
                    logger.warning(f"크기 불일치 조정 중: {num_regions} -> {max_objects}")
                    if num_regions > max_objects:
                        obj_embeddings = obj_embeddings[:, :max_objects, :]
                    else:
                        padding = torch.zeros(
                            obj_embeddings.shape[0], 
                            max_objects - num_regions, 
                            obj_embeddings.shape[-1], 
                            device=obj_embeddings.device
                        )
                        obj_embeddings = torch.cat([obj_embeddings, padding], dim=1)
                
                valid_mask = batch['valid_mask'].to(next(model.parameters()).device)
                loss = loss_fn(obj_embeddings, text_embeddings, class_ids.to(next(model.parameters()).device), valid_mask)
                logger.info(f"손실 계산 성공: {loss.item()}")
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

def main(config_path=None):
    """메인 디버깅 함수"""
    logger.info("===== 디버깅 시작 =====")
    
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
        
        logger.info("모델 초기화 성공")
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        model = None
    
    # 모델 아키텍처 검사
    if model is not None:
        inspect_model_architecture(model)
        
        # 더미 데이터로 포워드 테스트
        outputs = forward_dummy_data(model)
        
        # 손실 함수 분석
        if outputs is not None:
            analyze_loss_function(model, outputs)
    
    # 데이터셋 검사
    dataset = inspect_dataset(config_path)
    
    # 모델과 데이터셋 호환성 분석
    if model is not None and dataset is not None:
        analyze_model_and_dataset_compatibility(model, dataset)
    
    logger.info("===== 디버깅 완료 =====")
    
    # 해결 방법 제안
    logger.info("\n===== 해결 방안 제안 =====")
    logger.info("1. region_text_contrastive.py 파일의 forward 메소드에 크기 조정 로직 추가:")
    suggestion = """
    def forward(self, 
                region_features: torch.Tensor, 
                text_embeddings: torch.Tensor, 
                region_labels: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 기존 코드...
        batch_size, num_regions, embed_dim = region_features.shape
        num_classes = text_embeddings.shape[1]
        max_objects = region_labels.shape[1]
        
        # 크기 조정 로직 추가
        if num_regions != max_objects:
            print(f"Adjusting region_features from {num_regions} to {max_objects}")
            if num_regions > max_objects:
                region_features = region_features[:, :max_objects, :]
            else:
                padding = torch.zeros(batch_size, max_objects - num_regions, embed_dim, 
                                     device=region_features.device)
                region_features = torch.cat([region_features, padding], dim=1)
                
                if valid_mask is not None:
                    padding_mask = torch.zeros(batch_size, max_objects - num_regions, 
                                             dtype=torch.bool, device=valid_mask.device)
                    valid_mask = torch.cat([valid_mask, padding_mask], dim=1)
        
        # 기존 코드 계속...
    """
    logger.info(suggestion)
    
    logger.info("2. 모델 architecture 수정:")
    logger.info("   - model/yolo_clip.py에서 객체 특징 수와 데이터셋의 max_objects 값을 일치시키기")
    
    logger.info("3. 데이터셋 설정 변경:")
    logger.info("   - 데이터셋의 max_objects 값을 모델이 생성하는 리전 수에 맞게 조정")
    
    return model, dataset

if __name__ == "__main__":
    # 설정 파일 경로를 지정하거나 None으로 두어 기본값 사용
    config_path = None  # "path/to/your/config.yaml"
    main(config_path)