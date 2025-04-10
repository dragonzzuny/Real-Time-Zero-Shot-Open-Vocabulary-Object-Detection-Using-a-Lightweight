import torch

# GPU 사용 가능한지 확인
if torch.cuda.is_available():
    print("✅ GPU 사용 가능")
    print("사용 중인 GPU:", torch.cuda.get_device_name(0))
    print("현재 디바이스:", torch.cuda.current_device())
else:
    print("❌ GPU 사용 불가 (CPU로 작동 중)")

import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("CLIP 디바이스:", next(model.parameters()).device)
