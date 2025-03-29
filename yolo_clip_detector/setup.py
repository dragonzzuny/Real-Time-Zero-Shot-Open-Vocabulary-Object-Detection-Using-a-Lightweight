# yolo_clip_detector/setup.py 파일 내용
from setuptools import setup, find_packages

setup(
    name="yolo_clip_detector",
    version="0.1.0",
    description="YOLO-CLIP: Open-Vocabulary Object Detection with Vision-Language Model",
    author="YourName",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/yolo_clip_detector",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.18.0",
        "opencv-python>=4.5.1",
        "Pillow>=8.0.0",
        "PyYAML>=5.4.1",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "transformers>=4.11.0",
        "timm>=0.5.4",
        "ftfy>=6.0.3",
        "regex",
        "tensorboard>=2.5.0",
        "pycocotools>=2.0.2",
        "albumentations>=1.1.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "yolo-clip-train=yolo_clip_detector.train:main",
            "yolo-clip-detect=yolo_clip_detector.detect:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)