"""
C2-VLM: Contextual Contrastive Vision-Language Model
"""

__version__ = "0.1.0"
__author__ = "Xiang Li"
__email__ = "lixiang007666@gmail.com"

from .models.c2_vlm import C2VLM
from .models.vision_encoder import VisionEncoder
from .models.text_encoder import TextEncoder
from .models.fusion_module import FusionModule

__all__ = [
    "C2VLM",
    "VisionEncoder", 
    "TextEncoder",
    "FusionModule",
]