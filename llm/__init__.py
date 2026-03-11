"""
LLM module for VLD-RAG project.
Contains wrappers for various vision-language models.
"""

from .qwen3_vl_4b_instruct import Qwen3VL4BInstruct
from .internvl3_5_4b import InternVL35_4B

__all__ = ["Qwen3VL4BInstruct", "InternVL35_4B"]
