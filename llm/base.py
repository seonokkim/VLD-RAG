"""
Base class for LLM models in VLD-RAG project.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional
from pathlib import Path
from PIL import Image


class BaseLLM(ABC):
    """
    Base class for all LLM models in the VLD-RAG project.
    
    This class defines the common interface that all LLM implementations
    should follow for consistency across different model types.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None
    ):
        """
        Initialize the LLM model.
        
        Args:
            model_path: Path to the model directory
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            torch_dtype: Torch dtype to use (None for auto-detection)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.torch_dtype = torch_dtype
    
    @abstractmethod
    def answer_question(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0
    ) -> str:
        """
        Answer a question about an image.
        
        Args:
            image: PIL Image, image path (str), or Path object
            question: Question to ask about the image
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling
            temperature: Temperature for sampling
            
        Returns:
            str: Answer to the question
        """
        pass
    
    def judge(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
        search_results: str
    ) -> bool:
        """
        Judge if the question can be answered based on the image and search results.
        
        This is an optional method. If not implemented, JudgeAgent will use
        can_answer_question method if available.
        
        Args:
            image: PIL Image, image path (str), or Path object
            question: Question to ask
            search_results: Search results text
            
        Returns:
            bool: True if the question can be answered, False otherwise
        """
        # Default implementation: use can_answer_question if available
        if hasattr(self, "can_answer_question"):
            return self.can_answer_question(image, question, search_results)
        else:
            raise NotImplementedError(
                "judge method not implemented and can_answer_question not available"
            )
    
    def chat(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0
    ) -> str:
        """
        Chat with the model (alias for answer_question for consistency).
        
        Args:
            image: PIL Image, image path (str), or Path object
            question: Question to ask
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling
            temperature: Temperature for sampling
            
        Returns:
            str: Response from the model
        """
        return self.answer_question(
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature
        )
