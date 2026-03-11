"""
Qwen3-VL-4B-Instruct model wrapper for vision-language tasks.
"""

import logging
from typing import Union, Optional
from pathlib import Path
from PIL import Image
import torch

from llm.base import BaseLLM

try:
    from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration, AutoProcessor
    # Try to import AutoModelForImageTextToText (newer API) or fallback to AutoModelForVision2Seq
    try:
        from transformers import AutoModelForImageTextToText
        AUTO_MODEL_AVAILABLE = True
        USE_NEW_API = True
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
        AUTO_MODEL_AVAILABLE = True
        USE_NEW_API = False
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AUTO_MODEL_AVAILABLE = False
    USE_NEW_API = False
    print("Warning: transformers not installed. Install with: pip install transformers")

logger = logging.getLogger(__name__)


class Qwen3VL4BInstruct(BaseLLM):
    """
    Qwen3-VL-4B-Instruct model wrapper for vision-language tasks.
    
    This class provides a simple interface to use Qwen3-VL-4B-Instruct model
    for answering questions about images.
    """
    
    def __init__(
        self,
        model_path: str = "../models/models-small-3b-6b/Qwen3-VL-4B-Instruct",
        device: Optional[str] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None
    ):
        """
        Initialize Qwen3-VL-4B-Instruct model.
        
        Args:
            model_path: Path to the model directory
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            torch_dtype: Torch dtype to use (None for auto-detection, or string like 'float16', 'bfloat16')
        """
        super().__init__(model_path, device, torch_dtype)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed. "
                "Install with: pip install transformers torch"
            )
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {model_path}\n"
                f"Please check if the model is downloaded to the correct location."
            )
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Handle torch_dtype (can be string or torch.dtype)
        if isinstance(torch_dtype, str):
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            self.torch_dtype = dtype_map.get(torch_dtype.lower())
        elif torch_dtype is None:
            if self.device == "cuda":
                self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype
        
        logger.info(f"Loading Qwen3-VL-4B-Instruct from {model_path}")
        logger.info(f"Device: {self.device}, Dtype: {self.torch_dtype}")
        
        # Load processor
        try:
            self.processor = Qwen2VLProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Qwen2VLProcessor failed: {e}, trying AutoProcessor")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to load processor: {e}, {e2}")
        
        # Load model
        # Try Qwen2VLForConditionalGeneration first, then fallback to AutoModel
        # Use dtype parameter (newer API) with fallback to torch_dtype for compatibility
        load_kwargs = {
            "trust_remote_code": True,
        }
        
        # Try dtype first (newer API), fallback to torch_dtype if needed
        try:
            load_kwargs["dtype"] = self.torch_dtype
        except:
            load_kwargs["torch_dtype"] = self.torch_dtype
        
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                str(self.model_path),
                **load_kwargs
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Qwen2VLForConditionalGeneration failed: {e}, trying AutoModel")
            try:
                if AUTO_MODEL_AVAILABLE:
                    # Use AutoModelForImageTextToText (or AutoModelForVision2Seq as fallback)
                    if USE_NEW_API:
                        logger.info("Using AutoModelForImageTextToText (new API)")
                    else:
                        logger.info("Using AutoModelForVision2Seq (deprecated, but available)")
                    
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        str(self.model_path),
                        **load_kwargs
                    ).to(self.device)
                    self.model.eval()
                else:
                    # Last resort: try AutoModel
                    from transformers import AutoModel
                    logger.warning("Trying AutoModel as last resort")
                    self.model = AutoModel.from_pretrained(
                        str(self.model_path),
                        **load_kwargs
                    ).to(self.device)
                    self.model.eval()
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e}, {e2}")
        
        logger.info("Model loaded successfully")
    
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
        # Load image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Apply chat template
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"apply_chat_template failed: {e}, using direct prompt")
            text = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # Process inputs - handle different processor APIs
        try:
            # Try with images parameter only (no videos)
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
        except Exception as e:
            logger.warning(f"First processing attempt failed: {e}, trying alternative method")
            try:
                # Try with messages directly
                inputs = self.processor(
                    messages=messages,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
            except Exception as e2:
                logger.error(f"All processing methods failed: {e}, {e2}")
                raise
        
        # Generate
        with torch.no_grad():
            try:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None
                )
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise
        
        # Decode output
        try:
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        except Exception as e:
            logger.warning(f"Decoding failed: {e}, trying alternative method")
            # Fallback: decode full sequence
            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            # Remove input text if present
            if text in output_text:
                output_text = output_text.replace(text, "").strip()
        
        return output_text.strip()
    
    def can_answer_question(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
        search_results: Optional[str] = None
    ) -> bool:
        """
        Judge if the question can be answered based on the image and search results.
        
        Args:
            image: PIL Image, image path (str), or Path object
            question: Question to ask
            search_results: Optional search results text
            
        Returns:
            bool: True if the question can be answered, False otherwise
        """
        if search_results:
            prompt = f"""Determine if you can answer the question based on the following image.

Question: {question}

Search results:
{search_results}

Analyze the image to:
1. Check if the image contains an answer to the question
2. Check if the search results contain the correct answer to the question

Response format:
- If there is no answer to the question in the image and no answer in the search results: false
- If there is an answer to the question in the search results: true

You must output only "true" or "false"."""
        else:
            prompt = f"""Determine if you can answer the question based on the following image.

Question: {question}

Analyze the image to check if there is an answer to the question.

Response format:
- If there is an answer to the question: true
- If there is no answer to the question: false

You must output only "true" or "false"."""
        
        try:
            response = self.answer_question(image, prompt, max_new_tokens=128)
            response_lower = response.strip().lower()
            
            if "true" in response_lower:
                return True
            elif "false" in response_lower:
                return False
            else:
                # Default: false if unclear
                logger.warning(f"Unclear response: {response}, defaulting to False")
                return False
        except Exception as e:
            logger.error(f"Error in can_answer_question: {e}", exc_info=True)
            return False
