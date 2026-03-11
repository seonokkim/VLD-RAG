"""
InternVL3-5-4B model wrapper for vision-language tasks.
"""

import logging
from typing import Union, Optional
from pathlib import Path
from PIL import Image
import torch

from llm.base import BaseLLM

try:
    from transformers import (
        AutoProcessor, 
        AutoModelForCausalLM, 
        AutoModelForVision2Seq,
        AutoTokenizer,
        AutoImageProcessor
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers")

logger = logging.getLogger(__name__)


class InternVL35_4B(BaseLLM):
    """
    InternVL3-5-4B model wrapper for vision-language tasks.
    
    This class provides a simple interface to use InternVL3-5-4B model
    for answering questions about images.
    """
    
    def __init__(
        self,
        model_path: str = "../models/models-small-3b-6b/InternVL3_5-4B",
        device: Optional[str] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None
    ):
        """
        Initialize InternVL3-5-4B model.
        
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
                # Use float16 instead of bfloat16 for better compatibility
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype
        
        logger.info(f"Loading InternVL3-5-4B from {model_path}")
        logger.info(f"Device: {self.device}, Dtype: {self.torch_dtype}")
        
        # Load processor - try different methods for InternVL
        try:
            # First try AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"AutoProcessor failed: {e}, trying separate tokenizer and image_processor")
            try:
                # Try loading tokenizer and image processor separately
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True
                )
                self.image_processor = AutoImageProcessor.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True
                )
                # Create a simple processor-like object
                class SimpleProcessor:
                    def __init__(self, tokenizer, image_processor):
                        self.tokenizer = tokenizer
                        self.image_processor = image_processor
                    
                    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
                        # Process image
                        if images is not None:
                            pixel_values = self.image_processor(images, return_tensors=return_tensors)['pixel_values']
                        else:
                            pixel_values = None
                        
                        # Process text
                        if text is not None:
                            text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
                        else:
                            text_inputs = {}
                        
                        # Combine inputs
                        inputs = {**text_inputs}
                        if pixel_values is not None:
                            inputs['pixel_values'] = pixel_values
                        
                        return type('Inputs', (), inputs)()
                    
                    def batch_decode(self, token_ids, skip_special_tokens=True, **kwargs):
                        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
                
                self.processor = SimpleProcessor(self.tokenizer, self.image_processor)
            except Exception as e2:
                raise RuntimeError(f"Failed to load processor: {e}, {e2}")
        
        # Load model - InternVL uses custom model class, try AutoModel first
        try:
            # InternVL models typically use AutoModel with trust_remote_code
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"AutoModel failed: {e}, trying AutoModelForCausalLM")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True,
                    torch_dtype=self.torch_dtype
                ).to(self.device)
                self.model.eval()
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e}, {e2}")
        
        # Set img_context_token_id if needed (InternVL3.5 requirement)
        if hasattr(self.model, 'img_context_token_id') and self.model.img_context_token_id is None:
            # Try to get img_context_token_id from tokenizer
            if hasattr(self.processor, 'tokenizer'):
                tokenizer = self.processor.tokenizer
            elif hasattr(self, 'tokenizer'):
                tokenizer = self.tokenizer
            else:
                tokenizer = None
            
            if tokenizer is not None:
                # Try to find image token ID
                # InternVL3.5 typically uses special tokens for images
                if hasattr(tokenizer, 'convert_tokens_to_ids'):
                    # Try common image token names
                    for img_token in ['<image>', '<IMAGE>', '<img>', '<IMG>', '[IMAGE]']:
                        try:
                            img_id = tokenizer.convert_tokens_to_ids(img_token)
                            if img_id != tokenizer.unk_token_id:
                                self.model.img_context_token_id = img_id
                                logger.info(f"Set img_context_token_id to {img_id} (token: {img_token})")
                                break
                        except:
                            continue
                
                # If still None, try to get from tokenizer's special tokens
                if self.model.img_context_token_id is None and hasattr(tokenizer, 'special_tokens_map'):
                    special_tokens = tokenizer.special_tokens_map
                    if 'image_token' in special_tokens:
                        try:
                            img_id = tokenizer.convert_tokens_to_ids(special_tokens['image_token'])
                            self.model.img_context_token_id = img_id
                            logger.info(f"Set img_context_token_id to {img_id} from special_tokens_map")
                        except:
                            pass
        
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
        
        # Create messages format for InternVL3.5
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Process inputs for InternVL3.5
        # InternVL3.5 requires messages format and process_vision_info
        try:
            # Method 1: Use messages directly with processor (InternVL3.5 preferred method)
            if hasattr(self.processor, 'process_vision_info'):
                # Process vision info first
                image_inputs, video_inputs = self.processor.process_vision_info(messages)
                
                # Apply chat template
                if hasattr(self.processor, 'apply_chat_template'):
                    text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    text = f"<image>\n{question}"
                
                # Process with text and images
                # InternVL3.5 processor expects text as string (not list) when using process_vision_info
                inputs = self.processor(
                    text=text,  # Use text as string, not list
                    images=image_inputs if image_inputs else [image],
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt"
                )
            else:
                # Fallback: try messages format directly
                if hasattr(self.processor, '__call__'):
                    inputs = self.processor(
                        messages=messages,
                        return_tensors="pt",
                        padding=True
                    )
                else:
                    # Last fallback: standard processing
                    inputs = self.processor(
                        images=[image],
                        text=question,
                        padding=True,
                        return_tensors="pt"
                    )
            
            # Convert to dict and move to device
            if hasattr(inputs, '__dict__'):
                inputs_dict = {}
                for k, v in inputs.__dict__.items():
                    if not k.startswith('_'):
                        if hasattr(v, 'to'):
                            inputs_dict[k] = v.to(self.device)
                        else:
                            inputs_dict[k] = v
            else:
                inputs_dict = {}
                for k, v in inputs.items():
                    if hasattr(v, 'to'):
                        inputs_dict[k] = v.to(self.device)
                    else:
                        inputs_dict[k] = v
            
            # Ensure input_ids exists
            if 'input_ids' not in inputs_dict or inputs_dict['input_ids'] is None:
                # Try to tokenize text manually if needed
                if hasattr(self.processor, 'tokenizer'):
                    tokenizer = self.processor.tokenizer
                elif hasattr(self, 'tokenizer'):
                    tokenizer = self.tokenizer
                else:
                    tokenizer = None
                
                if tokenizer is not None:
                    # Get text from messages or use question
                    if hasattr(self.processor, 'apply_chat_template'):
                        text = self.processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        text = question
                    
                    text_inputs = tokenizer(text, return_tensors="pt", padding=True)
                    inputs_dict['input_ids'] = text_inputs['input_ids'].to(self.device)
                    if 'attention_mask' in text_inputs:
                        inputs_dict['attention_mask'] = text_inputs['attention_mask'].to(self.device)
            
            inputs = inputs_dict
            
        except Exception as e:
            logger.warning(f"First processing attempt failed: {e}, trying alternative method")
            try:
                # Alternative: try direct processing
                processed = self.processor(
                    images=[image],
                    text=question,
                    return_tensors="pt",
                    padding=True
                )
                # Convert to dict and move to device
                if hasattr(processed, '__dict__'):
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                              for k, v in processed.__dict__.items() if not k.startswith('_')}
                else:
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                              for k, v in processed.items()}
                
                # Ensure input_ids exists
                if 'input_ids' not in inputs or inputs['input_ids'] is None:
                    raise ValueError("input_ids is None after processing")
                    
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
            # Extract only the generated tokens (remove input tokens)
            if 'input_ids' in inputs:
                input_length = inputs['input_ids'].shape[1]
                generated_ids_trimmed = generated_ids[:, input_length:]
            elif hasattr(inputs, 'input_ids'):
                input_length = inputs.input_ids.shape[1]
                generated_ids_trimmed = generated_ids[:, input_length:]
            else:
                generated_ids_trimmed = generated_ids
            
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
            # Remove input prompt if present
            if prompt in output_text:
                output_text = output_text.replace(prompt, "").strip()
        
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
