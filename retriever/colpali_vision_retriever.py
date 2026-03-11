"""
ColPali Vision Retriever for VLD-RAG system.

Provides retrieval functionality using ColPali vision encoder for query encoding
and vector-based document retrieval.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import ColPaliForRetrieval, SiglipProcessor, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None

from retriever.scorer import EmbeddingScorer
from retriever.vector_loader import VectorLoader
from retriever.db_context import RetrieverDbContext


logger = logging.getLogger(__name__)


class ColPaliVisionRetriever:
    """
    ColPali-based vision retriever for document search.
    
    Uses ColPali model to encode queries (text or image) and retrieves
    similar documents using vector similarity.
    """
    
    def __init__(
        self,
        model_name: str = "vidore/colpaligemma-3b-mix-448-base",
        device: str = "cuda",
        db_context: Optional[RetrieverDbContext] = None,
        embeddings_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        source: str = "auto"
    ):
        """
        Initialize ColPali vision retriever.
        
        Args:
            model_name: HuggingFace model name or path, or adapter path
            device: Device to use ('cpu', 'cuda', etc.)
            db_context: Database context for loading from database
            embeddings_dir: Directory containing NPZ embedding files
            results_dir: Directory containing JSON embedding files
            source: Source to load embeddings from ('database', 'npz', 'json', 'auto')
        """
        self.model_name = model_name
        self.device = self._normalize_device(device)
        self.source = source
        
        self.vector_loader = VectorLoader(db_context, embeddings_dir, results_dir)
        self.scorer = EmbeddingScorer()
        
        self._embeddings_cache: Dict[str, Dict] = {}
        self._cache_loaded = False
        
        self._model: Optional[ColPaliForRetrieval] = None
        self._processor: Optional[SiglipProcessor] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_loaded = False
    
    def _normalize_device(self, device: str) -> str:
        """Normalize device string."""
        if device == "gpu":
            return "cuda"
        if device == "cuda" and (torch is None or not torch.cuda.is_available()):
            logger.warning("CUDA not available. Falling back to CPU.")
            return "cpu"
        return device
    
    def load_model(self):
        """Load ColPali model, processor, and tokenizer."""
        if self._model_loaded:
            return
        
        adapter_path = None
        base_model_id = None
        
        if not self.model_name.startswith('/') and '/' in self.model_name and not Path(self.model_name).exists():
            base_model_id = self.model_name
        else:
            model_path_obj = Path(self.model_name)
            adapter_config_path = model_path_obj / "adapter_config.json"
            config_path = model_path_obj / "config.json"
            
            if adapter_config_path.exists() and not config_path.exists():
                adapter_path = self.model_name
                logger.info(f"Detected adapter-only directory: {adapter_path}")
                
                try:
                    import json
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        base_model_id = adapter_config.get('base_model_name_or_path')
                        if not base_model_id:
                            raise ValueError("Could not find base_model_name_or_path in adapter config")
                except Exception as e:
                    raise ValueError(f"Failed to read base model from adapter config: {e}")
            else:
                base_model_id = self.model_name
        
        logger.info(f"Loading ColPali model: {base_model_id} (device: {self.device})")
        
        self._model = ColPaliForRetrieval.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if adapter_path and PEFT_AVAILABLE:
            logger.info(f"Loading adapter from: {adapter_path}")
            self._model = PeftModel.from_pretrained(self._model, adapter_path)
            self._model = self._model.merge_and_unload()
        
        self._model = self._model.to(self.device)
        self._model.eval()
        
        self._processor = SiglipProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        
        self._model_loaded = True
        logger.info("ColPali model loaded successfully")
    
    def _text_to_image(self, text: str, width: int = 800, height: int = 600) -> Image.Image:
        """Convert text to image for ColPali encoding."""
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        words = text.split()
        y = 20
        for word in words[:50]:
            draw.text((20, y), word, fill='black', font=font)
            y += 25
            if y > height - 30:
                break
        
        return image
    
    def encode_text(
        self,
        query_text: str,
        embedding_mode: str = "multi_vector"
    ) -> Dict:
        """
        Generate query embedding from text using ColPali.
        
        Args:
            query_text: Query text string
            embedding_mode: 'single_vector' or 'multi_vector'
        
        Returns:
            Dictionary with embedding data
        """
        if not self._model_loaded:
            self.load_model()
        
        try:
            with torch.no_grad():
                text_inputs = self._tokenizer(
                    query_text, return_tensors='pt', padding=True, 
                    truncation=True, max_length=512
                )
                text_inputs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in text_inputs.items()
                }
                
                if hasattr(self._model, 'vlm') and hasattr(self._model.vlm, 'language_model'):
                    lang_model = self._model.vlm.language_model
                    lang_outputs = lang_model(**text_inputs)
                    
                    if hasattr(lang_outputs, 'hidden_states') and lang_outputs.hidden_states:
                        hidden_states = lang_outputs.hidden_states[-1]
                        
                        if embedding_mode == "multi_vector":
                            token_embeddings = hidden_states[:, :, :1152].float().cpu().numpy()[0]
                            pooled = np.mean(token_embeddings, axis=0)
                            pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
                            
                            return {
                                'embedding_mode': 'multi_vector',
                                'token_embeddings': token_embeddings.astype(np.float32),
                                'pooled_embedding': pooled.astype(np.float32),
                                'embedding_dim': 1152,
                                'num_tokens': token_embeddings.shape[0]
                            }
                        else:
                            pooled = hidden_states.mean(axis=1)[:, :1152].float().cpu().numpy()[0]
                            pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
                            
                            return {
                                'embedding_mode': 'single_vector',
                                'embedding': pooled.astype(np.float32),
                                'pooled_embedding': pooled.astype(np.float32),
                                'embedding_dim': 1152,
                                'num_tokens': 1
                            }
            
            logger.info("Using text-to-image fallback method")
            question_image = self._text_to_image(query_text)
            return self.encode_image(question_image, embedding_mode)
        
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}", exc_info=True)
            raise
    
    def encode_image(
        self,
        image: Image.Image,
        embedding_mode: str = "multi_vector"
    ) -> Dict:
        """
        Generate query embedding from image using ColPali.
        
        Args:
            image: PIL Image
            embedding_mode: 'single_vector' or 'multi_vector'
        
        Returns:
            Dictionary with embedding data
        """
        if not self._model_loaded:
            self.load_model()
        
        try:
            if hasattr(self._model, 'vlm') and hasattr(self._model.vlm, 'vision_tower'):
                vision_encoder = self._model.vlm.vision_tower
            else:
                raise ValueError("Cannot find vision encoder in model")
            
            image_inputs = self._processor(images=image, return_tensors="pt")
            image_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in image_inputs.items()
            }
            
            with torch.no_grad():
                vision_outputs = vision_encoder(**image_inputs)
                
                if hasattr(vision_outputs, 'last_hidden_state'):
                    vision_hidden = vision_outputs.last_hidden_state[0]
                    
                    if embedding_mode == "multi_vector":
                        pooled = vision_hidden.mean(axis=0).float().cpu().numpy()
                        pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
                        
                        return {
                            'embedding_mode': 'multi_vector',
                            'token_embeddings': vision_hidden.float().cpu().numpy().astype(np.float32),
                            'pooled_embedding': pooled.astype(np.float32),
                            'embedding_dim': 1152,
                            'num_tokens': vision_hidden.shape[0]
                        }
                    else:
                        pooled = vision_hidden.mean(axis=0).float().cpu().numpy()
                        pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
                        
                        return {
                            'embedding_mode': 'single_vector',
                            'embedding': pooled.astype(np.float32),
                            'pooled_embedding': pooled.astype(np.float32),
                            'embedding_dim': pooled.shape[0],
                            'num_tokens': 1
                        }
                else:
                    raise ValueError("Vision encoder output does not have 'last_hidden_state'")
        
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}", exc_info=True)
            raise
    
    def load_embeddings(
        self,
        page_id: Optional[str] = None,
        embedding_mode: Optional[str] = None,
        doc_no: Optional[str] = None,
        force_reload: bool = False
    ) -> Dict[str, Dict]:
        """Load embeddings into cache."""
        if not self._cache_loaded or force_reload:
            self._embeddings_cache = self.vector_loader.load_all(
                source=self.source,
                page_id=page_id,
                embedding_mode=embedding_mode,
                doc_no=doc_no
            )
            self._cache_loaded = True
            logger.info(f"Loaded {len(self._embeddings_cache)} embeddings into cache")
        
        return self._embeddings_cache
    
    def search(
        self,
        query: Union[str, Image.Image],
        top_k: int = 10,
        embedding_mode: Optional[str] = None,
        doc_no: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents using query (text or image).
        
        Args:
            query: Query text string or PIL Image
            top_k: Number of top results to return
            embedding_mode: Filter documents by embedding_mode (optional)
            doc_no: Filter documents by doc_no (optional)
            min_score: Minimum similarity score threshold
            
        Returns:
            List of (page_id, score, embedding_data) tuples, sorted by score (descending)
        """
        if embedding_mode is None:
            embedding_mode = "multi_vector"
        
        if isinstance(query, str):
            query_embedding = self.encode_text(query, embedding_mode=embedding_mode)
        elif isinstance(query, Image.Image):
            query_embedding = self.encode_image(query, embedding_mode=embedding_mode)
        else:
            raise ValueError(f"Query must be str or Image.Image, got {type(query)}")
        
        self.load_embeddings(embedding_mode=embedding_mode, doc_no=doc_no)
        
        candidate_embeddings = self._embeddings_cache.copy()
        if embedding_mode:
            candidate_embeddings = {
                pid: emb for pid, emb in candidate_embeddings.items()
                if emb.get('embedding_mode') == embedding_mode
            }
        if doc_no:
            candidate_embeddings = {
                pid: emb for pid, emb in candidate_embeddings.items()
                if emb.get('doc_no') == doc_no
            }
        
        results = []
        query_mode = query_embedding.get('embedding_mode')
        
        for page_id, doc_embedding in candidate_embeddings.items():
            try:
                doc_mode = doc_embedding.get('embedding_mode')
                
                if query_mode and doc_mode and query_mode != doc_mode:
                    continue
                
                if not query_mode:
                    query_mode = doc_mode
                
                if query_mode == 'single_vector':
                    if 'embedding' not in query_embedding or 'embedding' not in doc_embedding:
                        if 'pooled_embedding' in query_embedding and 'pooled_embedding' in doc_embedding:
                            score = self.scorer.cosine_similarity(
                                query_embedding['pooled_embedding'],
                                doc_embedding['pooled_embedding']
                            )
                        else:
                            continue
                    else:
                        score = self.scorer.score_single_vector(
                            query_embedding['embedding'],
                            doc_embedding['embedding']
                        )
                elif query_mode == 'multi_vector':
                    if 'token_embeddings' not in query_embedding or 'token_embeddings' not in doc_embedding:
                        continue
                    score = self.scorer.score_multi_vector(
                        query_embedding['token_embeddings'],
                        doc_embedding['token_embeddings']
                    )
                else:
                    logger.warning(f"Unknown embedding mode: {query_mode}")
                    continue
                
                if score >= min_score:
                    results.append((page_id, score, doc_embedding))
            
            except Exception as e:
                logger.warning(f"Failed to score {page_id}: {e}")
                continue
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_embedding(self, page_id: str) -> Optional[Dict]:
        """Get embedding data for a specific page_id."""
        self.load_embeddings()
        return self._embeddings_cache.get(page_id)
    
    def clear_cache(self):
        """Clear the embeddings cache."""
        self._embeddings_cache.clear()
        self._cache_loaded = False
        logger.info("Embeddings cache cleared")
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded embeddings."""
        self.load_embeddings()
        
        stats = {
            'total_embeddings': len(self._embeddings_cache),
            'by_mode': {},
            'by_source': {},
            'by_doc': {}
        }
        
        for page_id, emb in self._embeddings_cache.items():
            mode = emb.get('embedding_mode', 'unknown')
            stats['by_mode'][mode] = stats['by_mode'].get(mode, 0) + 1
            
            source = emb.get('source', 'unknown')
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            
            doc_no = emb.get('doc_no', 'unknown')
            stats['by_doc'][doc_no] = stats['by_doc'].get(doc_no, 0) + 1
        
        return stats
