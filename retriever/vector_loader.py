"""
Vector loader for retriever system.

Loads embedding vectors from various sources:
- Database (EmbeddingResult, ColPaliEmbedding, SigLIPEmbedding)
- NPZ files from outputs/embeddings directory
- JSON files from results directory
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from retriever.db_context import RetrieverDbContext


logger = logging.getLogger(__name__)


class VectorLoader:
    """Load embedding vectors from various sources."""
    
    def __init__(
        self,
        db_context: Optional[RetrieverDbContext] = None,
        embeddings_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None
    ):
        """
        Initialize vector loader.
        
        Args:
            db_context: Database context for loading from database
            embeddings_dir: Directory containing NPZ embedding files
            results_dir: Directory containing JSON embedding files
        """
        self.db_context = db_context
        
        base_dir = Path(__file__).parent.parent
        self.embeddings_dir = embeddings_dir or (base_dir / "outputs" / "embeddings")
        self.results_dir = results_dir or (base_dir / "results")
    
    def load_from_database(
        self,
        page_id: Optional[str] = None,
        embedding_mode: Optional[str] = None,
        doc_no: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Load embeddings from database."""
        if self.db_context is None:
            raise ValueError("Database context not provided")
        
        embeddings = {}
        
        try:
            query = self.db_context.embedding_results.select()
            
            if page_id:
                query = query.where(self.db_context.embedding_results.page_id == page_id)
            if embedding_mode:
                query = query.where(self.db_context.embedding_results.embedding_mode == embedding_mode)
            if doc_no:
                query = query.where(self.db_context.embedding_results.doc_no == doc_no)
            
            for result in query:
                embedding_data = {
                    'page_id': result.page_id,
                    'file_name': result.file_name,
                    'page_no': result.page_no,
                    'doc_no': result.doc_no,
                    'embedding_mode': result.embedding_mode,
                    'embedding_dim': result.embedding_dim,
                    'num_tokens': result.num_tokens,
                    'metadata': result.metadata or {}
                }
                
                if result.embedding_mode == 'single_vector':
                    if result.embedding:
                        embedding_data['embedding'] = np.array(result.embedding, dtype=np.float32)
                        embedding_data['pooled_embedding'] = embedding_data['embedding']
                    elif result.pooled_embedding:
                        embedding_data['embedding'] = np.array(result.pooled_embedding, dtype=np.float32)
                        embedding_data['pooled_embedding'] = embedding_data['embedding']
                
                elif result.embedding_mode == 'multi_vector':
                    if result.token_embeddings_path:
                        npz_path = Path(result.token_embeddings_path)
                        if npz_path.exists():
                            npz_data = np.load(npz_path)
                            embedding_data['token_embeddings'] = npz_data['token_embeddings'].astype(np.float32)
                            if 'pooled_embedding' in npz_data:
                                embedding_data['pooled_embedding'] = npz_data['pooled_embedding'].astype(np.float32)
                            else:
                                embedding_data['pooled_embedding'] = np.mean(
                                    embedding_data['token_embeddings'], axis=0
                                ).astype(np.float32)
                    elif result.embedding_path:
                        npz_path = Path(result.embedding_path)
                        if npz_path.exists():
                            npz_data = np.load(npz_path)
                            embedding_data['token_embeddings'] = npz_data['token_embeddings'].astype(np.float32)
                            if 'pooled_embedding' in npz_data:
                                embedding_data['pooled_embedding'] = npz_data['pooled_embedding'].astype(np.float32)
                            else:
                                embedding_data['pooled_embedding'] = np.mean(
                                    embedding_data['token_embeddings'], axis=0
                                ).astype(np.float32)
                    
                    if 'pooled_embedding' not in embedding_data and result.pooled_embedding:
                        embedding_data['pooled_embedding'] = np.array(result.pooled_embedding, dtype=np.float32)
                
                embeddings[result.page_id] = embedding_data
        
        except Exception as e:
            logger.error(f"Failed to load embeddings from database: {e}", exc_info=True)
        
        return embeddings
    
    def load_from_npz(
        self,
        page_id: Optional[str] = None,
        embedding_mode: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Load embeddings from NPZ files."""
        embeddings = {}
        
        if not self.embeddings_dir.exists():
            logger.warning(f"Embeddings directory does not exist: {self.embeddings_dir}")
            return embeddings
        
        for npz_path in self.embeddings_dir.glob("*.npz"):
            filename = npz_path.stem
            
            if page_id and page_id not in filename:
                continue
            
            if embedding_mode:
                if embedding_mode == 'single_vector' and 'single' not in filename.lower():
                    continue
                elif embedding_mode == 'multi_vector' and 'multi' not in filename.lower():
                    continue
            
            try:
                npz_data = np.load(npz_path)
                
                if 'token_embeddings' in npz_data:
                    token_embeddings = npz_data['token_embeddings'].astype(np.float32)
                    pooled_embedding = npz_data.get('pooled_embedding', None)
                    if pooled_embedding is None:
                        pooled_embedding = np.mean(token_embeddings, axis=0).astype(np.float32)
                    else:
                        pooled_embedding = pooled_embedding.astype(np.float32)
                    
                    embedding_data = {
                        'page_id': self._extract_page_id_from_filename(filename),
                        'embedding_mode': 'multi_vector',
                        'token_embeddings': token_embeddings,
                        'pooled_embedding': pooled_embedding,
                        'embedding_dim': token_embeddings.shape[1],
                        'num_tokens': token_embeddings.shape[0],
                        'source': 'npz',
                        'file_path': str(npz_path)
                    }
                elif 'embedding' in npz_data:
                    embedding = npz_data['embedding'].astype(np.float32)
                    embedding_data = {
                        'page_id': self._extract_page_id_from_filename(filename),
                        'embedding_mode': 'single_vector',
                        'embedding': embedding,
                        'pooled_embedding': embedding,
                        'embedding_dim': embedding.shape[0],
                        'source': 'npz',
                        'file_path': str(npz_path)
                    }
                else:
                    logger.warning(f"Unknown NPZ format in {npz_path}")
                    continue
                
                embeddings[embedding_data['page_id']] = embedding_data
            
            except Exception as e:
                logger.warning(f"Failed to load {npz_path}: {e}")
                continue
        
        return embeddings
    
    def load_from_json(
        self,
        page_id: Optional[str] = None,
        embedding_mode: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Load embeddings from JSON files."""
        embeddings = {}
        
        if not self.results_dir.exists():
            logger.warning(f"Results directory does not exist: {self.results_dir}")
            return embeddings
        
        for json_path in self.results_dir.glob("*.json"):
            filename = json_path.name
            
            if page_id and page_id not in filename:
                continue
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'results' not in data or len(data['results']) == 0:
                    continue
                
                result = data['results'][0]
                
                mode = result.get('embedding_mode')
                if not mode:
                    if 'token_embeddings' in result:
                        mode = 'multi_vector'
                    elif 'embedding' in result:
                        mode = 'single_vector'
                    else:
                        continue
                
                if embedding_mode and mode != embedding_mode:
                    continue
                
                page_id_from_file = self._extract_page_id_from_json(data, result)
                
                embedding_data = {
                    'page_id': page_id_from_file,
                    'embedding_mode': mode,
                    'embedding_dim': result.get('embedding_dim', 1152),
                    'metadata': {
                        'timestamp': result.get('timestamp'),
                        'model': result.get('model'),
                        'image_size': result.get('image_size'),
                    }
                }
                
                if mode == 'single_vector':
                    if 'embedding' in result:
                        embedding_data['embedding'] = np.array(result['embedding'], dtype=np.float32)
                        embedding_data['pooled_embedding'] = embedding_data['embedding']
                
                elif mode == 'multi_vector':
                    if 'token_embeddings' in result:
                        embedding_data['token_embeddings'] = np.array(result['token_embeddings'], dtype=np.float32)
                        embedding_data['num_tokens'] = result.get('num_tokens', len(embedding_data['token_embeddings']))
                        
                        if 'pooled_embedding' in result:
                            embedding_data['pooled_embedding'] = np.array(result['pooled_embedding'], dtype=np.float32)
                        else:
                            embedding_data['pooled_embedding'] = np.mean(
                                embedding_data['token_embeddings'], axis=0
                            ).astype(np.float32)
                
                embedding_data['source'] = 'json'
                embedding_data['file_path'] = str(json_path)
                embeddings[embedding_data['page_id']] = embedding_data
            
            except Exception as e:
                logger.warning(f"Failed to load {json_path}: {e}")
                continue
        
        return embeddings
    
    def load_all(
        self,
        source: str = "auto",
        page_id: Optional[str] = None,
        embedding_mode: Optional[str] = None,
        doc_no: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Load embeddings from all available sources or specified source."""
        embeddings = {}
        
        if source == "auto":
            if self.db_context:
                try:
                    db_embeddings = self.load_from_database(page_id, embedding_mode, doc_no)
                    embeddings.update(db_embeddings)
                except Exception as e:
                    logger.warning(f"Failed to load from database: {e}")
            
            npz_embeddings = self.load_from_npz(page_id, embedding_mode)
            embeddings.update(npz_embeddings)
            
            json_embeddings = self.load_from_json(page_id, embedding_mode)
            embeddings.update(json_embeddings)
        
        elif source == "database":
            embeddings = self.load_from_database(page_id, embedding_mode, doc_no)
        
        elif source == "npz":
            embeddings = self.load_from_npz(page_id, embedding_mode)
        
        elif source == "json":
            embeddings = self.load_from_json(page_id, embedding_mode)
        
        else:
            raise ValueError(f"Unknown source: {source}")
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {source}")
        return embeddings
    
    def _extract_page_id_from_filename(self, filename: str) -> str:
        """Extract page_id from filename."""
        parts = filename.split('_')
        
        if 'page' in parts:
            page_idx = parts.index('page')
            if page_idx > 0:
                if page_idx >= 2:
                    doc_no = parts[0]
                    page_no = parts[1] if parts[1].startswith('p') else parts[page_idx - 1]
                    if page_no.startswith('p'):
                        page_no = page_no[1:]
                    return f"{doc_no}_{page_no}_page_{page_no}"
        
        return filename
    
    def _extract_page_id_from_json(self, full_data: Dict, result: Dict) -> str:
        """Extract page_id from JSON data."""
        if 'input_file' in full_data:
            input_file = full_data['input_file']
            if 'page_' in input_file:
                parts = Path(input_file).stem.split('_')
                if 'page' in parts:
                    page_idx = parts.index('page')
                    if page_idx > 0:
                        doc_no = parts[0]
                        page_no = parts[page_idx + 1] if page_idx + 1 < len(parts) else result.get('page_no', 0)
                        return f"{doc_no}_{page_no}_page_{page_no}"
        
        page_no = result.get('page_no', 0)
        return f"unknown_{page_no}_page_{page_no}"
