"""
BM25 Text Retriever using rank-bm25.

Requires: pip install rank-bm25
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False
    BM25Okapi = None


logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25-based text retriever for sparse text retrieval.
    """

    def __init__(
        self,
        corpus: Optional[List[Dict]] = None,
        corpus_file: Optional[Union[str, Path]] = None,
        tokenizer=None
    ):
        """
        Initialize BM25 retriever.

        Args:
            corpus: List of documents, each as dict with 'id' and 'text' keys
            corpus_file: Path to JSON file containing corpus
            tokenizer: Custom tokenizer function (optional)
        """
        if not RANK_BM25_AVAILABLE:
            raise ImportError(
                "rank-bm25 library is required. Install with: pip install rank-bm25"
            )

        self.tokenizer = tokenizer or self._default_tokenizer
        self.corpus = []
        self.doc_ids = []
        self.bm25 = None

        if corpus_file:
            self.corpus, self.doc_ids = self._load_corpus_from_file(corpus_file)
        elif corpus:
            self.corpus = corpus
            self.doc_ids = [doc['id'] for doc in corpus]
        else:
            raise ValueError("Either corpus or corpus_file must be provided")

        self._build_index()

    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenizer: lowercase whitespace split."""
        if not text:
            return []
        return text.lower().split()

    def _load_corpus_from_file(self, corpus_file: Union[str, Path]) -> Tuple[List[Dict], List[str]]:
        """Load corpus from JSON file."""
        corpus_path = Path(corpus_file)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            corpus = data
        elif isinstance(data, dict):
            if 'corpus' in data:
                corpus = data['corpus']
            elif 'documents' in data:
                corpus = data['documents']
            elif 'results' in data:
                corpus = data['results']
            else:
                raise ValueError(f"Unknown corpus format in {corpus_file}")
        else:
            raise ValueError(f"Invalid corpus format in {corpus_file}")

        for i, doc in enumerate(corpus):
            if 'id' not in doc:
                doc['id'] = f"doc_{i}"
            if 'text' not in doc:
                for key in ['ocr_text', 'markdown_text', 'content', 'body']:
                    if key in doc:
                        doc['text'] = doc[key]
                        break
                if 'text' not in doc:
                    raise ValueError(f"Document {doc.get('id', i)} has no text field")

        doc_ids = [doc['id'] for doc in corpus]
        logger.info(f"Loaded {len(corpus)} documents from {corpus_file}")

        return corpus, doc_ids

    def _build_index(self):
        """Build BM25 index."""
        tokenized_corpus = [self.tokenizer(doc['text']) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_id_filter: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for query.

        Args:
            query: Query string
            top_k: Number of results to return
            doc_id_filter: Optional document ID prefix to filter results

        Returns:
            List of (doc_id, score) tuples
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not initialized")

        tokenized_query = self.tokenizer(query)
        if not tokenized_query:
            logger.warning("Empty query after tokenization")
            return []

        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(scores):
            doc_id = self.doc_ids[i]
            if doc_id_filter and not doc_id.startswith(doc_id_filter):
                continue
            results.append((doc_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def add_documents(self, documents: List[Dict]):
        """Add new documents and rebuild BM25 index."""
        if not documents:
            return

        for doc in documents:
            if 'id' not in doc or 'text' not in doc:
                raise ValueError("Each document must have 'id' and 'text' keys")

        self.corpus.extend(documents)
        self.doc_ids.extend([doc['id'] for doc in documents])
        self._build_index()

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID."""
        for doc in self.corpus:
            if doc['id'] == doc_id:
                return doc
        return None

    def list_doc_ids(self) -> List[str]:
        """List all document IDs."""
        return list(self.doc_ids)
