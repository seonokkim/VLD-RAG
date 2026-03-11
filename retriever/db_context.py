"""
Database context for retriever system.

Provides database connection management for loading embeddings from database.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from playhouse.postgres_ext import PostgresqlExtDatabase

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger(__name__)


def load_db_config(config_path: Optional[str] = None) -> dict:
    """Load database configuration from config.yaml file."""
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using defaults.")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('postgres', {})
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
        return {}


class RetrieverDbContext:
    """Database context for retriever tables."""
    
    def __init__(
        self,
        database: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize database context.
        
        Args:
            database: Database name (if None, loads from config)
            host: Database host (if None, loads from config)
            port: Database port (if None, loads from config)
            user: Database user (if None, loads from config)
            password: Database password (if None, tries to get from env or config)
            config_path: Path to config.yaml
        """
        db_config = load_db_config(config_path)
        
        host = host or db_config.get('host', 'localhost')
        port = port or db_config.get('port', 5432)
        database = database or db_config.get('dbname', 'rag_local')
        user = user or db_config.get('user', 'postgres')
        
        if password is None:
            password_env = db_config.get('password_env', 'PGPASSWORD_LOCAL')
            password = os.getenv(password_env) or os.getenv('PGPASSWORD_LOCAL') or os.getenv('PGPASSWORD')
        
        self.connect_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        
        self.database = PostgresqlExtDatabase(
            database,
            user=user,
            password=password,
            host=host,
            port=port
        )
        
        try:
            from database.entities import (
                EmbeddingResult,
                ColPaliEmbedding,
                SigLIPEmbedding,
                Chunk,
                Page,
                Document
            )
            
            self.embedding_results = EmbeddingResult
            self.colpali_embeddings = ColPaliEmbedding
            self.siglip_embeddings = SigLIPEmbedding
            self.chunks = Chunk
            self.pages = Page
            self.documents = Document
            
            for model in [EmbeddingResult, ColPaliEmbedding, SigLIPEmbedding, Chunk, Page, Document]:
                model._meta.database = self.database
        except ImportError as e:
            logger.warning(f"Failed to import database entities: {e}")
            self.embedding_results = None
            self.colpali_embeddings = None
            self.siglip_embeddings = None
            self.chunks = None
            self.pages = None
            self.documents = None
    
    def connect(self):
        """Connect to database."""
        try:
            self.database.connect()
            logger.info(f"Connected to database: {self.database.database}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if not self.database.is_closed():
            self.database.close()
            logger.info("Database connection closed")
    
    def connection_context(self):
        """Context manager for database connection."""
        return self.database.connection_context()
