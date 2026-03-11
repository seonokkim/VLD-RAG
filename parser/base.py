"""
Base interface for page parsers.
All parser engines must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image

from .schema import PageParse


class PageParser(ABC):
    """Base class for all page parsers"""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize parser.
        
        Args:
            device: Device to use ('cpu', 'gpu', or 'cuda')
        """
        self.device = device
        self._initialized = False
    
    @abstractmethod
    def initialize(self):
        """
        Initialize the model (called once per process).
        This should load the model and prepare it for inference.
        """
        pass
    
    @abstractmethod
    def parse_page(
        self,
        doc_id: str,
        page_no: int,
        image: Image.Image,
        image_path: Optional[str] = None
    ) -> PageParse:
        """
        Parse a single page.
        
        Args:
            doc_id: Document identifier
            page_no: Page number (0-based)
            image: PIL Image of the page
            image_path: Optional path to image file
        
        Returns:
            PageParse object with parsed results
        """
        pass
    
    def ensure_initialized(self):
        """Ensure the parser is initialized before use"""
        if not self._initialized:
            self.initialize()
            self._initialized = True
