"""
Common output schema for parser engines.
Unifies output from PP-StructureV3 and Dolphin parsers.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, List

BlockType = Literal["text", "title", "table", "figure", "formula", "list", "footer", "header"]


@dataclass
class BBox:
    """Bounding box with normalized coordinates (0..1)"""
    x1: float  # normalized 0..1
    y1: float
    x2: float
    y2: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2
        }
    
    @classmethod
    def from_pixel(cls, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> "BBox":
        """Create BBox from pixel coordinates"""
        return cls(
            x1=x1 / width,
            y1=y1 / height,
            x2=x2 / width,
            y2=y2 / height
        )


@dataclass
class Block:
    """Document block (text, table, figure, etc.)"""
    block_id: str
    page_no: int
    type: BlockType
    bbox: BBox
    text: Optional[str] = None
    table_html: Optional[str] = None
    table_json: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None
    source_engine: Optional[str] = None  # "ppstructurev3" or "dolphin"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {
            "block_id": self.block_id,
            "page_no": self.page_no,
            "type": self.type,
            "bbox": self.bbox.to_dict(),
        }
        if self.text is not None:
            result["text"] = self.text
        if self.table_html is not None:
            result["table_html"] = self.table_html
        if self.table_json is not None:
            result["table_json"] = self.table_json
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.source_engine is not None:
            result["source_engine"] = self.source_engine
        return result


@dataclass
class PageParse:
    """Parsed page result"""
    doc_id: str
    page_no: int
    image_uri: str
    width: int
    height: int
    blocks: List[Block] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "doc_id": self.doc_id,
            "page_no": self.page_no,
            "image_uri": self.image_uri,
            "width": self.width,
            "height": self.height,
            "blocks": [block.to_dict() for block in self.blocks]
        }
