from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """Simple document class — replaces langchain_core.documents.Document."""

    page_content: str = ""
    metadata: dict = field(default_factory=dict)
    id: Optional[str] = None
