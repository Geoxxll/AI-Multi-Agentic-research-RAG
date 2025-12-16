from langchain_core.documents import Document
from typing import Annotated, TypedDict
from dataclasses import dataclass, field
from utils.utils import reduce_docs

@dataclass(kw_only=True)
class ResearchAgentState:
    question: str
    queries: list[str] = field(default_factory=list)
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)

class QueryState(TypedDict):
    query: str