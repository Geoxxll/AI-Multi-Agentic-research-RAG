from dataclasses import dataclass, field
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing import Annotated, Literal, TypedDict
from utils.utils import reduce_docs
from langchain_core.documents import Document
from pydantic import BaseModel
@dataclass(kw_only=True)
class InputState:
    messages: Annotated[list[AnyMessage], add_messages]
    user_question: str

class Router(BaseModel):
    logic: str
    type: Literal["more-info", "research", "general"]

@dataclass(kw_only=True)
class AgentState(InputState):
    router: Router = field(default_factory=lambda: Router(type="general", logic=""))
    steps: list[str] = field(default_factory=list)
    original_steps: list[str] = field(default_factory=list)
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    post_processed_docs: list[Document] = field(default_factory=list)
    distilled_docs: Annotated[list[str], reduce_docs] = field(default_factory=list)

@dataclass(kw_only=True)
class DistillAgentState(InputState):
    doc: str
