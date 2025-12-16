from research_graph.graph_state import ResearchAgentState, QueryState
from langchain_core.runnables import RunnableConfig
from utils.utils import config
from langchain_openai import ChatOpenAI
from utils.prompt import GENERATE_QUERIES_SYSTEM_PROMPT
from typing import TypedDict, cast
from langgraph.graph import StateGraph, START, END
from utils.signature_extractor import paper_signature
from RAG.retriever_utils import retrieve
from langgraph.types import Send
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = config["llm"]["gpt_4o_mini"]
HEADERS_TO_SPLIT_ON = config["retriever"]["headers_to_split_on"]
FILE_PTH = config["retriever"]["file_pth"]
# TODO:
#     Input (step == user_question)
#     decomposite the single question / step to a more refined step
#     parallel retrieval

async def generate_queries(
    state: ResearchAgentState, *, config: RunnableConfig
):
    # Print node boundaries so subgraph activity is visible in console
    print(f"\n============ ENTER NODE (research_graph): generate_queries ============\n")
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)
    system_prompt = GENERATE_QUERIES_SYSTEM_PROMPT.format(
        paper_signature=paper_signature
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.question}
    ]
    class Queries(TypedDict):
        queries: list[str]
    response = cast(Queries, await model.with_structured_output(Queries).ainvoke(messages))
    print("👉 Here is generated queries:\n" + "\n".join(response['queries']) + f"\nbased on user question: {state.question}")
    print("\n------------ END generate_queries ------------\n")
    # ensure returned shape is simple list of queries
    return {"queries": response["queries"]}

async def research_over_document(
    state: QueryState, *, config: RunnableConfig
):
    logger.info("---RETRIEVING DOCUMENTS---")
    logger.info(f"Query for the retrieval process: {state['query']}")
    retrieved_docs = retrieve(headers_to_split_on=HEADERS_TO_SPLIT_ON, query=state['query'], file_pth=FILE_PTH)
    print(f"👉 Research for query: {state['query']} completed..")
    return {"documents": retrieved_docs}

def retrieve_in_parallell(
        state: ResearchAgentState
):
    return [Send("research_over_document", QueryState(query=query)) for query in state.queries]
    

builder = StateGraph(ResearchAgentState)
builder.add_node(generate_queries)
builder.add_edge(START, "generate_queries")
builder.add_node(research_over_document)
builder.add_conditional_edges("generate_queries", retrieve_in_parallell, path_map=["research_over_document"])
builder.add_edge("research_over_document", END)

researcher_graph = builder.compile()