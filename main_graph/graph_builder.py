from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from utils.utils import config, align_evidence_to_steps, write_step_from_evidence
from utils.signature_extractor import paper_signature
from utils.prompt import ROUTER_SYSTEM_PROMPT, CREATE_PLAN_SYSTEM_PROMPT, ANSWER_GENERAL_QUERY_SYSTEM_PROPT, GENERATE_RESPONSE_SYSTEM_PROMPT, DOCUMENT_DISTILLATION_SYSTEM_PROMPT
from typing import Any, Literal, TypedDict, cast
import logging
from main_graph.graph_state import InputState, AgentState, Router, DistillAgentState
from research_graph.graph_builder import researcher_graph
from RAG.post_processor import PostProcessor
from langchain_core.messages import AIMessage
"""
user_input -> query_router -> general_query ----------------------------------------------------------------------------|
                    ^       -> research_query -> decompose_query_to_steps -> conduct_research -> check_finished -> response -> check_hallucination
                    |       -> ask_for_more_info                                     |________________|                 ^_________________|
                    |_________________|


"""

MODEL_NAME = config["llm"]["gpt_4o_mini"]
TEMPERATURE = config["llm"]["temperature"]

async def query_router(
        state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    system_prompt = ROUTER_SYSTEM_PROMPT.format(
        paper_signature=paper_signature
    )
    model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, streaming=True)
    messages = [
        {"role": "system", "content": system_prompt}
    ] + state.messages
    logging.info("---ANALYZE AND ROUTE QUERY---")
    logging.info(f"MESSAGES: {state.messages}")
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    return {"router": response}

def router(state: AgentState) -> Literal["research_query", "general_query", "more_info"]:
    type = state.router.type
    if type not in ["research", "general", "more_info"]:
        raise ValueError(f"Invalid return value: {type}")
    if type == "research":
        return "research_query"
    elif type == "general":
        return "general_query"
    else:
        return "more_info"
async def create_research_plan(
        state: AgentState, *, config: RunnableConfig
):
    class Plan(TypedDict):
        steps: list[str]
    model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, streaming=True)
    system_prompt = CREATE_PLAN_SYSTEM_PROMPT.format(
        paper_signature=paper_signature
    )
    messages = [
        {"role": "system", "content": system_prompt}
    ] + state.messages
    response = cast(Plan, await model.with_structured_output(Plan).ainvoke(messages))
    # return {"steps": response["steps"], "documents": "delete"}
    return {"steps": response["steps"], "original_steps": response["steps"]}

async def conduct_research(
        state: AgentState, *, config: RunnableConfig
):
    step = state.steps[0]
    result = await researcher_graph.ainvoke({"question": step})
    docs = result["documents"]
    logging.info(f"\n{len(docs)} documents retrieved in total for the step: {step}.")    
    return {"documents": result["documents"], "steps": state.steps[1:]}

def check_research_finished(
        state: AgentState
) -> Literal["conduct_research", "post_process_document"]:
    if len(state.steps or []) > 0:
        return "conduct_research"
    return "post_process_document"

async def distill_retrieved_document(
        state: DistillAgentState, *, config: RunnableConfig
):
    model = ChatOpenAI(model=MODEL_NAME, temperature=0, streaming=False)
    system_prompt = DOCUMENT_DISTILLATION_SYSTEM_PROMPT.format(
        user_query=state.user_question,
        document=state.doc,
    )
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    class Distilled_doc(TypedDict):
        facts: list[str]
    response = await model.with_structured_output(Distilled_doc).ainvoke(messages)
    # print(f"📝 Distilled docs: {response}")
    return {"distilled_docs": response["facts"]}

def post_process_document(
        state: AgentState, *, config: RunnableConfig
):
    raw_retrieved_docs = state.documents
    print(f"😈 Number of retrieved documents before post process: {len(raw_retrieved_docs)}")
    post_processor = PostProcessor(raw_retrieved_docs=raw_retrieved_docs)
    post_processed_docs = post_processor.dedup_by_content()
    print(f"🥳 Number of retrieved documents after post process: {len(post_processed_docs)}")
    return {"post_processed_docs": post_processed_docs}

def distill_document_in_parallel(state: AgentState):
    print("🧐 Sending docuemnt to distill_retrieved_document node !!!!!!")
    return [
        Send("distill_retrieved_document", DistillAgentState(doc=d.page_content, user_question=state.user_question, messages=state.messages)) for d in state.post_processed_docs
    ]
# async def respond(
#         state: AgentState, *, config: RunnableConfig
# ):
#     model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, streaming=True)
#     context = ""
#     distill_docs = state.distilled_docs
#     for idx, doc in enumerate(distill_docs):
#         num_doc = idx + 1
#         context += "Evidence " + str(num_doc) + ":\n" + doc + "\n--------------------\n\n"

#     print(f"📝 The following text is distilled documents:\n{context}")
#     system_prompt = GENERATE_RESPONSE_SYSTEM_PROMPT.format(
#         research_plan=state.original_steps,
#         context=context
#     )
#     messages = [
#         {"role": "system", "content": GENERATE_RESPONSE_SYSTEM_PROMPT}
#     ] + state.messages
#     response = await model.ainvoke(messages)
#     return {"messages": [response]}

async def respond(
        state: AgentState, *, config: RunnableConfig
):
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)

    evidence = state.distilled_docs
    steps = state.original_steps
    print("📝 The following text is distilled documents:")
    print("\n".join([f"Evidence {i+1}: {e}" for i, e in enumerate(evidence)]))

    # -------- Stage A: Evidence → Step alignment --------
    alignment = await align_evidence_to_steps(
        model=model,
        steps=steps,
        evidence=evidence
    )

    # -------- Stage B: Evidence-first generation --------

    final_answer = "\n\n---------------------------------------------------\n\nTo answer your research inquire based on the submitted paper.\n\n"

    for idx, step in enumerate(steps):
        final_answer += f"### {step}\n"
        ev_ids = alignment.get(str(idx), [])
        selected_evidence = [evidence[i] for i in ev_ids]

        result = await write_step_from_evidence(
            model=model,
            step=step,
            selected_evidence=selected_evidence
        )

        final_answer += "- " + result["paragraph"] + "\n\n"
        final_answer += f"👉📝 supported by {ev_ids}\n\n"
        final_answer += "---------------------------------------------------------------\n"
    return {
        "messages": [AIMessage(content=final_answer)]
    }
        

async def answer_general_query(
        state: AgentState, *, config=RunnableConfig
):
    model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, streaming=True)
    system_prompt = ANSWER_GENERAL_QUERY_SYSTEM_PROPT.format(
        logic=state.router.logic
    )
    messages = [
        {"role": "system", "content": system_prompt}
    ] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def ask_for_more_info(
        state: AgentState, *, config=RunnableConfig
):
    pass
builder = StateGraph(AgentState, input=InputState)
builder.add_node(query_router)
builder.add_node(create_research_plan)
builder.add_node(answer_general_query)
builder.add_node(ask_for_more_info)
builder.add_node(conduct_research)
builder.add_node(respond)
builder.add_node(distill_retrieved_document)
builder.add_node(post_process_document)

builder.add_edge(START, "query_router")
builder.add_conditional_edges(
    "query_router", 
    router, 
    {"general_query": "answer_general_query", "research_query": "create_research_plan", "more_info": "ask_for_more_info"}
)
builder.add_edge("create_research_plan", "conduct_research")    
builder.add_conditional_edges("conduct_research", check_research_finished)
# Path map should reference node *names* (strings), not function objects.
builder.add_conditional_edges("post_process_document", distill_document_in_parallel, path_map=["distill_retrieved_document"])
builder.add_edge("distill_retrieved_document", "respond")
builder.add_edge("respond", END)

graph = builder.compile()