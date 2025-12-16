import asyncio
from main_graph.graph_state import InputState
from main_graph.graph_builder import graph
from utils.utils import new_uuid

thread = {"configurable": {"thread_id": new_uuid()}}

async def process_query(query):
    inputState = InputState(messages=query, user_question=query)
    prev_node = None
    async for c, metadata in graph.astream(input=inputState, stream_mode="messages", config=thread):
        node = metadata.get("langgraph_node") or metadata.get("step")
        if node != prev_node:
            if prev_node is not None:
                print(f"\n------------ END {prev_node} ------------\n")
            print(f"\n============ ENTER NODE: {node} ============\n")
            prev_node = node
        if c.content:
            # 流式输出
            for char in c.content:
                print(char, end="", flush=True)
                await asyncio.sleep(0.001)
    print()  # 换行
    
"""What is the high-level workflow of ZCTG for automatically chapter video and generate title for each chapter?"""

async def main():
    import builtins
    input_fn = builtins.input
    print("Enter your query (type '-q' to quit):")
    while True:
        query = input_fn("> ")
        if query.strip().lower() == "-q":
            print("Exiting...")
            break
        await process_query(query)

if __name__ == "__main__":
    asyncio.run(main())
