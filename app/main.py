# %%
from typing import Annotated

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
import json
from pprint import pprint
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition


def print_line():
    print(
        "================================================================================"
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]


# class BasicToolNode:
#     def __init__(self, tools: list) -> None:
#         self.tools_by_name = {tool.name: tool for tool in tools}

#     def __call__(self, inputs: dict):
#         if messages := inputs.get("messages", []):
#             message = messages[-1]
#         else:
#             raise ValueError("No message found in input")
#         outputs = []
#         for tool_call in message.tool_calls:
#             tool_result = self.tools_by_name[tool_call["name"]].invoke(
#                 tool_call["args"]
#             )
#             outputs.append(
#                 ToolMessage(
#                     content=json.dumps(tool_result),
#                     name=tool_call["name"],
#                     tool_call_id=tool_call["id"],
#                 )
#             )
#         return {"messages": outputs}


graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o-mini")

tool = DuckDuckGoSearchResults(num_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)
memory = MemorySaver()


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def route_tools(
    state: State,
):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


tool_node = ToolNode(tools)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    # route_tools,
    # {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],
)

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass


config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    ):
        if "messages" in event:
            event["messages"][-1].pretty_print()


# %%
while True:
    try:
        if graph.get_state(config).next:
            print_line()
            print("press enter for continue")
            input()
            for event in graph.stream(None, config, stream_mode="values"):
                if "messages" in event:
                    event["messages"][-1].pretty_print()

        print_line()
        user_input = input("input: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

snapshot = graph.get_state(config)
pprint(snapshot._asdict())
pprint(f"Next node: {snapshot.next}")
