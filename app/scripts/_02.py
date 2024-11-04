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


def line_print():
    print(
        "================================================================================"
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]


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

graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")


graph = graph_builder.compile(memory,interrupt_before=["tools"])


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
            # line_print()
            # print("press enter for continue")
            # input()
            # for event in graph.stream(None, config, stream_mode="values"):
            #     if "messages" in event:
            #         event["messages"][-1].pretty_print()
            snapshot = graph.get_state(config)
            existing_message = snapshot.values["messages"][-1]
            existing_message.pretty_print()
            answer = (
                "LangGraph is a library for building stateful, multi-actor applications with LLMs."
            )
            new_messages = [
                ToolMessage(content=answer, tool_call_id=existing_message.tool_calls[0]["id"]),
                AIMessage(content=answer),
            ]

            new_messages[-1].pretty_print()
            graph.update_state(
                config,
                {"messages": new_messages},
                as_node="chatbot",
            )

            print("\n\nLast 2 messages;")
            print(graph.get_state(config).values["messages"][-2:])

        line_print()
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

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     pass
