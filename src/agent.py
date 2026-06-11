from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, Annotated, List
import operator

from src.tools import search_sbc, search_spd

# Bind tools
tools = [search_sbc, search_spd]

# LLM (use the same one you're already using)
from src.generator import create_llm
llm = create_llm()
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[List, operator.add]


# System Prompt
system_prompt = SystemMessage(
    content="""You are a helpful healthcare benefits assistant.
You have access to two specialized tools:
- search_sbc: Use for questions about Summary of Benefits and Coverage (deductibles, copays, covered services, etc.)
- search_spd: Use for detailed plan rules, definitions, exclusions, eligibility, etc.

Decide which tool(s) to use based on the question. 
You can use both if needed. 
If no relevant information is found, clearly say so."""
)


def agent_node(state: AgentState):
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,  # routes to tools or END
        {"tools": "tools", "__end__": END}
    )
    workflow.add_edge("tools", "agent")   # after tool call, go back to agent

    return workflow.compile()


# Create the agent
voice_agent = build_agent()


# Simple wrapper for easy calling
def run_agent(query: str) -> str:
    """Main function to call the agent"""
    inputs = {"messages": [HumanMessage(content=query)]}
    
    result = voice_agent.invoke(inputs)
    final_message = result["messages"][-1]
    
    if hasattr(final_message, "content"):
        return final_message.content
    return str(final_message)