# src/agent.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List
import operator

from src.tools import search_sbc, search_spd
from src.generator import create_llm

llm = create_llm()
tools = [search_sbc, search_spd]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

system_prompt = SystemMessage(content="""
You are a helpful healthcare benefits assistant with access to SBC and SPD documents.
- Use search_sbc for questions about Summary of Benefits and Coverage (coverage details, deductibles, copays, etc.)
- Use search_spd for detailed plan rules, eligibility, exclusions, definitions.
- You can call both tools if the question needs information from both.
- Always answer based on retrieved information. Be clear and professional.
""")

def agent_node(state: AgentState):
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()

# Global agent instance
voice_agent = build_agent()

def run_agent(query: str) -> str:
    """Easy function to call the agent from main.py or anywhere"""
    if not query or not query.strip():
        return "Please ask a question."
    
    inputs = {"messages": [HumanMessage(content=query)]}
    result = voice_agent.invoke(inputs)
    final_response = result["messages"][-1].content
    return final_response