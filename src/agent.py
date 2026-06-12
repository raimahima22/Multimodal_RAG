# # src/agent.py
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.tools import tool
# from typing import TypedDict, Annotated, List
# import operator
# from dotenv import load_dotenv

# from src.tools import search_sbc, search_spd
# from src.generator import create_llm

# load_dotenv()
# llm = create_llm()

# # ── Wrap tools properly for LangGraph ─────────────────────────────
# @tool
# def search_sbc_tool(query: str) -> str:
#     """Search the Summary of Benefits and Coverage (SBC) document.
#     Use this for questions about deductibles, copays, coinsurance, covered services,
#     out-of-pocket maximums, and quick benefit summaries."""
#     return search_sbc(query)

# @tool
# def search_spd_tool(query: str) -> str:
#     """Search the Summary Plan Description (SPD) document.
#     Use this for detailed plan rules, eligibility, exclusions, definitions,
#     claim procedures, and in-depth policy information."""
#     return search_spd(query)

# tools = [search_sbc_tool, search_spd_tool]
# llm_with_tools = llm.bind_tools(tools)

# class AgentState(TypedDict):
#     messages: Annotated[List, operator.add]

# system_prompt = SystemMessage(content="""
# You are an expert healthcare benefits assistant with access to two key documents:
# - SBC (Summary of Benefits and Coverage): Best for quick benefit details, costs, and coverage.
# - SPD (Summary Plan Description): Best for detailed rules, eligibility, exclusions, and procedures.

# Rules:
# - Use search_sbc_tool for simple coverage questions (deductibles, copays, what’s covered).
# - Use search_spd_tool for detailed explanations, limitations, or definitions.
# - You may call both tools if the question requires information from both documents.
# - Always base your final answer strictly on the tool results.
# - Be clear, professional, and concise. Use bullet points when helpful.
# - If no relevant information is found, clearly say so.
# """)

# def agent_node(state: AgentState):
#     messages = [system_prompt] + state["messages"]
#     response = llm_with_tools.invoke(messages)
#     return {"messages": [response]}

# def build_agent():
#     workflow = StateGraph(AgentState)
#     workflow.add_node("agent", agent_node)
#     workflow.add_node("tools", ToolNode(tools))

#     workflow.set_entry_point("agent")
#     workflow.add_conditional_edges(
#         "agent",
#         tools_condition,
#         {"tools": "tools", "__end__": END}
#     )
#     workflow.add_edge("tools", "agent")

#     return workflow.compile()

# # Global compiled agent
# agent = build_agent()

# def run_agent(query: str) -> str:
#     """Call this from main.py"""
#     if not query or not query.strip():
#         return "Please ask a question about your benefits."
    
#     inputs = {"messages": [HumanMessage(content=query)]}
    
#     try:
#         result = agent.invoke(inputs, {"recursion_limit": 12})
#         final_message = result["messages"][-1]
        
#         # If it's a tool call response, return the content
#         if hasattr(final_message, "content"):
#             return final_message.content
#         return str(final_message)
        
#     except Exception as e:
#         return f" Error processing your request: {str(e)}"

# src/agent.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv

from src.tools import search_sbc, search_spd
from src.generator import create_llm

load_dotenv()
llm = create_llm()

# ── Tool wrappers ─────────────────────────────────────────────────────────────

@tool
def search_sbc_tool(query: str) -> str:
    """Search the Summary of Benefits and Coverage (SBC) document.
    Use this for questions about deductibles, copays, coinsurance, covered services,
    out-of-pocket maximums, and quick benefit summaries."""
    return search_sbc(query)


@tool
def search_spd_tool(query: str) -> str:
    """Search the Summary Plan Description (SPD) document.
    Use this for detailed plan rules, eligibility, exclusions, definitions,
    claim procedures, and in-depth policy information."""
    return search_spd(query)


tools = [search_sbc_tool, search_spd_tool]
llm_with_tools = llm.bind_tools(tools)

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

# ── System prompt ──────────────────────────────────────────────────────────────

system_prompt = SystemMessage(content="""
You are an expert healthcare benefits assistant with access to two key documents:
- SBC (Summary of Benefits and Coverage): Best for quick benefit details, costs, and coverage.
- SPD (Summary Plan Description): Best for detailed rules, eligibility, exclusions, and procedures.

Rules:
- Use search_sbc_tool for simple coverage questions (deductibles, copays, what's covered).
- Use search_spd_tool for detailed explanations, limitations, or definitions.
- You may call both tools if the question requires information from both documents.
- Always base your final answer strictly on the tool results.
- Be clear, professional, and concise. Use bullet points when helpful.
- If the tool results contain "No relevant information found" for ALL tools called,
  respond with: "I'm sorry, I couldn't find relevant information about that in your
  plan documents. Please contact your HR department or benefits administrator for help."
""")

# ── Nodes ─────────────────────────────────────────────────────────────────────

def agent_node(state: AgentState):
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ── Graph ─────────────────────────────────────────────────────────────────────

def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# Compiled once at import time
agent = build_agent()


def run_agent(query: str) -> str:
    """Entry point called by main.py and voice.py."""
    if not query or not query.strip():
        return "Please ask a question about your benefits."

    inputs = {"messages": [HumanMessage(content=query)]}

    try:
        result = agent.invoke(inputs, {"recursion_limit": 12})
        final_message = result["messages"][-1]
        if hasattr(final_message, "content"):
            return final_message.content
        return str(final_message)
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"