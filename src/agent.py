# src/agent.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, List
import operator

from src.tools import search_sbc, search_spd
from src.generator import create_llm

llm = create_llm()
tools = [search_sbc, search_spd]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

# Stronger system prompt for better tool routing
system_prompt = SystemMessage(content="""
You are a precise healthcare benefits assistant.

You have two specialized tools:
- **search_sbc**: Best for Summary of Benefits and Coverage documents. Use this for questions about:
  - Deductibles, copays, coinsurance, out-of-pocket maximums
  - What services are covered, preventive care, prescription drugs
  - Coverage summaries and costs

- **search_spd**: Best for Summary Plan Description documents. Use this for:
  - Detailed plan rules, eligibility, definitions, exclusions
  - How the plan works, prior authorization, appeals, network rules
  - In-depth policy explanations

Rules:
- Choose **only the most relevant tool(s)** based on the query.
- Use `search_sbc` for general benefit/cost questions.
- Use `search_spd` for detailed rules and procedures.
- You may call **both tools** if the question clearly needs information from both documents.
- If unsure, call the most likely one first.
- Always base your final answer on the tool results. Do not hallucinate.
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
    
    # Conditional routing: after agent → tools or end
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END}
    )
    
    # After tools, go back to agent for final answer
    workflow.add_edge("tools", "agent")

    return workflow.compile()

# Global agent
voice_agent = build_agent()

def run_agent(query: str) -> str:
    """Main function to run the agent"""
    if not query or not query.strip():
        return "Please ask a question about your benefits."

    inputs = {"messages": [HumanMessage(content=query)]}
    
    try:
        result = voice_agent.invoke(inputs, {"recursion_limit": 30})
        final_message = result["messages"][-1]
        return final_message.content
    except Exception as e:
        return f"Error processing your question: {str(e)}"