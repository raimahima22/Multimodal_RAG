
# src/agent.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List, Dict
import operator
import json
from dotenv import load_dotenv

from src.tools import search_sbc, search_spd
from src.generator import create_llm

load_dotenv()
llm = create_llm()

_NO_RESULT_PHRASES = [
    "no relevant information found",
    "answer not found in provided documents",
]
 
def _is_empty_result(text: str) -> bool:
    """Return True when a tool result carries no useful information."""
    lowered = text.lower().strip()
    return any(phrase in lowered for phrase in _NO_RESULT_PHRASES)
 
_FALLBACK_MESSAGE = (
    "I'm sorry, I wasn't able to find relevant information in the "
    "available healthcare benefit documents for your question. "
    "Please try rephrasing, or contact your plan administrator directly."
)


# ========================== TOOLS with Logging ==========================
@tool
def search_sbc_tool(query: str) -> str:
    """Use ONLY for quick benefit summaries, costs, deductibles, copays, coinsurance, out-of-pocket maximums."""
    print(f" [TOOL CALL] search_sbc_tool → Query: {query[:80]}...")
    result = search_sbc(query)
    print(f" SBC Tool completed | Result length: {len(result)} chars")
    return json.dumps({"answer": result, "source": "SBC"})


@tool
def search_spd_tool(query: str) -> str:
    """Use ONLY for detailed plan rules, eligibility, exclusions, limitations, claim procedures, definitions, appeals."""
    print(f" [TOOL CALL] search_spd_tool → Query: {query[:80]}...")
    result = search_spd(query)
    print(f" SPD Tool completed | Result length: {len(result)} chars")
    return json.dumps({"answer": result, "source": "SPD"})


tools = [search_sbc_tool, search_spd_tool]
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    sources: Annotated[List[str], operator.add]


# ========================== ROUTER with Logging ==========================
def route_query(query: str) -> str:
    """Force routing decision every time"""
    print(f" [ROUTER] Analyzing query: {query[:100]}...")
    
    prompt = f"""
    You are a strict router. Classify the CURRENT question into exactly one category.

    Question: {query}

    Choose ONLY one:
    - SBC → costs, deductibles, copays, coverage summary, out-of-pocket, quick benefits
    - SPD → rules, eligibility, exclusions, procedures, definitions, legal, claims

    Answer with only one word: SBC or SPD
    """
    try:
        response = llm.invoke(prompt)
        decision = response.content.strip().upper()
        result = "sbc" if "SBC" in decision else "spd"
        print(f" [ROUTER] Decided: {result.upper()}")
        return result
    except Exception as e:
        print(f" Router failed: {e} → defaulting to SPD")
        return "spd"


# ========================== NODES ==========================
def agent_node(state: AgentState):
    last_query = state["messages"][-1].content
    suggested = route_query(last_query)
    
    system = SystemMessage(content=f"""
You are a healthcare benefits assistant.
Router strongly recommends using the **{suggested.upper()}** tool.
Follow the router. You MUST use the correct tool. Do not refuse.
    """)
    
    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tools_node(state: AgentState):
    print(" Executing tool(s)...")
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)
    print(" Tool execution finished")
    
    sources = []
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage) and msg.content:
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and "source" in data:
                    sources.append(data["source"])
            except:
                pass
    return {"messages": result["messages"], "sources": sources}


def final_answer_node(state: AgentState):
    sources = list(dict.fromkeys(state.get("sources", [])))
    
    # Get the last message (tool result)
    last_message = state["messages"][-1]
    
    # If the last message is a JSON string from tool, clean it
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    try:
        if content.strip().startswith('{') and '"answer"' in content:
            data = json.loads(content)
            clean_answer = data.get("answer", content)
        else:
            clean_answer = content
    except:
        clean_answer = content
    
    # ── Fallback: both (or the only) tool returned nothing useful ───────────
    if _is_empty_result(clean_answer):
        print(" [FALLBACK] Tool returned no useful result — using fallback message.")
        return {"messages": [AIMessage(content=_FALLBACK_MESSAGE)]}

    # Create nice final response
    if sources:
        final_content = f"{clean_answer}\n\n**Sources:** {', '.join(sources)}"
    else:
        final_content = clean_answer

    # print(f" Final clean answer generated | Sources: {sources}")
    return {"messages": [AIMessage(content=final_content)]}


# ========================== BUILD ==========================
def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("final_answer", final_answer_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "final_answer"})
    workflow.add_edge("tools", "final_answer")
    workflow.add_edge("final_answer", END)
    
    return workflow.compile()


agent = build_agent()


def run_agent(query: str) -> Dict:
    print(f"\n{'='*60}")
    print(f" NEW QUERY: {query}")
    print(f"{'='*60}\n")
    
    if not query or not query.strip():
        return {"answer": "Please ask a question about your benefits.", "sources": []}
    
    inputs = {"messages": [HumanMessage(content=query)], "sources": []}
    
    try:
        result = agent.invoke(inputs, {"recursion_limit": 20})
        final_answer = result["messages"][-1].content
        sources = result.get("sources", [])
        
        # print(f" FINAL SOURCES USED: {sources}")
        print(f"{'='*60}\n")
        
        return {"answer": final_answer, "sources": sources}
        
    except Exception as e:
        print(f" Agent Error: {e}")
        return {"answer": f"Error: {str(e)}", "sources": []}