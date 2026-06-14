# src/agent.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List, Dict, Optional
import operator
import json
import re
from dotenv import load_dotenv
from src.tools import search_sbc, search_spd, search_both_parallel 
from src.tools import search_sbc, search_spd
from src.generator import create_llm

load_dotenv()
llm = create_llm()

# Phrases that indicate the tool returned nothing useful
_NO_RESULT_PHRASES = [
    "no relevant information found",
    "answer not found in provided documents",
    "not found in sbc documents",
    "not found in spd documents",
]

_VAGUE_ANSWER_PHRASES = [
    "i don't know",
    "i cannot answer",
    "i am unable to",
    "i'm unable to",
    "no information available",
]

_FALLBACK_MESSAGE = (
    "I wasn't able to find relevant information in the available healthcare "
    "benefit documents for your question. Please try rephrasing, or contact "
    "your plan administrator directly."
)


def _is_empty_result(text: str) -> bool:
    """
    Detect whether a tool output indicates no useful retrieved content.
    """
    lowered = text.lower().strip()
    return any(phrase in lowered for phrase in _NO_RESULT_PHRASES)


def _is_vague_answer(text: str) -> bool:
    """
    Detect whether an LLM response is non-informative or uncertain.
    """
    lowered = text.lower().strip()
    return any(phrase in lowered for phrase in _VAGUE_ANSWER_PHRASES)


def format_answer(text: str) -> str:
    """Remove markdown symbols and clean up formatting."""
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[\*\-•]\s*", "", text, flags=re.MULTILINE)
    text = text.replace("**", "")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# Tool definitions

@tool
def search_sbc_tool(query: str) -> str:
    """Search SBC documents for costs, deductibles, copays, coinsurance,
    out-of-pocket maximums, coverage summaries, and quick benefit tables."""
    print(f"[TOOL] search_sbc_tool → {query[:80]}")
    try:
        result = search_sbc(query)
        print(f"[TOOL] SBC done | {len(result)} chars")
        return json.dumps({"answer": result, "source": "SBC"})
    except Exception as e:
        print(f"[TOOL] SBC error: {e}")
        return json.dumps({"answer": "No relevant information found in SBC documents.", "source": "SBC"})


@tool
def search_spd_tool(query: str) -> str:
    """Search SPD documents for eligibility rules, exclusions, claim procedures,
    definitions, appeals, HIPAA, PPO plan details, and maximum benefit rules."""
    print(f"[TOOL] search_spd_tool → {query[:80]}")
    try:
        result = search_spd(query)
        print(f"[TOOL] SPD done | {len(result)} chars")
        return json.dumps({"answer": result, "source": "SPD"})
    except Exception as e:
        print(f"[TOOL] SPD error: {e}")
        return json.dumps({"answer": "No relevant information found in SPD documents.", "source": "SPD"})


tools = [search_sbc_tool, search_spd_tool]
llm_with_tools = llm.bind_tools(tools)


# Agent State (shared across graph nodes)

class AgentState(TypedDict):
    """
    Shared state object used across LangGraph nodes.
    """
    messages: Annotated[List, operator.add]
    sources: Annotated[List[str], operator.add]
    routing: Optional[str]          # "sbc" | "spd" | "both"


# Router — determines which tool(s) the query needs

# Keywords that strongly suggest each document type
_SBC_KEYWORDS = {
    "deductible", "copay", "coinsurance", "premium", "out-of-pocket",
    "oop", "coverage summary", "cost", "costs", "benefit summary",
    "covered services", "in-network", "out-of-network", "cost sharing",
}
_SPD_KEYWORDS = {
    "eligib", "exclusion", "claim", "appeal", "hipaa", "ppo", "hmo",
    "definition", "rule", "procedure", "legal", "policy", "enrollment",
    "termination", "cobra", "continuation", "dependent", "beneficiar",
}


def route_query(query: str) -> str:
    """
    Classify query as 'sbc', 'spd', or 'both'.

    Strategy:
    1. Keyword pre-scan (fast, no LLM call needed for clear-cut cases).
    2. If ambiguous, ask the LLM.
    3. If the LLM hits an error, default to 'spd' (more comprehensive doc).
    """
    lowered = query.lower()
    sbc_hit = any(kw in lowered for kw in _SBC_KEYWORDS)
    spd_hit = any(kw in lowered for kw in _SPD_KEYWORDS)

    if sbc_hit and spd_hit:
        print("[ROUTER] Keyword match → BOTH")
        return "both"
    if sbc_hit:
        print("[ROUTER] Keyword match → SBC")
        return "sbc"
    if spd_hit:
        print("[ROUTER] Keyword match → SPD")
        return "spd"

    # Ambiguous — ask the LLM
    prompt = f"""Classify this healthcare question into ONE category.

Question: {query}

Categories:
- SBC  → costs, deductibles, copays, coinsurance, out-of-pocket, coverage summary
- SPD  → eligibility, exclusions, claim procedures, definitions, appeals, legal, policy
- BOTH → the question clearly needs information from BOTH documents

Reply with exactly one word: SBC, SPD, or BOTH"""

    try:
        response = llm.invoke(prompt)
        decision = response.content.strip().upper()
        if "BOTH" in decision:
            result = "both"
        elif "SBC" in decision:
            result = "sbc"
        else:
            result = "spd"
        print(f"[ROUTER] LLM decided → {result.upper()}")
        return result
    except Exception as e:
        print(f"[ROUTER] LLM error: {e} → defaulting to SPD")
        return "spd"


# Nodes

def agent_node(state: AgentState):
    """
    Decides which tool(s) to call.

    For 'both' queries: manually construct two tool calls and short-circuit
    the normal tool_condition flow by injecting fake AI tool-call messages
    that the tools_node will execute.
    """
    last_query = state["messages"][-1].content
    routing = route_query(last_query)

    if routing == "both":
        # Build a system prompt that forces calling BOTH tools
        system = SystemMessage(content="""You are a healthcare benefits assistant.
The user's question requires BOTH the SBC and SPD documents.
You MUST call search_sbc_tool AND search_spd_tool in the same response.
Do not answer without calling both tools.""")
    elif routing == "sbc":
        system = SystemMessage(content="""You are a healthcare benefits assistant.
This question is about costs, coverage, or benefits summaries.
You MUST call search_sbc_tool to answer it. Do not call search_spd_tool.""")
    else:
        system = SystemMessage(content="""You are a healthcare benefits assistant.
This question is about plan rules, eligibility, exclusions, or procedures.
You MUST call search_spd_tool to answer it. Do not call search_sbc_tool.""")

    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "routing": routing}


def tools_node(state: AgentState):
    """Execute tool calls and track sources."""
    print("[TOOLS] Executing tool calls...")
    tool_executor = ToolNode(tools)
    result = tool_executor.invoke(state)
    print("[TOOLS] Done")

    sources = []
    for msg in result.get("messages", []):
        content = msg.content if hasattr(msg, "content") else ""
        if not content:
            continue
        # ToolMessage content can be a list of dicts or a plain string
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    try:
                        data = json.loads(item.get("text", "{}"))
                        if "source" in data:
                            sources.append(data["source"])
                    except Exception:
                        pass
        else:
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "source" in data:
                    sources.append(data["source"])
            except Exception:
                pass

    return {"messages": result["messages"], "sources": sources}


def final_answer_node(state: AgentState):
    """
    Merge tool results into a clean final answer.

    Handles three cases:
    1. Single tool result  → clean and return.
    2. Two tool results    → merge answers with a section separator.
    3. All results empty   → return the fallback message.
    """
    # Collect all tool result messages
    tool_results = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict):
                    tool_results.append(data)
            except Exception:
                tool_results.append({"answer": msg.content, "source": "Unknown"})

    # Deduplicate sources
    sources = list(dict.fromkeys(state.get("sources", [])))

    if not tool_results:
        # No tool was called — LLM answered directly (or refused)
        last = state["messages"][-1]
        raw = last.content if hasattr(last, "content") else str(last)
        if _is_vague_answer(raw) or _is_empty_result(raw):
            return {"messages": [AIMessage(content=_FALLBACK_MESSAGE)]}
        cleaned = format_answer(raw)
        return {"messages": [AIMessage(content=cleaned)]}

    # Separate empty vs useful results
    useful = [r for r in tool_results if not _is_empty_result(r.get("answer", ""))]
    empty  = [r for r in tool_results if     _is_empty_result(r.get("answer", ""))]

    if not useful:
        print("[FALLBACK] All tools returned no useful result.")
        return {"messages": [AIMessage(content=_FALLBACK_MESSAGE)]}

    if len(useful) == 1:
        answer = format_answer(useful[0]["answer"])
    else:
        # Merge two answers with a clear separator
        parts = []
        for r in useful:
            src = r.get("source", "Document")
            body = format_answer(r["answer"])
            parts.append(f"From {src}:\n{body}")
        answer = "\n\n---\n\n".join(parts)

    if empty:
        not_found = ", ".join(r.get("source", "?") for r in empty)
        answer += f"\n\nNote: No relevant information was found in {not_found}."

    if sources:
        final_content = f"{answer}\n\nSource documents used: {', '.join(sources)}"
    else:
        final_content = answer

    return {"messages": [AIMessage(content=final_content)]}


def build_agent():
    """
    Builds and compiles the LangGraph workflow.

    Flow:
        agent → tools → final_answer → END
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("final_answer", final_answer_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": "final_answer"},
    )
    workflow.add_edge("tools", "final_answer")
    workflow.add_edge("final_answer", END)

    return workflow.compile()


agent = build_agent()


def run_agent(query: str) -> Dict:
    """
    Run the agent on a user query.

    Returns:
        {
          "answer":      str,
          "sources":     List[str],   # deduplicated
          "token_usage": dict | None,
        }
    """
    print(f"\n{'='*60}")
    print(f"[AGENT] Query: {query}")
    print(f"{'='*60}\n")

    if not query or not query.strip():
        return {"answer": "Please ask a question about your benefits.", "sources": [], "token_usage": None}

    inputs = {
        "messages": [HumanMessage(content=query)],
        "sources": [],
        "routing": None,
        "token_usage": None,
    }

    try:
        result = agent.invoke(inputs, {"recursion_limit": 20})
        final_answer = result["messages"][-1].content
        sources = list(dict.fromkeys(result.get("sources", [])))   # deduplicated here too
        token_usage = result.get("token_usage")

        print(f"[AGENT] Done | Sources: {sources}")
        print(f"{'='*60}\n")

        return {"answer": final_answer, "sources": sources, "token_usage": token_usage}

    except Exception as e:
        print(f"[AGENT] Error: {e}")
        return {"answer": f"Error processing your question: {str(e)}", "sources": [], "token_usage": None}