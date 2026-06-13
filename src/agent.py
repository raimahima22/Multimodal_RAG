# # # src/agent.py
# # from langgraph.graph import StateGraph, END
# # from langgraph.prebuilt import ToolNode, tools_condition
# # from langchain_core.messages import HumanMessage, SystemMessage
# # from langchain_core.tools import tool
# # from typing import TypedDict, Annotated, List
# # import operator
# # from dotenv import load_dotenv

# # from src.tools import search_sbc, search_spd
# # from src.generator import create_llm

# # load_dotenv()
# # llm = create_llm()

# # # ── Wrap tools properly for LangGraph ─────────────────────────────
# # @tool
# # def search_sbc_tool(query: str) -> str:
# #     """Search the Summary of Benefits and Coverage (SBC) document.
# #     Use this for questions about deductibles, copays, coinsurance, covered services,
# #     out-of-pocket maximums, and quick benefit summaries."""
# #     return search_sbc(query)

# # @tool
# # def search_spd_tool(query: str) -> str:
# #     """Search the Summary Plan Description (SPD) document.
# #     Use this for detailed plan rules, eligibility, exclusions, definitions,
# #     claim procedures, and in-depth policy information."""
# #     return search_spd(query)

# # tools = [search_sbc_tool, ]
# # llm_with_tools = llm.bind_tools(tools)

# # class AgentState(TypedDict):
# #     messages: Annotated[List, operator.add]

# # system_prompt = SystemMessage(content="""
# # You are an expert healthcare benefits assistant with access to two key documents:
# # - SBC (Summary of Benefits and Coverage): Best for quick benefit details, costs, and coverage.
# # - SPD (Summary Plan Description): Best for detailed rules, eligibility, exclusions, and procedures.

# # Rules:
# # - Use search_sbc_tool for simple coverage questions (deductibles, copays, what’s covered).
# # - Use search_spd_tool for detailed explanations, limitations, or definitions.
# # - You may call both tools if the question requires information from both documents.
# # - Always base your final answer strictly on the tool results.
# # - Be clear, professional, and concise. Use bullet points when helpful.
# # - If no relevant information is found, clearly say so.
# # """)

# # def agent_node(state: AgentState):
# #     messages = [system_prompt] + state["messages"]
# #     response = llm_with_tools.invoke(messages)
# #     return {"messages": [response]}

# # def build_agent():
# #     workflow = StateGraph(AgentState)
# #     workflow.add_node("agent", agent_node)
# #     workflow.add_node("tools", ToolNode(tools))

# #     workflow.set_entry_point("agent")
# #     workflow.add_conditional_edges(
# #         "agent",
# #         tools_condition,
# #         {"tools": "tools", "__end__": END}
# #     )
# #     workflow.add_edge("tools", "agent")

# #     return workflow.compile()

# # # Global compiled agent
# # agent = build_agent()

# # def run_agent(query: str) -> str:
# #     """Call this from main.py"""
# #     if not query or not query.strip():
# #         return "Please ask a question about your benefits."
    
# #     inputs = {"messages": [HumanMessage(content=query)]}
    
# #     try:
# #         result = agent.invoke(inputs, {"recursion_limit": 12})
# #         final_message = result["messages"][-1]
        
# #         # If it's a tool call response, return the content
# #         if hasattr(final_message, "content"):
# #             return final_message.content
# #         return str(final_message)
        
# #     except Exception as e:
# #         return f" Error processing your request: {str(e)}"

# # src/agent.py
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from langchain_core.tools import tool
# from typing import TypedDict, Annotated, List, Dict
# import operator
# import json

# from src.tools import search_sbc, search_spd
# from src.generator import create_llm

# # load_dotenv()
# llm = create_llm()


# # ========================== TOOLS (with source tracking) ==========================
# @tool
# def search_sbc_tool(query: str) -> str:
#     """Search the Summary of Benefits and Coverage (SBC) document.
#     Use for quick benefit summaries, costs, deductibles, copays, etc."""
#     result = search_sbc(query)
#     # Return structured output so we can track source
#     return json.dumps({
#         "answer": result,
#         "source": "SBC",
#         "tool": "search_sbc_tool"
#     })


# @tool
# def search_spd_tool(query: str) -> str:
#     """Search the Summary Plan Description (SPD) document.
#     Use for detailed rules, exclusions, eligibility, procedures, etc."""
#     result = search_spd(query)
#     return json.dumps({
#         "answer": result,
#         "source": "SPD",
#         "tool": "search_spd_tool"
#     })


# tools = [search_sbc_tool, search_spd_tool]
# llm_with_tools = llm.bind_tools(tools)


# class AgentState(TypedDict):
#     messages: Annotated[List, operator.add]
#     sources: Annotated[List[str], operator.add]   # Track which sources were used


# system_prompt = SystemMessage(content="""
# You are an expert healthcare benefits assistant.

# You have two tools:
# - search_sbc_tool → Best for quick benefit & cost information (SBC)
# - search_spd_tool → Best for detailed plan rules and legal information (SPD)

# You can call one or both tools when needed.
# After getting results, synthesize a clear final answer and mention the source(s) used.
# """)


# def agent_node(state: AgentState):
#     messages = [system_prompt] + state["messages"]
#     response = llm_with_tools.invoke(messages)
#     return {"messages": [response]}


# def tools_node(state: AgentState):
#     """Custom tools node to extract sources"""
#     tool_node = ToolNode(tools)
#     result = tool_node.invoke(state)
    
#     sources = []
#     for msg in result["messages"]:
#         if isinstance(msg, AIMessage) and msg.content:
#             try:
#                 data = json.loads(msg.content)
#                 if isinstance(data, dict) and "source" in data:
#                     sources.append(data["source"])
#             except:
#                 pass  # fallback if not json
    
#     return {
#         "messages": result["messages"],
#         "sources": sources
#     }


# def final_answer_node(state: AgentState):
#     """Generate clean final answer with source information"""
#     messages = state["messages"]
#     sources = list(dict.fromkeys(state.get("sources", [])))  # unique sources
    
#     # Get the last AI message (before final synthesis)
#     last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    
#     if last_ai and last_ai.tool_calls:
#         # Let the LLM create a nice final response
#         final_prompt = SystemMessage(content=f"""
#         Summarize the tool results into a clear, professional answer.
#         Sources used: {", ".join(sources) if sources else "None"}
#         Do not mention tool names. Just be helpful and transparent.
#         """)
        
#         response = llm.invoke([final_prompt] + messages[-4:])  # recent context
#         final_content = response.content
#     else:
#         final_content = last_ai.content if last_ai else "No answer generated."
    
#     # Append source info
#     if sources:
#         source_text = f"\n\n**Sources:** {', '.join(sources)}"
#         final_content += source_text
    
#     return {"messages": [AIMessage(content=final_content)]}


# def build_agent():
#     workflow = StateGraph(AgentState)
    
#     workflow.add_node("agent", agent_node)
#     workflow.add_node("tools", tools_node)
#     workflow.add_node("final_answer", final_answer_node)
    
#     workflow.set_entry_point("agent")
    
#     workflow.add_conditional_edges(
#         "agent",
#         tools_condition,
#         {"tools": "tools", "__end__": "final_answer"}
#     )
    
#     workflow.add_edge("tools", "final_answer")
#     workflow.add_edge("final_answer", END)
    
#     return workflow.compile()


# agent = build_agent()


# def run_agent(query: str) -> Dict:
#     """Run the agent and return both answer and sources"""
#     if not query or not query.strip():
#         return {"answer": "Please ask a question about your benefits.", "sources": []}
    
#     inputs = {"messages": [HumanMessage(content=query)], "sources": []}
    
#     try:
#         result = agent.invoke(inputs, {"recursion_limit": 15})
#         final_message = result["messages"][-1]
        
#         return {
#             "answer": final_message.content,
#             "sources": result.get("sources", [])
#         }
        
#     except Exception as e:
#         return {
#             "answer": f"Error processing your request: {str(e)}",
#             "sources": []
#         }
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
    final_content = state["messages"][-1].content if state["messages"] else "No answer."
    
    if sources:
        final_content += f"\n\n**Sources:** {', '.join(sources)}"
    
    print(f" Final Answer Generated | Sources: {sources}")
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
        
        print(f" FINAL SOURCES USED: {sources}")
        print(f"{'='*60}\n")
        
        return {"answer": final_answer, "sources": sources}
        
    except Exception as e:
        print(f" Agent Error: {e}")
        return {"answer": f"Error: {str(e)}", "sources": []}