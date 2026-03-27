"""
agent_llm/graph.py — Full LangGraph multimodal pipeline.
Uses LangGraph's built-in ToolNode + tools_condition for proper tool routing.

CRITICAL: tools_condition and ToolNode both look for state["messages"] — that is
the hard-coded key inside LangGraph's prebuilt internals.  Using any other key
(e.g. "lc_messages") causes the error:
  "No messages found in input state to tool_edge"
So the message list MUST live under the key "messages" in ChatState.

Single-model strategy:
  • llama-3.3-70b-versatile supports local tool use + parallel tool calls.
  • Tools are always bound so tools_condition can inspect tool_calls on every turn.
  • After tool results are in the history the model naturally produces a plain
    text answer (no more tool_calls) → tools_condition routes to END → finalize.
"""

import os
import re
import base64
from io import BytesIO
from typing import Optional, TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage,
)
from langgraph.graph.message import add_messages
from langsmith import traceable

from backend.config import get_settings
from backend.agent_llm.tools import ALL_TOOLS

settings = get_settings()

os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
os.environ["LANGCHAIN_API_KEY"]    = settings.langchain_api_key
os.environ["LANGCHAIN_PROJECT"]    = settings.langchain_project


# ── State ─────────────────────────────────────────────────────────────────────
# "messages" MUST be the exact key name — tools_condition / ToolNode require it.
# add_messages reducer lets each node append to the list instead of replacing it.

class ChatState(TypedDict):
    raw_text:             Optional[str]
    raw_image_b64:        Optional[str]
    raw_image_media_type: Optional[str]
    raw_audio_b64:        Optional[str]
    transcribed_text:     Optional[str]
    image_description:    Optional[str]
    merged_input:         Optional[str]
    messages:             Annotated[list[BaseMessage], add_messages]  # ← MUST be "messages"
    chat_history:         list
    final_response:       Optional[str]
    tools_called:         list
    tool_results:         list
    error:                Optional[str]


# ── Models ────────────────────────────────────────────────────────────────────

# Single LLM instance — created once at import time (no per-call overhead).
# bind_tools makes it capable of emitting tool_calls that ToolNode can execute.
_LLM = ChatGroq(
    model       = "llama-3.3-70b-versatile",
    temperature = 0.7,
    api_key     = settings.groq_api_key,
).bind_tools(ALL_TOOLS)

def _vision_llm():
    return ChatGroq(
        model       = "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature = 0.3,
        api_key     = settings.groq_api_key,
    )


# ── Nodes ─────────────────────────────────────────────────────────────────────

@traceable(name="audio_transcription_node")
def audio_transcription_node(state: ChatState) -> dict:
    try:
        from groq import Groq
        client      = Groq(api_key=settings.groq_api_key)
        audio_bytes = base64.b64decode(state["raw_audio_b64"])
        transcript  = client.audio.transcriptions.create(
            file            = ("audio.webm", BytesIO(audio_bytes), "audio/webm"),
            model           = "whisper-large-v3",
            response_format = "text",
            language        = "en",
        )
        return {"transcribed_text": transcript.strip()}
    except Exception as e:
        return {"transcribed_text": "", "error": f"Audio transcription failed: {e}"}


@traceable(name="image_processing_node")
def image_processing_node(state: ChatState) -> dict:
    try:
        media_type = state.get("raw_image_media_type", "image/jpeg")
        ctx = ""
        if state.get("raw_text"):
            ctx += f" The user says: '{state['raw_text']}'."
        if state.get("transcribed_text"):
            ctx += f" They said via voice: '{state['transcribed_text']}'."
        prompt = (f"{ctx} Analyze this image thoroughly.").strip()

        response = _vision_llm().invoke([HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{state['raw_image_b64']}"}},
            {"type": "text", "text": prompt},
        ])])
        return {"image_description": response.content}
    except Exception as e:
        return {"image_description": "", "error": f"Image processing failed: {e}"}


def merge_inputs_node(state: ChatState) -> dict:
    parts = []
    if state.get("transcribed_text"):
        parts.append(f"[Voice]: {state['transcribed_text']}")
    if state.get("image_description"):
        parts.append(f"[Image]: {state['image_description']}")
    if state.get("raw_text"):
        parts.append(f"[Text]: {state['raw_text']}")

    merged = "\n\n".join(parts) if parts else "Hello"

    system = SystemMessage(content=(
        "You are NOVA, a helpful AI assistant. "
        "You understand text, images, and voice messages. "
        "You have access to tools for weather, stocks, currency, web search, and Wikipedia. "
        "Use them whenever the user asks for real-time or factual information. "
        "Provide clean, natural, conversational text responses."
    ))

    history = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in state.get("chat_history", [])
    ]

    # Build the full initial message list.
    # add_messages will accumulate further messages on top of this.
    new_messages = [system] + history + [HumanMessage(content=merged)]
    return {"merged_input": merged, "messages": new_messages}   # ← "messages"


@traceable(name="llm_node")
def llm_node(state: ChatState) -> dict:
    """
    Invoke _LLM.  tools_condition (called after this node) inspects the
    returned AIMessage:
      • tool_calls present → routes to "tools"
      • no tool_calls      → routes to END → "finalize"
    """
    response   = _LLM.invoke(state["messages"])                 # ← "messages"
    tool_calls = getattr(response, "tool_calls", []) or []
    called     = [tc["name"] for tc in tool_calls if isinstance(tc, dict)]

    return {
        "messages":     [response],                             # ← "messages"
        "tools_called": list(state.get("tools_called", [])) + called,
    }


# ── ToolNode wrapper ──────────────────────────────────────────────────────────
# LangGraph's ToolNode reads tool_calls from state["messages"][-1] and writes
# ToolMessage results back under state["messages"].
# We wrap it only to also populate our own tool_results list for the API response.

_base_tool_node = ToolNode(ALL_TOOLS)

def tool_executor_node(state: ChatState) -> dict:
    """Execute tools via ToolNode, then capture results for the API response."""
    # ToolNode returns {"messages": [ToolMessage, ...]}
    result   = _base_tool_node.invoke(state)
    new_msgs = result.get("messages", [])

    tool_results = list(state.get("tool_results", []))
    for msg in new_msgs:
        if isinstance(msg, ToolMessage):
            tool_results.append({"tool": msg.name, "result": msg.content})

    return {
        "messages":     new_msgs,   # ← "messages" — add_messages appends them
        "tool_results": tool_results,
    }


def finalize_node(state: ChatState) -> dict:
    """Pick the last non-empty AIMessage as the final response."""
    for msg in reversed(state["messages"]):                     # ← "messages"
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            # Strip any leaked XML tags (safety net)
            content = re.sub(r"<[a-z_]+>.*?</[a-z_]+>", "", content, flags=re.DOTALL)
            content = re.sub(r"<[a-z_]+\s*/>", "", content)
            content = content.strip()
            if content:
                return {"final_response": content}
    return {"final_response": "I'm sorry, I could not generate a response."}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_inputs(state: ChatState) -> str:
    if state.get("raw_audio_b64"): return "audio_transcription"
    if state.get("raw_image_b64"): return "image_processing"
    return "merge_inputs"

def after_audio(state: ChatState) -> str:
    return "image_processing" if state.get("raw_image_b64") else "merge_inputs"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ChatState)

    g.add_node("audio_transcription", audio_transcription_node)
    g.add_node("image_processing",    image_processing_node)
    g.add_node("merge_inputs",        merge_inputs_node)
    g.add_node("llm",                 llm_node)
    g.add_node("tools",               tool_executor_node)
    g.add_node("finalize",            finalize_node)

    # Entry routing
    g.set_conditional_entry_point(route_inputs, {
        "audio_transcription": "audio_transcription",
        "image_processing":    "image_processing",
        "merge_inputs":        "merge_inputs",
    })
    g.add_conditional_edges("audio_transcription", after_audio, {
        "image_processing": "image_processing",
        "merge_inputs":     "merge_inputs",
    })
    g.add_edge("image_processing", "merge_inputs")
    g.add_edge("merge_inputs",     "llm")

    # tools_condition reads state["messages"][-1].tool_calls:
    #   • tool_calls present  → "tools"
    #   • no tool_calls       → END  (remapped to "finalize" below)
    g.add_conditional_edges(
        "llm",
        tools_condition,
        {
            "tools": "tools",
            END:     "finalize",
        },
    )

    g.add_edge("tools",    "llm")       # after tools → LLM synthesises answer
    g.add_edge("finalize", END)

    return g.compile()


chatbot_graph = build_graph()