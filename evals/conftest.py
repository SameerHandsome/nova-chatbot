"""
evals/conftest.py — Shared eval infrastructure.

Exports:
  - judge_llm()        : low-temp Groq LLM for judging
  - score_response()   : returns int 1-5 from judge LLM
  - PASS_THRESHOLD     : minimum score to pass (3/5)
  - make_graph_state() : builds a clean ChatState dict for graph invocation
  - assert_pass()      : assertion helper with printed score
"""

import pytest
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from backend.config import get_settings

settings        = get_settings()
PASS_THRESHOLD  = 3   # score out of 5 to be considered passing


# ── Judge LLM ─────────────────────────────────────────────────────────────────

def judge_llm() -> ChatGroq:
    """
    Separate low-temperature LLM used exclusively for scoring.
    Never the same model instance as the one being evaluated.
    """
    return ChatGroq(
        model       = "llama-3.3-70b-versatile",
        temperature = 0.0,
        api_key     = settings.groq_api_key,
    )


# ── Scorer ────────────────────────────────────────────────────────────────────

def score_response(question: str, answer: str, criteria: str) -> int:
    """
    Ask the judge LLM to score `answer` on a 1-5 scale given `criteria`.

    Scale:
      5 = Excellent — fully satisfies all criteria
      4 = Good      — mostly satisfies with minor gaps
      3 = Acceptable — partially satisfies (PASS threshold)
      2 = Poor      — mostly fails
      1 = Fail      — completely fails

    Returns: int (1-5)
    """
    llm    = judge_llm()
    prompt = (
        f"You are an objective AI evaluator. Score the AI response below on a scale of 1-5.\n\n"
        f"SCORING CRITERIA:\n{criteria}\n\n"
        f"SCALE:\n"
        f"  5 = Excellent (fully satisfies all criteria)\n"
        f"  4 = Good (mostly satisfies, minor gaps)\n"
        f"  3 = Acceptable (partially satisfies — minimum passing)\n"
        f"  2 = Poor (mostly fails)\n"
        f"  1 = Fail (completely fails)\n\n"
        f"QUESTION ASKED: {question}\n\n"
        f"AI RESPONSE: {answer}\n\n"
        f"Respond with ONLY a single integer: 1, 2, 3, 4, or 5. Nothing else."
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    try:
        return int(resp.content.strip())
    except ValueError:
        return 1


# ── State Builder ─────────────────────────────────────────────────────────────

def make_graph_state(
    text:             str  = None,
    image_b64:        str  = None,
    image_description:str  = None,
    transcribed_text: str  = None,
    audio_b64:        str  = None,
    chat_history:     list = None,
) -> dict:
    """Build a clean ChatState for direct graph node invocation in evals."""
    return {
        "raw_text":             text,
        "raw_image_b64":        image_b64,
        "raw_image_media_type": "image/jpeg" if image_b64 else None,
        "raw_audio_b64":        audio_b64,
        "transcribed_text":     transcribed_text,
        "image_description":    image_description,
        "merged_input":         None,
        "lc_messages":          [],
        "chat_history":         chat_history or [],
        "final_response":       None,
        "tools_called":         [],
        "tool_results":         [],
        "error":                None,
    }


# ── Assert Helper ─────────────────────────────────────────────────────────────

def assert_pass(score: int, label: str):
    status = "✓ PASS" if score >= PASS_THRESHOLD else "✗ FAIL"
    print(f"\n  [{label}] Score: {score}/5  {status}")
    assert score >= PASS_THRESHOLD, (
        f"EVAL FAILED: [{label}] scored {score}/5 "
        f"(minimum passing: {PASS_THRESHOLD}/5)"
    )


# ── Pytest Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def graph_state():
    return make_graph_state


@pytest.fixture
def scorer():
    return score_response


@pytest.fixture
def passer():
    return assert_pass