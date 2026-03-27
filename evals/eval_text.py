"""
evals/eval_text.py — Text Modality Quality Evaluation

Industry metrics used:
  1. BLEU (Bilingual Evaluation Understudy) — n-gram overlap between
     response and reference answer. Standard in NLP since Papineni 2002.
     Library: evaluate (HuggingFace)

  2. ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation — Longest
     Common Subsequence) — measures recall of key content from reference.
     Library: evaluate (HuggingFace)

  3. BERTScore — Semantic similarity using BERT embeddings. Captures meaning
     even when wording differs. F1 >= 0.85 is generally considered good.
     Library: evaluate (HuggingFace) or bert-score

  4. Coherence + Tool Use — LLM-as-judge for multi-turn coherence and
     correct tool invocation (non-automated metrics where reference is unclear).

Run: pytest evals/eval_text.py -v -s -m eval
"""

import pytest
import evaluate   # pip install evaluate
from evals.conftest import (
    make_graph_state, score_response, assert_pass, PASS_THRESHOLD
)
from backend.agent_llm.graph import merge_inputs_node, llm_node, finalize_node


pytestmark = pytest.mark.eval

# ── Metric Loaders (cached at module level) ────────────────────────────────────

_bleu   = None
_rouge  = None
_bert   = None


def get_bleu():
    global _bleu
    if _bleu is None:
        _bleu = evaluate.load("bleu")
    return _bleu


def get_rouge():
    global _rouge
    if _rouge is None:
        _rouge = evaluate.load("rouge")
    return _rouge


def get_bertscore():
    global _bert
    if _bert is None:
        _bert = evaluate.load("bertscore")
    return _bert


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_text(text: str, history: list = None) -> str:
    state = make_graph_state(text=text, chat_history=history or [])
    state = merge_inputs_node(state)
    state = llm_node(state)
    state = finalize_node(state)
    return state["final_response"]


def _run_full(text: str) -> dict:
    """Returns full state (includes tools_called)."""
    state = make_graph_state(text=text)
    state = merge_inputs_node(state)
    state = llm_node(state)
    # Run tool executor if tools called
    from backend.agent_llm.graph import tool_executor_node, should_use_tools
    if should_use_tools(state) == "tool_executor":
        state = tool_executor_node(state)
        state = llm_node(state)
    state = finalize_node(state)
    return state


# ── 1. BLEU Score ─────────────────────────────────────────────────────────────

class TestBLEU:
    """
    BLEU measures n-gram precision overlap between prediction and reference.
    BLEU >= 0.3 is acceptable for open-ended generation.
    BLEU >= 0.5 is good.
    We use sentence-level BLEU (smoothed).
    """

    BLEU_THRESHOLD = 0.25   # relaxed for open-ended generation

    def test_bleu_factual_definition(self):
        """
        Response to a factual definition question should have enough
        n-gram overlap with a reference answer.
        """
        question  = "What is machine learning in one sentence?"
        reference = (
            "Machine learning is a branch of artificial intelligence that enables "
            "computers to learn from data and improve their performance without being "
            "explicitly programmed."
        )
        response = _run_text(question)

        result = get_bleu().compute(
            predictions = [response],
            references  = [[reference]],
        )
        bleu_score = result["bleu"]
        print(f"\n  [text:bleu:factual_definition] BLEU = {bleu_score:.4f} (threshold: {self.BLEU_THRESHOLD})")
        assert bleu_score >= self.BLEU_THRESHOLD, (
            f"BLEU {bleu_score:.4f} below threshold {self.BLEU_THRESHOLD}"
        )

    def test_bleu_short_answer(self):
        """Short factual answers should achieve higher BLEU against reference."""
        question  = "What does HTTP stand for?"
        reference = "HTTP stands for HyperText Transfer Protocol."
        response  = _run_text(question)

        result     = get_bleu().compute(predictions=[response], references=[[reference]])
        bleu_score = result["bleu"]
        print(f"\n  [text:bleu:short_answer] BLEU = {bleu_score:.4f}")
        assert bleu_score >= 0.15, f"BLEU {bleu_score:.4f} too low for simple factual answer"


# ── 2. ROUGE-L Score ──────────────────────────────────────────────────────────

class TestROUGEL:
    """
    ROUGE-L measures the longest common subsequence between prediction and reference.
    Good for evaluating whether key content is recalled.
    ROUGE-L >= 0.4 is acceptable, >= 0.6 is good.
    """

    ROUGE_L_THRESHOLD = 0.35

    def test_rouge_l_explanation(self):
        """
        Explanation of a concept — key terms should appear in response.
        """
        question  = "Explain what an API is."
        reference = (
            "An API (Application Programming Interface) is a set of rules and protocols "
            "that allows different software applications to communicate with each other. "
            "It defines the methods and data formats that programs can use to request "
            "and exchange information."
        )
        response = _run_text(question)

        result       = get_rouge().compute(predictions=[response], references=[reference])
        rouge_l      = result["rougeL"]
        print(f"\n  [text:rouge_l:explanation] ROUGE-L = {rouge_l:.4f} (threshold: {self.ROUGE_L_THRESHOLD})")
        assert rouge_l >= self.ROUGE_L_THRESHOLD, (
            f"ROUGE-L {rouge_l:.4f} below threshold {self.ROUGE_L_THRESHOLD}"
        )

    def test_rouge_l_summarization(self):
        """
        Summarizing a known topic — ROUGE-L checks recall of key information.
        """
        question  = "Briefly describe what Python is as a programming language."
        reference = (
            "Python is a high-level, interpreted programming language known for its "
            "clear syntax and readability. It supports multiple programming paradigms "
            "including procedural, object-oriented, and functional programming."
        )
        response = _run_text(question)

        result  = get_rouge().compute(predictions=[response], references=[reference])
        rouge_l = result["rougeL"]
        print(f"\n  [text:rouge_l:summarization] ROUGE-L = {rouge_l:.4f}")
        assert rouge_l >= self.ROUGE_L_THRESHOLD, (
            f"ROUGE-L {rouge_l:.4f} below threshold"
        )


# ── 3. BERTScore ──────────────────────────────────────────────────────────────

class TestBERTScore:
    """
    BERTScore computes similarity using contextual BERT embeddings.
    Captures semantic similarity even when wording differs from reference.
    F1 >= 0.85 is generally considered good quality for English text.
    """

    BERT_F1_THRESHOLD = 0.82   # slightly relaxed for open-ended generation

    def test_bertscore_semantic_similarity(self):
        """
        The response should be semantically close to the reference,
        even if using different wording.
        """
        question  = "What is the difference between supervised and unsupervised learning?"
        reference = (
            "Supervised learning uses labeled training data where the model learns "
            "to map inputs to known outputs. Unsupervised learning finds patterns in "
            "unlabeled data without predefined correct answers."
        )
        response = _run_text(question)

        result = get_bertscore().compute(
            predictions = [response],
            references  = [reference],
            lang        = "en",
        )
        f1 = result["f1"][0]
        print(f"\n  [text:bertscore:semantic_similarity] BERTScore F1 = {f1:.4f} (threshold: {self.BERT_F1_THRESHOLD})")
        assert f1 >= self.BERT_F1_THRESHOLD, (
            f"BERTScore F1 {f1:.4f} below threshold {self.BERT_F1_THRESHOLD}"
        )


# ── 4. Tool Use + Coherence (LLM-as-Judge) ────────────────────────────────────

class TestToolUseAndCoherence:
    """
    Tool invocation and multi-turn coherence cannot be measured with n-gram metrics.
    These use LLM-as-judge scoring.
    """

    def test_tool_called_for_live_data_query(self):
        """Stock price query must invoke the stock tool."""
        result = _run_full("What is the current stock price of Microsoft (MSFT)?")

        tools = result.get("tools_called", [])
        assert "stock_tool" in tools, (
            f"Expected stock_tool to be called. Tools called: {tools}"
        )

        score = score_response(
            question = "What is the current stock price of Microsoft?",
            answer   = result["final_response"],
            criteria = (
                "The response must include a dollar price for MSFT and indicate it is "
                "real-time or recent data. A response that says 'I don't have real-time data' "
                "without attempting to use the tool scores 2."
            ),
        )
        assert_pass(score, "text:tool_use:stock_query")

    def test_multi_turn_coherence(self):
        """
        The model must recall prior context in a multi-turn conversation.
        Tests conversation coherence — a core chatbot quality.
        """
        history = [
            {"role": "user",      "content": "My name is Usman and I work in fintech."},
            {"role": "assistant", "content": "Nice to meet you, Usman! Fintech is a great field."},
        ]
        response = _run_text(
            "Based on what I told you, what industry do I work in?",
            history=history,
        )

        score = score_response(
            question = "What industry do I work in? (prior context: Usman, fintech)",
            answer   = response,
            criteria = (
                "The response must correctly recall 'fintech' as the industry from prior context. "
                "Not recalling the industry scores 1. Correctly stating fintech scores 4-5."
            ),
        )
        assert_pass(score, "text:coherence:multi_turn")