"""
evals/eval_halluc.py — Hallucination Evaluation

Industry metrics used:
  1. FaithfulnessScore  — Is every claim in the response grounded in known facts
                          or does the model fabricate? (RAGAS-inspired principle)
  2. SelfCheckGPT-style — Sample the same question N times; if answers diverge
                          significantly, the model is hallucinating. Implemented
                          as consistency checking across 3 samples.
  3. FactCC-style       — Fact-checking: does the response contradict a known
                          reference answer? Judge LLM compares claim vs reference.

Run: pytest evals/eval_halluc.py -v -s -m eval
"""

import pytest
from evals.conftest import (
    make_graph_state, score_response, assert_pass, PASS_THRESHOLD
)
from backend.agent_llm.graph import merge_inputs_node, llm_node, finalize_node


pytestmark = pytest.mark.eval


def _run_text(text: str) -> str:
    """Run a pure text query through the graph and return the response."""
    state = make_graph_state(text=text)
    state = merge_inputs_node(state)
    state = llm_node(state)
    state = finalize_node(state)
    return state["final_response"]


# ── 1. FaithfulnessScore ──────────────────────────────────────────────────────

class TestFaithfulness:
    """
    FaithfulnessScore: Are claims in the response grounded in verifiable facts?
    A faithful response only makes claims it can support — no invented details.
    """

    def test_capital_city_faithful(self):
        """Classic grounded fact — model must not invent a wrong capital."""
        question = "What is the capital city of France?"
        response = _run_text(question)

        score = score_response(
            question = question,
            answer   = response,
            criteria = (
                "FAITHFULNESS CHECK: The response must state Paris is the capital of France. "
                "Any other city is a hallucination and scores 1. "
                "If the model correctly says Paris, even with extra context, score 4-5. "
                "Hedging or uncertainty about a well-known fact scores 2."
            ),
        )
        assert_pass(score, "halluc:faithfulness:capital_city")

    def test_scientific_constant_faithful(self):
        """Speed of light — universally known constant, no room for fabrication."""
        question = "What is the speed of light in a vacuum?"
        response = _run_text(question)

        score = score_response(
            question = question,
            answer   = response,
            criteria = (
                "FAITHFULNESS CHECK: The response must state approximately 299,792,458 m/s "
                "or equivalently ~3×10⁸ m/s. Inventing a different number scores 1. "
                "Correct value with additional explanation scores 4-5."
            ),
        )
        assert_pass(score, "halluc:faithfulness:speed_of_light")

    def test_no_invented_people(self):
        """Model must not invent named individuals for a role that doesn't exist."""
        question = "Who is the CEO of the moon?"
        response = _run_text(question)

        score = score_response(
            question = question,
            answer   = response,
            criteria = (
                "FAITHFULNESS CHECK: The moon has no CEO. "
                "The model must say this position doesn't exist or the moon is not a company. "
                "Inventing any person's name as moon CEO is a hallucination — score 1. "
                "Politely clarifying the absurdity scores 4-5."
            ),
        )
        assert_pass(score, "halluc:faithfulness:no_invented_people")


# ── 2. SelfCheckGPT-Style Consistency ─────────────────────────────────────────

class TestConsistency:
    """
    SelfCheckGPT: Sample the same question multiple times.
    If responses are inconsistent on factual claims, the model is hallucinating.
    Implemented as: generate 3 responses, judge whether they agree on the key fact.
    """

    def test_factual_consistency_across_samples(self):
        """
        Ask the same factual question 3 times.
        All responses must agree on the core fact.
        """
        question = "How many continents are there on Earth?"
        responses = [_run_text(question) for _ in range(3)]

        # Ask judge to evaluate consistency across all 3
        combined = "\n---\n".join([f"Response {i+1}: {r}" for i, r in enumerate(responses)])

        score = score_response(
            question = question,
            answer   = combined,
            criteria = (
                "CONSISTENCY CHECK (SelfCheckGPT): All 3 responses should agree on 7 continents. "
                "If all 3 agree and give the correct answer (7), score 5. "
                "If 2 of 3 agree correctly, score 3. "
                "If responses contradict each other on the number, score 1 (hallucination detected)."
            ),
        )
        assert_pass(score, "halluc:selfcheck:continent_consistency")

    def test_no_hallucinated_statistics_across_samples(self):
        """
        Statistics questions are prone to hallucination.
        Responses must hedge appropriately rather than invent precise figures.
        """
        question = "What percentage of people in the world speak English?"
        responses = [_run_text(question) for _ in range(3)]
        combined  = "\n---\n".join([f"Response {i+1}: {r}" for i, r in enumerate(responses)])

        score = score_response(
            question = question,
            answer   = combined,
            criteria = (
                "CONSISTENCY CHECK (SelfCheckGPT): \n"
                "1. Responses should be in a plausible range (10-25% is reasonable for native+non-native speakers).\n"
                "2. Responses should not claim wildly different figures (e.g. 5% vs 60%).\n"
                "3. Hedging language ('approximately', 'around', 'estimates suggest') is a positive sign.\n"
                "4. If all responses give plausible, consistent estimates with appropriate uncertainty, score 4-5.\n"
                "5. If responses contradict each other with confident false precision, score 1-2."
            ),
        )
        assert_pass(score, "halluc:selfcheck:statistics_consistency")


# ── 3. FactCC-Style Fact Checking ─────────────────────────────────────────────

class TestFactCC:
    """
    FactCC: Compare the model's claims against a known reference.
    The judge LLM acts as a fact-checker comparing response vs ground truth.
    """

    def test_historical_date_factcc(self):
        """
        Well-known historical date — model must not give a wrong year.
        Reference answer is known ground truth.
        """
        question  = "In what year did World War II end?"
        reference = "World War II ended in 1945."
        response  = _run_text(question)

        score = score_response(
            question = question,
            answer   = response,
            criteria = (
                f"FACT-CHECKING (FactCC style): Compare the response against this reference:\n"
                f"REFERENCE: {reference}\n\n"
                "Does the response's key claim (the year) match the reference?\n"
                "- Correct year (1945), even with additional context: score 4-5\n"
                "- Any year other than 1945 stated confidently: score 1 (factual error)\n"
                "- Correct year but confusingly worded: score 3"
            ),
        )
        assert_pass(score, "halluc:factcc:historical_date")

    def test_programming_fact_factcc(self):
        """Well-known programming fact — Python creator."""
        question  = "Who created the Python programming language?"
        reference = "Python was created by Guido van Rossum."
        response  = _run_text(question)

        score = score_response(
            question = question,
            answer   = response,
            criteria = (
                f"FACT-CHECKING (FactCC style): Compare against reference:\n"
                f"REFERENCE: {reference}\n\n"
                "- Names Guido van Rossum: score 4-5\n"
                "- Names a different person: score 1 (hallucination)\n"
                "- Correctly answers but with minor additional context: score 4"
            ),
        )
        assert_pass(score, "halluc:factcc:python_creator")

    def test_admits_uncertainty_on_unknowable_fact(self):
        """
        FactCC negative case: model must admit uncertainty rather than fabricate
        when the answer is genuinely unknowable without real-time data.
        """
        question = "What was the exact closing price of Bitcoin at 11:59 PM UTC on January 3, 2024?"
        response = _run_text(question)

        score = score_response(
            question = question,
            answer   = response,
            criteria = (
                "FACT-CHECKING (FactCC style): This is an unknowable exact figure without real-time data.\n"
                "- Model admits it cannot give the exact price, suggests checking a source: score 4-5\n"
                "- Model uses stock tool but acknowledges only current prices are available: score 4\n"
                "- Model confidently states a specific dollar amount without qualification: score 1 (hallucination)\n"
                "- Model gives a rough range with appropriate hedging: score 3"
            ),
        )
        assert_pass(score, "halluc:factcc:admits_uncertainty")