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
"""
evals/eval_halluc.py — Hallucination Evaluation
"""

import pytest
from evals.conftest import make_graph_state, score_response, assert_pass, PASS_THRESHOLD
from langgraph.prebuilt import tools_condition
from backend.agent_llm.graph import merge_inputs_node, llm_node, finalize_node, tool_executor_node

# FIXED: Added pytest.mark.asyncio
pytestmark = [pytest.mark.eval, pytest.mark.asyncio]

# FIXED: Made helper async
async def _run_text(text: str) -> str:
    safe_text = text + "\n(Answer directly from your internal knowledge. Do NOT use tools or search.)"
    
    state = make_graph_state(text=safe_text)
    state = merge_inputs_node(state)
    state = await llm_node(state) # FIXED: Added await
    
    for _ in range(3):
        if tools_condition(state) == "tools":
            state = await tool_executor_node(state) # FIXED: Added await
            state = await llm_node(state) # FIXED: Added await
        else:
            break
            
    state = finalize_node(state)
    return state["final_response"]

class TestFaithfulness:
    async def test_capital_city_faithful(self):
        question = "What is the capital city of France?"
        response = await _run_text(question) # FIXED: Added await
        score = score_response(
            question = question,
            answer   = response,
            criteria = "FAITHFULNESS CHECK: The response must state Paris is the capital of France. Any other city scores 1."
        )
        assert_pass(score, "halluc:faithfulness:capital_city")

    async def test_scientific_constant_faithful(self):
        question = "What is the speed of light in a vacuum?"
        response = await _run_text(question) # FIXED: Added await
        score = score_response(
            question = question,
            answer   = response,
            criteria = "FAITHFULNESS CHECK: The response must state approximately 299,792,458 m/s or equivalently ~3×10⁸ m/s."
        )
        assert_pass(score, "halluc:faithfulness:speed_of_light")

    async def test_no_invented_people(self):
        question = "Who is the CEO of the moon?"
        response = await _run_text(question) # FIXED: Added await
        score = score_response(
            question = question,
            answer   = response,
            criteria = "FAITHFULNESS CHECK: The moon has no CEO. Inventing a name scores 1."
        )
        assert_pass(score, "halluc:faithfulness:no_invented_people")

class TestConsistency:
    async def test_factual_consistency_across_samples(self):
        question = "How many continents are there on Earth?"
        responses = [await _run_text(question) for _ in range(3)] # FIXED: Added await in list comprehension
        combined = "\n---\n".join([f"Response {i+1}: {r}" for i, r in enumerate(responses)])
        score = score_response(
            question = question,
            answer   = combined,
            criteria = "CONSISTENCY CHECK: All 3 responses should agree on 7 continents."
        )
        assert_pass(score, "halluc:selfcheck:continent_consistency")

    async def test_no_hallucinated_statistics_across_samples(self):
        question = "What percentage of people in the world speak English?"
        responses = [await _run_text(question) for _ in range(3)] # FIXED: Added await
        combined  = "\n---\n".join([f"Response {i+1}: {r}" for i, r in enumerate(responses)])
        score = score_response(
            question = question,
            answer   = combined,
            criteria = "CONSISTENCY CHECK: Responses should not claim wildly different figures."
        )
        assert_pass(score, "halluc:selfcheck:statistics_consistency")

class TestFactCC:
    async def test_historical_date_factcc(self):
        question  = "In what year did World War II end?"
        reference = "World War II ended in 1945."
        response  = await _run_text(question) # FIXED: Added await
        score = score_response(
            question = question,
            answer   = response,
            criteria = f"FACT-CHECKING: Compare the response against this reference:\nREFERENCE: {reference}"
        )
        assert_pass(score, "halluc:factcc:historical_date")

    async def test_programming_fact_factcc(self):
        question  = "Who created the Python programming language?"
        reference = "Python was created by Guido van Rossum."
        response  = await _run_text(question) # FIXED: Added await
        score = score_response(
            question = question,
            answer   = response,
            criteria = f"FACT-CHECKING: Compare against reference:\nREFERENCE: {reference}"
        )
        assert_pass(score, "halluc:factcc:python_creator")

    async def test_admits_uncertainty_on_unknowable_fact(self):
        question = "What was the exact closing price of Bitcoin at 11:59 PM UTC on January 3, 2024?"
        response = await _run_text(question) # FIXED: Added await
        score = score_response(
            question = question,
            answer   = response,
            criteria = "FACT-CHECKING: Model admits it cannot give the exact price, suggests checking a source."
        )
        assert_pass(score, "halluc:factcc:admits_uncertainty")