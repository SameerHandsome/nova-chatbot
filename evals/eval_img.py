"""
evals/eval_img.py — Image Modality Quality Evaluation

Industry metrics used:
  1. CLIPScore — Measures alignment between image and text description
                 using CLIP embeddings. Higher = better image-text match.
                 We approximate this by scoring how well the response
                 references the image content described.

  2. Entity Recall — Did the response mention the key entities present
                     in the image? Precision metric from VQA evaluation.
                     We compute: entities_mentioned / entities_expected

  3. CIDEr-style (Consensus-based Image Description Evaluation) —
                 Measures how well the response matches a reference description
                 using TF-IDF weighted n-grams. Originally for image captioning.
                 We use ROUGE-L as a practical approximation since we don't have
                 multiple reference captions.

  4. LLM-as-judge — For OCR and compositional image+text queries where
                    automated metrics don't capture intent accuracy.

Note: We mock image_description (pre-filled) since evals run without
real image uploads. This tests the pipeline's use of image context,
not the vision LLM itself (which is tested separately in production).

Run: pytest evals/eval_img.py -v -s -m eval
"""
"""
evals/eval_img.py — Image Modality Quality Evaluation
"""

import pytest
import evaluate
from evals.conftest import make_graph_state, score_response, assert_pass, PASS_THRESHOLD
from langgraph.prebuilt import tools_condition
from backend.agent_llm.graph import merge_inputs_node, llm_node, finalize_node, tool_executor_node

# FIXED: Added pytest.mark.asyncio
pytestmark = [pytest.mark.eval, pytest.mark.asyncio]
_rouge = None

def get_rouge():
    global _rouge
    if _rouge is None:
        _rouge = evaluate.load("rouge")
    return _rouge

# FIXED: Made helper async
async def _run_with_image(image_description: str, text: str = None) -> str:
    state = make_graph_state(
        text              = text,
        image_b64         = "MOCKED_BASE64",
        image_description = image_description,
    )
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

def _entity_recall(response: str, expected_entities: list) -> float:
    response_lower = response.lower()
    mentioned = sum(1 for entity in expected_entities if entity.lower() in response_lower)
    return mentioned / len(expected_entities) if expected_entities else 0.0

class TestCLIPScore:
    ENTITY_RECALL_THRESHOLD = 0.6
    async def test_clip_animal_scene(self):
        img_desc  = "A golden retriever dog catching a red ball on a sandy beach with ocean waves in the background."
        expected  = ["dog", "beach", "ball"]
        response  = await _run_with_image(img_desc) # FIXED: Added await
        recall    = _entity_recall(response, expected)
        assert recall >= self.ENTITY_RECALL_THRESHOLD

    async def test_clip_urban_scene(self):
        img_desc = "A busy city street at night with glowing neon signs, yellow taxis, and tall skyscrapers lit up."
        expected = ["street", "city", "night", "neon"]
        response = await _run_with_image(img_desc) # FIXED: Added await
        recall   = _entity_recall(response, expected)
        assert recall >= self.ENTITY_RECALL_THRESHOLD

    async def test_clip_food_scene(self):
        img_desc  = "A white plate containing grilled salmon, steamed broccoli, and lemon wedges on a wooden table."
        expected  = ["salmon", "broccoli", "lemon", "plate"]
        response  = await _run_with_image(img_desc) # FIXED: Added await
        recall    = _entity_recall(response, expected)
        assert recall >= self.ENTITY_RECALL_THRESHOLD

class TestEntityRecall:
    async def test_entity_recall_chart_image(self):
        img_desc  = "A bar chart showing quarterly sales: Q1=$120K, Q2=$145K, Q3=$98K, Q4=$210K."
        expected  = ["Q1", "Q2", "Q3", "Q4"]
        response  = await _run_with_image(img_desc, text="Summarize this chart.") # FIXED: Added await
        recall    = _entity_recall(response, expected)
        assert recall >= 0.75

    async def test_entity_recall_product_label(self):
        img_desc  = "A food product label showing: Product Name: 'NovaBars', Calories: 250, Protein: 12g, Sugar: 8g."
        expected  = ["novabars", "250", "protein", "sugar"]
        response  = await _run_with_image(img_desc, text="What information is on this label?") # FIXED: Added await
        recall    = _entity_recall(response, expected)
        assert recall >= 0.6

class TestCIDEr:
    ROUGE_L_THRESHOLD = 0.15

    async def test_cider_nature_scene_description(self):
        img_desc  = "A mountain lake surrounded by pine trees at sunset, with orange and purple reflections on the water."
        reference = (
            "The image shows a serene mountain lake surrounded by pine trees during sunset. "
            "The water reflects the warm orange and purple colors of the evening sky."
        )
        response = await _run_with_image(img_desc) # FIXED: Added await
        result  = get_rouge().compute(predictions=[response], references=[reference])
        assert result["rougeL"] >= self.ROUGE_L_THRESHOLD

    async def test_cider_image_plus_text_question(self):
        img_desc = "A red sports car parked in front of a modern glass building."
        question = "What color is the car and what type of building is behind it?"
        response = await _run_with_image(img_desc, text=question) # FIXED: Added await
        score = score_response(
            question = question, answer = response,
            criteria = "State the car is red and the building is modern/glass. Missing either scores <= 2."
        )
        assert_pass(score, "img:cider:image_plus_text")

class TestOCRAndComposition:
    async def test_ocr_text_extraction(self):
        img_desc = "A road sign with bold white text on a red background that reads 'STOP'."
        response = await _run_with_image(img_desc, text="What does the sign say?") # FIXED: Added await
        assert "stop" in response.lower() or "STOP" in response
        score = score_response(
            question = "What does the sign say?", answer = response,
            criteria = "The sign says STOP. Response must state the word STOP."
        )
        assert_pass(score, "img:ocr:stop_sign")

    async def test_compositional_image_and_voice(self):
        img_desc    = "A line graph showing Bitcoin price rising from $30K to $65K over 6 months."
        transcript  = "What trend does this graph show and is it positive or negative?"
        state = make_graph_state(
            image_b64 = "MOCKED", image_description = img_desc, transcribed_text = transcript,
        )
        state = merge_inputs_node(state)
        state = await llm_node(state) # FIXED: Added await here directly
        state = finalize_node(state)
        score = score_response(
            question = transcript, answer = state["final_response"],
            criteria = "Response must identify an upward/positive trend and reference Bitcoin."
        )
        assert_pass(score, "img:compositional:image_plus_voice")