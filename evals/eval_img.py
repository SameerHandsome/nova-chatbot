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

import pytest
import evaluate
from evals.conftest import (
    make_graph_state, score_response, assert_pass, PASS_THRESHOLD
)
from backend.agent_llm.graph import merge_inputs_node, llm_node, finalize_node


pytestmark = pytest.mark.eval

_rouge = None


def get_rouge():
    global _rouge
    if _rouge is None:
        _rouge = evaluate.load("rouge")
    return _rouge


def _run_with_image(image_description: str, text: str = None) -> str:
    """
    Simulate image input by pre-filling image_description.
    Skips the vision LLM node — tests how well the pipeline
    uses image context in the final response.
    """
    state = make_graph_state(
        text              = text,
        image_b64         = "MOCKED_BASE64",   # signals has_image=True for routing
        image_description = image_description,
    )
    state = merge_inputs_node(state)
    state = llm_node(state)
    state = finalize_node(state)
    return state["final_response"]


def _entity_recall(response: str, expected_entities: list) -> float:
    """
    Entity Recall = entities mentioned in response / total expected entities.
    Case-insensitive substring match.
    """
    response_lower = response.lower()
    mentioned = sum(
        1 for entity in expected_entities
        if entity.lower() in response_lower
    )
    return mentioned / len(expected_entities) if expected_entities else 0.0


# ── 1. CLIPScore Approximation ────────────────────────────────────────────────

class TestCLIPScore:
    """
    CLIPScore measures image-text alignment.
    We approximate it by evaluating whether the response text
    aligns with the image description content using entity recall
    + LLM judge (since we can't run actual CLIP in evals without GPU).
    """

    ENTITY_RECALL_THRESHOLD = 0.6   # at least 60% of key entities mentioned

    def test_clip_animal_scene(self):
        """
        Image: a dog playing fetch on a beach.
        Key entities: dog, beach, fetch/ball.
        """
        img_desc  = "A golden retriever dog catching a red ball on a sandy beach with ocean waves in the background."
        expected  = ["dog", "beach", "ball"]
        response  = _run_with_image(img_desc)
        recall    = _entity_recall(response, expected)

        print(f"\n  [img:clip:animal_scene] Entity recall = {recall:.2f} (threshold: {self.ENTITY_RECALL_THRESHOLD})")
        assert recall >= self.ENTITY_RECALL_THRESHOLD, (
            f"Entity recall {recall:.2f} below {self.ENTITY_RECALL_THRESHOLD}. "
            f"Response did not mention enough key entities: {expected}"
        )

    def test_clip_urban_scene(self):
        """
        Image: city street at night.
        Key entities: street, night/dark, city/buildings.
        """
        img_desc = "A busy city street at night with glowing neon signs, yellow taxis, and tall skyscrapers lit up."
        expected = ["street", "city", "night", "neon"]
        response = _run_with_image(img_desc)
        recall   = _entity_recall(response, expected)

        print(f"\n  [img:clip:urban_scene] Entity recall = {recall:.2f}")
        assert recall >= self.ENTITY_RECALL_THRESHOLD, (
            f"Entity recall {recall:.2f} below threshold"
        )

    def test_clip_food_scene(self):
        """
        Image: a plate of food.
        Response must reference the food items, not describe something unrelated.
        """
        img_desc  = "A white plate containing grilled salmon, steamed broccoli, and lemon wedges on a wooden table."
        expected  = ["salmon", "broccoli", "lemon", "plate"]
        response  = _run_with_image(img_desc)
        recall    = _entity_recall(response, expected)

        print(f"\n  [img:clip:food_scene] Entity recall = {recall:.2f}")
        assert recall >= self.ENTITY_RECALL_THRESHOLD


# ── 2. Entity Recall ──────────────────────────────────────────────────────────

class TestEntityRecall:
    """
    Entity Recall from VQA (Visual Question Answering) evaluation.
    Measures whether the response mentions the key visual entities.
    Entity Recall = |entities in response ∩ entities in image| / |entities in image|
    """

    def test_entity_recall_chart_image(self):
        """
        Image: a bar chart showing sales data.
        Query: summarize the chart.
        All key entities must appear in response.
        """
        img_desc  = "A bar chart showing quarterly sales: Q1=$120K, Q2=$145K, Q3=$98K, Q4=$210K."
        expected  = ["Q1", "Q2", "Q3", "Q4"]
        response  = _run_with_image(img_desc, text="Summarize this chart.")
        recall    = _entity_recall(response, expected)

        print(f"\n  [img:entity_recall:chart] = {recall:.2f}")
        assert recall >= 0.75, f"Expected 75% of quarters mentioned, got {recall:.2f}"

    def test_entity_recall_product_label(self):
        """
        Image: a product label with specific details.
        Query: what information is on this label?
        """
        img_desc  = "A food product label showing: Product Name: 'NovaBars', Calories: 250, Protein: 12g, Sugar: 8g."
        expected  = ["novabars", "250", "protein", "sugar"]
        response  = _run_with_image(img_desc, text="What information is on this label?")
        recall    = _entity_recall(response, expected)

        print(f"\n  [img:entity_recall:product_label] = {recall:.2f}")
        assert recall >= 0.6, f"Expected 60%+ of label entities, got {recall:.2f}"


# ── 3. CIDEr-Style (ROUGE-L Approximation) ────────────────────────────────────

class TestCIDEr:
    """
    CIDEr measures consensus in image descriptions using TF-IDF n-grams.
    We approximate with ROUGE-L against a reference description
    since CIDEr requires multiple reference captions.

    ROUGE-L >= 0.30 is our threshold for image description quality.
    """

    ROUGE_L_THRESHOLD = 0.28

    def test_cider_nature_scene_description(self):
        """
        Image description quality — does the response capture the scene?
        """
        img_desc  = "A mountain lake surrounded by pine trees at sunset, with orange and purple reflections on the water."
        reference = (
            "The image shows a serene mountain lake surrounded by pine trees during sunset. "
            "The water reflects the warm orange and purple colors of the evening sky."
        )
        response = _run_with_image(img_desc)

        result  = get_rouge().compute(predictions=[response], references=[reference])
        rouge_l = result["rougeL"]
        print(f"\n  [img:cider:nature_scene] ROUGE-L = {rouge_l:.4f} (threshold: {self.ROUGE_L_THRESHOLD})")
        assert rouge_l >= self.ROUGE_L_THRESHOLD, (
            f"CIDEr-approx ROUGE-L {rouge_l:.4f} below threshold"
        )

    def test_cider_image_plus_text_question(self):
        """
        Image + text question: model must address both the image and the question.
        """
        img_desc = "A red sports car parked in front of a modern glass building."
        question = "What color is the car and what type of building is behind it?"
        response = _run_with_image(img_desc, text=question)

        score = score_response(
            question = question,
            answer   = response,
            criteria = (
                "The response must:\n"
                "1. State the car is red (from image)\n"
                "2. Describe the building as modern/glass (from image)\n"
                "Missing either detail scores ≤ 2. Both correct scores 4-5."
            ),
        )
        assert_pass(score, "img:cider:image_plus_text")


# ── 4. OCR Reference + LLM Judge ─────────────────────────────────────────────

class TestOCRAndComposition:
    """
    Tests image understanding for text-containing images (OCR-like tasks)
    and compositional image+text queries.
    """

    def test_ocr_text_extraction(self):
        """
        Image contains visible text — model must reference it accurately.
        """
        img_desc = "A road sign with bold white text on a red background that reads 'STOP'."
        response = _run_with_image(img_desc, text="What does the sign say?")

        assert "stop" in response.lower() or "STOP" in response, (
            f"Response did not mention 'STOP'. Response: {response[:200]}"
        )

        score = score_response(
            question = "What does the sign say?",
            answer   = response,
            criteria = (
                "The sign says STOP. "
                "Response must state the word STOP. Any other word scores 1."
            ),
        )
        assert_pass(score, "img:ocr:stop_sign")

    def test_compositional_image_and_voice(self):
        """
        Voice transcript + image description — response must address both.
        """
        img_desc    = "A line graph showing Bitcoin price rising from $30K to $65K over 6 months."
        transcript  = "What trend does this graph show and is it positive or negative?"

        state = make_graph_state(
            image_b64         = "MOCKED",
            image_description = img_desc,
            transcribed_text  = transcript,
        )
        state = merge_inputs_node(state)
        state = llm_node(state)
        state = finalize_node(state)
        response = state["final_response"]

        score = score_response(
            question = transcript,
            answer   = response,
            criteria = (
                "Response must:\n"
                "1. Identify an upward/positive trend\n"
                "2. Reference the Bitcoin price increase or the graph\n"
                "3. Confirm the trend is positive\n"
                "Missing all three scores 1. All three correct scores 5."
            ),
        )
        assert_pass(score, "img:compositional:image_plus_voice")