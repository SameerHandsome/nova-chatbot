"""
evals/eval_audio.py — Audio Modality Quality Evaluation

Industry metrics used:
  1. WER (Word Error Rate) — Standard ASR (Automatic Speech Recognition) metric.
     WER = (Substitutions + Deletions + Insertions) / Total Words in Reference
     WER = 0.0 is perfect. WER <= 0.10 (10%) is considered production-quality.
     Library: jiwer

  2. CER (Character Error Rate) — Same as WER but at character level.
     Better for short utterances and languages with compound words.
     CER <= 0.08 is good. Library: jiwer

  3. Intent Accuracy — Did the system correctly identify the user's intent
     from the spoken input and take the right action (call the right tool)?
     Binary metric: correct tool called = 1, wrong/no tool = 0.
     Extended with LLM judge for nuance scoring.

  4. Robustness to Disfluency — Does the pipeline handle filler words
     ('um', 'uh', 'like', 'you know') without losing intent?
     Measured as intent accuracy on disfluent transcripts.

Note: WER/CER are computed on the transcript itself (comparing Whisper output
to ground truth). In eval mode we inject mock transcripts to test the pipeline's
handling of transcription quality, not Whisper itself.

Run: pytest evals/eval_audio.py -v -s -m eval
"""

import pytest
from jiwer import wer, cer   # pip install jiwer
from evals.conftest import (
    make_graph_state, score_response, assert_pass, PASS_THRESHOLD
)
from backend.agent_llm.graph import (
    merge_inputs_node, llm_node, finalize_node,
    should_use_tools, tool_executor_node,
)


pytestmark = pytest.mark.eval


def _run_with_transcript(transcript: str, image_description: str = None) -> dict:
    """Simulate audio input with a pre-built transcript. Returns full state."""
    state = make_graph_state(
        audio_b64         = "MOCKED_AUDIO",
        transcribed_text  = transcript,
        image_description = image_description,
        image_b64         = "MOCKED" if image_description else None,
    )
    state = merge_inputs_node(state)
    state = llm_node(state)

    # Run ReAct loop if tools called
    for _ in range(3):   # max 3 tool call rounds
        if should_use_tools(state) == "tool_executor":
            state = tool_executor_node(state)
            state = llm_node(state)
        else:
            break

    state = finalize_node(state)
    return state


# ── 1. WER (Word Error Rate) ──────────────────────────────────────────────────

class TestWER:
    """
    WER measures transcription accuracy:
      WER = (S + D + I) / N
      S = substitutions, D = deletions, I = insertions, N = reference word count
    WER <= 0.10 is production quality for clean speech.
    WER <= 0.20 is acceptable for noisy/accented speech.
    """

    WER_THRESHOLD = 0.10   # 10% max word error rate for clean speech

    def test_wer_clean_speech_transcript(self):
        """
        Simulates a clean speech transcript vs ground truth.
        Whisper on clean English audio typically achieves WER < 5%.
        """
        reference   = "What is the current price of Apple stock?"
        hypothesis  = "What is the current price of Apple stock?"   # perfect transcript

        error_rate = wer(reference, hypothesis)
        print(f"\n  [audio:wer:clean_speech] WER = {error_rate:.4f} (threshold: {self.WER_THRESHOLD})")
        assert error_rate <= self.WER_THRESHOLD, (
            f"WER {error_rate:.4f} exceeds threshold {self.WER_THRESHOLD}"
        )

    def test_wer_minor_transcription_errors(self):
        """
        Minor errors (one word wrong) should still be within acceptable WER.
        """
        reference  = "Convert one hundred dollars to Pakistani rupees"
        hypothesis = "Convert one hundred dollar to Pakistani rupees"  # 'dollar' vs 'dollars'

        error_rate = wer(reference, hypothesis)
        print(f"\n  [audio:wer:minor_errors] WER = {error_rate:.4f}")
        assert error_rate <= 0.20, (
            f"WER {error_rate:.4f} too high even for minor errors"
        )

    def test_wer_complete_mismatch_detected(self):
        """
        Completely wrong transcript should have high WER — validates metric works.
        """
        reference  = "What is the weather in London today?"
        hypothesis = "Play some music on Spotify please"

        error_rate = wer(reference, hypothesis)
        print(f"\n  [audio:wer:mismatch_detection] WER = {error_rate:.4f}")
        # WER should be HIGH for completely wrong transcription
        assert error_rate > 0.5, (
            "WER should be high for completely wrong transcript — metric may not be working"
        )


# ── 2. CER (Character Error Rate) ────────────────────────────────────────────

class TestCER:
    """
    CER is more granular than WER — useful for short utterances.
    CER <= 0.08 is good quality.
    """

    CER_THRESHOLD = 0.08

    def test_cer_clean_transcript(self):
        """Perfect transcript should have CER = 0.0."""
        reference  = "Show me the stock price for Tesla"
        hypothesis = "Show me the stock price for Tesla"

        error_rate = cer(reference, hypothesis)
        print(f"\n  [audio:cer:clean] CER = {error_rate:.4f} (threshold: {self.CER_THRESHOLD})")
        assert error_rate <= self.CER_THRESHOLD

    def test_cer_near_perfect_transcript(self):
        """One character off — still within CER threshold."""
        reference  = "Convert 500 dollars to euros"
        hypothesis = "Convert 500 dollar to euros"   # missing 's'

        error_rate = cer(reference, hypothesis)
        print(f"\n  [audio:cer:near_perfect] CER = {error_rate:.4f}")
        assert error_rate <= 0.10, (
            f"CER {error_rate:.4f} too high for near-perfect transcript"
        )


# ── 3. Intent Accuracy ────────────────────────────────────────────────────────

class TestIntentAccuracy:
    """
    Intent Accuracy = correct tool invoked / total queries
    Binary: 1 if correct tool called, 0 if not.
    Extended with LLM judge for response quality given the intent.
    """

    def test_weather_intent_calls_weather_tool(self):
        """Voice query about weather must trigger weather_tool."""
        transcript = "What is the weather like in Karachi right now?"
        result     = _run_with_transcript(transcript)

        assert "weather_tool" in result["tools_called"], (
            f"Intent: weather. Expected weather_tool. Got: {result['tools_called']}"
        )
        print(f"\n  [audio:intent:weather] ✓ weather_tool called")

        score = score_response(
            question = transcript,
            answer   = result["final_response"],
            criteria = (
                "Response must provide weather information for Karachi. "
                "A refusal to provide weather data scores 1. "
                "Weather data provided with city name scores 4-5."
            ),
        )
        assert_pass(score, "audio:intent:weather_quality")

    def test_currency_intent_calls_currency_tool(self):
        """Voice query about currency conversion must trigger currency_tool."""
        transcript = "How much is 200 US dollars in Pakistani rupees?"
        result     = _run_with_transcript(transcript)

        assert "currency_tool" in result["tools_called"], (
            f"Intent: currency. Expected currency_tool. Got: {result['tools_called']}"
        )
        print(f"\n  [audio:intent:currency] ✓ currency_tool called")

    def test_stock_intent_calls_stock_tool(self):
        """Voice query about stock must trigger stock_tool."""
        transcript = "What is the stock price of Google?"
        result     = _run_with_transcript(transcript)

        assert "stock_tool" in result["tools_called"], (
            f"Intent: stock. Expected stock_tool. Got: {result['tools_called']}"
        )
        print(f"\n  [audio:intent:stock] ✓ stock_tool called")


# ── 4. Robustness to Disfluency ───────────────────────────────────────────────

class TestDisfluencyRobustness:
    """
    Real speech contains filler words: 'um', 'uh', 'like', 'you know'.
    The pipeline must extract intent correctly despite disfluencies.
    Disfluency robustness = intent accuracy on disfluent transcripts.
    """

    def test_weather_intent_with_filler_words(self):
        """Filler words must not confuse weather intent."""
        transcript = "Um, can you, uh, tell me the weather like, in Dubai today?"
        result     = _run_with_transcript(transcript)

        assert "weather_tool" in result["tools_called"], (
            f"Disfluent weather query failed. Tools called: {result['tools_called']}"
        )
        print(f"\n  [audio:disfluency:weather_filler] ✓ Intent preserved despite fillers")

    def test_currency_intent_with_hesitation(self):
        """Hesitation and repetition must not confuse currency intent."""
        transcript = "I need to, um, convert — convert like fifty, uh, fifty dollars to euros"
        result     = _run_with_transcript(transcript)

        assert "currency_tool" in result["tools_called"], (
            f"Disfluent currency query failed. Tools called: {result['tools_called']}"
        )
        print(f"\n  [audio:disfluency:currency_hesitation] ✓ Intent preserved")

    def test_audio_plus_image_combined(self):
        """
        Voice + image: transcript asks about the image.
        Response must reference both the image content and answer the voice question.
        """
        transcript  = "Um, can you tell me more about what you see in this picture?"
        img_desc    = "A historic stone castle on a cliff overlooking the sea at dusk."
        result      = _run_with_transcript(transcript, image_description=img_desc)
        response    = result["final_response"]

        score = score_response(
            question = transcript,
            answer   = response,
            criteria = (
                "Response must describe the image content — castle, cliff, sea, or dusk. "
                "A generic response that ignores the image scores 1. "
                "Detailed description of castle and setting scores 4-5."
            ),
        )
        assert_pass(score, "audio:combined:voice_plus_image")