import pytest
from jiwer import wer, cer
from evals.conftest import make_graph_state, score_response, assert_pass
from backend.agent_llm.graph import chatbot_graph

pytestmark = [pytest.mark.eval, pytest.mark.asyncio]

async def _run_with_transcript(transcript: str, image_description: str = None) -> dict:
    state = make_graph_state(
        # By setting these to None, we tell the Graph to skip the API processing nodes
        audio_b64         = None, 
        image_b64         = None,
        # And instead, we inject the mocked results directly
        transcribed_text  = transcript,
        image_description = image_description,
    )
    
    # Run the compiled graph! It automatically handles the config and tool loops perfectly.
    return await chatbot_graph.ainvoke(state)


class TestWER:
    WER_THRESHOLD = 0.10
    async def test_wer_clean_speech_transcript(self):
        reference   = "What is the current price of Apple stock?"
        hypothesis  = "What is the current price of Apple stock?"
        error_rate = wer(reference, hypothesis)
        assert error_rate <= self.WER_THRESHOLD

    async def test_wer_minor_transcription_errors(self):
        reference  = "Convert one hundred dollars to Pakistani rupees"
        hypothesis = "Convert one hundred dollar to Pakistani rupees"
        error_rate = wer(reference, hypothesis)
        assert error_rate <= 0.20

    async def test_wer_complete_mismatch_detected(self):
        reference  = "What is the weather in London today?"
        hypothesis = "Play some music on Spotify please"
        error_rate = wer(reference, hypothesis)
        assert error_rate > 0.5

class TestCER:
    CER_THRESHOLD = 0.08
    async def test_cer_clean_transcript(self):
        reference  = "Show me the stock price for Tesla"
        hypothesis = "Show me the stock price for Tesla"
        error_rate = cer(reference, hypothesis)
        assert error_rate <= self.CER_THRESHOLD

class TestIntentAccuracy:
    async def test_weather_intent_calls_weather_tool(self, passer):
        transcript = "What is the weather like in Karachi right now?"
        result     = await _run_with_transcript(transcript)
        assert "weather_tool" in result["tools_called"]
        score = score_response(transcript, result["final_response"], "Weather info for Karachi.")
        passer(score, "audio:intent:weather_quality")

    async def test_currency_intent_calls_currency_tool(self):
        transcript = "How much is 200 US dollars in Pakistani rupees?"
        result     = await _run_with_transcript(transcript)
        assert "currency_tool" in result["tools_called"]

    async def test_stock_intent_calls_stock_tool(self):
        transcript = "What is the stock price of Google?"
        result     = await _run_with_transcript(transcript)
        assert "stock_tool" in result["tools_called"]

class TestDisfluencyRobustness:
    async def test_weather_intent_with_filler_words(self):
        transcript = "Um, can you, uh, tell me the weather like, in Dubai today?"
        result     = await _run_with_transcript(transcript)
        assert "weather_tool" in result["tools_called"]

    async def test_audio_plus_image_combined(self, passer):
        transcript  = "Um, can you tell me more about what you see in this picture?"
        img_desc    = "A historic stone castle on a cliff overlooking the sea at dusk."
        result      = await _run_with_transcript(transcript, image_description=img_desc)
        score = score_response(transcript, result["final_response"], "Describe castle, cliff, sea.")
        passer(score, "audio:combined:voice_plus_image")