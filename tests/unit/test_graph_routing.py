"""
tests/unit/test_graph_routing.py — LangGraph routing logic unit tests.
All 7 input combos + tool loop routing. No LLM calls made.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from backend.agent_llm.graph import (
    route_inputs,
    after_audio,
    should_use_tools,
)


def _state(**kwargs):
    """Build a minimal ChatState dict with provided fields."""
    base = {
        "raw_text":             None,
        "raw_image_b64":        None,
        "raw_image_media_type": None,
        "raw_audio_b64":        None,
        "transcribed_text":     None,
        "image_description":    None,
        "merged_input":         None,
        "lc_messages":          [],
        "chat_history":         [],
        "final_response":       None,
        "tools_called":         [],
        "tool_results":         [],
        "error":                None,
    }
    return {**base, **kwargs}


class TestRouteInputs:
    def test_text_only_goes_to_merge(self):
        state = _state(raw_text="hello")
        assert route_inputs(state) == "merge_inputs"

    def test_audio_only_goes_to_audio_transcription(self):
        state = _state(raw_audio_b64="abc==")
        assert route_inputs(state) == "audio_transcription"

    def test_image_only_goes_to_image_processing(self):
        state = _state(raw_image_b64="abc==")
        assert route_inputs(state) == "image_processing"

    def test_audio_plus_text_goes_to_audio_first(self):
        """Audio takes priority — text merged later."""
        state = _state(raw_audio_b64="abc==", raw_text="also this")
        assert route_inputs(state) == "audio_transcription"

    def test_image_plus_text_goes_to_image_first(self):
        state = _state(raw_image_b64="abc==", raw_text="describe this")
        assert route_inputs(state) == "image_processing"

    def test_audio_plus_image_goes_to_audio_first(self):
        state = _state(raw_audio_b64="abc==", raw_image_b64="img==")
        assert route_inputs(state) == "audio_transcription"

    def test_all_three_goes_to_audio_first(self):
        state = _state(raw_audio_b64="a==", raw_image_b64="i==", raw_text="t")
        assert route_inputs(state) == "audio_transcription"

    def test_no_inputs_goes_to_merge(self):
        """Edge case — empty state should still reach merge."""
        state = _state()
        assert route_inputs(state) == "merge_inputs"


class TestAfterAudio:
    def test_image_present_after_audio_goes_to_image(self):
        state = _state(raw_image_b64="img==")
        assert after_audio(state) == "image_processing"

    def test_no_image_after_audio_goes_to_merge(self):
        state = _state(raw_image_b64=None)
        assert after_audio(state) == "merge_inputs"

    def test_empty_string_image_goes_to_merge(self):
        state = _state(raw_image_b64="")
        assert after_audio(state) == "merge_inputs"


class TestShouldUseTools:
    def test_tool_call_present_routes_to_executor(self):
        msg = MagicMock(spec=AIMessage)
        msg.tool_calls = [{"name": "weather_tool", "args": {"location": "Lahore"}, "id": "tc1"}]
        state = _state(lc_messages=[msg])
        assert should_use_tools(state) == "tool_executor"

    def test_no_tool_call_routes_to_finalize(self):
        msg = MagicMock(spec=AIMessage)
        msg.tool_calls = []
        state = _state(lc_messages=[msg])
        assert should_use_tools(state) == "finalize"

    def test_empty_tool_calls_attr_routes_to_finalize(self):
        msg = MagicMock(spec=AIMessage)
        del msg.tool_calls   # simulate missing attribute
        msg.tool_calls = []
        state = _state(lc_messages=[msg])
        assert should_use_tools(state) == "finalize"

    def test_multiple_tool_calls_routes_to_executor(self):
        msg = MagicMock(spec=AIMessage)
        msg.tool_calls = [
            {"name": "stock_tool",    "args": {"symbol": "AAPL"}, "id": "tc1"},
            {"name": "weather_tool",  "args": {"location": "NYC"}, "id": "tc2"},
        ]
        state = _state(lc_messages=[msg])
        assert should_use_tools(state) == "tool_executor"