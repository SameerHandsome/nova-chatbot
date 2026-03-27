"""
tests/integration/test_tool_calls.py — Tool invocation integration tests.
Verifies the correct tool is triggered for each query type.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.agent_llm.graph import should_use_tools, route_inputs
from backend.agent_llm.tools import ALL_TOOLS


class TestToolRegistry:
    def test_all_five_tools_registered(self):
        tool_names = [t.name for t in ALL_TOOLS]
        assert "weather_tool"  in tool_names
        assert "currency_tool" in tool_names
        assert "stock_tool"    in tool_names
        assert "wikipedia"     in tool_names or any("wiki" in n.lower() for n in tool_names)
        assert len(ALL_TOOLS) == 5

    def test_all_tools_have_descriptions(self):
        for tool in ALL_TOOLS:
            assert tool.description, f"{tool.name} has no description"

    def test_all_tools_have_names(self):
        for tool in ALL_TOOLS:
            assert tool.name, "Tool missing name"


class TestToolRouting:
    """
    These tests mock the LLM response to simulate tool calls
    and verify the graph routing works correctly.
    """

    def _msg_with_tool_call(self, tool_name, args):
        msg = MagicMock()
        msg.tool_calls = [{"name": tool_name, "args": args, "id": "call-1"}]
        msg.content    = ""
        return msg

    def _msg_without_tool_call(self, content="Final answer."):
        msg = MagicMock()
        msg.tool_calls = []
        msg.content    = content
        return msg

    def test_weather_query_routes_to_tool_executor(self):
        msg   = self._msg_with_tool_call("weather_tool", {"location": "London"})
        state = {"lc_messages": [msg]}
        assert should_use_tools(state) == "tool_executor"

    def test_stock_query_routes_to_tool_executor(self):
        msg   = self._msg_with_tool_call("stock_tool", {"symbol": "AAPL"})
        state = {"lc_messages": [msg]}
        assert should_use_tools(state) == "tool_executor"

    def test_currency_query_routes_to_tool_executor(self):
        msg   = self._msg_with_tool_call("currency_tool", {"query": "100 USD to PKR"})
        state = {"lc_messages": [msg]}
        assert should_use_tools(state) == "tool_executor"

    def test_final_answer_routes_to_finalize(self):
        msg   = self._msg_without_tool_call("Here is the answer.")
        state = {"lc_messages": [msg]}
        assert should_use_tools(state) == "finalize"

    def test_multiple_tool_calls_in_one_message(self):
        """LLM can call multiple tools at once — still routes to executor."""
        msg = MagicMock()
        msg.tool_calls = [
            {"name": "weather_tool", "args": {"location": "NYC"}, "id": "c1"},
            {"name": "stock_tool",   "args": {"symbol": "TSLA"},  "id": "c2"},
        ]
        msg.content = ""
        state = {"lc_messages": [msg]}
        assert should_use_tools(state) == "tool_executor"


class TestToolExecutorNode:
    def test_unknown_tool_returns_error_message(self):
        from backend.agent_llm.graph import tool_executor_node

        msg = MagicMock()
        msg.tool_calls = [{"name": "nonexistent_tool", "args": {}, "id": "bad-1"}]
        msg.content    = ""

        state = {
            "lc_messages":  [msg],
            "tools_called": [],
            "tool_results": [],
            "raw_text": None, "raw_image_b64": None,
            "raw_image_media_type": None, "raw_audio_b64": None,
            "transcribed_text": None, "image_description": None,
            "merged_input": None, "chat_history": [],
            "final_response": None, "error": None,
        }

        result = tool_executor_node(state)

        # Should not crash — should record an error result
        assert len(result["tool_results"]) == 1
        assert "nonexistent_tool" in result["tool_results"][0]["tool"]
        assert "not found" in result["tool_results"][0]["result"].lower()

    def test_tool_results_accumulate_across_turns(self):
        """Previous tool results should be preserved in state."""
        from backend.agent_llm.graph import tool_executor_node

        msg = MagicMock()
        msg.tool_calls = [{"name": "nonexistent_tool", "args": {}, "id": "c1"}]
        msg.content    = ""

        existing_results = [{"tool": "weather_tool", "result": "Sunny 24°C"}]

        state = {
            "lc_messages":  [msg],
            "tools_called": ["weather_tool"],
            "tool_results": existing_results,
            "raw_text": None, "raw_image_b64": None,
            "raw_image_media_type": None, "raw_audio_b64": None,
            "transcribed_text": None, "image_description": None,
            "merged_input": None, "chat_history": [],
            "final_response": None, "error": None,
        }

        result = tool_executor_node(state)

        # Existing + new result
        assert len(result["tool_results"]) == 2
        assert result["tool_results"][0]["tool"] == "weather_tool"