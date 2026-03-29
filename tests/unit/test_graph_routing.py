import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition
from langgraph.graph import END

def test_routing_to_tools():
    """
    Test that tools_condition routes to 'tools' when the last message 
    contains tool_calls.
    """
    state = {
        "messages": [
            HumanMessage(content="What is the weather in London?"),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "weather_tool",
                    "args": {"location": "London"},
                    "id": "call_123"
                }]
            )
        ]
    }
    # In your graph configuration, "tools" is mapped to the tool node
    assert tools_condition(state) == "tools"

def test_routing_to_finalize():
    """
    Test that tools_condition routes to END when there are no tool calls.
    """
    state = {
        "messages": [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there! How can I help you today?")
        ]
    }
    # tools_condition returns the END constant if no tool_calls exist
    assert tools_condition(state) == END

def test_routing_with_empty_messages():
    """
    Verify that tools_condition raises a ValueError when the messages list is empty.
    """
    state = {"messages": []}
    
    # LangGraph's prebuilt tools_condition requires at least one message
    with pytest.raises(ValueError, match="No messages found"):
        tools_condition(state)