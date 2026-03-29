import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition
from langgraph.graph import END

# Import only what still exists in your graph.py
from backend.agent_llm.graph import route_inputs

def test_integration_routing_logic():
    """
    Integration test ensuring the routing logic correctly identifies 
    when to call tools versus when to finalize the conversation.
    """
    
    # Scenario 1: LLM wants to use a tool
    tool_state = {
        "messages": [
            HumanMessage(content="What is the weather in Tokyo?"),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "weather_tool",
                    "args": {"location": "Tokyo"},
                    "id": "call_999"
                }]
            )
        ]
    }
    
    # tools_condition should return "tools"
    # Note: In your graph.py, this is mapped to the "tool_executor" node
    assert tools_condition(tool_state) == "tools"

def test_integration_finalize_logic():
    """
    Integration test ensuring the flow moves to END (finalize) 
    when no tools are needed.
    """
    
    # Scenario 2: LLM provides a direct answer
    chat_state = {
        "messages": [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi! How can I help you today?")
        ]
    }
    
    # tools_condition should return the END constant
    # In your graph.py, END is mapped to the "finalize" node
    assert tools_condition(chat_state) == END

@pytest.mark.asyncio
async def test_route_inputs_integration():
    """
    Test the entry point router that handles multimodal inputs.
    """
    # Test case for text-only input
    state_text = {
        "raw_text": "Hello",
        "raw_image_b64": None,
        "raw_audio_b64": None
    }
    assert route_inputs(state_text) == "merge_inputs"

    # Test case for image input
    state_image = {
        "raw_text": None,
        "raw_image_b64": "base64_data...",
        "raw_audio_b64": None
    }
    assert route_inputs(state_image) == "image_processing"