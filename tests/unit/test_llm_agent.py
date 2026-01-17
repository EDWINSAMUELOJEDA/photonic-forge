"""Tests for the LLM agent infrastructure."""

import pytest
from photonic_forge.agent.llm import LLMAgent, Message, MockLLMClient
from photonic_forge.agent.tools import ToolRegistry


def test_tool_registry():
    """Test registering and retrieving tools."""
    registry = ToolRegistry()

    def my_tool(x: int, y: int = 1) -> int:
        """Adds two numbers."""
        return x + y

    registry.register_function(my_tool)
    
    # Check definition
    defs = registry.get_definitions()
    assert len(defs) == 1
    assert defs[0].name == "my_tool"
    assert defs[0].description == "Adds two numbers."
    assert "x" in defs[0].parameters
    assert defs[0].parameters["x"]["type"] == "integer"
    assert defs[0].parameters["x"]["required"] is True

    # Execute
    result = registry.execute("my_tool", {"x": 5, "y": 2})
    assert result == 7


def test_llm_agent_loop():
    """Test the agent conversation loop with a mock client."""
    registry = ToolRegistry()
    
    def greet(name: str) -> str:
        return f"Hello, {name}!"
        
    registry.register_function(greet)
    
    # Mock conversation:
    # 1. User: "Hi"
    # 2. Assistant: (Calls tool greet)
    # 3. Tool: "Hello, World!"
    # 4. Assistant: "The tool said hello."
    
    mock_responses = [
        # Response 1: Call tool
        Message(
            role="assistant",
            content="",
            tool_calls=[{"name": "greet", "arguments": {"name": "World"}}]
        ),
        # Response 2: Final answer (after tool execution)
        Message(
            role="assistant",
            content="The tool said hello."
        )
    ]
    
    client = MockLLMClient(mock_responses)
    agent = LLMAgent(client, registry)
    
    response = agent.chat("Hi")
    
    assert response == "The tool said hello."
    
    # Check history: User, Assistant(Tool Call), ToolOutput, Assistant(Final)
    assert len(agent.history) == 4
    assert agent.history[0].role == "user"
    assert agent.history[1].tool_calls is not None
    assert agent.history[2].role == "tool"
    assert "Hello, World!" in agent.history[2].content
    assert agent.history[3].content == "The tool said hello."
