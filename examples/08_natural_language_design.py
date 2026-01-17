"""Example of natural language design interaction.

Demonstrates how the LLM Agent can orchestrate tools to create designs
based on user requests. Includes conversation logging.
"""

import json
from pathlib import Path

from photonic_forge.agent import (
    ConversationLogger,
    FunctionTool,
    LLMAgent,
    Message,
    MockLLMClient,
    ToolRegistry,
)
from photonic_forge.core import Waveguide


def create_waveguide(length: float, width: float) -> str:
    """Create a waveguide and return its summary."""
    wg = Waveguide(start=(0, 0), end=(length, 0), width=width)
    return f"Created Waveguide(length={length*1e6:.2f}um, width={width*1e9:.0f}nm)"


def run_example():
    # 1. Setup Tools
    registry = ToolRegistry()
    registry.register_function(create_waveguide)

    # 2. Setup Mock LLM (Simulating a real interaction)
    # Scenario: User asks for a 10um waveguide with 500nm width.
    
    responses = [
        # LLM decides to call the tool
        Message(
            role="assistant",
            content="I will create a waveguide with length 10um and width 500nm.",
            tool_calls=[
                {
                    "name": "create_waveguide",
                    "arguments": {"length": 10e-6, "width": 500e-9}
                }
            ]
        ),
        # LLM confirms completion
        Message(
            role="assistant",
            content="I've created the waveguide as requested."
        )
    ]
    
    client = MockLLMClient(responses)
    
    # 3. Initialize Agent
    agent = LLMAgent(
        client=client,
        registry=registry,
        system_prompt="You are an expert photonic designer."
    )
    
    # 4. Run Interaction
    user_request = "Create a straight waveguide, 10 microns long and 500nm wide."
    print(f"User: {user_request}")
    
    response = agent.chat(user_request)
    print(f"Agent: {response}")
    
    # 5. Log Conversation (Data Moat)
    logger = ConversationLogger(log_dir="data/conversations")
    log_path = logger.log_conversation(
        conversation_id="demo-session-01",
        messages=agent.history,
        metadata={"user_intent": "create_waveguide"}
    )
    
    print(f"\nConversation logged to: {log_path}")
    
    # Verify log content
    with open(log_path) as f:
        log_data = json.load(f)
        params = log_data["messages"][2]["tool_outputs"][0]["output"] # 0:User, 1:Asst(Call), 2:Tool(Out)
        print(f"Confirmed tool execution: {params}")


if __name__ == "__main__":
    run_example()
