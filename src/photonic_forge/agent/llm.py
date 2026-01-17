"""LLM Agent for natural language design interaction.

Manages conversation history and orchestrates tool usage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from photonic_forge.optimize import YieldEstimator
from photonic_forge.agent.tools import ToolRegistry


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_outputs: list[dict[str, Any]] | None = None


class LLMClient(Protocol):
    """Protocol for an LLM provider client."""

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Message:
        """Get the next message completion."""
        ...


class MockLLMClient:
    """Mock client for testing."""
    
    def __init__(self, responses: list[Message] | None = None):
        self.responses = responses or []
        self._call_count = 0

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Message:
        if self._call_count < len(self.responses):
            response = self.responses[self._call_count]
            self._call_count += 1
            return response
        return Message(role="assistant", content="I allow myself no further responses.")


class LLMAgent:
    """Agent that uses an LLM to accomplish tasks."""

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        system_prompt: str = "You are a helpful photonic design assistant.",
    ):
        self.client = client
        self.registry = registry
        self.system_prompt = system_prompt
        self.history: list[Message] = []

    def chat(self, user_input: str) -> str:
        """Process a user message and return the final response."""
        
        # Add user message
        self.history.append(Message(role="user", content=user_input))
        
        # Main loop: Think/Call Tools -> Repeat until done
        while True:
            # Prepare context
            messages = [Message(role="system", content=self.system_prompt)] + self.history
            
            # specific dict format for tools based on definitions
            tool_definitions = [
                {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.parameters
                }
                for d in self.registry.get_definitions()
            ]

            # Get LLM response
            response = self.client.complete(messages, tools=tool_definitions)
            self.history.append(response)

            # If no tool calls, we are done
            if not response.tool_calls:
                return response.content

            # Execute tools
            tool_outputs = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["arguments"]
                
                try:
                    result = self.registry.execute(tool_name, args)
                    output = str(result)
                except Exception as e:
                    output = f"Error: {e}"
                
                tool_outputs.append({
                    "name": tool_name,
                    "output": output
                })

            # Append tool outputs as a new message (simulating the 'tool' role or user response with data)
            # For simplicity in this generic implementation, we'll format it as a user/system message 
            # or strictly follow the underlying client's expected format.
            # Here, we will append a message with role 'tool' if the client supported it, 
            # but for this generic protocol, let's assume we send it back as an observation.
            
            # NOTE: In a real OpenAI/Anthropic client, this part is specific. 
            # We will use a generic 'tool' role for now.
            tool_content = "\n".join(
                f"Tool '{t['name']}' returned: {t['output']}" for t in tool_outputs
            )
            self.history.append(Message(role="tool", content=tool_content, tool_outputs=tool_outputs))
            
            # Loop continues to let LLM process the tool output
