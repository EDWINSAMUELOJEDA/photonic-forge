"""Tool definitions and registry for the design agent.

Defines the interface for tools that the agent can use and a registry
for managing them.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")


@dataclass
class ToolDefinition:
    """Metadata about a tool."""
    name: str
    description: str
    parameters: dict[str, Any]


class Tool(Protocol):
    """Protocol for an agent tool."""

    @property
    def name(self) -> str:
        """Name of the tool."""
        ...

    @property
    def description(self) -> str:
        """Description of what the tool does."""
        ...

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the tool."""
        ...


class FunctionTool:
    """Wraps a python function as a Tool."""

    def __init__(self, func: Callable[..., Any], name: str | None = None, description: str | None = None):
        self.func = func
        self._name = name or func.__name__
        self._description = description or func.__doc__ or ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def get_definition(self) -> ToolDefinition:
        """Get the schema definition of the tool."""
        sig = inspect.signature(self.func)
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Basic type inference
            param_type = "string"
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            
            params[param_name] = {
                "type": param_type,
                "required": param.default == inspect.Parameter.empty
            }

        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=params
        )



def analyze_yield(nominal_value: float) -> str:
    """Analyze manufacturing yield for a parameter."""
    # Mock estimator for now since we don't have a live surrogate in this file
    return f"Estimated Yield: 98.5% (Robust to +/- 10% variation)"

class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: dict[str, FunctionTool] = {}
        # Auto-register basic tools
        self.register_function(analyze_yield)

    def register(self, tool: FunctionTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_function(self, func: Callable[..., Any], name: str | None = None, description: str | None = None) -> None:
        """Register a python function as a tool."""
        tool = FunctionTool(func, name, description)
        self.register(tool)

    def get_tool(self, name: str) -> FunctionTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_definitions(self) -> list[ToolDefinition]:
        """Get definitions for all registered tools."""
        return [tool.get_definition() for tool in self._tools.values()]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        return tool(**arguments)
