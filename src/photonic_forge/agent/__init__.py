"""Agent module for PhotonicForge.

Provides automated design exploration and intelligent agents
for photonic device optimization.
"""

from photonic_forge.optimize import OptimizationResult

from .designer import AgentConfig, DesignAgent, DesignCandidate, run_agent_exploration
from .llm import LLMAgent, LLMClient, Message, MockLLMClient
from .logger import ConversationLogger
from .strategy import (
    ExplorationResult,
    ExplorationStrategy,
    HybridStrategy,
    LatinHypercubeStrategy,
    LocalSearchStrategy,
    RandomStrategy,
)
from .tools import FunctionTool, Tool, ToolDefinition, ToolRegistry

__all__ = [
    # Designer
    "DesignCandidate",
    "AgentConfig",
    "DesignAgent",
    "run_agent_exploration",
    # LLM Agent
    "LLMAgent",
    "LLMClient",
    "Message",
    "MockLLMClient",
    # Tools
    "Tool",
    "ToolDefinition",
    "ToolRegistry",
    "FunctionTool",
    # Logger
    "ConversationLogger",
    # Strategies
    "ExplorationResult",
    "ExplorationStrategy",
    "RandomStrategy",
    "LatinHypercubeStrategy",
    "LocalSearchStrategy",
    "HybridStrategy",
]
