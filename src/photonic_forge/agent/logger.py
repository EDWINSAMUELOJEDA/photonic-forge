"""Logger for agent conversations to support Data Moat strategy.

Captures design intent, agent actions, and results for future training.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from photonic_forge.agent.llm import Message


@dataclass
class ConversationLog:
    """A full conversation record."""
    conversation_id: str
    timestamp: float
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]


class ConversationLogger:
    """Logs conversation data to disk."""

    def __init__(self, log_dir: str | Path = "data/conversations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_conversation(
        self,
        conversation_id: str,
        messages: list[Message],
        metadata: dict[str, Any] | None = None
    ) -> Path:
        """Save a conversation to a JSONL file."""
        
        # Convert internal Message objects to dicts
        msg_dicts = []
        for msg in messages:
            m_dict = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.tool_calls:
                m_dict["tool_calls"] = msg.tool_calls
            if msg.tool_outputs:
                m_dict["tool_outputs"] = msg.tool_outputs
            msg_dicts.append(m_dict)

        log_entry = ConversationLog(
            conversation_id=conversation_id,
            timestamp=time.time(),
            messages=msg_dicts,
            metadata=metadata or {}
        )

        filename = f"{conversation_id}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(asdict(log_entry), f, indent=2)
            
        return filepath
