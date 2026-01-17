"""Tests for the Conversation Logger."""

import json
from pathlib import Path

from photonic_forge.agent.llm import Message
from photonic_forge.agent.logger import ConversationLogger


def test_conversation_logger(tmp_path):
    """Test logging a simple conversation."""
    logger = ConversationLogger(log_dir=tmp_path)
    
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there")
    ]
    
    log_file = logger.log_conversation(
        conversation_id="test-conv-123",
        messages=messages,
        metadata={"user_id": "test_user"}
    )
    
    assert log_file.exists()
    
    with open(log_file) as f:
        data = json.load(f)
        
    assert data["conversation_id"] == "test-conv-123"
    assert data["metadata"]["user_id"] == "test_user"
    assert len(data["messages"]) == 2
    assert data["messages"][0]["content"] == "Hello"
