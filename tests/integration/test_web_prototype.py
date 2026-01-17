"""Integration tests for the Web Prototype."""

from fastapi.testclient import TestClient
from photonic_forge.serv.api import app


def test_chat_endpoint_waveguide():
    """Test generating a waveguide via chat."""
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "Make a 20um waveguide"})
        assert response.status_code == 200
        data = response.json()
        
        assert "response" in data
        assert "waveguide" in data["response"]
        assert data["vis_data"] is not None
        assert "epsilon" in data["vis_data"]
        assert data["vis_data"]["params"]["dx"] > 0

def test_chat_endpoint_yield():
    """Test checking yield via chat."""
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "Check yield"})
        assert response.status_code == 200
        data = response.json()
        
        assert "response" in data
    # The heuristic agent replies with yield info 
    # (actually the tool execution is mocked in HeuristicLLM.complete but 
    # the generic agent loop executes the tool.
    # The heuristic mocked 'assistant' message has tool calls.)
    
    # Wait, in api.py:HeuristicLLM, we return a Message with tool_calls.
    # The Agent loop in llm.py executes them and appends a tool output message.
    # But currently HeuristicLLM.complete() is called.
    # The loop calls complete() -> gets tool call -> executes tool -> appends output.
    # Then loop calls complete() AGAIN with the history (User, Asst, Tool).
    # Does HeuristicLLM handle the 'Tool' message?
    # In my api.py, HeuristicLLM only checks the LAST message.
    # If the last message is from tool (role='tool'), it should probably generate a final answer.
    
    # I need to check api.py HeuristicLLM logic for tool outputs.
    # I suspect it might loop forever if I didn't handle the 'tool' role case.
    # Let's inspect api.py before running.
    pass
