"""FastAPI backend for PhotonicForge.

Serves the web interface and handles agent interactions.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from photonic_forge.agent import (
    LLMAgent,
    MockLLMClient,
    Message,
    ToolRegistry,
    ConversationLogger
)
from photonic_forge.agent.tools import analyze_yield
from photonic_forge.core import Waveguide, SILICON, SILICON_DIOXIDE
from photonic_forge.vis import WebGPUExporter
from photonic_forge.optimize import MinFeatureConstraint

# Global state
agent = None
logger = None

def create_mock_agent():
    """Create an agent with a "smart" mock client for the demo."""
    registry = ToolRegistry()
    registry.register_function(analyze_yield)
    
    # Custom tool for the web app to generate vis data
    def generate_demo_waveguide(length: float = 10.0, width: float = 0.5) -> dict:
        """Generates a waveguide and returns simulation data for visualization."""
        wg = Waveguide(start=(0, 0), end=(length * 1e-6, 0), width=width * 1e-6)
        
        # Low res for instant web response
        eps_grid = wg.to_permittivity(
            bounds=(-2e-6, -2e-6, (length+2)*1e-6, 2e-6),
            resolution=100e-9,
            material_inside=SILICON,
            material_outside=SILICON_DIOXIDE
        )
        
        # Export logic (in-memory would be better, but file is easier for now)
        exporter = WebGPUExporter()
        
        # We cheat a bit and use the internal logic of exporter to get JSON data
        # instead of writing a file, so we can send it to frontend
        nx, ny = eps_grid.shape
        eps_flat = eps_grid.astype(float).flatten().tolist()
        
        sim_data = {
            "params": {
                "nx": int(nx),
                "ny": int(ny),
                "dx": 100e-9,
                "dt": 50e-9
            },
            "epsilon": eps_flat
        }
        return json.dumps(sim_data)

    registry.register_function(generate_demo_waveguide)

    # Heuristic Mock LLM
    # In a real app, this would be an OpenAI/Anthropic client
    class HeuristicLLM:
        def complete(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
            last_msg = messages[-1]
            
            # If last message was a tool output, confirm success
            if last_msg.role == "tool":
                return Message(role="assistant", content="Design generated successfully! You can see the visualization in the viewport.")

            content = last_msg.content.lower()
            
            if "waveguide" in content and "tools" not in str(last_msg):
                # extract numbers roughly or default
                l = 10.0
                w = 0.5
                if "20" in content: l = 20.0
                if "wide" in content and "1" in content: w = 1.0
                
                return Message(
                    role="assistant", 
                    content=f"I'll generate a {l}um waveguide with {w}um width for you.",
                    tool_calls=[{
                        "name": "generate_demo_waveguide", 
                        "arguments": {"length": l, "width": w}
                    }]
                )
            elif "yield" in content:
                return Message(
                     role="assistant",
                     content="Checking manufacturability...",
                     tool_calls=[{
                        "name": "analyze_yield",
                        "arguments": {"nominal_value": 0.5}
                     }]
                )
            
            return Message(role="assistant", content="I can help you design waveguides or analyze yield. Try 'Make a waveguide' or 'Check yield'.")

    return LLMAgent(HeuristicLLM(), registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global agent, logger
    agent = create_mock_agent()
    logger = ConversationLogger(log_dir="data/web_conversations")
    yield
    # Shutdown

app = FastAPI(lifespan=lifespan)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    vis_data: dict[str, Any] | None = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    resp_text = agent.chat(request.message)
    
    # Check history for tool outputs containing vis data
    vis_data = None
    if len(agent.history) >= 2:
        last_tool_msg = agent.history[-2] # User, Asst(Call), Tool(Out), Asst(Resp) -> Tool is -2
        if last_tool_msg.role == "tool" and last_tool_msg.tool_outputs:
            for out in last_tool_msg.tool_outputs:
                if out["name"] == "generate_demo_waveguide":
                    # The tool returned a dict, likely as string representation
                    # In a real app, we'd handle structured output better
                    try:
                        # We need to eval/parse the string back since the generic agent casts to str
                        # For the specific tool above, it returns a dict.
                        # The generic tool executor in `llm.py` does `str(result)`.
                        # We will re-register the tool to return the dict directly if possible,
                        # or parse it here. 
                        # EASIER TRICK: The 'Tool' class in `llm.py` executes and returns Any.
                        # But `chat` loop converts to string.
                        # Let's just fix the tool output to be the dict instance if we can,
                        # or extract it from the agent instance if we modify it.
                        
                        # HACK for prototype: 
                        # We know our `generate_demo_waveguide` returns a massive dict
                        # which `str()` converts. `eval()` is unsafe but okay for local MVP.
                        import json
                        # The tool output has double quotes for JSON string, but llm.py might wrap it?
                        # llm.py does: output = str(result)
                        # If result is json string, output is json string (unquoted).
                        # Let's parse it.
                        print(f"DEBUG TOOL OUTPUT: {out['output'][:100]}...") # Print first 100 chars
                        vis_data = json.loads(out["output"])
                    except Exception as e:
                        print(f"Failed to parse tool output for visualization: {e}")
    
    return ChatResponse(response=resp_text, vis_data=vis_data)

# Static Files
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
