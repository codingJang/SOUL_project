import os
import sys
import time
import json
import asyncio
import numpy as np
import glob
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import from the original application
from env import MacroSimRayRLlibEnv, N
from ray.rllib.policy.policy import Policy
from configs.color_schemes import color_scheme


class CheckpointInfo(BaseModel):
    path: str
    display_name: str
    full_path: str
    timestamp: List[str]


class SimulationState(BaseModel):
    is_playing: bool
    step: int
    current_data: Optional[Dict] = None
    is_terminated: bool = False


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.disconnect(connection)


class SimulationManager:
    def __init__(self):
        self.checkpoint_path = None
        self.env = None
        self.policies = {}
        self.state = {}
        self.observations = None
        self.is_playing = False
        self.is_running = False
        self.step = 0
        self.items = [
            "interest_rates",
            'gdp',
            'dem_after_shock',
            'price_lvl',
            'delta_price_lvl',
            'affinity',
            'delta_affinity'
        ]
        self.history = {key: [deque(maxlen=100) for _ in range(N)] for key in self.items}
        self.colors = ['#87CEEB', '#FFB6C1', '#98FB98', '#DDA0DD', '#F0E68C', '#FFCCCB', '#B0E0E6']
        self.current_data = None
        self.is_terminated = False
        self.speed_multiplier = 1.0  # Speed multiplier for simulation

    def find_available_checkpoints(self):
        """Find all available checkpoints in the models directory."""
        models_dir = "./models/"
        checkpoints = []
        
        if not os.path.exists(models_dir):
            print(f"Models directory '{models_dir}' not found!")
            return []
        
        # Find all APPO experiment directories
        appo_dirs = glob.glob(os.path.join(models_dir, "APPO_*"))
        
        for appo_dir in appo_dirs:
            # Find the experiment subdirectory
            exp_subdirs = glob.glob(os.path.join(appo_dir, "APPO_macro_sim_*"))
            
            for exp_subdir in exp_subdirs:
                # Find checkpoint directories
                checkpoint_dirs = glob.glob(os.path.join(exp_subdir, "checkpoint_*"))
                
                for checkpoint_dir in checkpoint_dirs:
                    # Check if policies directory exists
                    policies_dir = os.path.join(checkpoint_dir, "policies")
                    if os.path.exists(policies_dir):
                        appo_name = os.path.basename(appo_dir)
                        checkpoint_name = os.path.basename(checkpoint_dir)
                        
                        checkpoints.append({
                            'path': checkpoint_dir,
                            'display_name': f"{appo_name}/{os.path.basename(exp_subdir)}/{checkpoint_name}",
                            'full_path': checkpoint_dir,
                            'timestamp': appo_name.split('_')[-2:] if '_' in appo_name else ['', '']
                        })
        
        # Sort by timestamp (most recent first)
        checkpoints.sort(key=lambda x: ''.join(x['timestamp']), reverse=True)
        return checkpoints

    def load_checkpoint(self, checkpoint_path: str):
        """Load a specific checkpoint."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        
        # Initialize environment
        self.env = MacroSimRayRLlibEnv(render_mode='human')
        self.policies = {}
        self.state = {}
        self.observations, _ = self.env.reset()
        
        # Load policies for each agent
        for i in range(0, N):
            policy_path = os.path.join(checkpoint_path, "policies", f"agent_{i}")
            if not os.path.exists(policy_path):
                raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")
            
            self.policies[f'agent_{i}'] = Policy.from_checkpoint(policy_path)
            self.state[f'agent_{i}'] = [
                np.zeros([256], np.float32) for _ in range(2)
            ]
        
        self.step = 0
        self.is_terminated = False
        # Reset history
        self.history = {key: [deque(maxlen=100) for _ in range(N)] for key in self.items}
        
        return True

    def step_simulation(self):
        """Run one step of the simulation."""
        if not self.env or self.is_terminated:
            return None
        
        # Get actions for all agents
        actions = {}
        for i in range(N):
            action, state_out, _ = self.policies[f'agent_{i}'].compute_single_action(
                self.observations[f'agent_{i}'], 
                self.state[f'agent_{i}']
            )
            actions[f'agent_{i}'] = action
            self.state[f'agent_{i}'] = state_out

        # Step the environment
        self.observations, _, terminateds, _, _ = self.env.step(actions)
        self.is_terminated = terminateds.get('agent_0', False)
        
        # Get render data
        render = self.env.render(mode='human')
        if render is None:
            return None

        # Process data for web display
        current_data = {}
        for item in self.items:
            if item not in render:
                continue
                
            if item in ['interest_rates', 'gdp', 'affinity', 'delta_affinity']:
                value = render[item]
            else:
                value = np.exp(render[item])
            
            # Convert numpy arrays to lists for JSON serialization
            if hasattr(value, 'tolist'):
                current_data[item] = value.tolist()
            else:
                current_data[item] = [float(value)] * N if not hasattr(value, '__len__') else [float(v) for v in value]
            
            # Update history
            for x in range(N):
                val = current_data[item][x] if len(current_data[item]) > x else current_data[item][0]
                self.history[item][x].append(val)

        self.current_data = current_data
        self.step += 1
        return current_data

    def get_history_data(self, item: str = None):
        """Get historical data for plotting."""
        if item and item in self.history:
            return {item: [list(agent_history) for agent_history in self.history[item]]}
        else:
            return {key: [list(agent_history) for agent_history in value] 
                   for key, value in self.history.items()}

    def get_state(self):
        """Get current simulation state."""
        return {
            "is_playing": self.is_playing,
            "step": self.step,
            "current_data": self.current_data,
            "is_terminated": self.is_terminated,
            "checkpoint_loaded": self.checkpoint_path is not None
        }

    def cleanup(self):
        """Clean up resources."""
        if self.env:
            self.env.close()
        self.is_running = False


# Global instances
sim_manager = SimulationManager()
manager = ConnectionManager()

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting SOUL Project Web Application")
    yield
    # Shutdown
    print("Shutting down...")
    sim_manager.cleanup()

# Create FastAPI app
app = FastAPI(
    title="SOUL Project - Load and Play Web Interface",
    description="Web interface for loading and running AI economic simulations",
    version="1.0.0",
    lifespan=lifespan
)

# Setup templates and static files
templates = Jinja2Templates(directory="src/templates")
app.mount("/static", StaticFiles(directory="src/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/checkpoints", response_model=List[CheckpointInfo])
async def get_checkpoints():
    """Get all available checkpoints."""
    checkpoints = sim_manager.find_available_checkpoints()
    return checkpoints


@app.post("/load_checkpoint")
async def load_checkpoint(request: Request):
    """Load a specific checkpoint."""
    try:
        form = await request.form()
        checkpoint_path = form.get("checkpoint_path")
        if not checkpoint_path:
            raise HTTPException(status_code=400, detail="checkpoint_path is required")
        
        sim_manager.load_checkpoint(checkpoint_path)
        return {"success": True, "message": f"Checkpoint loaded: {checkpoint_path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/start_simulation")
async def start_simulation():
    """Start the simulation."""
    if not sim_manager.checkpoint_path:
        raise HTTPException(status_code=400, detail="No checkpoint loaded")
    
    sim_manager.is_playing = True
    return {"success": True, "message": "Simulation started"}


@app.post("/pause_simulation")
async def pause_simulation():
    """Pause the simulation."""
    sim_manager.is_playing = False
    return {"success": True, "message": "Simulation paused"}


@app.post("/set_speed")
async def set_speed(request: Request):
    """Set simulation speed multiplier."""
    try:
        data = await request.json()
        speed = float(data.get('speed', 1.0))
        if speed <= 0:
            raise HTTPException(status_code=400, detail="Speed must be positive")
        sim_manager.speed_multiplier = speed
        return {"success": True, "message": f"Speed set to {speed}x"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/simulation_state")
async def get_simulation_state():
    """Get current simulation state."""
    return sim_manager.get_state()


@app.get("/history/{item}")
async def get_history(item: str):
    """Get historical data for a specific item."""
    return sim_manager.get_history_data(item)


@app.get("/history")
async def get_all_history():
    """Get all historical data."""
    return sim_manager.get_history_data()


@app.get("/color_scheme")
async def get_color_scheme():
    """Get color scheme for consistent visualization."""
    # Get colors for agents (using fallback palette which is designed for this)
    agent_colors = color_scheme.fallback_palette[:N]  # Get first N colors for agents
    
    return {
        "agent_colors": agent_colors,
        "chart_colors": {
            "interest_rates": agent_colors,
            "gdp": agent_colors, 
            "price_lvl": agent_colors,
            "affinity": agent_colors
        },
        "regional_colors": color_scheme.regional_colors,
        "num_agents": N
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send current state
            state = sim_manager.get_state()
            await manager.send_personal_message(json.dumps(state), websocket)
            
            # If playing, step the simulation
            if sim_manager.is_playing and not sim_manager.is_terminated:
                data = sim_manager.step_simulation()
                if data:
                    await manager.send_personal_message(
                        json.dumps({"type": "simulation_data", "data": data}), 
                        websocket
                    )
            
            await asyncio.sleep(1.0 / sim_manager.speed_multiplier)  # Update based on speed multiplier
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def run_simulation_loop():
    """Background task to run simulation."""
    while True:
        if sim_manager.is_playing and not sim_manager.is_terminated:
            data = sim_manager.step_simulation()
            if data:
                await manager.broadcast(json.dumps({"type": "simulation_data", "data": data}))
        await asyncio.sleep(1.0 / sim_manager.speed_multiplier)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 