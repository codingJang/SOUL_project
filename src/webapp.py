import os
import sys
import time
import json
import asyncio
import numpy as np
import glob
import threading
import subprocess
import zipfile
import tempfile
import shutil
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import from the original application
from env import MacroSimRayRLlibEnv, N
from ray.rllib.policy.policy import Policy
from configs.color_schemes import color_scheme
from configs.rllib_train_config import RLlibTrainConfig
from configs.environment_config import MacroSimEnvConfig


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


class TrainingConfig(BaseModel):
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    gamma_min: float = 0.9
    gamma_max: float = 0.9999
    clip_param: float = 0.2
    train_batch_size: int = 512
    max_timesteps: int = 10000000
    num_samples: int = 20
    time_budget_hours: float = 4.0
    checkpoint_frequency: int = 1000


class TrainingStatus(BaseModel):
    is_training: bool = False
    iteration: int = 0
    timesteps_total: int = 0
    episode_reward_mean: float = 0.0
    training_config: Optional[TrainingConfig] = None
    progress_percentage: float = 0.0
    eta_hours: float = 0.0
    logs: List[str] = []


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


class TrainingManager:
    def __init__(self):
        self.is_training = False
        self.training_process = None
        self.training_thread = None
        self.training_config = TrainingConfig()
        self.training_status = TrainingStatus()
        self.logs = deque(maxlen=1000)  # Store last 1000 log entries
        self.start_time = None
        self.progress_metrics = {
            'iterations': [],
            'timesteps': [],
            'rewards': [],
            'agent_metrics': {}
        }
        
    def start_training(self, config: TrainingConfig):
        """Start training with given configuration."""
        if self.is_training:
            raise ValueError("Training is already in progress")
        
        self.training_config = config
        self.is_training = True
        self.start_time = time.time()
        self.training_status.is_training = True
        self.training_status.training_config = config
        self.logs.clear()
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self._run_training, daemon=True)
        self.training_thread.start()
        
        self.add_log("Training started with new configuration")
        return True
    
    def stop_training(self):
        """Stop the current training process."""
        if not self.is_training:
            return False
        
        self.is_training = False
        self.training_status.is_training = False
        
        if self.training_process:
            self.training_process.terminate()
            self.training_process = None
        
        self.add_log("Training stopped by user")
        return True
    
    def reset_training(self):
        """Reset training state."""
        self.stop_training()
        self.training_status = TrainingStatus()
        self.logs.clear()
        self.progress_metrics = {
            'iterations': [],
            'timesteps': [],
            'rewards': [],
            'agent_metrics': {}
        }
        self.add_log("Training state reset")
        return True
    
    def get_status(self):
        """Get current training status."""
        return self.training_status
    
    def get_logs(self, limit: int = 100):
        """Get recent training logs."""
        return list(self.logs)[-limit:]
    
    def add_log(self, message: str):
        """Add a log entry with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        self.training_status.logs = self.get_logs(50)  # Keep last 50 in status
    
    def _run_training(self):
        """Run the training process in a separate thread."""
        try:
            # Import Ray modules here to avoid conflicts with the simulation environment
            import ray
            from ray import tune, air, train
            from ray.rllib.algorithms.appo import APPOConfig
            from ray.tune.registry import register_env
            
            self.add_log("Initializing Ray for training...")
            
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(num_gpus=0, log_to_driver=False, logging_level='ERROR')
            
            # Environment setup
            def env_creator(env_config):
                from env import MacroSimRayRLlibEnv
                env_config_obj = MacroSimEnvConfig()
                return MacroSimRayRLlibEnv(render_mode='array', config=env_config.get('env_config', env_config_obj))
            
            env_name = "macro_sim_rllib_env_v0"
            register_env(env_name, env_creator)
            
            self.add_log("Environment registered successfully")
            
            # Create temporary environment for configuration
            env_config_obj = MacroSimEnvConfig()
            temp_env = env_creator({'env_config': env_config_obj})
            
            # Configure APPO
            config = (
                APPOConfig()
                .training(
                    lr=tune.loguniform(self.training_config.lr_min, self.training_config.lr_max),
                    gamma=tune.uniform(self.training_config.gamma_min, self.training_config.gamma_max),
                    clip_param=self.training_config.clip_param,
                    train_batch_size=self.training_config.train_batch_size
                )
                .environment(env=env_name, clip_actions=True, env_config={'env_config': env_config_obj})
                .rollouts(num_rollout_workers=1)  # Use fewer workers for web interface
                .framework(framework="torch")
                .resources(num_learner_workers=1, num_cpus_for_local_worker=1)
                .multi_agent(
                    policies=temp_env.get_agent_ids(),
                    policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
                )
                .debugging(log_level="ERROR")
            )
            
            config.model['use_lstm'] = True
            
            self.add_log(f"Starting training with {self.training_config.num_samples} samples")
            
            # Custom callback for progress tracking
            class WebTrainingCallback:
                def __init__(self, training_manager):
                    self.training_manager = training_manager
                
                def __call__(self, trial_id, result):
                    if not self.training_manager.is_training:
                        return True  # Stop if training was cancelled
                    
                    iteration = result.get("training_iteration", 0)
                    timesteps = result.get("timesteps_total", 0)
                    reward_mean = result.get("episode_reward_mean", 0)
                    
                    # Update status
                    self.training_manager.training_status.iteration = iteration
                    self.training_manager.training_status.timesteps_total = timesteps
                    self.training_manager.training_status.episode_reward_mean = reward_mean
                    
                    # Calculate progress
                    progress = min(100, (timesteps / self.training_manager.training_config.max_timesteps) * 100)
                    self.training_manager.training_status.progress_percentage = progress
                    
                    # Calculate ETA
                    if timesteps > 0 and self.training_manager.start_time:
                        elapsed = time.time() - self.training_manager.start_time
                        estimated_total = elapsed * (self.training_manager.training_config.max_timesteps / timesteps)
                        eta = max(0, estimated_total - elapsed) / 3600  # Convert to hours
                        self.training_manager.training_status.eta_hours = eta
                    
                    # Store metrics
                    self.training_manager.progress_metrics['iterations'].append(iteration)
                    self.training_manager.progress_metrics['timesteps'].append(timesteps)
                    self.training_manager.progress_metrics['rewards'].append(reward_mean)
                    
                    # Log progress every 5 iterations
                    if iteration % 5 == 0:
                        self.training_manager.add_log(
                            f"Iteration {iteration}: {timesteps:,} timesteps, avg reward: {reward_mean:.3f}"
                        )
                    
                    return False  # Continue training
            
            # Stop function
            def stop_fn(trial_id: str, result: dict) -> bool:
                if not self.is_training:
                    return True
                
                timesteps_reached = result["timesteps_total"] >= self.training_config.max_timesteps
                if timesteps_reached:
                    self.add_log("Training completed: Maximum timesteps reached")
                
                return timesteps_reached
            
            # Create tuner
            tuner = tune.Tuner(
                "APPO",
                run_config=air.RunConfig(
                    storage_path=os.path.abspath("models"),
                    checkpoint_config=train.CheckpointConfig(
                        checkpoint_frequency=self.training_config.checkpoint_frequency
                    ),
                    stop=stop_fn,
                    verbose=0
                ),
                tune_config=tune.TuneConfig(
                    num_samples=self.training_config.num_samples,
                    time_budget_s=int(self.training_config.time_budget_hours * 3600),
                    max_concurrent_trials=1
                ),
                param_space=config.to_dict()
            )
            
            # Run training
            self.add_log("Training started successfully")
            result = tuner.fit()
            
            if self.is_training:  # Only log completion if not manually stopped
                self.add_log("Training completed successfully!")
                best_result = result.get_best_result()
                if best_result:
                    final_reward = best_result.metrics.get("episode_reward_mean", 0)
                    self.add_log(f"Best result - Final reward: {final_reward:.3f}")
            
        except Exception as e:
            self.add_log(f"Training error: {str(e)}")
            print(f"Training error: {e}")
        finally:
            self.is_training = False
            self.training_status.is_training = False
            if ray.is_initialized():
                ray.shutdown()
    
    def get_progress_metrics(self):
        """Get training progress metrics for visualization."""
        return self.progress_metrics


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
                            'timestamp': appo_name.split('_')[-2:] if '_' in appo_name else ['', ''],
                            'type': 'trained'
                        })
        
        # Find uploaded checkpoints
        uploaded_dir = os.path.join(models_dir, "uploaded")
        if os.path.exists(uploaded_dir):
            uploaded_checkpoints = glob.glob(os.path.join(uploaded_dir, "*"))
            for checkpoint_dir in uploaded_checkpoints:
                if os.path.isdir(checkpoint_dir):
                    policies_dir = os.path.join(checkpoint_dir, "policies")
                    if os.path.exists(policies_dir):
                        checkpoint_name = os.path.basename(checkpoint_dir)
                        checkpoints.append({
                            'path': checkpoint_dir,
                            'display_name': f"[Uploaded] {checkpoint_name}",
                            'full_path': checkpoint_dir,
                            'timestamp': ['uploaded', ''],
                            'type': 'uploaded'
                        })
        
        # Sort by timestamp (most recent first), with uploaded models at the top
        checkpoints.sort(key=lambda x: (x['type'] != 'uploaded', ''.join(x['timestamp'])), reverse=True)
        return checkpoints

    def validate_checkpoint_structure(self, checkpoint_dir: str) -> bool:
        """Validate that the checkpoint has the required structure."""
        try:
            policies_dir = os.path.join(checkpoint_dir, "policies")
            if not os.path.exists(policies_dir):
                return False
            
            # Check for required agent policies
            required_agents = [f"agent_{i}" for i in range(N)]
            for agent in required_agents:
                agent_dir = os.path.join(policies_dir, agent)
                if not os.path.exists(agent_dir):
                    return False
                
                # Check for required files in agent directory
                required_files = ["policy_state.pkl"]  # Minimum required file
                for req_file in required_files:
                    if not os.path.exists(os.path.join(agent_dir, req_file)):
                        return False
            
            return True
        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def extract_and_validate_checkpoint(self, zip_file_path: str, checkpoint_name: str) -> str:
        """Extract and validate uploaded checkpoint zip file."""
        models_dir = "./models/"
        uploaded_dir = os.path.join(models_dir, "uploaded")
        
        # Create uploaded directory if it doesn't exist
        os.makedirs(uploaded_dir, exist_ok=True)
        
        # Create temporary extraction directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the checkpoint directory in extracted files
            # Look for a directory containing "policies" subdirectory
            checkpoint_dir = None
            for root, dirs, files in os.walk(temp_dir):
                if "policies" in dirs:
                    checkpoint_dir = root
                    break
            
            if not checkpoint_dir:
                raise ValueError("No valid checkpoint structure found in zip file")
            
            # Validate checkpoint structure
            if not self.validate_checkpoint_structure(checkpoint_dir):
                raise ValueError("Invalid checkpoint structure - missing required agent policies")
            
            # Create final destination
            final_dir = os.path.join(uploaded_dir, checkpoint_name)
            
            # Remove existing if present
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            
            # Move to final location
            shutil.move(checkpoint_dir, final_dir)
            
            return final_dir
            
        except Exception as e:
            raise ValueError(f"Failed to extract or validate checkpoint: {str(e)}")
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

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
training_manager = TrainingManager()
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
    training_manager.stop_training()

# Create FastAPI app
app = FastAPI(
    title="SOUL Project - Economic Simulation & Training Platform",
    description="Web interface for loading, training, and running AI economic simulations",
    version="2.0.0",
    lifespan=lifespan
)

# Setup templates and static files
templates = Jinja2Templates(directory="src/templates")
app.mount("/static", StaticFiles(directory="src/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# === SIMULATION ENDPOINTS (Load & Play Tab) ===

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


@app.post("/upload_checkpoint")
async def upload_checkpoint(file: UploadFile = File(...), checkpoint_name: str = Form(...)):
    """Upload and extract a checkpoint zip file."""
    try:
        # Validate file type
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Only ZIP files are allowed")
        
        # Validate checkpoint name
        if not checkpoint_name or not checkpoint_name.strip():
            raise HTTPException(status_code=400, detail="Checkpoint name is required")
        
        # Clean checkpoint name (remove potentially dangerous characters)
        checkpoint_name = "".join(c for c in checkpoint_name.strip() if c.isalnum() or c in (' ', '-', '_')).strip()
        if not checkpoint_name:
            raise HTTPException(status_code=400, detail="Invalid checkpoint name")
        
        # Create temporary file for upload
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        try:
            # Save uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            
            # Extract and validate checkpoint
            final_path = sim_manager.extract_and_validate_checkpoint(temp_file.name, checkpoint_name)
            
            return {
                "success": True, 
                "message": f"Checkpoint '{checkpoint_name}' uploaded successfully",
                "checkpoint_path": final_path
            }
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# === TRAINING ENDPOINTS (Train Tab) ===

@app.post("/start_training")
async def start_training(request: Request):
    """Start training with given configuration."""
    try:
        data = await request.json()
        
        # Create training configuration from request data
        config = TrainingConfig(
            lr_min=float(data.get('lr_min', 1e-5)),
            lr_max=float(data.get('lr_max', 1e-3)),
            gamma_min=float(data.get('gamma_min', 0.9)),
            gamma_max=float(data.get('gamma_max', 0.9999)),
            clip_param=float(data.get('clip_param', 0.2)),
            train_batch_size=int(data.get('train_batch_size', 512)),
            max_timesteps=int(data.get('max_timesteps', 10000000)),
            num_samples=int(data.get('num_samples', 20)),
            time_budget_hours=float(data.get('time_budget_hours', 4.0)),
            checkpoint_frequency=int(data.get('checkpoint_frequency', 1000))
        )
        
        training_manager.start_training(config)
        return {"success": True, "message": "Training started"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/stop_training")
async def stop_training():
    """Stop the current training process."""
    try:
        result = training_manager.stop_training()
        if result:
            return {"success": True, "message": "Training stopped"}
        else:
            return {"success": False, "message": "No training in progress"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/reset_training")
async def reset_training():
    """Reset training state."""
    try:
        training_manager.reset_training()
        return {"success": True, "message": "Training state reset"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/training_status")
async def get_training_status():
    """Get current training status."""
    return training_manager.get_status()


@app.get("/training_logs")
async def get_training_logs(limit: int = 100):
    """Get recent training logs."""
    return {"logs": training_manager.get_logs(limit)}


@app.get("/training_metrics")
async def get_training_metrics():
    """Get training progress metrics."""
    return training_manager.get_progress_metrics()


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
            # Send current state for simulation
            state = sim_manager.get_state()
            
            # Add training status to the state
            training_status = training_manager.get_status()
            state.update({
                "training_status": training_status.dict(),
                "training_logs": training_manager.get_logs(10)  # Send last 10 logs
            })
            
            await manager.send_personal_message(json.dumps(state), websocket)
            
            # If playing, step the simulation
            if sim_manager.is_playing and not sim_manager.is_terminated:
                data = sim_manager.step_simulation()
                if data:
                    await manager.send_personal_message(
                        json.dumps({"type": "simulation_data", "data": data}), 
                        websocket
                    )
            
            # Send training metrics if training is active
            if training_manager.is_training:
                metrics = training_manager.get_progress_metrics()
                await manager.send_personal_message(
                    json.dumps({"type": "training_metrics", "data": metrics}),
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False) 