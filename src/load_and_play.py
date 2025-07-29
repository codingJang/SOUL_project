import os
import sys
import time
import numpy as np
import threading
import glob
from datetime import datetime
from env import *
from collections import deque
from ray.rllib.policy.policy import Policy
from UI.ui_mainwindow import Ui_MainWindow
from PySide6.QtWidgets import QApplication, QMainWindow
import platform


def find_available_checkpoints():
    """Find all available checkpoints in the models directory."""
    models_dir = "./models/"
    checkpoints = []
    
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found!")
        return []
    
    # Find all APPO experiment directories
    appo_dirs = glob.glob(os.path.join(models_dir, "APPO_*"))
    
    for appo_dir in appo_dirs:
        # Find the experiment subdirectory (usually contains the actual checkpoints)
        exp_subdirs = glob.glob(os.path.join(appo_dir, "APPO_macro_sim_*"))
        
        for exp_subdir in exp_subdirs:
            # Find checkpoint directories
            checkpoint_dirs = glob.glob(os.path.join(exp_subdir, "checkpoint_*"))
            
            for checkpoint_dir in checkpoint_dirs:
                # Check if policies directory exists
                policies_dir = os.path.join(checkpoint_dir, "policies")
                if os.path.exists(policies_dir):
                    # Extract timestamp from directory name for sorting
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


def select_checkpoint():
    """Interactive checkpoint selection."""
    checkpoints = find_available_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found in ./models/ directory!")
        return None
    
    print("\nAvailable checkpoints:")
    print("=" * 50)
    
    for i, checkpoint in enumerate(checkpoints):
        print(f"{i + 1}: {checkpoint['display_name']}")
    
    print(f"\nMost recent checkpoint: {checkpoints[0]['display_name']}")
    print("=" * 50)
    
    while True:
        user_input = input(f"\nSelect checkpoint (1-{len(checkpoints)}) or press Enter for most recent: ").strip()
        
        if not user_input:
            # Use most recent checkpoint
            selected = checkpoints[0]
            print(f"Using most recent checkpoint: {selected['display_name']}")
            return selected['full_path']
        
        try:
            selection = int(user_input)
            if 1 <= selection <= len(checkpoints):
                selected = checkpoints[selection - 1]
                print(f"Selected checkpoint: {selected['display_name']}")
                return selected['full_path']
            else:
                print(f"Please enter a number between 1 and {len(checkpoints)}")
        except ValueError:
            print("Please enter a valid number or press Enter for default")


class MainWindow(QMainWindow):
    def __init__(self, checkpoint_path):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.checkpoint_path = checkpoint_path

        # Init variables
        self.items = [
            "interest_rates",
            'gdp',
            'dem_after_shock',
            'price_lvl',
            'delta_price_lvl',
            'affinity',
            'delta_affinity'
        ]
        self.N = N
        self.nStep = 0
        self.is_Playing = False
        self.is_Running = True
        self.history = { key:[deque(maxlen=100) for _ in range(self.N)] for key in self.items }
        self.colors = ['#87CEEB', '#FFB6C1', '#98FB98', '#DDA0DD', '#F0E68C', '#FFCCCB', '#B0E0E6']  # Pastel colors: sky blue, light pink, pale green, plum, khaki, misty rose, powder blue
        self.currentAction = 150

        # Reset UI default values
        self.ui.cmbFilter.addItems(self.items)

        # Link components listeners
        self.ui.cmbFilter.currentIndexChanged.connect(self.On_FilterChanged)
        self.ui.btnPlay.clicked.connect(self.on_simulate_event)
        self.ui.btnPlay.setText("Start")
        self.thread = threading.Thread(target=self.process).start()

        # load env
        self.load_env()

    def process(self):
        while self.is_Running:
            time.sleep(1)
            if self.is_Playing:
                self.on_simulate()

    def On_FilterChanged(self):
        self.update_history()

    def on_simulate_event(self, event):
        self.is_Playing = not self.is_Playing
        self.ui.btnPlay.setText("Pause" if self.is_Playing else "Play")

    def on_simulate(self):
        # Get actions for AI agents
        # actions = {f'agent_{i}': self.policies[f'agent_{i}'].compute_single_action(self.observations[f'agent_{i}'])[0] for i in range(1, self.N)}
        # actions['agent_0'] = np.array([human_action], dtype=np.float32)
        actions = {}
        for i in range(self.N):
            action, state_out, _ = self.policies[f'agent_{i}'].compute_single_action(self.observations[f'agent_{i}'], self.state[f'agent_{i}'])
            actions[f'agent_{i}'] = action
            self.state[f'agent_{i}'] = state_out

        # Step the environment
        self.observations, _, terminateds, _, _ = self.env.step(actions)
        
        self.ui.btnPlay.setEnabled(not terminateds['agent_0'])

        # Get render data for plotting (use 'human' mode to get the dictionary)
        render = self.env.render(mode='human')
        if render is None:
            print("Warning: render() returned None")
            return

        # Debug: Print available keys in render data
        if hasattr(self, '_debug_printed') is False:
            print(f"Available render keys: {list(render.keys())}")
            print(f"Expected items: {self.items}")
            self._debug_printed = True

        for item in self.items:
            if item not in render:
                print(f"Warning: '{item}' not found in render data")
                continue
            # value = np.random.rand(self.N)
            if item in ['interest_rates', 'gdp', 'affinity', 'delta_affinity']:
                value = render[item]
            else:
                value = np.exp(render[item])
            
            # Update history
            for x in range(self.N):
                self.history[item][x].append(value[x] if hasattr(value, '__getitem__') and len(value) > x else value)
            
            # Update UI widget
            ui_name = f'wdt_{item}'
            try:
                if 'affinity' in item:
                    getattr(self.ui, ui_name).ShowDepthPlot(value, self.N, self.colors)
                else:
                    getattr(self.ui, ui_name).ShowPlot(value, self.N, self.colors)
            except Exception as e:
                print(f"Error updating widget {ui_name}: {e}")
        self.update_history()

    def update_history(self):
        label = self.ui.cmbFilter.currentText()
        value = self.history[label]
        getattr(self.ui, "wdt_history").ShowHistoryPlot(value, self.N, self.colors, label)

    def load_env(self):
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        self.env = MacroSimRayRLlibEnv(render_mode='human')
        self.policies = {}
        self.state = {}
        self.observations, _ = self.env.reset()
        
        for i in range(0, self.N):
            checkpoint_path = os.path.join(self.checkpoint_path, "policies", f"agent_{i}")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")
            
            self.policies[f'agent_{i}'] = Policy.from_checkpoint(checkpoint_path)

            self.state[f'agent_{i}'] = [
                np.zeros([256], np.float32) for _ in range(2)
            ]

    def closeEvent(self, *args, **kwargs):
        self.env.close()
        self.is_Running = False

if __name__ == '__main__':
    # CLI interaction for checkpoint selection
    print("SOUL Project - Load and Play")
    print("=" * 30)
    
    checkpoint_path = select_checkpoint()
    if checkpoint_path is None:
        print("No valid checkpoint selected. Exiting...")
        sys.exit(1)
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    print("Starting application...")
    
    app = QApplication(sys.argv)
    mainWin = MainWindow(checkpoint_path)
    mainWin.resize(1200, 600)
    mainWin.showMaximized()
    sys.exit(app.exec_())