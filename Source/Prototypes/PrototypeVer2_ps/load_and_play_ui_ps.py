import os
import sys
import time
import numpy as np
import threading
from combined_env import *
# from combined_env import *
from collections import deque
from ray.rllib.policy.policy import Policy
from UI.Ui_ui_mainwindow import Ui_MainWindow
from PySide2.QtWidgets import (QApplication, QMainWindow)
import platform


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Init variables
        self.items = [
            "interest_rates",
            'gdp',
            'one_plus_gdp_growth_rate',
            'price_lvl',
            'one_plus_inf_rate',
            'affinity',
            'delta_affinity'
        ]
        self.N = N
        self.nStep = 0
        self.is_Playing = False
        self.is_Running = True
        self.history = { key:[deque(maxlen=100) for _ in range(self.N)] for key in self.items }
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'pink', 'cyan']
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
        self.env.render(mode='human')

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

        render = self.env.render()
        if render is None:
            return

        for item in self.items:
            if item not in render:
                continue
            # value = np.random.rand(self.N)
            if item in ['interest_rates', 'one_plus_gdp_growth_rate', 'one_plus_inf_rate', 'affinity', 'delta_affinity']:
                value = render[item]
            else:
                value = np.exp(render[item])
            for x in range(self.N):
                self.history[item][x].append(value[x])
            ui_name = f'wdt_{item}'
            if 'affinity' in item:
                getattr(self.ui, ui_name).ShowDepthPlot(value, self.N, self.colors)
            else:
                getattr(self.ui, ui_name).ShowPlot(value, self.N, self.colors)
        self.update_history()

    def update_history(self):
        label = self.ui.cmbFilter.currentText()
        value = self.history[label]
        getattr(self.ui, "wdt_history").ShowHistoryPlot(value, self.N, self.colors, label)

    def load_env(self):
        my_checkpoint_path = "~/ray_results/APPO_2024-01-17_12-48-34/APPO_combined_environment_b9dfc_00000_0_gamma=0.9013,lr=0.0000_2024-01-17_12-48-34/checkpoint_000003"
        if platform.node() == "jang-yejun-ui-MacBookAir.local":
            my_checkpoint_path = "~/Desktop/checkpoint_000002"
        self.env = CombinedEnv(render_mode='human')
        self.policies = {}
        self.state = {}
        self.observations, _ = self.env.reset()
        # checkpoint_path = os.path.expanduser(f"~/ray_results/{my_checkpoint_path}/policies/")
        # self.policies[f'default_policy'] = Policy.from_checkpoint(checkpoint_path)
        for i in range(0, self.N):
            
            checkpoint_path = os.path.expanduser(f"{my_checkpoint_path}/policies/agent_{i}")
            self.policies[f'agent_{i}'] = Policy.from_checkpoint(checkpoint_path)

            self.state[f'agent_{i}'] = [
                np.zeros([256], np.float32) for _ in range(2)
            ]

    def closeEvent(self, *args, **kwargs):
        self.env.close()
        self.is_Running = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.resize(1200, 600)
    mainWin.showMaximized()
    sys.exit(app.exec_())
