import os
import sys
import numpy as np
from economics_env import *
from collections import deque
from ray.rllib.policy.policy import Policy
from UI.Ui_ui_mainwindow import Ui_MainWindow
from PySide2.QtWidgets import (QApplication, QMainWindow)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Init variables
        self.items = [
            # "agents",
            "interest_rates",
            'gdp',
            'dem_after_shock',
            'prev_price_lvl',
            'price_lvl'
        ]
        self.N = 2
        self.history = { key:[deque(maxlen=100) for _ in range(self.N)] for key in self.items }
        self.colors = ['blue', 'orange', 'green', 'red']
        self.currentAction = 150

        # Reset UI default values
        self.ui.cmbFilter.addItems(self.items)
        self.ui.action_slider.setRange(0, 200)
        self.ui.action_slider.setSingleStep(1)
        self.ui.action_slider.setValue(self.currentAction)
        self.ui.txt_slider.setText(str(self.currentAction / 10.0))
        self.ui.groupBox.setVisible(False)

        # Link components listeners
        # Connect slider's valueChanged signal to update_line_edit slot
        self.ui.action_slider.valueChanged.connect(self.update_line_edit)
        # Connect line edit's editingFinished signal to update_slider_value slot
        self.ui.txt_slider.editingFinished.connect(self.update_slider_value)
        self.ui.cmbFilter.currentIndexChanged.connect(self.On_FilterChanged)
        self.ui.btnPlay.clicked.connect(self.on_simulate)
        self.update_line_edit(self.currentAction)

        # load env
        self.load_env()

    def update_line_edit(self, value):
        # Convert integer value to float for precision
        float_value = value / 10.0
        self.ui.txt_slider.setText(str(float_value))
        self.currentAction = float_value

    def update_slider_value(self):
        try:
            # Try to convert the text to a float and then to an integer for the slider
            float_value = float(self.ui.txt_slider.text())
            slider_value = min(200, max(0, int(float_value * 10)))
            self.ui.action_slider.setValue(slider_value)
            self.currentAction = float_value
        except ValueError:
            # Handle the case where the entered value is not a valid float
            pass

    def On_FilterChanged(self):
        self.update_history()

    def on_simulate(self, event):
        self.env.render(mode='human')
        human_action = np.log(np.clip(self.currentAction / (20-self.currentAction), 1e-5, 1-1e-5))

        # Get actions for AI agents
        actions = {f'agent_{i}': self.policies[f'agent_{i}'].compute_single_action(self.observations[f'agent_{i}'])[0] for i in range(1, self.N)}
        actions['agent_0'] = np.array([human_action], dtype=np.float32)

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
            if item in ['interest_rates', 'gdp']:
                value = render[item]
            else:
                value = np.exp(render[item])
            for x in range(self.N):
                self.history[item][x].append(value[x])
            ui_name = f'wdt_{item}'
            getattr(self.ui, ui_name).ShowPlot(value, self.N, self.colors)
        self.update_history()

    def update_history(self):
        label = self.ui.cmbFilter.currentText()
        value = self.history[label]
        getattr(self.ui, "wdt_history").ShowHistoryPlot(value, self.N, self.colors, label)

    def load_env(self):
        my_experiment_name = "APPO_2023-11-28_23-19-22"
        my_trial_name = "APPO_economics_environment_90392_00000_0_gamma=0.9444,lr=0.0000_2023-11-28_23-19-22"
        checkpoint_name = "checkpoint_000009"
        self.env = EconomicsEnv()
        self.policies = {}
        self.observations, _ = self.env.reset()
        for i in range(1, self.N):
            checkpoint_path = os.path.expanduser(f"~/ray_results/{my_experiment_name}/{my_trial_name}/{checkpoint_name}/policies/agent_{i}")
            self.policies[f'agent_{i}'] = Policy.from_checkpoint(checkpoint_path)

    def closeEvent(self, *args, **kwargs):
        self.env.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.resize(1200, 600)
    mainWin.show()
    sys.exit(app.exec_())