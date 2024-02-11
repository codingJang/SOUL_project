from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np


class ItemPlotWidget(FigureCanvas):
    def __init__(self, parent=None):
        super(ItemPlotWidget, self).__init__(Figure())       
        self.setParent(parent)
        self.figure = Figure(figsize=[10.0, 3.0], dpi=100)

        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.axs = [self.ax]
        # self.figure.tight_layout()


    def ShowPlot(self, value, N, colors):
        plt.close('all')
        for ax in self.axs:
            ax.clear()

        # Define different colors for each bar
        

        labels = [f'{i}' for i in range(1, N+1)]
        self.ax.bar(labels, value, color=colors)
        self.draw()

    def ShowDepthPlot(self, value, N, colors):
        plt.close('all')
        for ax in self.axs:
            ax.clear()

        # Define different colors for each bar
        # self.ax.imshow(value)
        # self.draw()
        # a = np.random.random((16, 16))
        ax.imshow(value, cmap='hot', interpolation='nearest')
        self.draw()

    def ShowHistoryPlot(self, value, N, colors, label):
        plt.close('all')
        for ax in self.axs:
            ax.clear()

        for i in range(N):
            self.ax.plot(value[i], color=colors[i%N%len(colors)])
        self.ax.set_title(label)
        self.draw()