import matplotlib
matplotlib.use('Qt5Agg')  # Ensure Qt5Agg backend is used for PySide6 compatibility
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Import unified color scheme
sys.path.append(str(Path(__file__).parent.parent.parent / 'configs'))
from color_schemes import get_country_color, get_country_colors, create_country_color_map


class ItemPlotWidget(FigureCanvas):
    def __init__(self, parent=None):
        super(ItemPlotWidget, self).__init__(Figure())       
        self.setParent(parent)
        self.figure = Figure(figsize=[10.0, 3.0], dpi=100)

        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.axs = [self.ax]
        self.colorbar = None  # Store reference to colorbar
        # self.figure.tight_layout()


    def ShowPlot(self, value, N, colors=None, countries=None):
        """Show bar plot with unified color scheme."""
        plt.close('all')
        for ax in self.axs:
            ax.clear()

        # Use unified colors if countries are provided
        if countries is not None:
            plot_colors = get_country_colors(countries[:N], 'matplotlib')
        elif colors is not None:
            plot_colors = colors
        else:
            # Default color scheme
            plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:N]

        labels = countries[:N] if countries else [f'{i}' for i in range(1, N+1)]
        bars = self.ax.bar(labels, value, color=plot_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Rotate labels if they are country codes
        if countries:
            self.ax.set_xticklabels(labels, rotation=45, ha='right')
        
        self.ax.grid(True, alpha=0.3)
        self.draw()

    def ShowDepthPlot(self, value, N, colors=None, colormap='RdBu_r'):
        """Show heatmap with unified color scheme."""
        plt.close('all')
        
        # Remove existing colorbar if it exists
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
            
        for ax in self.axs:
            ax.clear()

        # Display the 2D affinity matrix as a heatmap with better colormap
        im = self.ax.imshow(value, cmap=colormap, interpolation='nearest', aspect='auto')
        
        # Add colorbar and store reference
        self.colorbar = plt.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        
        self.ax.set_title('Affinity Matrix')
        self.draw()

    def ShowHistoryPlot(self, value, N, colors=None, label='History', countries=None):
        """Show line plot with unified color scheme."""
        plt.close('all')
        for ax in self.axs:
            ax.clear()

        # Use unified colors if countries are provided
        if countries is not None:
            plot_colors = get_country_colors(countries[:N], 'matplotlib')
        elif colors is not None:
            plot_colors = colors
        else:
            # Default color scheme
            plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for i in range(N):
            color = plot_colors[i % len(plot_colors)]
            country_label = countries[i] if countries and i < len(countries) else f'Series {i+1}'
            self.ax.plot(value[i], color=color, linewidth=2, alpha=0.8, label=country_label)
        
        self.ax.set_title(label, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Add legend if countries are provided
        if countries:
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self.draw()