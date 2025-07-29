#!/usr/bin/env python3
"""
Network Visualization for Political Data

Creates interactive network visualizations from political adjacency matrices including:
- IGO (Intergovernmental Organizations) membership networks - tracking multilateral cooperation 
  patterns from institutional memberships spanning 1815-2014
- DCAD (Defense Cooperation Agreement Dataset) defense cooperation networks - bilateral defense
  agreements coordinating routine defense relations

IGO Citation: Pevehouse, Jon C.W., Timothy Nordstron, Roseanne W McManus, Anne Spencer Jamison, 
"Tracking Organizations in the World: The Correlates of War IGO Version 3.0 datasets", 
Journal of Peace Research.

DCAD Citation: Kinne, Brandon J. 2020. "The Defense Cooperation Agreement Dataset (DCAD)," 
The Journal of Conflict Resolution 64(4): 729-755.

The IGO dataset captures membership in intergovernmental organizations with at least 3 
nation-states, while DCAD captures bilateral defense cooperation agreements that coordinate
routine defense relations including joint exercises, peacekeeping, defense research, 
weapons programs, and officer exchanges.
"""

import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import pandas as pd
import webbrowser
import os
import sys
from pathlib import Path
import argparse

class GraphVisualizer:
    """Class for creating and visualizing network graphs from adjacency matrices."""
    
    def __init__(self, dataframe_list, labels=None):
        """
        Initialize with list of adjacency matrices.
        
        Args:
            dataframe_list: List of pandas DataFrames representing adjacency matrices
            labels: Optional list of labels for each graph
        """
        self.dataframe_list = dataframe_list
        self.labels = labels or [f"Graph {i}" for i in range(len(dataframe_list))]
        self.graph_list = None

    def create_graphs(self):
        """Convert each dataframe to a networkx graph."""
        self.graph_list = []
        
        for i, df in enumerate(self.dataframe_list):
            print(f"Creating graph {i+1}/{len(self.dataframe_list)}: {self.labels[i]}")
            
            # Create graph from adjacency matrix
            G = nx.from_pandas_adjacency(df)
            
            # Remove self-loops
            G.remove_edges_from(nx.selfloop_edges(G))
            
            # Filter weak connections if needed
            if hasattr(self, 'min_weight') and self.min_weight > 0:
                edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) 
                                 if d.get('weight', 0) < self.min_weight]
                G.remove_edges_from(edges_to_remove)
            
            # Add graph statistics as attributes
            G.graph['name'] = self.labels[i]
            G.graph['nodes'] = G.number_of_nodes()
            G.graph['edges'] = G.number_of_edges()
            G.graph['density'] = nx.density(G)
            
            self.graph_list.append(G)
            
        return self.graph_list

    def visualize_graphs(self, output_dir="./", open_browser=False):
        """Convert each graph into a pyvis network and save as HTML."""
        if self.graph_list is None:
            self.create_graphs()
        
        output_files = []
        
        for i, G in enumerate(self.graph_list):
            print(f"Visualizing graph {i+1}/{len(self.graph_list)}: {self.labels[i]}")
            
            # Add edge titles for hover information
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 0)
                data['title'] = f"Weight: {weight:.3f}" if weight != int(weight) else f"Weight: {int(weight)}"
            
            # Create pyvis network
            net = Network(
                notebook=False, 
                cdn_resources='remote',
                height="600px",
                width="100%",
                bgcolor="#ffffff",
                font_color="black"
            )
            
            # Configure physics
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
              }
            }
            """)
            
            # Convert networkx to pyvis
            net.from_nx(G)
            
            # Save file
            filename = f"graph_{i}_{self.labels[i].lower().replace(' ', '_')}.html"
            filepath = os.path.join(output_dir, filename)
            net.save_graph(filepath)
            
            output_files.append(filepath)
            print(f"Saved visualization: {filename}")
            
            # Optionally open in browser
            if open_browser:
                webbrowser.open('file://' + os.path.realpath(filepath))
        
        return output_files

    def print_graph_statistics(self):
        """Print statistics for all graphs."""
        if self.graph_list is None:
            self.create_graphs()
        
        print("\nGraph Statistics:")
        print("=" * 50)
        
        for i, G in enumerate(self.graph_list):
            print(f"\n{self.labels[i]}:")
            print(f"  Nodes: {G.number_of_nodes()}")
            print(f"  Edges: {G.number_of_edges()}")
            print(f"  Density: {nx.density(G):.4f}")
            
            if G.number_of_edges() > 0:
                # Calculate additional metrics
                if nx.is_connected(G):
                    print(f"  Average path length: {nx.average_shortest_path_length(G):.3f}")
                    print(f"  Diameter: {nx.diameter(G)}")
                else:
                    print(f"  Connected components: {nx.number_connected_components(G)}")
                
                print(f"  Average clustering: {nx.average_clustering(G):.4f}")
                
                # Top nodes by degree
                degrees = dict(G.degree())
                top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top 5 nodes by degree: {top_nodes}")

def load_political_data():
    """Load the standard political data matrices."""
    data_files = {
        'IGO_scaled_important': '../results/preprocessed/IGO_adjmat_scaled_top1pct_important.csv',
        'DCAD_top5pct': '../results/preprocessed/DCAD_adjmat_top5pct.csv',
        'IGO_full': '../results/preprocessed/IGO_adjmat_scaled.csv',
        'DCAD_full': '../results/preprocessed/DCAD_adjmat.csv'
    }
    
    dataframes = []
    labels = []
    
    for label, filepath in data_files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0)
                dataframes.append(df)
                labels.append(label)
                print(f"Loaded {label}: {df.shape}")
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
        else:
            print(f"Warning: File not found: {filepath}")
    
    return dataframes, labels

def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize political network data')
    parser.add_argument('--output-dir', default='./', help='Output directory for HTML files')
    parser.add_argument('--open-browser', action='store_true', help='Open visualizations in browser')
    parser.add_argument('--stats-only', action='store_true', help='Only print statistics, don\'t create visualizations')
    parser.add_argument('--min-weight', type=float, default=0, help='Minimum edge weight to include')
    
    args = parser.parse_args()
    
    print("Starting political data visualization...")
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Load data
        print("Loading political data...")
        dataframes, labels = load_political_data()
        
        if not dataframes:
            print("Error: No data files found!")
            sys.exit(1)
        
        # Create visualizer
        visualizer = GraphVisualizer(dataframes, labels)
        visualizer.min_weight = args.min_weight
        
        # Print statistics
        visualizer.print_graph_statistics()
        
        if not args.stats_only:
            # Create visualizations
            print(f"\nCreating visualizations in {args.output_dir}...")
            os.makedirs(args.output_dir, exist_ok=True)
            
            output_files = visualizer.visualize_graphs(
                output_dir=args.output_dir,
                open_browser=args.open_browser
            )
            
            print(f"\nVisualization completed!")
            print(f"Created {len(output_files)} HTML files:")
            for filepath in output_files:
                print(f"  {filepath}")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 