#!/usr/bin/env python3
"""
Unified Color Scheme Demonstration Script

This script demonstrates how to use the unified color scheme across all
visualization types in the SOUL project, including matplotlib, plotly,
and UI widgets.

Run this script to:
1. See the color palette in action
2. Test color consistency across different visualization libraries
3. Generate example plots with unified colors
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo

# Add configs directory to path
sys.path.append(str(Path(__file__).parent.parent / 'configs'))
from color_schemes import (
    get_country_color, 
    get_country_colors, 
    create_country_color_map,
    color_scheme,
    COUNTRY_REGION_MAP
)

def demo_color_palette():
    """Demonstrate the color palette for important countries."""
    print("=== Unified Color Scheme Demo ===\n")
    
    important_countries = ['USA', 'CHN', 'JPN', 'GER', 'UKG', 'FRN', 'RUS', 'KOR', 'IND', 'AUL']
    
    print("Country Colors:")
    for country in important_countries:
        hex_color = get_country_color(country, 'hex')
        rgb_color = get_country_color(country, 'rgb')
        plotly_color = get_country_color(country, 'plotly')
        print(f"  {country:3}: {hex_color:7} | RGB: {rgb_color} | Plotly: {plotly_color}")
    
    print(f"\nRegional Colors:")
    for region, color in color_scheme.regional_colors.items():
        print(f"  {region:12}: {color}")
    
    return important_countries

def create_matplotlib_demo(countries, output_dir="./results/"):
    """Create matplotlib demonstration plots."""
    print(f"\nCreating matplotlib demo plots...")
    
    # Create sample economic data
    np.random.seed(42)
    years = np.arange(2010, 2021)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Unified Color Scheme - Matplotlib Demo', fontsize=16, fontweight='bold')
    
    # Get country colors
    country_colors = get_country_colors(countries, 'matplotlib')
    
    # 1. Time series plot
    ax1 = axes[0, 0]
    for i, country in enumerate(countries[:5]):
        gdp_data = 1000 + np.cumsum(np.random.normal(10, 30, len(years)))
        ax1.plot(years, gdp_data, label=country, color=country_colors[i], linewidth=2, alpha=0.8)
    
    ax1.set_title('Economic Indicators Over Time')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GDP (Billions USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Bar chart
    ax2 = axes[0, 1]
    trade_volumes = np.random.uniform(50, 200, len(countries[:8]))
    bars = ax2.bar(countries[:8], trade_volumes, color=country_colors[:8], alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    ax2.set_title('Trade Volumes by Country')
    ax2.set_xlabel('Country')
    ax2.set_ylabel('Trade Volume')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot (PCA-style)
    ax3 = axes[1, 0]
    x_coords = np.random.normal(0, 1, len(countries))
    y_coords = np.random.normal(0, 1, len(countries))
    
    scatter = ax3.scatter(x_coords, y_coords, c=country_colors[:len(countries)], 
                         s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    for i, country in enumerate(countries):
        ax3.annotate(country, (x_coords[i], y_coords[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax3.set_title('Country Clustering (Demo)')
    ax3.set_xlabel('Principal Component 1')
    ax3.set_ylabel('Principal Component 2')
    ax3.grid(True, alpha=0.3)
    
    # 4. Regional analysis
    ax4 = axes[1, 1]
    regions = list(set(COUNTRY_REGION_MAP.get(c, 'Other') for c in countries))
    region_data = [sum(1 for c in countries if COUNTRY_REGION_MAP.get(c, 'Other') == region) for region in regions]
    regional_colors = [color_scheme.get_regional_color(region, 'matplotlib') for region in regions]
    
    wedges, texts, autotexts = ax4.pie(region_data, labels=regions, colors=regional_colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax4.set_title('Regional Distribution')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'unified_colors_matplotlib_demo.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    
    plt.close()

def create_plotly_demo(countries, output_dir="./results/"):
    """Create plotly demonstration plots."""
    print(f"\nCreating plotly demo plots...")
    
    # Create sample data
    np.random.seed(42)
    years = list(range(2010, 2021))
    
    # Get country colors for plotly
    country_colors = get_country_colors(countries, 'plotly')
    
    # Create interactive time series
    fig = go.Figure()
    
    for i, country in enumerate(countries[:5]):
        gdp_data = 1000 + np.cumsum(np.random.normal(10, 30, len(years)))
        fig.add_trace(go.Scatter(
            x=years,
            y=gdp_data,
            mode='lines+markers',
            name=country,
            line=dict(color=country_colors[i], width=3),
            hovertemplate=f'<b>{country}</b><br>' +
                         'Year: %{x}<br>' +
                         'GDP: $%{y:.1f}B<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title='Economic Indicators Over Time (Interactive)',
        xaxis_title='Year',
        yaxis_title='GDP (Billions USD)',
        hovermode='x unified',
        width=1000,
        height=600,
        font=dict(size=12)
    )
    
    # Save interactive plot
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, 'unified_colors_plotly_demo.html')
    pyo.plot(fig, filename=html_path, auto_open=False)
    print(f"  Saved: {html_path}")

def create_comparison_table(countries):
    """Create a comparison table showing color consistency."""
    print(f"\nColor Consistency Table:")
    print(f"{'Country':<8} {'Hex':<8} {'Matplotlib RGB':<20} {'Plotly':<20}")
    print("-" * 70)
    
    for country in countries:
        hex_color = get_country_color(country, 'hex')
        rgb_color = get_country_color(country, 'matplotlib')
        plotly_color = get_country_color(country, 'plotly')
        
        rgb_str = f"({rgb_color[0]:.2f}, {rgb_color[1]:.2f}, {rgb_color[2]:.2f})"
        print(f"{country:<8} {hex_color:<8} {rgb_str:<20} {plotly_color:<20}")

def demo_ui_integration():
    """Demonstrate how to integrate with UI widgets."""
    print(f"\nUI Integration Example:")
    print("To use unified colors in UI widgets, update your calls:")
    print("\n# Before:")
    print("widget.ShowPlot(values, N, custom_colors)")
    print("\n# After:")
    print("countries = ['USA', 'CHN', 'JPN']")
    print("widget.ShowPlot(values, N, countries=countries)")
    print("\n# Or manually get colors:")
    print("colors = get_country_colors(countries, 'matplotlib')")
    print("widget.ShowPlot(values, N, colors=colors)")

def test_fallback_colors():
    """Test fallback colors for unknown countries."""
    print(f"\nTesting fallback colors for unknown countries:")
    
    unknown_countries = ['XYZ', 'ABC', 'DEF']
    for country in unknown_countries:
        color = get_country_color(country)
        print(f"  {country}: {color} (fallback)")

def main():
    """Run the complete demonstration."""
    output_dir = Path(__file__).parent.parent / 'data' / 'results' / 'color_demo'
    
    # Demo color palette
    countries = demo_color_palette()
    
    # Create matplotlib demo
    create_matplotlib_demo(countries, output_dir)
    
    # Create plotly demo
    create_plotly_demo(countries, output_dir)
    
    # Show comparison table
    create_comparison_table(countries[:5])
    
    # Demo UI integration
    demo_ui_integration()
    
    # Test fallback colors
    test_fallback_colors()
    
    print(f"\n=== Demo Complete ===")
    print(f"Check the output directory for generated plots: {output_dir}")
    print(f"\nTo use the unified color scheme in your code:")
    print(f"  from configs.color_schemes import get_country_color, get_country_colors")
    print(f"  colors = get_country_colors(['USA', 'CHN', 'JPN'], 'matplotlib')")

if __name__ == "__main__":
    main() 