#!/usr/bin/env python3
"""
Economic Data Visualization
Creates comprehensive visualizations for economic indicators and trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
import argparse
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import warnings
warnings.filterwarnings('ignore')

# Import unified color scheme
sys.path.append(str(Path(__file__).parent.parent.parent / 'configs'))
from color_schemes import get_country_color, get_country_colors, create_country_color_map

# Set style for matplotlib
plt.style.use('default')
# Remove the hardcoded seaborn palette - we'll use our unified colors instead

def load_and_prepare_data():
    """Load and prepare datasets for visualization."""
    datasets = {}
    
    data_files = {
        'WB_GEM': '../WB_GEM.csv',
        'IMF_BOP': '../IMF_BOP.csv',
        'IMF_CPI': '../IMF_CPI.csv',
        'IMF_MFS': '../IMF_MFS.csv'
    }
    
    for name, filepath in data_files.items():
        if os.path.exists(filepath):
            print(f"Loading {name}...")
            try:
                # Load with size limit for large files
                df = pd.read_csv(filepath, nrows=10000 if name == 'IMF_MFS' else None)
                datasets[name] = df
                print(f"  Shape: {df.shape}")
            except Exception as e:
                print(f"  Warning: Could not load {filepath}: {e}")
    
    return datasets

def parse_time_data(df, time_col):
    """Parse time column handling various formats."""
    if 'period' in time_col.lower():
        # Handle quarterly format
        def parse_quarter(q_str):
            if pd.isna(q_str):
                return pd.NaT
            try:
                if 'Q' in str(q_str):
                    year, quarter = str(q_str).split('-Q')
                    month = int(quarter) * 3 - 2
                    return pd.Timestamp(int(year), month, 1)
                else:
                    return pd.to_datetime(q_str, errors='coerce')
            except:
                return pd.NaT
        
        return df[time_col].apply(parse_quarter)
    else:
        return pd.to_datetime(df[time_col], errors='coerce')

def create_time_series_plots(datasets, output_dir="../results/"):
    """Create time series visualization plots."""
    
    print("\nCreating time series plots...")
    
    for dataset_name, df in datasets.items():
        print(f"  Processing {dataset_name}...")
        
        # Detect columns
        time_cols = [col for col in df.columns if any(word in col.lower() for word in ['period', 'date', 'time'])]
        country_cols = [col for col in df.columns if any(word in col.lower() for word in ['country', 'ref_area'])]
        value_cols = [col for col in df.columns if any(word in col.lower() for word in ['value', 'original_value'])]
        
        if not (time_cols and country_cols and value_cols):
            print(f"    Skipping {dataset_name}: missing required columns")
            continue
        
        time_col = time_cols[0]
        country_col = country_cols[0]
        value_col = value_cols[0]
        
        # Parse time and prepare data
        df_clean = df.copy()
        df_clean['parsed_time'] = parse_time_data(df_clean, time_col)
        df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
        df_clean = df_clean.dropna(subset=['parsed_time', value_col])
        
        if len(df_clean) == 0:
            continue
        
        # Get top countries by data availability
        country_counts = df_clean[country_col].value_counts()
        top_countries = country_counts.head(8).index.tolist()
        
        # Get unified colors for these countries
        country_color_map = create_country_color_map(top_countries, 'matplotlib')
        
        # Create time series plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Economic Indicators Over Time: {dataset_name}', fontsize=16)
        
        # 1. Time series for top countries
        ax1 = axes[0, 0]
        for country in top_countries[:5]:
            country_data = df_clean[df_clean[country_col] == country].sort_values('parsed_time')
            # Ensure both time and value columns have valid data and same length
            country_data = country_data.dropna(subset=['parsed_time', value_col])
            if len(country_data) > 1:
                try:
                    # Additional safety check to ensure arrays have same length
                    x_data = country_data['parsed_time'].values
                    y_data = country_data[value_col].values
                    if len(x_data) == len(y_data) and len(x_data) > 1:
                        # Use unified color scheme
                        color = country_color_map.get(country, '#1f77b4')  # Default fallback
                        ax1.plot(x_data, y_data, label=country, alpha=0.8, linewidth=2, color=color)
                except Exception as e:
                    print(f"    Warning: Could not plot {country}: {e}")
                    continue
        
        ax1.set_title('Time Series by Country (Top 5)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Annual trends (if enough data)
        ax2 = axes[0, 1]
        df_clean['year'] = df_clean['parsed_time'].dt.year
        yearly_data = df_clean.groupby('year')[value_col].agg(['mean', 'std']).reset_index()
        
        if len(yearly_data) > 1:
            ax2.errorbar(yearly_data['year'], yearly_data['mean'], 
                        yerr=yearly_data['std'], capsize=5, alpha=0.7, color='#2ca02c')
            ax2.set_title('Annual Trends (Mean ± Std)')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Value')
            ax2.grid(True, alpha=0.3)
        
        # 3. Distribution over time periods
        ax3 = axes[1, 0]
        recent_years = df_clean['year'].max() - np.arange(5)
        recent_years = recent_years[recent_years >= df_clean['year'].min()]
        
        recent_data = df_clean[df_clean['year'].isin(recent_years)]
        if len(recent_data) > 0:
            ax3.boxplot([recent_data[recent_data['year'] == year][value_col].dropna() 
                        for year in sorted(recent_years)], 
                       labels=sorted(recent_years))
            ax3.set_title('Value Distribution (Recent Years)')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Value')
            ax3.grid(True, alpha=0.3)
        
        # 4. Growth rates
        ax4 = axes[1, 1]
        # Calculate growth rates for top countries
        for country in top_countries[:3]:
            country_data = df_clean[df_clean[country_col] == country].sort_values('parsed_time')
            # Ensure clean data for growth rate calculation
            country_data = country_data.dropna(subset=['parsed_time', value_col])
            if len(country_data) > 2:  # Need at least 3 points for meaningful growth rates
                try:
                    # Calculate growth rates and align with time data
                    country_data = country_data.reset_index(drop=True)
                    growth_rates = country_data[value_col].pct_change()
                    
                    # Remove first row (NaN from pct_change) and any remaining NaN values
                    valid_mask = ~growth_rates.isna()
                    if valid_mask.sum() > 0:
                        time_data = country_data['parsed_time'][valid_mask]
                        growth_data = growth_rates[valid_mask]
                        
                        if len(time_data) == len(growth_data) and len(time_data) > 0:
                            # Use unified color scheme
                            color = country_color_map.get(country, '#1f77b4')  # Default fallback
                            ax4.plot(time_data, growth_data, 
                                   label=f'{country} Growth Rate', alpha=0.7, color=color)
                except Exception as e:
                    print(f"    Warning: Could not plot growth rates for {country}: {e}")
                    continue
        
        ax4.set_title('Growth Rates')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Growth Rate')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'time_series_{dataset_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {plot_path}")
        
        plt.close()

def create_interactive_plots(datasets, output_dir="../results/"):
    """Create interactive plotly visualizations."""
    
    print("\nCreating interactive plots...")
    
    # Create interactive_plots subdirectory
    interactive_dir = os.path.join(output_dir, "interactive_plots")
    os.makedirs(interactive_dir, exist_ok=True)
    
    for dataset_name, df in datasets.items():
        print(f"  Creating interactive plot for {dataset_name}...")
        
        # Detect columns
        time_cols = [col for col in df.columns if any(word in col.lower() for word in ['period', 'date', 'time'])]
        country_cols = [col for col in df.columns if any(word in col.lower() for word in ['country', 'ref_area'])]
        value_cols = [col for col in df.columns if any(word in col.lower() for word in ['value', 'original_value'])]
        
        if not (time_cols and country_cols and value_cols):
            continue
        
        time_col = time_cols[0]
        country_col = country_cols[0]
        value_col = value_cols[0]
        
        # Prepare data
        df_clean = df.copy()
        df_clean['parsed_time'] = parse_time_data(df_clean, time_col)
        df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
        df_clean = df_clean.dropna(subset=['parsed_time', value_col])
        
        if len(df_clean) == 0:
            continue
        
        # Get top countries by data availability
        country_counts = df_clean[country_col].value_counts()
        top_countries = country_counts.head(10).index.tolist()
        df_top = df_clean[df_clean[country_col].isin(top_countries)]
        
        # Get unified colors for these countries
        country_color_map = create_country_color_map(top_countries, 'plotly')
        
        # Create interactive time series plot
        fig = go.Figure()
        
        for country in top_countries:
            country_data = df_top[df_top[country_col] == country].sort_values('parsed_time')
            if len(country_data) > 0:
                # Use unified color scheme
                color = country_color_map.get(country, 'rgb(31, 119, 180)')  # Default fallback
                fig.add_trace(go.Scatter(
                    x=country_data['parsed_time'],
                    y=country_data[value_col],
                    mode='lines+markers',
                    name=country,
                    line=dict(width=2, color=color),
                    hovertemplate=f'<b>{country}</b><br>' +
                                  'Date: %{x}<br>' +
                                  'Value: %{y:.2f}<br>' +
                                  '<extra></extra>'
                ))
        
        fig.update_layout(
            title=f'Interactive Time Series: {dataset_name}',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            width=1000,
            height=600
        )
        
        # Save interactive plot to interactive_plots subdirectory
        html_path = os.path.join(interactive_dir, f'interactive_timeseries_{dataset_name.lower()}.html')
        pyo.plot(fig, filename=html_path, auto_open=False)
        print(f"    Saved: {html_path}")

def create_correlation_analysis(datasets, output_dir="../results/"):
    """Create correlation analysis between different economic indicators."""
    
    print("\nCreating correlation analysis...")
    
    # Try to merge datasets for correlation analysis
    merged_data = None
    
    for dataset_name, df in datasets.items():
        # Detect columns
        time_cols = [col for col in df.columns if any(word in col.lower() for word in ['period', 'date', 'time'])]
        country_cols = [col for col in df.columns if any(word in col.lower() for word in ['country', 'ref_area'])]
        value_cols = [col for col in df.columns if any(word in col.lower() for word in ['value', 'original_value'])]
        
        if not (time_cols and country_cols and value_cols):
            continue
        
        time_col = time_cols[0]
        country_col = country_cols[0]
        value_col = value_cols[0]
        
        # Prepare data
        df_clean = df.copy()
        df_clean['parsed_time'] = parse_time_data(df_clean, time_col)
        df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
        df_clean = df_clean.dropna(subset=['parsed_time', value_col])
        
        if len(df_clean) == 0:
            continue
        
        # Create year column and aggregate by country-year
        df_clean['year'] = df_clean['parsed_time'].dt.year
        yearly_data = df_clean.groupby([country_col, 'year'])[value_col].mean().reset_index()
        yearly_data = yearly_data.rename(columns={value_col: f'{dataset_name}_value'})
        
        if merged_data is None:
            merged_data = yearly_data
        else:
            merged_data = merged_data.merge(yearly_data, on=[country_col, 'year'], how='outer')
    
    if merged_data is not None and len(merged_data.columns) > 3:
        # Create correlation matrix
        value_columns = [col for col in merged_data.columns if col.endswith('_value')]
        
        if len(value_columns) > 1:
            corr_matrix = merged_data[value_columns].corr()
            
            # Plot correlation heatmap with unified color scheme
            plt.figure(figsize=(10, 8))
            # Use a professional colormap that's accessibility-friendly
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
            plt.title('Correlation Between Economic Indicators', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            corr_path = os.path.join(output_dir, 'correlation_matrix.png')
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            print(f"  Saved correlation matrix: {corr_path}")
            
            plt.close()

def create_dashboard_summary(datasets, output_dir="../results/"):
    """Create a summary dashboard with key visualizations."""
    
    print("\nCreating summary dashboard...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Economic Data Dashboard', fontsize=20)
    
    dataset_stats = {}
    
    for i, (dataset_name, df) in enumerate(datasets.items()):
        if i >= 6:  # Limit to 6 datasets for layout
            break
            
        ax = axes[i // 3, i % 3]
        
        # Detect columns
        time_cols = [col for col in df.columns if any(word in col.lower() for word in ['period', 'date', 'time'])]
        country_cols = [col for col in df.columns if any(word in col.lower() for word in ['country', 'ref_area'])]
        value_cols = [col for col in df.columns if any(word in col.lower() for word in ['value', 'original_value'])]
        
        if time_cols and country_cols and value_cols:
            value_col = value_cols[0]
            country_col = country_cols[0]
            
            values = pd.to_numeric(df[value_col], errors='coerce').dropna()
            
            if len(values) > 0:
                # Create histogram with unified color scheme
                # Use a consistent color for histograms
                ax.hist(values, bins=30, alpha=0.7, edgecolor='black', color='#2ca02c')
                ax.set_title(f'{dataset_name}\n({len(df[country_col].unique())} countries)')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Store stats
                dataset_stats[dataset_name] = {
                    'countries': len(df[country_col].unique()),
                    'records': len(df),
                    'mean': values.mean(),
                    'std': values.std()
                }
            else:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(dataset_name)
        else:
            ax.text(0.5, 0.5, 'Missing columns', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(dataset_name)
    
    # Remove empty subplots
    for j in range(len(datasets), 6):
        axes[j // 3, j % 3].remove()
    
    plt.tight_layout()
    
    # Save dashboard
    dashboard_path = os.path.join(output_dir, 'economic_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    print(f"  Saved dashboard: {dashboard_path}")
    
    plt.close()
    
    # Save summary statistics
    if dataset_stats:
        summary_path = os.path.join(output_dir, 'dataset_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Economic Datasets Summary\n")
            f.write("="*40 + "\n\n")
            
            for name, stats in dataset_stats.items():
                f.write(f"{name}:\n")
                f.write(f"  Countries: {stats.get('countries', 'N/A')}\n")
                f.write(f"  Records: {stats.get('records', 'N/A')}\n")
                f.write(f"  Mean value: {stats.get('mean', 0):.2f}\n")
                f.write(f"  Std deviation: {stats.get('std', 0):.2f}\n\n")
        
        print(f"  Saved summary: {summary_path}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Create visualizations for economic data')
    parser.add_argument('--output-dir', default='../results/', help='Output directory for visualizations')
    parser.add_argument('--interactive', action='store_true', help='Create interactive plots (requires plotly)')
    parser.add_argument('--static-only', action='store_true', help='Create only static plots')
    
    args = parser.parse_args()
    
    print("Starting economic data visualization...")
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load datasets
        datasets = load_and_prepare_data()
        
        if not datasets:
            print("Error: No datasets found!")
            sys.exit(1)
        
        print(f"\nLoaded {len(datasets)} datasets for visualization")
        
        # Create static plots
        if not args.interactive:
            create_time_series_plots(datasets, args.output_dir)
            create_correlation_analysis(datasets, args.output_dir)
            create_dashboard_summary(datasets, args.output_dir)
        
        # Create interactive plots if requested
        if args.interactive and not args.static_only:
            try:
                create_interactive_plots(datasets, args.output_dir)
            except ImportError:
                print("Warning: plotly not available for interactive plots")
            except Exception as e:
                print(f"Warning: Could not create interactive plots: {e}")
        
        print(f"\nEconomic data visualization completed!")
        print(f"Results saved in: {args.output_dir}")
        
        # Print summary
        total_plots = len(datasets) * (3 if not args.static_only else 2)
        print(f"\nSummary:")
        print(f"- Created visualizations for {len(datasets)} datasets")
        print(f"- Generated approximately {total_plots} plot files")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 