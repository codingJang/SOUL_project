#!/usr/bin/env python3
"""
Economic Data Exploration
Analyzes and summarizes the structure and content of economic datasets.
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

def load_datasets():
    """Load all available economic datasets."""
    datasets = {}
    
    # Main datasets
    data_files = {
        'main': '../df.csv',
        'main_scaled': '../df_scaled.csv',
        'train': '../train_df.csv',
        'test': '../test_df.csv',
        'val': '../val_df.csv',
        'wb_gem': '../WB_GEM.csv',
        'imf_bop': '../IMF_BOP.csv',
        'imf_cpi': '../IMF_CPI.csv',
        'imf_mfs': '../IMF_MFS.csv'
    }
    
    for name, filepath in data_files.items():
        if os.path.exists(filepath):
            try:
                print(f"Loading {name}...")
                
                # For large files, load only a sample first
                if name in ['main', 'main_scaled', 'train']:
                    df = pd.read_csv(filepath, nrows=1000)  # Sample first
                    print(f"  Loaded sample of {name}: {df.shape}")
                else:
                    df = pd.read_csv(filepath)
                    print(f"  Loaded {name}: {df.shape}")
                
                datasets[name] = df
                
            except Exception as e:
                print(f"  Warning: Could not load {filepath}: {e}")
        else:
            print(f"  Warning: File not found: {filepath}")
    
    return datasets

def analyze_dataset_structure(df, name):
    """Analyze the structure of a dataset."""
    analysis = {
        'name': name,
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        analysis['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        analysis['categorical_summary'] = {}
        for col in categorical_cols:
            unique_count = df[col].nunique()
            analysis['categorical_summary'][col] = {
                'unique_count': unique_count,
                'top_values': df[col].value_counts().head().to_dict() if unique_count < 100 else None
            }
    
    return analysis

def detect_time_series_columns(df):
    """Detect potential time series columns."""
    time_cols = []
    
    for col in df.columns:
        if any(time_word in col.lower() for time_word in ['date', 'time', 'year', 'period', 'quarter']):
            time_cols.append(col)
        
        # Try to parse as datetime
        try:
            pd.to_datetime(df[col].dropna().head(100))
            time_cols.append(col)
        except:
            pass
    
    return list(set(time_cols))

def detect_country_columns(df):
    """Detect potential country identifier columns."""
    country_cols = []
    
    for col in df.columns:
        if any(country_word in col.lower() for country_word in ['country', 'nation', 'ref_area', 'cty', 'ctry']):
            country_cols.append(col)
    
    return country_cols

def detect_indicator_columns(df):
    """Detect potential economic indicator columns."""
    indicator_cols = []
    
    for col in df.columns:
        if any(indicator_word in col.lower() for indicator_word in 
               ['gdp', 'inflation', 'cpi', 'unemployment', 'trade', 'export', 'import', 
                'balance', 'indicator', 'value', 'rate', 'index']):
            indicator_cols.append(col)
    
    return indicator_cols

def create_data_summary_report(datasets, output_dir="./"):
    """Create a comprehensive data summary report."""
    
    print("\nCreating data summary report...")
    
    report = {
        'generation_time': datetime.now().isoformat(),
        'datasets': {}
    }
    
    for name, df in datasets.items():
        print(f"  Analyzing {name}...")
        
        analysis = analyze_dataset_structure(df, name)
        analysis['time_columns'] = detect_time_series_columns(df)
        analysis['country_columns'] = detect_country_columns(df)
        analysis['indicator_columns'] = detect_indicator_columns(df)
        
        report['datasets'][name] = analysis
    
    # Save detailed report
    report_path = os.path.join(output_dir, 'data_summary_report.txt')
    with open(report_path, 'w') as f:
        f.write("Economic Data Summary Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {report['generation_time']}\n\n")
        
        for name, analysis in report['datasets'].items():
            f.write(f"\nDataset: {analysis['name']}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Shape: {analysis['shape']}\n")
            f.write(f"Memory Usage: {analysis['memory_usage']:.2f} MB\n")
            f.write(f"Columns: {len(analysis['columns'])}\n")
            
            # Time series info
            if analysis['time_columns']:
                f.write(f"Time columns: {analysis['time_columns']}\n")
            
            # Country info  
            if analysis['country_columns']:
                f.write(f"Country columns: {analysis['country_columns']}\n")
            
            # Indicator info
            if analysis['indicator_columns']:
                f.write(f"Indicator columns: {analysis['indicator_columns'][:10]}...")
                if len(analysis['indicator_columns']) > 10:
                    f.write(f" (+{len(analysis['indicator_columns'])-10} more)")
                f.write("\n")
            
            # Missing data
            null_cols = {col: pct for col, pct in analysis['null_percentage'].items() if pct > 0}
            if null_cols:
                f.write(f"Missing data: {len(null_cols)} columns have missing values\n")
                high_missing = {col: pct for col, pct in null_cols.items() if pct > 50}
                if high_missing:
                    f.write(f"High missing (>50%): {list(high_missing.keys())}\n")
            
            f.write("\n")
    
    print(f"  Saved detailed report: {report_path}")
    
    return report

def plot_dataset_overview(datasets, output_dir="./"):
    """Create overview plots of the datasets."""
    
    print("\nCreating overview plots...")
    
    # Dataset sizes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Dataset sizes
    names = list(datasets.keys())
    sizes = [df.shape[0] * df.shape[1] for df in datasets.values()]
    rows = [df.shape[0] for df in datasets.values()]
    cols = [df.shape[1] for df in datasets.values()]
    
    ax1.bar(names, sizes)
    ax1.set_title('Dataset Sizes (Total Elements)')
    ax1.set_ylabel('Total Elements')
    ax1.tick_params(axis='x', rotation=45)
    
    # Rows vs Columns
    ax2.scatter(rows, cols, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax2.annotate(name, (rows[i], cols[i]), xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Number of Rows')
    ax2.set_ylabel('Number of Columns')
    ax2.set_title('Dataset Dimensions')
    
    # Missing data percentages
    missing_data = {}
    for name, df in datasets.items():
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        missing_data[name] = missing_pct
    
    ax3.bar(missing_data.keys(), missing_data.values())
    ax3.set_title('Missing Data Percentage')
    ax3.set_ylabel('% Missing')
    ax3.tick_params(axis='x', rotation=45)
    
    # Memory usage
    memory_usage = {}
    for name, df in datasets.items():
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        memory_usage[name] = memory_mb
    
    ax4.bar(memory_usage.keys(), memory_usage.values())
    ax4.set_title('Memory Usage (MB)')
    ax4.set_ylabel('Memory (MB)')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'dataset_overview.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved overview plot: {plot_path}")
    
    return plot_path

def analyze_time_series_data(datasets, output_dir="./"):
    """Analyze time series patterns in the datasets."""
    
    print("\nAnalyzing time series patterns...")
    
    time_series_info = {}
    
    for name, df in datasets.items():
        time_cols = detect_time_series_columns(df)
        
        if time_cols:
            print(f"  Found time series in {name}: {time_cols}")
            
            for time_col in time_cols:
                try:
                    # Try to parse the time column
                    if 'period' in time_col.lower():
                        # Handle quarterly data like "1987-Q1"
                        time_series = pd.to_datetime(df[time_col], errors='coerce')
                    else:
                        time_series = pd.to_datetime(df[time_col], errors='coerce')
                    
                    if time_series.notna().sum() > 0:
                        min_date = time_series.min()
                        max_date = time_series.max()
                        
                        time_series_info[f"{name}_{time_col}"] = {
                            'dataset': name,
                            'column': time_col,
                            'min_date': min_date,
                            'max_date': max_date,
                            'date_range_years': (max_date - min_date).days / 365.25 if pd.notna(min_date) and pd.notna(max_date) else None,
                            'valid_dates': time_series.notna().sum(),
                            'total_records': len(df)
                        }
                        
                except Exception as e:
                    print(f"    Could not parse {time_col} in {name}: {e}")
    
    # Save time series analysis
    if time_series_info:
        ts_path = os.path.join(output_dir, 'time_series_analysis.txt')
        with open(ts_path, 'w') as f:
            f.write("Time Series Analysis\n")
            f.write("="*30 + "\n\n")
            
            for ts_name, info in time_series_info.items():
                f.write(f"{ts_name}:\n")
                f.write(f"  Dataset: {info['dataset']}\n")
                f.write(f"  Column: {info['column']}\n")
                f.write(f"  Date range: {info['min_date']} to {info['max_date']}\n")
                f.write(f"  Years covered: {info['date_range_years']:.1f}\n" if info['date_range_years'] else "  Years covered: N/A\n")
                f.write(f"  Valid dates: {info['valid_dates']}/{info['total_records']} ({info['valid_dates']/info['total_records']*100:.1f}%)\n\n")
        
        print(f"  Saved time series analysis: {ts_path}")
    
    return time_series_info

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Explore economic datasets')
    parser.add_argument('--output-dir', default='./', help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true', help='Skip creating plots')
    
    args = parser.parse_args()
    
    print("Starting economic data exploration...")
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load datasets
        print("Loading economic datasets...")
        datasets = load_datasets()
        
        if not datasets:
            print("Error: No datasets found!")
            sys.exit(1)
        
        print(f"\nLoaded {len(datasets)} datasets")
        
        # Create summary report
        report = create_data_summary_report(datasets, args.output_dir)
        
        # Analyze time series
        time_series_info = analyze_time_series_data(datasets, args.output_dir)
        
        # Create plots if requested
        if not args.no_plots:
            plot_path = plot_dataset_overview(datasets, args.output_dir)
        
        print(f"\nData exploration completed!")
        print(f"Results saved in: {args.output_dir}")
        
        # Print summary
        print(f"\nSummary:")
        for name, df in datasets.items():
            print(f"- {name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        if time_series_info:
            print(f"\nTime series detected in {len(time_series_info)} column(s)")
        
    except Exception as e:
        print(f"Error during data exploration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 