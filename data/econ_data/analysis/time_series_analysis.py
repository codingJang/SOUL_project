#!/usr/bin/env python3
"""
Time Series Analysis for Economic Data
Analyzes temporal patterns, trends, and seasonality in economic indicators.
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
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare time series data."""
    
    # Try to load different datasets
    datasets = {}
    
    # World Bank GEM data (quarterly GDP data)
    if os.path.exists('../WB_GEM.csv'):
        print("Loading World Bank GEM data...")
        wb_df = pd.read_csv('../WB_GEM.csv')
        datasets['WB_GEM'] = wb_df
        print(f"  Shape: {wb_df.shape}")
    
    # IMF datasets
    imf_files = {
        'IMF_BOP': '../IMF_BOP.csv',
        'IMF_CPI': '../IMF_CPI.csv', 
        'IMF_MFS': '../IMF_MFS.csv'
    }
    
    for name, filepath in imf_files.items():
        if os.path.exists(filepath):
            print(f"Loading {name}...")
            try:
                df = pd.read_csv(filepath)
                datasets[name] = df
                print(f"  Shape: {df.shape}")
            except Exception as e:
                print(f"  Warning: Could not load {filepath}: {e}")
    
    return datasets

def parse_time_column(df, time_col):
    """Parse various time column formats."""
    
    if 'period' in time_col.lower():
        # Handle quarterly format like "1987-Q1"
        try:
            return pd.to_datetime(df[time_col])
        except:
            # Try to parse manually
            def parse_quarter(q_str):
                if pd.isna(q_str):
                    return pd.NaT
                try:
                    year, quarter = q_str.split('-Q')
                    month = int(quarter) * 3 - 2  # Q1->1, Q2->4, Q3->7, Q4->10
                    return pd.Timestamp(int(year), month, 1)
                except:
                    return pd.NaT
            
            return df[time_col].apply(parse_quarter)
    else:
        return pd.to_datetime(df[time_col], errors='coerce')

def analyze_single_time_series(df, time_col, value_col, country_col=None, indicator_col=None):
    """Analyze a single time series."""
    
    # Parse time column
    time_series = parse_time_column(df, time_col)
    
    # Create analysis dataframe
    analysis_df = pd.DataFrame({
        'time': time_series,
        'value': pd.to_numeric(df[value_col], errors='coerce')
    })
    
    if country_col and country_col in df.columns:
        analysis_df['country'] = df[country_col]
    
    if indicator_col and indicator_col in df.columns:
        analysis_df['indicator'] = df[indicator_col]
    
    # Remove rows with missing time or value
    analysis_df = analysis_df.dropna(subset=['time', 'value'])
    
    if len(analysis_df) == 0:
        return None
    
    # Sort by time
    analysis_df = analysis_df.sort_values('time')
    
    return analysis_df

def calculate_time_series_statistics(ts_df, value_col='value'):
    """Calculate basic time series statistics."""
    
    values = ts_df[value_col]
    
    stats_dict = {
        'count': len(values),
        'mean': values.mean(),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),
        'range': values.max() - values.min(),
        'coefficient_of_variation': values.std() / values.mean() if values.mean() != 0 else np.nan
    }
    
    # Calculate growth rates
    if len(values) > 1:
        growth_rates = values.pct_change().dropna()
        stats_dict.update({
            'mean_growth_rate': growth_rates.mean(),
            'std_growth_rate': growth_rates.std(),
            'volatility': growth_rates.std() * np.sqrt(len(growth_rates))  # Annualized volatility proxy
        })
    
    # Trend analysis
    if len(values) > 2:
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        stats_dict.update({
            'trend_slope': slope,
            'trend_r_squared': r_value**2,
            'trend_p_value': p_value
        })
    
    return stats_dict

def analyze_country_time_series(datasets, output_dir="./"):
    """Analyze time series by country."""
    
    print("\nAnalyzing country-level time series...")
    
    country_analyses = {}
    
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
        
        # Parse time series
        ts_df = analyze_single_time_series(df, time_col, value_col, country_col)
        
        if ts_df is None or len(ts_df) == 0:
            print(f"    No valid time series data in {dataset_name}")
            continue
        
        # Group by country and analyze
        countries = ts_df['country'].unique()
        print(f"    Found {len(countries)} countries")
        
        dataset_analysis = {}
        
        for country in countries[:20]:  # Limit to first 20 countries for performance
            country_data = ts_df[ts_df['country'] == country].copy()
            
            if len(country_data) < 3:  # Need at least 3 points for meaningful analysis
                continue
            
            country_stats = calculate_time_series_statistics(country_data)
            country_stats['dataset'] = dataset_name
            country_stats['time_span_years'] = (country_data['time'].max() - country_data['time'].min()).days / 365.25
            
            dataset_analysis[country] = country_stats
        
        if dataset_analysis:
            country_analyses[dataset_name] = dataset_analysis
    
    # Save country analysis results
    if country_analyses:
        results_path = os.path.join(output_dir, 'country_time_series_analysis.txt')
        with open(results_path, 'w') as f:
            f.write("Country Time Series Analysis\n")
            f.write("="*40 + "\n\n")
            
            for dataset_name, countries in country_analyses.items():
                f.write(f"\nDataset: {dataset_name}\n")
                f.write("-" * 30 + "\n")
                
                # Summary statistics across countries
                all_means = [stats['mean'] for stats in countries.values() if not np.isnan(stats['mean'])]
                all_volatilities = [stats.get('volatility', np.nan) for stats in countries.values() if not np.isnan(stats.get('volatility', np.nan))]
                
                if all_means:
                    f.write(f"Countries analyzed: {len(countries)}\n")
                    f.write(f"Mean value range: {min(all_means):.2f} to {max(all_means):.2f}\n")
                    f.write(f"Average volatility: {np.mean(all_volatilities):.4f}\n" if all_volatilities else "")
                    f.write("\n")
                
                # Top countries by various metrics
                if len(countries) > 5:
                    # Highest mean values
                    top_by_mean = sorted(countries.items(), key=lambda x: x[1]['mean'], reverse=True)[:5]
                    f.write("Top 5 countries by mean value:\n")
                    for country, stats in top_by_mean:
                        f.write(f"  {country}: {stats['mean']:.2f}\n")
                    f.write("\n")
                    
                    # Highest volatility
                    volatile_countries = [(c, s) for c, s in countries.items() if 'volatility' in s and not np.isnan(s['volatility'])]
                    if volatile_countries:
                        top_by_volatility = sorted(volatile_countries, key=lambda x: x[1]['volatility'], reverse=True)[:5]
                        f.write("Top 5 most volatile countries:\n")
                        for country, stats in top_by_volatility:
                            f.write(f"  {country}: {stats['volatility']:.4f}\n")
                        f.write("\n")
        
        print(f"  Saved country analysis: {results_path}")
    
    return country_analyses

def plot_time_series_overview(datasets, output_dir="./"):
    """Create overview plots of time series data."""
    
    print("\nCreating time series overview plots...")
    
    for dataset_name, df in datasets.items():
        print(f"  Plotting {dataset_name}...")
        
        # Detect columns
        time_cols = [col for col in df.columns if any(word in col.lower() for word in ['period', 'date', 'time'])]
        country_cols = [col for col in df.columns if any(word in col.lower() for word in ['country', 'ref_area'])]
        value_cols = [col for col in df.columns if any(word in col.lower() for word in ['value', 'original_value'])]
        
        if not (time_cols and country_cols and value_cols):
            continue
        
        time_col = time_cols[0]
        country_col = country_cols[0]
        value_col = value_cols[0]
        
        # Parse time series
        ts_df = analyze_single_time_series(df, time_col, value_col, country_col)
        
        if ts_df is None or len(ts_df) == 0:
            continue
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Time Series Analysis: {dataset_name}', fontsize=16)
        
        # 1. Time series for top countries
        countries = ts_df['country'].value_counts().head(5).index
        ax1 = axes[0, 0]
        
        for country in countries:
            country_data = ts_df[ts_df['country'] == country]
            if len(country_data) > 1:
                ax1.plot(country_data['time'], country_data['value'], label=country, alpha=0.7)
        
        ax1.set_title('Time Series by Country (Top 5)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution of values
        ax2 = axes[0, 1]
        ax2.hist(ts_df['value'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Values')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Time coverage by country
        ax3 = axes[1, 0]
        country_coverage = ts_df.groupby('country')['time'].agg(['min', 'max', 'count']).sort_values('count', ascending=False).head(10)
        
        if len(country_coverage) > 0:
            y_pos = np.arange(len(country_coverage))
            ax3.barh(y_pos, country_coverage['count'])
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(country_coverage.index)
            ax3.set_title('Data Points by Country (Top 10)')
            ax3.set_xlabel('Number of Data Points')
        
        # 4. Temporal coverage
        ax4 = axes[1, 1]
        time_coverage = ts_df.set_index('time')['value'].resample('Y').count()
        ax4.plot(time_coverage.index, time_coverage.values, marker='o')
        ax4.set_title('Temporal Coverage')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Number of Data Points')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'time_series_overview_{dataset_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    Saved plot: {plot_path}")
        
        plt.close()

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze time series patterns in economic data')
    parser.add_argument('--output-dir', default='./', help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true', help='Skip creating plots')
    
    args = parser.parse_args()
    
    print("Starting time series analysis...")
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load datasets
        datasets = load_and_prepare_data()
        
        if not datasets:
            print("Error: No time series datasets found!")
            sys.exit(1)
        
        print(f"\nLoaded {len(datasets)} datasets for time series analysis")
        
        # Analyze country time series
        country_analyses = analyze_country_time_series(datasets, args.output_dir)
        
        # Create plots if requested
        if not args.no_plots:
            plot_time_series_overview(datasets, args.output_dir)
        
        print(f"\nTime series analysis completed!")
        print(f"Results saved in: {args.output_dir}")
        
        # Print summary
        total_series = sum(len(countries) for countries in country_analyses.values())
        print(f"\nSummary:")
        print(f"- Analyzed {total_series} country time series across {len(country_analyses)} datasets")
        
    except Exception as e:
        print(f"Error during time series analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 