#!/usr/bin/env python3
"""
Country Comparison Analysis for Economic Data
Compares economic indicators across different countries and regions.
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Import unified color scheme
sys.path.append(str(Path(__file__).parent.parent.parent / 'configs'))
from color_schemes import get_country_color, get_country_colors, create_country_color_map

def load_and_prepare_data():
    """Load and prepare data for country comparison."""
    datasets = {}
    
    # World Bank GEM data
    if os.path.exists('../WB_GEM.csv'):
        print("Loading World Bank GEM data...")
        wb_df = pd.read_csv('../WB_GEM.csv')
        datasets['WB_GEM'] = wb_df
    
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
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
    
    return datasets

def extract_country_indicators(datasets):
    """Extract country-level indicators from datasets."""
    
    country_data = {}
    
    for dataset_name, df in datasets.items():
        print(f"Processing {dataset_name}...")
        
        # Detect relevant columns
        country_cols = [col for col in df.columns if any(word in col.lower() for word in ['country', 'ref_area'])]
        value_cols = [col for col in df.columns if any(word in col.lower() for word in ['value', 'original_value'])]
        time_cols = [col for col in df.columns if any(word in col.lower() for word in ['period', 'date', 'time'])]
        indicator_cols = [col for col in df.columns if any(word in col.lower() for word in ['indicator', 'series'])]
        
        if not (country_cols and value_cols):
            print(f"  Skipping {dataset_name}: missing required columns")
            continue
        
        country_col = country_cols[0]
        value_col = value_cols[0]
        
        # Use all available data for each country to calculate meaningful statistics
        df_clean = df.copy()
        df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[value_col])
        
        if len(df_clean) == 0:
            continue
        
        # Group by country and calculate statistics using ALL available data points
        country_stats = df_clean.groupby(country_col)[value_col].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Filter out countries with insufficient data for meaningful statistics
        country_stats = country_stats[country_stats['count'] >= 2]
        
        country_stats['dataset'] = dataset_name
        
        # Calculate coefficient of variation with proper handling of edge cases
        # Avoid division by zero and handle NaN values
        country_stats['coefficient_of_variation'] = np.where(
            (country_stats['mean'] != 0) & (country_stats['mean'].notna()) & 
            (country_stats['std'].notna()) & (country_stats['std'] > 0),
            country_stats['std'] / np.abs(country_stats['mean']),
            np.nan
        )
        

        
        # Add indicator information if available
        if indicator_cols:
            indicator_col = indicator_cols[0]
            # Get most common indicator for each country
            country_indicators = df_clean.groupby(country_col)[indicator_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
            country_stats = country_stats.merge(country_indicators.reset_index(), on=country_col, how='left')
        
        country_data[dataset_name] = country_stats
    
    return country_data

def create_country_ranking(country_data, metric='mean'):
    """Create country rankings based on a specific metric."""
    
    rankings = {}
    
    for dataset_name, df in country_data.items():
        if metric in df.columns:
            # Sort by metric and create ranking
            sorted_df = df.sort_values(metric, ascending=False).reset_index(drop=True)
            sorted_df['rank'] = range(1, len(sorted_df) + 1)
            
            rankings[dataset_name] = sorted_df
    
    return rankings

def perform_country_clustering(country_data, n_clusters=5):
    """Perform clustering analysis on countries."""
    
    print(f"\nPerforming country clustering analysis...")
    
    clustering_results = {}
    
    for dataset_name, df in country_data.items():
        print(f"  Clustering {dataset_name}...")
        
        # Select numeric columns for clustering
        numeric_cols = ['mean', 'median', 'std', 'coefficient_of_variation']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 2:
            print(f"    Insufficient numeric columns for clustering")
            continue
        
        # Prepare data for clustering
        clustering_data = df[available_cols].dropna()
        if len(clustering_data) < n_clusters:
            print(f"    Insufficient data points for clustering ({len(clustering_data)} < {n_clusters})")
            continue
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        # Determine optimal number of clusters using silhouette score
        silhouette_scores = []
        cluster_range = range(2, min(8, len(clustering_data)))
        
        for n in cluster_range:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        if silhouette_scores:
            optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
            best_silhouette = max(silhouette_scores)
        else:
            optimal_clusters = n_clusters
            best_silhouette = None
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels back to dataframe
        clustered_df = clustering_data.copy()
        clustered_df['cluster'] = cluster_labels
        clustered_df['country'] = df.loc[clustering_data.index, df.columns[0]]  # Country column
        
        clustering_results[dataset_name] = {
            'data': clustered_df,
            'n_clusters': optimal_clusters,
            'silhouette_score': best_silhouette,
            'cluster_centers': scaler.inverse_transform(kmeans.cluster_centers_),
            'feature_names': available_cols
        }
    
    return clustering_results

def analyze_regional_patterns(country_data):
    """Analyze patterns by region (basic region assignment based on country codes)."""
    
    # Simple region mapping (this could be expanded with more sophisticated mapping)
    region_mapping = {
        'US': 'North America', 'CA': 'North America', 'MX': 'North America',
        'BR': 'South America', 'AR': 'South America', 'CL': 'South America',
        'DE': 'Europe', 'FR': 'Europe', 'IT': 'Europe', 'ES': 'Europe', 'UK': 'Europe', 'GB': 'Europe',
        'CN': 'Asia', 'JP': 'Asia', 'IN': 'Asia', 'KR': 'Asia', 'TH': 'Asia',
        'AU': 'Oceania', 'NZ': 'Oceania',
        'EG': 'Africa', 'ZA': 'Africa', 'NG': 'Africa'
    }
    
    regional_analysis = {}
    
    for dataset_name, df in country_data.items():
        # Add region information
        df_with_region = df.copy()
        country_col = df.columns[0]
        
        df_with_region['region'] = df_with_region[country_col].map(region_mapping)
        df_with_region['region'] = df_with_region['region'].fillna('Other')
        
        # Calculate regional statistics
        regional_stats = df_with_region.groupby('region').agg({
            'mean': ['mean', 'std', 'count'],
            'median': 'mean',
            'coefficient_of_variation': 'mean'
        }).round(4)
        
        regional_analysis[dataset_name] = {
            'data': df_with_region,
            'regional_stats': regional_stats
        }
    
    return regional_analysis

def plot_country_comparisons(country_data, rankings, output_dir="./"):
    """Create comparison plots for countries."""
    
    print("\nCreating country comparison plots...")
    
    for dataset_name, df in country_data.items():
        print(f"  Plotting {dataset_name}...")
        
        # Create multi-panel plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Country Comparison: {dataset_name}', fontsize=16)
        
        # 1. Top countries by mean value
        ax1 = axes[0, 0]
        top_countries = rankings[dataset_name].head(15) if dataset_name in rankings else df.head(15)
        
        if len(top_countries) > 0:
            country_col = top_countries.columns[0]
            ax1.barh(range(len(top_countries)), top_countries['mean'])
            ax1.set_yticks(range(len(top_countries)))
            ax1.set_yticklabels(top_countries[country_col])
            ax1.set_title('Top Countries by Mean Value')
            ax1.set_xlabel('Mean Value')
            ax1.invert_yaxis()
        
        # 2. Distribution of mean values
        ax2 = axes[0, 1]
        ax2.hist(df['mean'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Country Mean Values')
        ax2.set_xlabel('Mean Value')
        ax2.set_ylabel('Number of Countries')
        ax2.grid(True, alpha=0.3)
        
        # 3. Mean vs Volatility (CV)
        ax3 = axes[1, 0]
        if 'coefficient_of_variation' in df.columns:
            # Filter out NaN and infinite values for plotting
            plot_data = df[['mean', 'coefficient_of_variation']].dropna()
            plot_data = plot_data[np.isfinite(plot_data['coefficient_of_variation']) & 
                                np.isfinite(plot_data['mean'])]
            
            if len(plot_data) > 0:
                ax3.scatter(plot_data['mean'], plot_data['coefficient_of_variation'], alpha=0.6)
                ax3.set_xlabel('Mean Value')
                ax3.set_ylabel('Coefficient of Variation')
                ax3.set_title('Mean vs Volatility')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No valid data for plotting', 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Mean vs Volatility')
        else:
            ax3.text(0.5, 0.5, 'Coefficient of variation not available', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Mean vs Volatility')
        
        # 4. Country count and data quality
        ax4 = axes[1, 1]
        ax4.bar(['Countries', 'Valid Data Points'], [len(df), df['count'].sum()])
        ax4.set_title('Data Coverage')
        ax4.set_ylabel('Count')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'country_comparison_{dataset_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    Saved plot: {plot_path}")
        
        plt.close()

def plot_clustering_results(clustering_results, output_dir="./"):
    """Plot clustering analysis results."""
    
    print("\nCreating clustering plots...")
    
    for dataset_name, results in clustering_results.items():
        print(f"  Plotting clustering for {dataset_name}...")
        
        data = results['data']
        n_clusters = results['n_clusters']
        
        # Create clustering visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Country Clustering: {dataset_name}', fontsize=16)
        
        # 1. Scatter plot of first two features colored by cluster
        ax1 = axes[0]
        features = results['feature_names'][:2]  # Use first two features
        
        # Use a colorblind-friendly colormap
        scatter = ax1.scatter(data[features[0]], data[features[1]], 
                            c=data['cluster'], cmap='Set2', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel(features[0])
        ax1.set_ylabel(features[1])
        ax1.set_title(f'Clusters (Silhouette Score: {results["silhouette_score"]:.3f})')
        plt.colorbar(scatter, ax=ax1, label='Cluster')
        
        # 2. Cluster centers visualization
        ax2 = axes[1]
        centers = results['cluster_centers']
        feature_names = results['feature_names']
        
        # Use consistent colors for cluster centers
        cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        for i, center in enumerate(centers):
            color = cluster_colors[i % len(cluster_colors)]
            ax2.plot(range(len(feature_names)), center, 'o-', label=f'Cluster {i}', linewidth=2, color=color, markersize=8)
        
        ax2.set_xticks(range(len(feature_names)))
        ax2.set_xticklabels(feature_names, rotation=45)
        ax2.set_title('Cluster Centers')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'country_clustering_{dataset_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    Saved plot: {plot_path}")
        
        plt.close()

def save_analysis_results(country_data, rankings, clustering_results, regional_analysis, output_dir="./"):
    """Save detailed analysis results to files."""
    
    print("\nSaving analysis results...")
    
    # Country rankings
    rankings_path = os.path.join(output_dir, 'country_rankings.txt')
    with open(rankings_path, 'w') as f:
        f.write("Country Rankings Analysis\n")
        f.write("="*40 + "\n\n")
        
        for dataset_name, ranking_df in rankings.items():
            f.write(f"\nDataset: {dataset_name}\n")
            f.write("-" * 30 + "\n")
            f.write("Top 10 Countries by Mean Value:\n")
            
            country_col = ranking_df.columns[0]
            for i, row in ranking_df.head(10).iterrows():
                f.write(f"{row['rank']:2d}. {row[country_col]:<20} {row['mean']:>10.2f}\n")
            f.write("\n")
    
    # Clustering results
    if clustering_results:
        clustering_path = os.path.join(output_dir, 'country_clustering_analysis.txt')
        with open(clustering_path, 'w') as f:
            f.write("Country Clustering Analysis\n")
            f.write("="*40 + "\n\n")
            
            for dataset_name, results in clustering_results.items():
                f.write(f"\nDataset: {dataset_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Optimal clusters: {results['n_clusters']}\n")
                f.write(f"Silhouette score: {results['silhouette_score']:.4f}\n")
                
                # List countries by cluster
                data = results['data']
                for cluster_id in range(results['n_clusters']):
                    cluster_countries = data[data['cluster'] == cluster_id]['country'].tolist()
                    f.write(f"\nCluster {cluster_id} ({len(cluster_countries)} countries):\n")
                    for country in cluster_countries:
                        f.write(f"  {country}\n")
                f.write("\n")
    
    print(f"  Saved rankings: {rankings_path}")
    if clustering_results:
        print(f"  Saved clustering analysis: {clustering_path}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Compare economic indicators across countries')
    parser.add_argument('--output-dir', default='./', help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true', help='Skip creating plots')
    parser.add_argument('--clusters', type=int, default=5, help='Number of clusters for analysis')
    
    args = parser.parse_args()
    
    print("Starting country comparison analysis...")
    
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
        
        print(f"\nLoaded {len(datasets)} datasets for country comparison")
        
        # Extract country indicators
        country_data = extract_country_indicators(datasets)
        
        if not country_data:
            print("Error: No country data extracted!")
            sys.exit(1)
        
        # Create rankings
        rankings = create_country_ranking(country_data, metric='mean')
        
        # Perform clustering analysis
        clustering_results = perform_country_clustering(country_data, n_clusters=args.clusters)
        
        # Regional analysis
        regional_analysis = analyze_regional_patterns(country_data)
        
        # Create plots if requested
        if not args.no_plots:
            plot_country_comparisons(country_data, rankings, args.output_dir)
            if clustering_results:
                plot_clustering_results(clustering_results, args.output_dir)
        
        # Save detailed results
        save_analysis_results(country_data, rankings, clustering_results, regional_analysis, args.output_dir)
        
        print(f"\nCountry comparison analysis completed!")
        print(f"Results saved in: {args.output_dir}")
        
        # Print summary
        total_countries = sum(len(df) for df in country_data.values())
        print(f"\nSummary:")
        print(f"- Analyzed {total_countries} country records across {len(country_data)} datasets")
        if clustering_results:
            print(f"- Performed clustering analysis on {len(clustering_results)} datasets")
        
    except Exception as e:
        print(f"Error during country comparison analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 