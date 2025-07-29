#!/usr/bin/env python3
"""
Eigenanalysis of Political Networks

Performs correspondence analysis and principal component analysis on political 
adjacency matrices from:
- IGO (Intergovernmental Organizations) membership data - tracking multilateral cooperation
  patterns from institutional memberships (1815-2014)
- DCAD (Defense Cooperation Agreement Dataset) defense cooperation data - bilateral defense
  agreements coordinating routine defense relations (1980-2010)

IGO Citation: Pevehouse, Jon C.W., Timothy Nordstron, Roseanne W McManus, Anne Spencer Jamison, 
"Tracking Organizations in the World: The Correlates of War IGO Version 3.0 datasets", 
Journal of Peace Research.

DCAD Citation: Kinne, Brandon J. 2020. "The Defense Cooperation Agreement Dataset (DCAD)," 
The Journal of Conflict Resolution 64(4): 729-755.

IGO captures membership in intergovernmental organizations with at least 3 nation-states,
while DCAD captures bilateral defense cooperation agreements that institutionalize routine 
defense relations between countries through formal international agreements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import unified color scheme
sys.path.append(str(Path(__file__).parent.parent.parent / 'configs'))
from color_schemes import get_country_color, get_country_colors, create_country_color_map

def correspondence_analysis(matrix):
    """
    Perform correspondence analysis on a contingency table.
    
    Args:
        matrix: Input matrix as numpy array or pandas DataFrame
    
    Returns:
        dict: Results including coordinates, eigenvalues, and inertia
    """
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
    
    # Step 1: Compute row and column totals
    row_totals = matrix.sum(axis=1)
    column_totals = matrix.sum(axis=0)

    # Step 2: Compute total sum of the matrix
    N = matrix.sum()

    # Step 3: Compute relative frequencies
    P = matrix / N

    # Step 4: Compute expected frequencies under independence
    P_hat = np.outer(row_totals, column_totals) / N**2

    # Step 5: Compute deviation from independence
    D = P - P_hat

    # Step 6: Compute total inertia
    inertia = np.sum(D**2 / (P_hat + 1e-10))  # Add small epsilon to avoid division by zero

    # Step 7: Decompose the total inertia into orthogonal factors
    row_ratios = row_totals / N
    column_ratios = column_totals / N
    sqrt_row_ratios = np.sqrt(row_ratios + 1e-10)
    sqrt_col_ratios = np.sqrt(column_ratios + 1e-10)
    
    # Scaling D
    S = D / np.outer(sqrt_row_ratios, sqrt_col_ratios)
    
    # Singular Value Decomposition
    U, s, VT = np.linalg.svd(S, full_matrices=False)
    
    # First and second dimensions for rows and columns
    row_coord_dim1 = U[:, 0] / sqrt_row_ratios
    row_coord_dim2 = U[:, 1] / sqrt_row_ratios
    
    col_coord_dim1 = VT[0, :] / sqrt_col_ratios
    col_coord_dim2 = VT[1, :] / sqrt_col_ratios
    
    # Eigenvalues (squared singular values)
    eigenvalues = s**2
    
    return {
        'row_coord_dim1': row_coord_dim1,
        'row_coord_dim2': row_coord_dim2,
        'col_coord_dim1': col_coord_dim1,
        'col_coord_dim2': col_coord_dim2,
        'eigenvalues': eigenvalues,
        'total_inertia': inertia,
        'explained_variance_ratio': eigenvalues / inertia if inertia > 0 else np.zeros_like(eigenvalues)
    }

def perform_pca_analysis(matrix, n_components=None):
    """
    Perform PCA analysis on the matrix.
    
    Args:
        matrix: Input matrix as pandas DataFrame
        n_components: Number of components to extract
    
    Returns:
        dict: PCA results including components, explained variance, etc.
    """
    # Standardize the data
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    
    # Perform PCA
    if n_components is None:
        n_components = min(matrix.shape) - 1
    
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(matrix_scaled)
    
    return {
        'components': components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
        'eigenvalues': pca.explained_variance_,
        'feature_loadings': pca.components_,
        'scaler': scaler,
        'pca_model': pca
    }

def analyze_important_countries(igo_file, dcad_file, important_countries_file):
    """
    Analyze the important countries subset using both IGO and DCAD data.
    
    Args:
        igo_file: Path to IGO adjacency matrix
        dcad_file: Path to DCAD adjacency matrix  
        important_countries_file: Path to important countries list
    
    Returns:
        dict: Analysis results for both datasets
    """
    # Read important countries
    with open(important_countries_file, 'r') as file:
        text = file.read()
    important_countries = [country.strip() for country in text.split() if country.strip()]
    
    print(f"Important countries: {important_countries}")
    
    # Load matrices
    igo_df = pd.read_csv(igo_file, index_col=0)
    dcad_df = pd.read_csv(dcad_file, index_col=0)
    
    print(f"IGO matrix shape: {igo_df.shape}")
    print(f"DCAD matrix shape: {dcad_df.shape}")
    
    # Filter to important countries that exist in both datasets
    common_countries = []
    for country in important_countries:
        if country in igo_df.index and country in dcad_df.index:
            common_countries.append(country)
    
    print(f"Common important countries: {common_countries}")
    
    # Subset matrices
    igo_subset = igo_df.loc[common_countries, common_countries]
    dcad_subset = dcad_df.loc[common_countries, common_countries]
    
    # Perform analyses
    results = {}
    
    # IGO Analysis
    print("\nAnalyzing IGO data...")
    igo_ca = correspondence_analysis(igo_subset)
    igo_pca = perform_pca_analysis(igo_subset)
    
    results['igo'] = {
        'matrix': igo_subset,
        'correspondence_analysis': igo_ca,
        'pca': igo_pca
    }
    
    # DCAD Analysis
    print("Analyzing DCAD data...")
    dcad_ca = correspondence_analysis(dcad_subset)
    dcad_pca = perform_pca_analysis(dcad_subset)
    
    results['dcad'] = {
        'matrix': dcad_subset,
        'correspondence_analysis': dcad_ca,
        'pca': dcad_pca
    }
    
    results['countries'] = common_countries
    
    return results

def plot_correspondence_analysis(ca_results, countries, title, output_dir="./"):
    """Plot correspondence analysis results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get unified colors for countries
    country_color_map = create_country_color_map(countries, 'matplotlib')
    country_colors = [country_color_map.get(country, '#1f77b4') for country in countries]
    
    # Plot 1: Country positions in CA space
    ax1.scatter(ca_results['row_coord_dim1'], ca_results['row_coord_dim2'], 
               c=country_colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    for i, country in enumerate(countries):
        ax1.annotate(country, 
                    (ca_results['row_coord_dim1'][i], ca_results['row_coord_dim2'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Dimension 1', fontsize=12)
    ax1.set_ylabel('Dimension 2', fontsize=12)
    ax1.set_title(f'{title} - Country Positions', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Explained variance
    var_explained = ca_results['explained_variance_ratio'][:10]  # Top 10 dimensions
    ax2.bar(range(1, len(var_explained) + 1), var_explained, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax2.set_title(f'{title} - Explained Variance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"correspondence_analysis_{title.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    
    return filepath

def plot_pca_analysis(pca_results, countries, title, output_dir="./"):
    """Plot PCA analysis results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get unified colors for countries
    country_color_map = create_country_color_map(countries, 'matplotlib')
    country_colors = [country_color_map.get(country, '#1f77b4') for country in countries]
    
    # Plot 1: PCA scatter plot
    components = pca_results['components']
    ax1.scatter(components[:, 0], components[:, 1], 
               c=country_colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    for i, country in enumerate(countries):
        ax1.annotate(country, 
                    (components[i, 0], components[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel(f'PC1 ({pca_results["explained_variance_ratio"][0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca_results["explained_variance_ratio"][1]:.1%} variance)', fontsize=12)
    ax1.set_title(f'{title} - PCA', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Explained variance
    var_explained = pca_results['explained_variance_ratio']
    cumulative_var = pca_results['cumulative_variance_ratio']
    
    x = range(1, len(var_explained) + 1)
    ax2.bar(x, var_explained, alpha=0.7, label='Individual', color='#2ca02c', edgecolor='black')
    ax2.plot(x, cumulative_var, 'o-', label='Cumulative', color='#d62728', linewidth=2, markersize=6)
    ax2.set_xlabel('Principal Component', fontsize=12)
    ax2.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax2.set_title(f'{title} - Explained Variance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"pca_analysis_{title.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    
    return filepath

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Perform eigenanalysis on political data')
    parser.add_argument('--igo-file', default='../results/preprocessed/IGO_adjmat_important.csv',
                       help='Path to IGO adjacency matrix CSV file')
    parser.add_argument('--dcad-file', default='../results/preprocessed/DCAD_adjmat_important.csv',
                       help='Path to DCAD adjacency matrix CSV file')
    parser.add_argument('--countries-file', default='../important_countries.txt',
                       help='Path to important countries text file')
    parser.add_argument('--output-dir', default='./',
                       help='Output directory for plots and results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating plots')
    
    args = parser.parse_args()
    
    print("Starting eigenanalysis of political data...")
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Perform analysis
        results = analyze_important_countries(
            args.igo_file, 
            args.dcad_file, 
            args.countries_file
        )
        
        countries = results['countries']
        
        # Print results
        print(f"\nAnalysis Results:")
        print(f"Countries analyzed: {countries}")
        print(f"Number of countries: {len(countries)}")
        
        # IGO Results
        igo_ca = results['igo']['correspondence_analysis']
        igo_pca = results['igo']['pca']
        print(f"\nIGO Correspondence Analysis:")
        print(f"  Total inertia: {igo_ca['total_inertia']:.6f}")
        print(f"  First 3 dimensions explain: {igo_ca['explained_variance_ratio'][:3].sum():.1%} of variance")
        
        print(f"\nIGO PCA:")
        print(f"  First 3 components explain: {igo_pca['cumulative_variance_ratio'][2]:.1%} of variance")
        
        # DCAD Results  
        dcad_ca = results['dcad']['correspondence_analysis']
        dcad_pca = results['dcad']['pca']
        print(f"\nDCAD Correspondence Analysis:")
        print(f"  Total inertia: {dcad_ca['total_inertia']:.6f}")
        print(f"  First 3 dimensions explain: {dcad_ca['explained_variance_ratio'][:3].sum():.1%} of variance")
        
        print(f"\nDCAD PCA:")
        print(f"  First 3 components explain: {dcad_pca['cumulative_variance_ratio'][2]:.1%} of variance")
        
        # Create plots if requested
        if not args.no_plots:
            print(f"\nCreating plots in {args.output_dir}...")
            
            plot_correspondence_analysis(igo_ca, countries, "IGO", args.output_dir)
            plot_pca_analysis(igo_pca, countries, "IGO", args.output_dir)
            
            plot_correspondence_analysis(dcad_ca, countries, "DCAD", args.output_dir)
            plot_pca_analysis(dcad_pca, countries, "DCAD", args.output_dir)
        
        print(f"\nEigenanalysis completed!")
        
    except Exception as e:
        print(f"Error during eigenanalysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 