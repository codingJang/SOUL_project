#!/usr/bin/env python3
"""
Statistical significance testing for political groupings using permutation tests.

This script tests whether observed country groupings show significantly different
within-group vs. between-group strengths compared to random groupings, using
political network data from:
- IGO (Intergovernmental Organizations) membership networks - tracking multilateral 
  cooperation patterns from institutional memberships (1815-2014)
- DCAD (Defense Cooperation Agreement Dataset) defense cooperation networks - bilateral 
  defense agreements coordinating routine defense relations (1980-2010)

IGO Citation: Pevehouse, Jon C.W., Timothy Nordstron, Roseanne W McManus, Anne Spencer Jamison, 
"Tracking Organizations in the World: The Correlates of War IGO Version 3.0 datasets", 
Journal of Peace Research.

DCAD Citation: Kinne, Brandon J. 2020. "The Defense Cooperation Agreement Dataset (DCAD)," 
The Journal of Conflict Resolution 64(4): 729-755.

IGO provides data on membership in intergovernmental organizations with at least 3 
nation-states, while DCAD provides comprehensive data on bilateral defense cooperation 
agreements (DCAs) that coordinate routine defense relations between countries including 
joint exercises, peacekeeping operations, defense research, and policy coordination.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import sys

def calculate_partition_goodness(adjmat, partition):
    """
    Calculate the 'goodness' of a partition - ratio of within-group to between-group strength.
    
    Args:
        adjmat: Adjacency matrix (pandas DataFrame)
        partition: Partition assignments (pandas Series with same index as adjmat)
    
    Returns:
        float: Goodness measure (higher = more within-group strength relative to between-group)
    """
    n = len(adjmat)
    within_group_sum = 0
    between_group_sum = 0
    within_count = 0
    between_count = 0
    
    for i in range(n):
        for j in range(i+1, n):  # Only upper triangle to avoid double counting
            country_i = adjmat.index[i]
            country_j = adjmat.index[j]
            
            if country_i in partition.index and country_j in partition.index:
                edge_weight = adjmat.iloc[i, j]
                
                if partition[country_i] == partition[country_j]:
                    # Same group
                    within_group_sum += edge_weight
                    within_count += 1
                else:
                    # Different groups
                    between_group_sum += edge_weight
                    between_count += 1
    
    # Calculate average strengths
    avg_within = within_group_sum / within_count if within_count > 0 else 0
    avg_between = between_group_sum / between_count if between_count > 0 else 0
    
    # Goodness is ratio of within-group to between-group strength
    # Add small epsilon to avoid division by zero
    goodness = avg_within / (avg_between + 1e-10)
    
    return goodness

def permutation_test(adjmat, partition, n_permutations=1000):
    """
    Perform permutation test by randomly shuffling partition assignments.
    
    Args:
        adjmat: Adjacency matrix (pandas DataFrame)
        partition: True partition assignments (pandas Series)
        n_permutations: Number of random permutations to test
    
    Returns:
        tuple: (p_value, observed_goodness, list_of_permutation_goodness_values)
    """
    # Calculate observed goodness
    observed_goodness = calculate_partition_goodness(adjmat, partition)
    
    # Perform permutations
    goodness_values = []
    
    for i in range(n_permutations):
        if i % 100 == 0:
            print(f"Permutation {i}/{n_permutations}")
        
        # Create shuffled partition - use copy to avoid UserWarning
        shuffled_partition = partition.copy()
        shuffled_values = shuffled_partition.values.copy()
        np.random.shuffle(shuffled_values)
        shuffled_partition[:] = shuffled_values
        
        # Calculate goodness for shuffled partition
        goodness = calculate_partition_goodness(adjmat, shuffled_partition)
        goodness_values.append(goodness)
    
    # Calculate p-value (proportion of permutations with goodness >= observed)
    p_value = np.mean([g >= observed_goodness for g in goodness_values])
    
    return p_value, observed_goodness, goodness_values

def plot_combined_results(results_lv1, results_lv2, output_dir="./", dataset_name="IGO"):
    """Plot combined histogram with two subplots for lv1 and lv2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot lv1 results
    ax1.hist(results_lv1['goodness_values'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.axvline(results_lv1['observed_goodness'], color='red', linestyle='--', linewidth=2, 
                label=f'Observed: {results_lv1["observed_goodness"]:.4f}')
    ax1.set_xlabel('Goodness Measure')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Level 1 Groupings\n(p={results_lv1["p_value"]:.6f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot lv2 results
    ax2.hist(results_lv2['goodness_values'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(results_lv2['observed_goodness'], color='red', linestyle='--', linewidth=2, 
                label=f'Observed: {results_lv2["observed_goodness"]:.4f}')
    ax2.set_xlabel('Goodness Measure')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Level 2 Groupings\n(p={results_lv2["p_value"]:.6f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Distribution of Goodness Values from Permutation Test - {dataset_name}', fontsize=16)
    
    # Save plot
    filename = f"goodness_histogram_{dataset_name.lower()}.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {filename}")
    plt.close()
    
    return filepath

def plot_results(observed_goodness, goodness_values, output_dir="./", level="lv1", dataset_name="IGO"):
    """Plot histogram of goodness values with observed goodness marked (legacy function for individual plots)."""
    plt.figure(figsize=(10, 6))
    plt.hist(goodness_values, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(observed_goodness, color='red', linestyle='--', linewidth=2, 
                label=f'Observed Goodness: {observed_goodness:.4f}')
    plt.xlabel('Goodness Measure')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Goodness Values from Permutation Test ({level})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    filename = f"goodness_histogram_{level}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    plt.close()
    
    return filepath

def run_significance_test(adjmat_file, partition_file, level='lv1', n_permutations=2000, 
                         output_dir="./", plot=True, seed=None):
    """
    Run the complete significance test pipeline.
    
    Args:
        adjmat_file: Path to adjacency matrix CSV file
        partition_file: Path to partition CSV file
        level: Level of grouping to test ('lv1' or 'lv2')
        n_permutations: Number of permutations
        output_dir: Output directory for results
        plot: Whether to create histogram plot
        seed: Random seed for reproducibility
    
    Returns:
        dict: Results dictionary with p_value, observed_goodness, etc.
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"Loading adjacency matrix from {adjmat_file}...")
    adjmat_df = pd.read_csv(adjmat_file, index_col=0)
    
    print(f"Loading partition data from {partition_file}...")
    partition_df = pd.read_csv(partition_file, index_col='CtryAbb')
    partition_df = partition_df[level]
    
    # Remove countries with partition value 0 (unassigned)
    partition_df = partition_df[partition_df != 0]
    
    # Filter adjacency matrix to only include countries in partition
    index = list(partition_df.index)
    adjmat_df = adjmat_df.loc[index, index]
    
    print(f"Testing {len(index)} countries in {len(partition_df.unique())} groups")
    print(f"Running {n_permutations} permutations...")
    
    # Run permutation test
    p_value, observed_goodness, goodness_values = permutation_test(
        adjmat_df, partition_df, n_permutations=n_permutations
    )
    
    # Print results
    print(f"\nResults:")
    print(f"Observed goodness: {observed_goodness:.6f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print(f"Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")
    
    # Create individual plot if requested (for backward compatibility)
    plot_file = None
    if plot and level in ['lv1', 'lv2']:  # Only create individual plots if specifically requested
        plot_file = plot_results(observed_goodness, goodness_values, output_dir, level)
    
    # Save detailed results
    results = {
        'observed_goodness': observed_goodness,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'goodness_values': goodness_values,
        'level': level,
        'n_countries': len(index),
        'n_groups': len(partition_df.unique())
    }
    
    return results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Test statistical significance of political groupings')
    parser.add_argument('--adjmat', default='../results/preprocessed/IGO_adjmat.csv', 
                       help='Path to adjacency matrix CSV file')
    parser.add_argument('--partition', default='../grouping_countries.csv',
                       help='Path to partition CSV file')
    parser.add_argument('--level', nargs='+', default=['lv1'], choices=['lv1', 'lv2'],
                       help='Level(s) of grouping to test. Can specify one or multiple: --level lv1 or --level lv1 lv2')
    parser.add_argument('--permutations', type=int, default=2000,
                       help='Number of permutations to run')
    parser.add_argument('--output-dir', default='./',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip creating histogram plot')
    
    args = parser.parse_args()
    
    print("Starting significance test for political groupings...")
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Extract dataset name from adjacency matrix filename
        dataset_name = "IGO"  # default
        if "IGO" in args.adjmat.upper():
            dataset_name = "IGO"
        elif "DCAD" in args.adjmat.upper():
            dataset_name = "DCAD"
        
        # Test multiple levels if more than one specified
        if len(args.level) > 1:
            print(f"Running tests for levels: {', '.join(args.level)} to create combined plot...")
            
            results_by_level = {}
            
            for level in args.level:
                print("\n" + "="*50)
                print(f"TESTING LEVEL {level.upper()}")
                print("="*50)
                results_by_level[level] = run_significance_test(
                    adjmat_file=args.adjmat,
                    partition_file=args.partition,
                    level=level,
                    n_permutations=args.permutations,
                    output_dir=args.output_dir,
                    plot=False,  # Don't create individual plots
                    seed=args.seed
                )
            
            # Create combined plot if we have both lv1 and lv2
            if not args.no_plot and 'lv1' in results_by_level and 'lv2' in results_by_level:
                plot_combined_results(results_by_level['lv1'], results_by_level['lv2'], args.output_dir, dataset_name)
            elif not args.no_plot and len(results_by_level) == 1:
                # Single level, create individual plot
                level = list(results_by_level.keys())[0]
                results = results_by_level[level]
                plot_results(results, args.output_dir, level, dataset_name)
            
        else:
            # Run single level test
            level = args.level[0]
            print(f"Running test for {level} only...")
            results = run_significance_test(
                adjmat_file=args.adjmat,
                partition_file=args.partition,
                level=level,
                n_permutations=args.permutations,
                output_dir=args.output_dir,
                plot=not args.no_plot,
                seed=args.seed
            )
        
        print(f"\nSignificance test completed!")
        
    except Exception as e:
        print(f"Error during significance test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 