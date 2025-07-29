#!/usr/bin/env python3
"""
DCAD (Defense Cooperation Agreement Dataset) Data Preprocessing

The Defense Cooperation Agreement Dataset (DCAD) is a comprehensive, human-coded dataset 
of bilateral defense cooperation agreements (DCAs) covering all independent countries 
for the period 1980-2010. DCAs are formal international agreements that coordinate and 
institutionalize the routine, day-to-day defense relations of their signatories.

DCAD includes 1,872 unique agreements and is based on sources including the United Nations 
Treaty Series, World Treaty Index, individual country sources, and global newspaper 
archives. This script processes DCAD data into adjacency matrices for network analysis.

Dataset Citation:
Kinne, Brandon J. 2020. "The Defense Cooperation Agreement Dataset (DCAD)," 
The Journal of Conflict Resolution 64(4): 729-755.

Dataset Source: https://www.brandonkinne.com/dcad
Access via Correlates of War: https://correlatesofwar.org/

Note: This analysis filters agreements to those from 2000 onwards to focus on 
contemporary defense cooperation patterns.
"""

import numpy as np
import pandas as pd
import re
import sys
import os
from pathlib import Path

def read_country_codes():
    """Read country codes mapping."""
    abbs = pd.read_csv("../abb_ccode_names.csv", index_col='StateAbb').index
    abbs.name = None
    return abbs

def read_reference_files():
    """Read important countries and non-states lists."""
    with open('../important_countries.txt', 'r') as file:
        text = file.read()
    important_countries = re.split(r'\n\s*\n', text)
    important_countries = [country.strip() for country in important_countries if country.strip()]
    
    with open('../non_states.txt', 'r') as file:
        text = file.read()
    non_states = re.split(r'\n\s*\n', text)
    non_states = [state.strip() for state in non_states if state.strip()]
    
    return important_countries, non_states

def process_dcad_data(abbs):
    """Process DCAD dataset and create adjacency matrix."""
    print("Reading DCAD dataset...")
    dcad_df = pd.read_csv('../DCAD.csv')
    
    # Filter for agreements from 2000 onwards
    dcad_df = dcad_df[dcad_df.signYear >= 2000]
    print(f"Filtered to {len(dcad_df)} agreements from 2000 onwards")
    
    # Get unique country pairs
    country_pairs = set(zip(dcad_df.cowName1, dcad_df.cowName2))
    print(f"Found {len(country_pairs)} unique country pairs")
    
    # Create adjacency matrix
    adjmat_df = pd.DataFrame(np.zeros((len(abbs), len(abbs))), index=abbs, columns=abbs)
    
    # Fill in the adjacency matrix
    valid_pairs = 0
    invalid_pairs = 0
    
    for ctry1, ctry2 in country_pairs:
        if ctry1 in abbs and ctry2 in abbs:
            adjmat_df.loc[ctry1, ctry2] = 1
            adjmat_df.loc[ctry2, ctry1] = 1  # Make symmetric
            valid_pairs += 1
        else:
            invalid_pairs += 1
    
    print(f"Valid country pairs: {valid_pairs}")
    print(f"Invalid country pairs (not in abbreviation list): {invalid_pairs}")
    
    return adjmat_df

def create_filtered_matrices(adjmat_df, important_countries):
    """Create filtered versions of the adjacency matrix."""
    # Important countries subset
    important_in_data = [c for c in important_countries if c in adjmat_df.index]
    adjmat_important_df = adjmat_df.loc[important_in_data, important_in_data]
    
    # Top 5% connections (for important countries)
    # Since DCAD is binary, we'll use the important subset as "top 5%"
    adjmat_top5pct_df = adjmat_important_df.copy()
    
    return {
        'adjmat': adjmat_df,
        'adjmat_important': adjmat_important_df,
        'adjmat_top5pct': adjmat_top5pct_df
    }

def save_matrices(matrices, output_dir="../results/preprocessed/"):
    """Save all matrices to CSV files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = {
        'adjmat': 'DCAD_adjmat.csv',
        'adjmat_important': 'DCAD_adjmat_important.csv',
        'adjmat_top5pct': 'DCAD_adjmat_top5pct.csv'
    }
    
    for key, filename in output_files.items():
        filepath = os.path.join(output_dir, filename)
        matrices[key].to_csv(filepath)
        print(f"Saved {filename} to {filepath}")

def main():
    """Main processing function."""
    print("Starting DCAD data preprocessing...")
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Read input data
        print("Reading country codes...")
        abbs = read_country_codes()
        
        print("Reading reference files...")
        important_countries, non_states = read_reference_files()
        
        print("Processing DCAD data...")
        adjmat_df = process_dcad_data(abbs)
        
        print("Creating filtered matrices...")
        matrices = create_filtered_matrices(adjmat_df, important_countries)
        
        print("Saving matrices...")
        save_matrices(matrices)
        
        print("DCAD preprocessing completed successfully!")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"- Total countries: {len(abbs)}")
        print(f"- Important countries: {len([c for c in important_countries if c in abbs])}")
        print(f"- Total connections: {int(matrices['adjmat'].sum().sum() / 2)}")  # Divide by 2 for symmetric matrix
        print(f"- Connection density: {matrices['adjmat'].sum().sum() / (len(abbs) * (len(abbs) - 1)) * 100:.2f}%")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 