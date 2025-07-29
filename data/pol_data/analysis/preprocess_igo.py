#!/usr/bin/env python3
"""
IGO (Intergovernmental Organizations) Data Preprocessing

The Intergovernmental Organizations (IGO) dataset tracks the status and membership 
of intergovernmental organizations from 1815-2014. IGOs are international organizations 
that have at least 3 nation-states as their members. The dataset contains information 
collected at 5-year intervals from 1815-1965, and annually thereafter.

This comprehensive dataset enables analysis of multilateral cooperation patterns,
institutional membership networks, and the evolution of international governance
structures over two centuries.

Dataset Citation:
Pevehouse, Jon C.W., Timothy Nordstron, Roseanne W McManus, Anne Spencer Jamison, 
"Tracking Organizations in the World: The Correlates of War IGO Version 3.0 datasets", 
Journal of Peace Research.

Original Dataset:
Wallace, Michael, and J. David Singer. 1970. "International Governmental Organization 
in the Global System, 1815-1964." International Organization 24: 239-87.

Dataset Source: https://correlatesofwar.org/data-sets/igos/
Hosted by: Timothy Nordstrom (University of Mississippi), Jon Pevehouse (University of Wisconsin), 
and Megan Shannon (Colorado-Boulder) under the COW Data Hosting Program.

This script processes IGO membership data into adjacency matrices for network analysis
of multilateral cooperation patterns.
"""

import pandas as pd
import numpy as np
import re
import sys
import os
from pathlib import Path

def read_country_codes():
    """Read country codes mapping."""
    return pd.read_csv("../abb_ccode_names.csv", index_col='IGO_dataset').loc[:, 'StateAbb']

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

def read_igo_data():
    """Read and parse IGO dataset."""
    with open('../IGO.txt', 'r') as file:
        text = file.read()
    
    # Split the text by blank lines to separate the countries
    splitted_text = re.split(r'\n\s*\n|\n', text)
    splitted_text = [item.strip() for item in splitted_text if item.strip()]
    
    return splitted_text

def process_igo_memberships(splitted_text, names_to_abb):
    """Process IGO membership data and create adjacency matrix."""
    # Initialize dictionary to store IGO memberships
    igo_dict = {}
    current_country = None
    
    i = 0
    while i < len(splitted_text):
        item = splitted_text[i].strip()
        
        # Check if this is a country name
        if item in names_to_abb.index:
            current_country = names_to_abb[item]
            igo_dict[current_country] = []
            i += 1
            
            # Get IGO memberships for this country
            if i < len(splitted_text):
                igos = splitted_text[i].strip()
                if igos and current_country:
                    # Split by comma and clean up
                    igo_list = [igo.strip() for igo in igos.split(',')]
                    igo_dict[current_country] = igo_list
                i += 1
        else:
            i += 1
    
    return igo_dict

def create_adjacency_matrices(igo_dict, important_countries):
    """Create various adjacency matrices from IGO membership data."""
    # Get all countries
    all_countries = list(igo_dict.keys())
    
    # Create adjacency matrix based on shared IGO memberships
    n_countries = len(all_countries)
    adjmat = np.zeros((n_countries, n_countries))
    
    # Calculate shared IGO memberships
    for i, country1 in enumerate(all_countries):
        for j, country2 in enumerate(all_countries):
            if i != j:
                igos1 = set(igo_dict.get(country1, []))
                igos2 = set(igo_dict.get(country2, []))
                shared_igos = len(igos1.intersection(igos2))
                adjmat[i, j] = shared_igos
    
    # Create DataFrame
    adjmat_df = pd.DataFrame(adjmat, index=all_countries, columns=all_countries)
    
    # Create scaled version
    max_val = adjmat_df.max().max()
    if max_val > 0:
        adjmat_scaled_df = adjmat_df / max_val
    else:
        adjmat_scaled_df = adjmat_df.copy()
    
    # Create important countries subset
    important_in_data = [c for c in important_countries if c in all_countries]
    adjmat_important_df = adjmat_df.loc[important_in_data, important_in_data]
    adjmat_scaled_important_df = adjmat_scaled_df.loc[important_in_data, important_in_data]
    
    # Create top percentile matrices
    threshold_5pct = np.percentile(adjmat_scaled_df.values[adjmat_scaled_df.values > 0], 95)
    threshold_1pct = np.percentile(adjmat_scaled_df.values[adjmat_scaled_df.values > 0], 99)
    
    adjmat_scaled_top5pct_df = adjmat_scaled_df.copy()
    adjmat_scaled_top5pct_df[adjmat_scaled_top5pct_df < threshold_5pct] = 0
    
    adjmat_scaled_top1pct_important_df = adjmat_scaled_important_df.copy()
    adjmat_scaled_top1pct_important_df[adjmat_scaled_top1pct_important_df < threshold_1pct] = 0
    
    return {
        'adjmat': adjmat_df,
        'adjmat_scaled': adjmat_scaled_df,
        'adjmat_important': adjmat_important_df,
        'adjmat_scaled_important': adjmat_scaled_important_df,
        'adjmat_scaled_top5pct': adjmat_scaled_top5pct_df,
        'adjmat_scaled_top1pct_important': adjmat_scaled_top1pct_important_df
    }

def save_matrices(matrices, output_dir="../results/preprocessed/"):
    """Save all matrices to CSV files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = {
        'adjmat': 'IGO_adjmat.csv',
        'adjmat_scaled': 'IGO_adjmat_scaled.csv',
        'adjmat_important': 'IGO_adjmat_important.csv',
        'adjmat_scaled_important': 'IGO_adjmat_scaled_important.csv',
        'adjmat_scaled_top5pct': 'IGO_adjmat_scaled_top5pct.csv',
        'adjmat_scaled_top1pct_important': 'IGO_adjmat_scaled_top1pct_important.csv'
    }
    
    for key, filename in output_files.items():
        filepath = os.path.join(output_dir, filename)
        matrices[key].to_csv(filepath)
        print(f"Saved {filename} to {filepath}")

def main():
    """Main processing function."""
    print("Starting IGO data preprocessing...")
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Read input data
        print("Reading country codes...")
        names_to_abb = read_country_codes()
        
        print("Reading reference files...")
        important_countries, non_states = read_reference_files()
        
        print("Reading IGO data...")
        splitted_text = read_igo_data()
        
        print("Processing IGO memberships...")
        igo_dict = process_igo_memberships(splitted_text, names_to_abb)
        
        print("Creating adjacency matrices...")
        matrices = create_adjacency_matrices(igo_dict, important_countries)
        
        print("Saving matrices...")
        save_matrices(matrices)
        
        print("IGO preprocessing completed successfully!")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"- Countries processed: {len(igo_dict)}")
        print(f"- Important countries: {len([c for c in important_countries if c in igo_dict])}")
        print(f"- Max shared IGOs: {matrices['adjmat'].max().max()}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 