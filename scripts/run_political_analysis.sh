#!/bin/bash

# SOUL Project - Political Data Analysis Runner
# Convenience script to run political data analysis from the root directory

set -e  # Exit on any error

# Check if we're in the right directory
if [[ ! -f "data/pol_data/IGO.txt" ]] && [[ ! -f "data/pol_data/DCAD.csv" ]]; then
    echo "Error: This script must be run from the SOUL project root directory"
    echo "Expected to find: data/pol_data/IGO.txt and data/pol_data/DCAD.csv"
    exit 1
fi

echo "======================================================"
echo "SOUL Project - Political Data Analysis"
echo "======================================================"
echo ""
echo "Running political data analysis pipeline..."
echo "Working directory: $(pwd)"
echo ""
echo "Note: The pipeline will automatically activate the virtual environment"
echo "and use Python packages from .venv/ if available."
echo ""

# Run the political data analysis pipeline
bash data/pol_data/analysis/run_full_pipeline.sh

echo ""
echo "======================================================"
echo "Political data analysis completed!"
echo "======================================================"
echo ""
echo "Results are available in:"
echo "- data/pol_data/results/"
echo ""
echo "To run individual components:"
echo "- bash data/pol_data/analysis/run_preprocessing.sh"
echo "- bash data/pol_data/analysis/run_analysis.sh" 
echo "- bash data/pol_data/analysis/run_visualization.sh" 