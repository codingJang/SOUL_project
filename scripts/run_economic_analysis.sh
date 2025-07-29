#!/bin/bash

# SOUL Project - Economic Data Analysis Runner
# Convenience script to run economic data analysis from the root directory

set -e  # Exit on any error

# Check if we're in the right directory and have economic data
data_found=false
for file in "data/econ_data/WB_GEM.csv" "data/econ_data/IMF_BOP.csv" "data/econ_data/IMF_CPI.csv"; do
    if [[ -f "$file" ]]; then
        data_found=true
        break
    fi
done

if [[ "$data_found" = false ]]; then
    echo "Error: This script must be run from the SOUL project root directory"
    echo "Expected to find economic data files in: data/econ_data/"
    echo "Looking for: WB_GEM.csv, IMF_BOP.csv, IMF_CPI.csv, or IMF_MFS.csv"
    exit 1
fi

echo "======================================================"
echo "SOUL Project - Economic Data Analysis"
echo "======================================================"
echo ""
echo "Running economic data analysis pipeline..."
echo "Working directory: $(pwd)"
echo ""
echo "Note: The pipeline will automatically activate the virtual environment"
echo "and use Python packages from .venv/ if available."
echo ""

# Run the economic data analysis pipeline
bash data/econ_data/analysis/run_full_pipeline.sh

echo ""
echo "======================================================"
echo "Economic data analysis completed!"
echo "======================================================"
echo ""
echo "Results are available in:"
echo "- data/econ_data/results/"
echo ""
echo "To run individual components:"
echo "- bash data/econ_data/analysis/run_exploration.sh"
echo "- bash data/econ_data/analysis/run_time_series.sh"
echo "- bash data/econ_data/analysis/run_country_comparison.sh"
echo "- bash data/econ_data/analysis/run_visualization.sh" 