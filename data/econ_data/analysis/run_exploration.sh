#!/bin/bash

# Economic Data Exploration Pipeline
# Explores and summarizes the structure of economic datasets

set -e  # Exit on any error

# Detect script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "Economic Data Exploration Pipeline"
echo "========================================="
echo ""
echo "Analysis directory: $SCRIPT_DIR"
echo "Data directory: $DATA_DIR"
echo ""

# Activate virtual environment
if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    source "$ROOT_DIR/.venv/bin/activate"
fi

# Check if we're in the right directory structure
if [[ ! -f "$DATA_DIR/WB_GEM.csv" ]] && [[ ! -f "$DATA_DIR/IMF_BOP.csv" ]]; then
    echo "Error: Required data files not found in $DATA_DIR"
    echo "Expected at least one of: WB_GEM.csv, IMF_BOP.csv"
    exit 1
fi

# Change to analysis directory for script execution
cd "$SCRIPT_DIR"

# Create output directory for results
mkdir -p results

echo "Step 1: Data Structure Exploration..."
echo "------------------------------------"
python data_exploration.py --output-dir ../results/

if [[ $? -eq 0 ]]; then
    echo "✓ Data exploration completed successfully"
else
    echo "✗ Data exploration failed"
    exit 1
fi

echo ""
echo "========================================="
echo "Data exploration completed!"
echo "========================================="
echo ""
echo "Generated files in $SCRIPT_DIR/../results/:"
echo "- data_summary_report.txt (comprehensive data summary)"
echo "- time_series_analysis.txt (time series patterns)"
echo "- dataset_overview.png (visualization of dataset characteristics)"
echo ""
echo "Next steps:"
echo "- Run time series analysis: bash $SCRIPT_DIR/run_time_series.sh"
echo "- Run country comparison: bash $SCRIPT_DIR/run_country_comparison.sh"
echo "- Run visualizations: bash $SCRIPT_DIR/run_visualization.sh" 