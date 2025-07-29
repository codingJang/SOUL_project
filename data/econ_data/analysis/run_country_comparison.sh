#!/bin/bash

# Economic Country Comparison Analysis Pipeline
# Compares economic indicators across countries and regions

set -e  # Exit on any error

# Detect script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "Country Comparison Analysis Pipeline"
echo "========================================="
echo ""
echo "Analysis directory: $SCRIPT_DIR"
echo "Data directory: $DATA_DIR"
echo ""

# Activate virtual environment
if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    source "$ROOT_DIR/.venv/bin/activate"
fi

# Check if we have economic data
if [[ ! -f "$DATA_DIR/WB_GEM.csv" ]] && [[ ! -f "$DATA_DIR/IMF_BOP.csv" ]]; then
    echo "Error: No economic data files found in $DATA_DIR"
    echo "Expected at least one of: WB_GEM.csv, IMF_BOP.csv"
    exit 1
fi

# Change to analysis directory for script execution
cd "$SCRIPT_DIR"

# Create results directory
mkdir -p results

echo "Comparing economic indicators across countries..."
echo "==============================================="
echo ""
echo "Running country comparison analysis..."
python country_comparison.py --output-dir ../results/ --clusters 5

if [[ $? -eq 0 ]]; then
    echo "✓ Country comparison analysis completed successfully"
else
    echo "✗ Country comparison analysis failed"
    exit 1
fi

echo ""
echo "========================================="
echo "Country comparison analysis completed!"
echo "========================================="
echo ""
echo "Generated files in $SCRIPT_DIR/../results/:"
echo "- country_rankings.txt (country rankings by economic indicators)"
echo "- country_clustering_analysis.txt (clustering results)"
echo "- country_comparison_*.png (comparison plots by dataset)"
echo "- country_clustering_*.png (clustering visualizations)"
echo ""
echo "Analysis includes:"
echo "- Country rankings by mean values"
echo "- Clustering analysis to identify similar countries"
echo "- Regional pattern analysis"
echo "- Volatility and performance metrics"
echo ""
echo "Next steps:"
echo "- Create comprehensive visualizations: bash $SCRIPT_DIR/run_visualization.sh" 