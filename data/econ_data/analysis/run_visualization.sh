#!/bin/bash

# Economic Data Visualization Pipeline
# Creates comprehensive visualizations for economic indicators

set -e  # Exit on any error

# Detect script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "Economic Data Visualization Pipeline"
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

echo "Creating economic data visualizations..."
echo "======================================="

# Create static visualizations
echo ""
echo "1. Creating static visualizations..."
python economic_visualization.py --output-dir ../results/ --static-only

echo ""
echo "2. Creating interactive visualizations (if available)..."
python economic_visualization.py --output-dir ../results/ --interactive || {
    echo "Note: Interactive visualizations skipped (plotly not available)"
}

if [[ $? -eq 0 ]]; then
    echo "✓ Visualization pipeline completed successfully"
else
    echo "✗ Visualization pipeline failed"
    exit 1
fi

echo ""
echo "========================================="
echo "Economic visualization completed!"
echo "========================================="
echo ""
echo "Generated visualizations in $SCRIPT_DIR/../results/:"
echo ""
echo "Static Plots:"
echo "- time_series_*.png (time series analysis by dataset)"
echo "- correlation_matrix.png (indicator correlations)"
echo "- economic_dashboard.png (summary dashboard)"
echo "- dataset_summary.txt (statistical summary)"
echo ""
echo "Interactive Plots (if created):"
ls -la "$SCRIPT_DIR/../results/"*.html 2>/dev/null | grep -E "(interactive|dashboard)" || echo "- No interactive files created"
echo ""
echo "Open the HTML files in your web browser for interactive exploration."
echo ""
echo "Visualizations include:"
echo "- Time series trends and patterns"
echo "- Country comparisons and rankings"
echo "- Correlation analysis between indicators"
echo "- Distribution and statistical summaries" 