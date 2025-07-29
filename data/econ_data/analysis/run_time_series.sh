#!/bin/bash

# Economic Time Series Analysis Pipeline
# Analyzes temporal patterns and trends in economic data

set -e  # Exit on any error

# Detect script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "Economic Time Series Analysis Pipeline"
echo "========================================="
echo ""
echo "Analysis directory: $SCRIPT_DIR"
echo "Data directory: $DATA_DIR"
echo ""

# Activate virtual environment
if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    source "$ROOT_DIR/.venv/bin/activate"
fi

# Check if we have time series data
if [[ ! -f "$DATA_DIR/WB_GEM.csv" ]] && [[ ! -f "$DATA_DIR/IMF_BOP.csv" ]] && [[ ! -f "$DATA_DIR/IMF_CPI.csv" ]]; then
    echo "Error: No time series data files found in $DATA_DIR"
    echo "Expected at least one of: WB_GEM.csv, IMF_BOP.csv, IMF_CPI.csv"
    exit 1
fi

# Change to analysis directory for script execution
cd "$SCRIPT_DIR"

# Create results directory
mkdir -p results

echo "Analyzing temporal patterns in economic data..."
echo "=============================================="
python time_series_analysis.py --output-dir ../results/

if [[ $? -eq 0 ]]; then
    echo "✓ Time series analysis completed successfully"
else
    echo "✗ Time series analysis failed"
    exit 1
fi

echo ""
echo "========================================="
echo "Time series analysis completed!"
echo "========================================="
echo ""
echo "Generated files in $SCRIPT_DIR/../results/:"
echo "- country_time_series_analysis.txt (country-level statistics)"
echo "- time_series_overview_*.png (time series plots by dataset)"
echo ""
echo "Analysis includes:"
echo "- Trend analysis and growth rates"
echo "- Volatility measurements"
echo "- Temporal coverage assessment"
echo "- Country-specific time series statistics"
echo ""
echo "Next steps:"
echo "- Run country comparison: bash $SCRIPT_DIR/run_country_comparison.sh"
echo "- Create visualizations: bash $SCRIPT_DIR/run_visualization.sh" 