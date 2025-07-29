#!/bin/bash

# Complete Economic Data Analysis Pipeline
# Runs exploration, time series analysis, country comparison, and visualization

set -e  # Exit on any error

# Detect script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ANALYSIS_DIR="$SCRIPT_DIR"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================================"
echo "Complete Economic Data Analysis Pipeline"
echo "================================================================"
echo ""
echo "This script will run the complete economic analysis pipeline:"
echo "1. Data exploration and structure analysis"
echo "2. Time series pattern analysis"
echo "3. Country comparison and clustering"
echo "4. Comprehensive data visualization"
echo ""
echo "Script directory: $ANALYSIS_DIR"
echo "Data directory: $DATA_DIR"
echo "Working from: $(pwd)"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    source "$ROOT_DIR/.venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "Warning: Virtual environment not found at $ROOT_DIR/.venv/bin/activate"
    echo "Continuing with system Python..."
fi
echo ""

# Check dependencies
echo "Checking dependencies..."
command -v python >/dev/null 2>&1 || { echo "Error: python is required but not installed"; exit 1; }

# Check if we have the required Python packages
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, scipy" 2>/dev/null || {
    echo "Error: Required Python packages missing. Please install:"
    echo "  uv add pandas numpy matplotlib seaborn scikit-learn scipy"
    exit 1
}

echo "✓ Dependencies checked"
echo ""

# Check for economic data files
data_files_exist=false
for file in "$DATA_DIR/WB_GEM.csv" "$DATA_DIR/IMF_BOP.csv" "$DATA_DIR/IMF_CPI.csv" "$DATA_DIR/IMF_MFS.csv"; do
    if [[ -f "$file" ]]; then
        data_files_exist=true
        break
    fi
done

if [[ "$data_files_exist" = false ]]; then
    echo "Error: No economic data files found in $DATA_DIR!"
    echo "Expected files: WB_GEM.csv, IMF_BOP.csv, IMF_CPI.csv, IMF_MFS.csv"
    exit 1
fi

echo "✓ Economic data files found"
echo ""

# Ask for confirmation
read -p "Do you want to proceed with the full economic analysis pipeline? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Pipeline cancelled."
    exit 0
fi

# Record start time
start_time=$(date +%s)

echo ""
echo "Starting economic analysis pipeline at $(date)"
echo ""

# Change to analysis directory for execution
cd "$ANALYSIS_DIR"

# Step 1: Data Exploration
echo "================================================================"
echo "STEP 1: DATA EXPLORATION"
echo "================================================================"
bash "$ANALYSIS_DIR/run_exploration.sh"

# Step 2: Time Series Analysis
echo ""
echo "================================================================"
echo "STEP 2: TIME SERIES ANALYSIS"
echo "================================================================"
bash "$ANALYSIS_DIR/run_time_series.sh"

# Step 3: Country Comparison
echo ""
echo "================================================================"
echo "STEP 3: COUNTRY COMPARISON"
echo "================================================================"
bash "$ANALYSIS_DIR/run_country_comparison.sh"

# Step 4: Visualization
echo ""
echo "================================================================"
echo "STEP 4: VISUALIZATION"
echo "================================================================"
bash "$ANALYSIS_DIR/run_visualization.sh"

# Calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo ""
echo "================================================================"
echo "ECONOMIC ANALYSIS PIPELINE COMPLETED!"
echo "================================================================"
echo ""
echo "Total runtime: ${minutes}m ${seconds}s"
echo "Completed at: $(date)"
echo ""
echo "Results summary:"
echo "- Data exploration: $ANALYSIS_DIR/../results/data_summary_report.txt"
echo "- Time series analysis: $ANALYSIS_DIR/../results/country_time_series_analysis.txt"
echo "- Country comparison: $ANALYSIS_DIR/../results/country_rankings.txt"
echo "- Interactive plots: $ANALYSIS_DIR/../results/interactive_plots/"
echo ""
echo "Key outputs:"
echo "1. Data Structure Analysis:"
echo "   - Dataset summaries and statistics"
echo "   - Time series coverage analysis"
echo "   - Missing data assessment"
echo ""
echo "2. Time Series Analysis:"
echo "   - Trend and growth rate analysis"
echo "   - Volatility measurements"
echo "   - Temporal pattern visualization"
echo ""
echo "3. Country Comparison:"
echo "   - Country rankings by indicators"
echo "   - Clustering analysis"
echo "   - Regional pattern analysis"
echo ""
echo "4. Visualizations:"
echo "   - Static plots for detailed analysis"
echo "   - Interactive charts for exploration"
echo "   - Summary dashboard"
echo ""
echo "To explore the results:"
echo "1. View text reports in $ANALYSIS_DIR/../results/"
echo "2. Open PNG files for static visualizations"
echo "3. Open HTML files in a web browser for interactive exploration"
echo ""
echo "For help or to run individual components:"
echo "- bash $ANALYSIS_DIR/run_exploration.sh (data exploration only)"
echo "- bash $ANALYSIS_DIR/run_time_series.sh (time series analysis only)"
echo "- bash $ANALYSIS_DIR/run_country_comparison.sh (country comparison only)"
echo "- bash $ANALYSIS_DIR/run_visualization.sh (visualization only)" 