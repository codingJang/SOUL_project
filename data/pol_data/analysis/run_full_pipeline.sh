#!/bin/bash

# Complete Political Data Analysis Pipeline
# Runs preprocessing, analysis, and visualization in sequence

set -e  # Exit on any error

# Detect script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ANALYSIS_DIR="$SCRIPT_DIR"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================================"
echo "Complete Political Data Analysis Pipeline"
echo "================================================================"
echo ""
echo "This script will run the complete analysis pipeline:"
echo "1. Data preprocessing (IGO and DCAD)"
echo "2. Statistical analysis and significance testing"
echo "3. Eigenanalysis and dimensionality reduction"
echo "4. Interactive network visualization"
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
python -c "import pandas, numpy, networkx, matplotlib, sklearn" 2>/dev/null || {
    echo "Error: Required Python packages missing. Please install:"
    echo "  uv add pandas numpy networkx matplotlib scikit-learn pyvis"
    exit 1
}

echo "✓ Dependencies checked"
echo ""

# Check for required data files
if [[ ! -f "$DATA_DIR/IGO.txt" ]] || [[ ! -f "$DATA_DIR/DCAD.csv" ]]; then
    echo "Error: Required data files not found in $DATA_DIR"
    echo "Expected: IGO.txt, DCAD.csv"
    exit 1
fi

echo "✓ Required data files found"
echo ""

# Ask for confirmation
read -p "Do you want to proceed with the full pipeline? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Pipeline cancelled."
    exit 0
fi

# Record start time
start_time=$(date +%s)

echo ""
echo "Starting pipeline at $(date)"
echo ""

# Change to analysis directory for execution
cd "$ANALYSIS_DIR"

# Step 1: Preprocessing
echo "================================================================"
echo "STEP 1: DATA PREPROCESSING"
echo "================================================================"
bash "$ANALYSIS_DIR/run_preprocessing.sh"

# Step 2: Analysis
echo ""
echo "================================================================"
echo "STEP 2: STATISTICAL ANALYSIS"
echo "================================================================"
bash "$ANALYSIS_DIR/run_analysis.sh"

# Step 3: Visualization
echo ""
echo "================================================================"
echo "STEP 3: VISUALIZATION"
echo "================================================================"
bash "$ANALYSIS_DIR/run_visualization.sh"

# Calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo ""
echo "================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "================================================================"
echo ""
echo "Total runtime: ${minutes}m ${seconds}s"
echo "Completed at: $(date)"
echo ""
echo "Results summary:"
echo "- Adjacency matrices: ../results/preprocessed/IGO_adjmat*.csv, ../results/preprocessed/DCAD_adjmat*.csv"
echo "- Analysis results: $ANALYSIS_DIR/../results/"
echo "- Interactive visualizations: $ANALYSIS_DIR/../results/interactive_plots/"
echo ""
echo "To explore the results:"
echo "1. View static plots in $ANALYSIS_DIR/../results/"
echo "2. Open HTML files in $ANALYSIS_DIR/../results/interactive_plots/ with a web browser"
echo "3. Check the significance test results for statistical validation"
echo ""
echo "For help or to run individual components:"
echo "- bash $ANALYSIS_DIR/run_preprocessing.sh (data preprocessing only)"
echo "- bash $ANALYSIS_DIR/run_analysis.sh (analysis only)"
echo "- bash $ANALYSIS_DIR/run_visualization.sh (visualization only)" 