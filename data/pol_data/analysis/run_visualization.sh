#!/bin/bash

# Political Data Visualization Pipeline
# Creates interactive network visualizations from political data

set -e  # Exit on any error

# Detect script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "Political Data Visualization Pipeline"
echo "========================================="
echo ""
echo "Analysis directory: $SCRIPT_DIR"
echo "Data directory: $DATA_DIR"
echo ""

# Activate virtual environment
if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    source "$ROOT_DIR/.venv/bin/activate"
fi

# Check if preprocessing has been run
if [[ ! -f "../results/preprocessed/IGO_adjmat.csv" ]] || [[ ! -f "../results/preprocessed/DCAD_adjmat.csv" ]]; then
    echo "Error: Preprocessing not completed. Run preprocessing first:"
    echo "  bash $SCRIPT_DIR/run_preprocessing.sh"
    exit 1
fi

# Change to analysis directory for script execution
cd "$SCRIPT_DIR"

# Create results directory
mkdir -p ../results/interactive_plots

echo "Creating interactive network visualizations..."
echo "=============================================="

# Create visualizations with different filtering options
echo ""
echo "1. Full networks (all connections)..."
python visualization.py --output-dir ../results/interactive_plots/ --min-weight 0

echo ""
echo "2. Filtered networks (stronger connections only)..."
python visualization.py --output-dir ../results/interactive_plots/ --min-weight 0.1

echo ""
echo "3. High-quality connections only..."
python visualization.py --output-dir ../results/interactive_plots/ --min-weight 0.3

if [[ $? -eq 0 ]]; then
    echo "✓ Visualization pipeline completed successfully"
else
    echo "✗ Visualization pipeline failed"
    exit 1
fi

echo ""
echo "========================================="
echo "Visualization pipeline completed!"
echo "========================================="
echo ""
echo "Generated interactive HTML files in $SCRIPT_DIR/../results/interactive_plots/:"
ls -la "$SCRIPT_DIR/../results/interactive_plots/"*.html 2>/dev/null || echo "No HTML files found"
echo ""
echo "Open these files in your web browser to explore the networks interactively."
echo ""
echo "Files typically include:"
echo "- graph_0_igo_scaled_important.html (IGO important countries network)"
echo "- graph_1_dcad_top5pct.html (DCAD top connections network)"
echo "- Additional filtered versions with different connection thresholds" 