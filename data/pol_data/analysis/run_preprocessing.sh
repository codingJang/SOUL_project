#!/bin/bash

# Political Data Preprocessing Pipeline
# Runs IGO and DCAD data preprocessing to generate adjacency matrices

set -e  # Exit on any error

# Detect script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "Political Data Preprocessing Pipeline"
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
if [[ ! -f "$DATA_DIR/IGO.txt" ]] || [[ ! -f "$DATA_DIR/DCAD.csv" ]]; then
    echo "Error: Required data files not found in $DATA_DIR"
    echo "Expected: IGO.txt, DCAD.csv"
    exit 1
fi

# Change to analysis directory for script execution
cd "$SCRIPT_DIR"

# Create output directory for results
mkdir -p results

echo "Step 1: Preprocessing IGO data..."
echo "---------------------------------"
python preprocess_igo.py

if [[ $? -eq 0 ]]; then
    echo "✓ IGO preprocessing completed successfully"
else
    echo "✗ IGO preprocessing failed"
    exit 1
fi

echo ""
echo "Step 2: Preprocessing DCAD data..."
echo "----------------------------------"
python preprocess_dcad.py

if [[ $? -eq 0 ]]; then
    echo "✓ DCAD preprocessing completed successfully"
else
    echo "✗ DCAD preprocessing failed"
    exit 1
fi

echo ""
echo "========================================="
echo "Preprocessing pipeline completed!"
echo "========================================="
echo ""
echo "Generated files in $DATA_DIR:"
echo "- ../results/preprocessed/IGO_adjmat.csv (full IGO adjacency matrix)"
echo "- ../results/preprocessed/IGO_adjmat_scaled.csv (scaled IGO adjacency matrix)"
echo "- ../results/preprocessed/IGO_adjmat_important.csv (important countries only)"
echo "- ../results/preprocessed/IGO_adjmat_scaled_important.csv (scaled, important countries)"
echo "- ../results/preprocessed/IGO_adjmat_scaled_top5pct.csv (top 5% connections)"
echo "- ../results/preprocessed/IGO_adjmat_scaled_top1pct_important.csv (top 1%, important countries)"
echo "- ../results/preprocessed/DCAD_adjmat.csv (full DCAD adjacency matrix)"
echo "- ../results/preprocessed/DCAD_adjmat_important.csv (important countries only)"
echo "- ../results/preprocessed/DCAD_adjmat_top5pct.csv (top 5% connections)"
echo ""
echo "Next steps:"
echo "- Run analysis: bash $SCRIPT_DIR/run_analysis.sh"
echo "- Run visualization: bash $SCRIPT_DIR/run_visualization.sh" 