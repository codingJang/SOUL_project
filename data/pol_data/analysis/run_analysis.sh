#!/bin/bash

# Political Data Analysis Pipeline
# Runs significance tests and eigenanalysis on processed political data

set -e  # Exit on any error

# Detect script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "Political Data Analysis Pipeline"
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
mkdir -p results

echo "Step 1: Statistical Significance Testing..."
echo "==========================================="
echo ""
echo "Testing IGO groupings (both lv1 and lv2 for combined plot)..."
python significance_test.py --adjmat ../results/preprocessed/IGO_adjmat.csv --level lv1 lv2 --permutations 100 --output-dir ../results/ --seed 42

if [[ $? -eq 0 ]]; then
    echo "✓ Significance testing completed successfully for IGO"
else
    echo "✗ Significance testing failed for IGO"
    exit 1
fi

echo "Testing DCAD groupings (both lv1 and lv2 for combined plot)..."
python significance_test.py --adjmat ../results/preprocessed/DCAD_adjmat.csv --level lv1 lv2 --permutations 100 --output-dir ../results/ --seed 42

if [[ $? -eq 0 ]]; then
    echo "✓ Significance testing completed successfully for DCAD"
else
    echo "✗ Significance testing failed for DCAD"
    exit 1
fi

echo ""
echo "Step 2: Eigenanalysis and Correspondence Analysis..."
echo "==================================================="
python eigenanalysis.py --output-dir ../results/

if [[ $? -eq 0 ]]; then
    echo "✓ Eigenanalysis completed successfully"
else
    echo "✗ Eigenanalysis failed"
    exit 1
fi

echo ""
echo "Step 3: Network Statistics and Visualization Prep..."
echo "==================================================="
python visualization.py --stats-only --output-dir ../results/

if [[ $? -eq 0 ]]; then
    echo "✓ Network statistics completed successfully"
else
    echo "✗ Network statistics failed"
    exit 1
fi

echo ""
echo "========================================="
echo "Analysis pipeline completed!"
echo "========================================="
echo ""
echo "Generated files in $SCRIPT_DIR/../results/:"
echo "- goodness_histogram_igo.png (combined significance test plot for lv1 and lv2)"
echo "- correspondence_analysis_igo.png (IGO eigenanalysis)"
echo "- pca_analysis_igo.png (IGO PCA)"
echo "- correspondence_analysis_dcad.png (DCAD eigenanalysis)"
echo "- pca_analysis_dcad.png (DCAD PCA)"
echo ""
echo "Next steps:"
echo "- Create interactive visualizations: bash $SCRIPT_DIR/run_visualization.sh"
echo "- View results in $SCRIPT_DIR/../results/" 