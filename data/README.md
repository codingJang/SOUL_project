# SOUL Project Data Analysis

This directory contains organized data and analysis pipelines for the SOUL project, with separate analysis frameworks for political and economic data.

## Directory Structure

```
data/
├── pol_data/                    # Political data and analysis
│   ├── analysis/               # Analysis scripts and pipelines
│   │   ├── preprocess_igo.py         # IGO data preprocessing
│   │   ├── preprocess_dcad.py        # DCAD data preprocessing  
│   │   ├── visualization.py          # Network visualization
│   │   ├── significance_test.py      # Statistical significance testing
│   │   ├── eigenanalysis.py          # Eigenanalysis and correspondence analysis
│   │   ├── run_preprocessing.sh      # Preprocessing pipeline
│   │   ├── run_analysis.sh           # Analysis pipeline
│   │   ├── run_visualization.sh      # Visualization pipeline
│   │   └── run_full_pipeline.sh      # Complete pipeline
│   ├── archive/               # Archived jupyter notebooks and old files
│   └── [data files]           # IGO, DCAD, and country mapping files
├── econ_data/                 # Economic data and analysis
│   ├── analysis/              # Analysis scripts and pipelines
│   │   ├── data_exploration.py       # Data structure exploration
│   │   ├── time_series_analysis.py   # Time series pattern analysis
│   │   ├── country_comparison.py     # Country comparison and clustering
│   │   ├── economic_visualization.py # Comprehensive visualizations
│   │   ├── run_exploration.sh        # Data exploration pipeline
│   │   ├── run_time_series.sh        # Time series analysis pipeline
│   │   ├── run_country_comparison.sh # Country comparison pipeline
│   │   ├── run_visualization.sh      # Visualization pipeline
│   │   └── run_full_pipeline.sh      # Complete pipeline
│   ├── archive/               # Archived files (if any)
│   └── [data files]           # IMF, World Bank, and processed datasets
└── README.md                  # This file

scripts/                       # Convenience scripts for easy execution
├── run_political_analysis.sh  # Political data analysis runner
└── run_economic_analysis.sh   # Economic data analysis runner
```

## Quick Start

### Running from Root Directory (Recommended)

The easiest way to run the analysis pipelines is from the project root directory using the convenience scripts:

```bash
# Political data analysis
./scripts/run_political_analysis.sh

# Economic data analysis  
./scripts/run_economic_analysis.sh
```

**Note**: All scripts automatically activate the virtual environment (`.venv/`) and use the proper Python interpreter with installed packages.

### Political Data Analysis

To run the complete political data analysis pipeline:

```bash
# From root directory (recommended)
./scripts/run_political_analysis.sh

# Or from analysis directory
cd data/pol_data/analysis
./run_full_pipeline.sh

# Or using bash from anywhere
bash data/pol_data/analysis/run_full_pipeline.sh
```

Individual components:

```bash
# Data preprocessing
bash data/pol_data/analysis/run_preprocessing.sh

# Statistical analysis
bash data/pol_data/analysis/run_analysis.sh

# Interactive visualizations
bash data/pol_data/analysis/run_visualization.sh
```

### Economic Data Analysis

To run the complete economic data analysis pipeline:

```bash
# From root directory (recommended)
./scripts/run_economic_analysis.sh

# Or from analysis directory
cd data/econ_data/analysis
./run_full_pipeline.sh

# Or using bash from anywhere
bash data/econ_data/analysis/run_full_pipeline.sh
```

Individual components:

```bash
# Data exploration
bash data/econ_data/analysis/run_exploration.sh

# Time series analysis
bash data/econ_data/analysis/run_time_series.sh

# Country comparison
bash data/econ_data/analysis/run_country_comparison.sh

# Comprehensive visualization
bash data/econ_data/analysis/run_visualization.sh
```

## Dependencies

Install the required Python packages using uv:

```bash
# Install all required packages
uv add pandas numpy matplotlib seaborn scikit-learn scipy networkx pyvis

# Optional for interactive economic visualizations
uv add plotly
```

The shell scripts will automatically:
1. Activate the virtual environment (`.venv/`)
2. Use the correct Python interpreter with installed packages
3. Check for required dependencies before running

## Python Environment

All analysis scripts follow the uv workflow:

1. **Virtual Environment**: Scripts automatically activate `.venv/bin/activate` if available
2. **Python Commands**: Use `python` directly (not `uv run python3`) after venv activation
3. **Package Management**: Install packages with `uv add <package>` as needed

### Manual Python Execution

If you want to run Python scripts manually:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Then run scripts directly
cd data/pol_data/analysis
python significance_test.py --level lv1 --permutations 1000

# Or run from anywhere (after venv activation)
python data/econ_data/analysis/time_series_analysis.py --output-dir results/
```

## Political Data Analysis

### Data Sources
- **IGO (Intergovernmental Organizations)**: A comprehensive dataset tracking the status and membership of intergovernmental organizations from 1815-2014. IGOs are international organizations with at least 3 nation-states as members. Data collected at 5-year intervals from 1815-1965, annually thereafter. Enables analysis of multilateral cooperation patterns and institutional membership networks. *Citation: Pevehouse, Jon C.W., Timothy Nordstron, Roseanne W McManus, Anne Spencer Jamison, "Tracking Organizations in the World: The Correlates of War IGO Version 3.0 datasets", Journal of Peace Research.*
- **DCAD (Defense Cooperation Agreement Dataset)**: A comprehensive, human-coded dataset of bilateral defense cooperation agreements (DCAs) covering 1980-2010. DCAs are formal international agreements that coordinate defense relations including joint military exercises, peacekeeping operations, defense research, weapons programs, officer exchanges, and policy coordination. Contains 1,872 unique agreements. *Citation: Kinne, Brandon J. 2020. "The Defense Cooperation Agreement Dataset (DCAD)," The Journal of Conflict Resolution 64(4): 729-755.*
- **Country mappings**: Abbreviations, groupings, and important countries lists

### Analysis Components

1. **Preprocessing** (`preprocess_igo.py`, `preprocess_dcad.py`)
   - Converts raw data into adjacency matrices
   - Creates scaled and filtered versions
   - Focuses on important countries subset
   - IGO analysis captures multilateral institutional membership patterns
   - DCAD analysis filters to agreements from 2000 onwards for contemporary patterns

2. **Statistical Analysis** (`significance_test.py`)
   - Permutation tests for country grouping validity
   - Statistical significance assessment
   - Goodness measure calculations

3. **Eigenanalysis** (`eigenanalysis.py`)
   - Principal Component Analysis (PCA)
   - Correspondence analysis
   - Dimensionality reduction

4. **Visualization** (`visualization.py`)
   - Interactive network graphs
   - Country relationship visualization
   - Network statistics and metrics

### Outputs
- Adjacency matrices (CSV format)
- Statistical significance results
- Interactive HTML network visualizations
- Analysis plots and summaries

## Economic Data Analysis

### Data Sources
- **World Bank GEM**: Quarterly GDP and economic indicators
- **IMF BOP**: Balance of Payments data
- **IMF CPI**: Consumer Price Index data  
- **IMF MFS**: Monetary and Financial Statistics
- **Processed datasets**: Training, testing, and validation splits

### Analysis Components

1. **Data Exploration** (`data_exploration.py`)
   - Dataset structure analysis
   - Missing data assessment
   - Time series coverage analysis

2. **Time Series Analysis** (`time_series_analysis.py`)
   - Temporal pattern analysis
   - Trend and growth rate calculations
   - Volatility measurements
   - Country-level time series statistics

3. **Country Comparison** (`country_comparison.py`)
   - Cross-country indicator comparison
   - Clustering analysis to identify similar countries
   - Regional pattern analysis
   - Country ranking by economic performance

4. **Visualization** (`economic_visualization.py`)
   - Time series plots
   - Correlation analysis
   - Interactive dashboards
   - Summary statistics visualization

### Outputs
- Data summary reports
- Time series analysis results
- Country rankings and clustering results
- Static and interactive visualizations

## Usage Examples

### Running Specific Analysis

```bash
# Activate virtual environment first
source .venv/bin/activate

# Political data: Test significance of country groupings
cd data/pol_data/analysis
python significance_test.py --level lv1 --permutations 1000

# Economic data: Analyze time series patterns
cd data/econ_data/analysis  
python time_series_analysis.py --output-dir results/

# Economic data: Compare countries with clustering
python country_comparison.py --clusters 5 --output-dir results/
```

### Customizing Analysis

Most scripts accept command-line arguments for customization:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Political data with custom parameters
cd data/pol_data/analysis
python visualization.py --min-weight 0.2 --output-dir custom_results/

# Economic data with specific settings
cd data/econ_data/analysis
python country_comparison.py --clusters 8 --no-plots
```

## Flexible Execution

The analysis pipelines can be run from multiple locations:

1. **From root directory** (recommended): Use `./scripts/run_political_analysis.sh` or `./scripts/run_economic_analysis.sh`
2. **From analysis directories**: Use `./run_full_pipeline.sh` 
3. **From anywhere**: Use `bash data/[pol_data|econ_data]/analysis/run_full_pipeline.sh`

All scripts automatically detect their location and set correct paths, so you can run them from any directory while maintaining the same directory structure.

## Results Structure

After running the pipelines, results are organized as follows:

```
analysis/results/
├── [analysis_reports].txt      # Text summaries and statistics
├── [static_plots].png          # Static visualizations
├── visualizations/             # Interactive visualizations
│   └── [interactive_plots].html
└── [data_outputs].csv          # Processed data outputs
```

## Development and Customization

### Adding New Analysis

1. Create new Python script in the appropriate `analysis/` directory
2. Follow the existing pattern for argument parsing and output handling
3. Add corresponding shell script or integrate into existing pipelines

### Modifying Existing Analysis

- Python scripts use argparse for flexible configuration
- Shell scripts can be modified to change default parameters
- Output directories and file names can be customized

### Data Integration

- New datasets can be added to the respective data directories
- Update the loading functions in Python scripts to include new data sources
- Modify shell scripts to handle new data dependencies

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages using `uv add <package>`
2. **Virtual Environment**: Ensure `.venv/` exists by running `uv sync` in project root
3. **File Not Found**: Ensure you're running scripts from the correct directory
4. **Permission Denied**: Make shell scripts executable with `chmod +x *.sh`
5. **Memory Issues**: For large datasets, consider reducing sample sizes in scripts

### Virtual Environment Issues

If you encounter Python import errors:

```bash
# Ensure virtual environment exists
uv sync

# Check if packages are installed
source .venv/bin/activate
python -c "import pandas, numpy, matplotlib"

# Reinstall missing packages
uv add pandas numpy matplotlib seaborn scikit-learn scipy networkx pyvis
```

### Getting Help

- Run any script with `--help` to see available options
- Check the script source code for detailed documentation
- Review the output logs for specific error messages

## Archive

Old jupyter notebooks and legacy files have been moved to `archive/` directories for reference while maintaining a clean working environment.

---

For questions or issues, please refer to the individual script documentation or the main SOUL project repository. 