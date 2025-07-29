# Python Package Requirements

This document lists all Python package dependencies for the SOUL project and their specific use cases.

## Package Management

The project uses **uv** for Python package management. All dependencies are managed through:
- `pyproject.toml` - Main project configuration and dependencies
- `uv.lock` - Locked dependency versions for reproducible builds

## Installation

### Quick Setup
```bash
# Install all dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Adding New Packages
```bash
# Add a new package
uv add <package_name>

# Add development dependencies
uv add --dev <package_name>
```

## Core Dependencies

### Multi-Agent Simulation
```toml
gymnasium = "==0.28.1"          # RL environment framework
ray[rllib] = "==2.8.0"          # Distributed RL training
stable-baselines3 = ">=2.3.2"   # RL algorithms (PPO)
pettingzoo = ">=1.24.2"         # Multi-agent environments
supersuit = ">=3.9.3"           # Environment wrappers
torch = "==2.1.1"               # Neural network backend
```

### Data Analysis & Visualization
```toml
pandas = ">=2.3.1"              # Data manipulation
numpy = "==1.26.2"              # Numerical computing
matplotlib = "==3.8.2"          # Static plotting
seaborn = ">=0.13.2"            # Statistical visualization
plotly = ">=6.2.0"              # Interactive plotting
```

### Scientific Computing
```toml
scikit-learn = ">=1.7.1"        # Machine learning algorithms
scipy = ">=1.15.3"              # Scientific computing
```

### Network Analysis
```toml
networkx = ">=3.4.2"            # Network analysis
pyvis = ">=0.3.2"               # Interactive network visualization
```

### UI Components
```toml
PySide6 = "*"                   # Qt GUI framework
```

### Optimization & Experimentation
```toml
optuna = ">=4.4.0"              # Hyperparameter optimization
packaging = ">=25.0"            # Package utilities
setuptools = "<66"              # Build tools (version constraint)
```

## Package Usage by Component

### Multi-Agent Economic Simulation (`src/`)
- **gymnasium**: Environment interface and spaces
- **ray[rllib]**: APPO agent training and distributed computing
- **stable-baselines3**: PPO algorithm implementation
- **pettingzoo**: Multi-agent environment wrappers
- **supersuit**: Environment preprocessing and augmentation
- **torch**: Neural network models for agents
- **PySide6**: Real-time visualization UI

### Political Data Analysis (`data/pol_data/`)
- **pandas**: Data loading and manipulation for IGO/DCAD datasets
- **numpy**: Matrix operations for adjacency matrices
- **matplotlib**: Static plots for correspondence analysis and PCA
- **networkx**: Network analysis and graph algorithms
- **pyvis**: Interactive network visualizations
- **scikit-learn**: PCA and statistical analysis
- **scipy**: Statistical testing and significance analysis

### Economic Data Analysis (`data/econ_data/`)
- **pandas**: Time series data manipulation
- **numpy**: Numerical computations for economic indicators
- **matplotlib**: Time series and correlation plots
- **seaborn**: Statistical visualizations and heatmaps
- **plotly**: Interactive economic dashboards
- **scikit-learn**: Clustering and dimensionality reduction
- **scipy**: Statistical analysis and correlation testing

### Configuration Management (`configs/`)
- **packaging**: Version handling for platform detection
- **optuna**: Hyperparameter search range definitions

## Development Dependencies

Development packages (not in production requirements):
```bash
# Code quality
uv add --dev black            # Code formatting
uv add --dev isort            # Import sorting
uv add --dev flake8           # Linting
uv add --dev mypy             # Type checking

# Testing
uv add --dev pytest           # Testing framework
uv add --dev pytest-cov      # Coverage reporting

# Documentation
uv add --dev sphinx           # Documentation generation
uv add --dev sphinx-rtd-theme # Documentation theme
```

## Version Constraints

### Strict Version Pins
- **gymnasium**: `==0.28.1` - Ensures environment compatibility
- **ray[rllib]**: `==2.8.0` - APPO algorithm stability
- **torch**: `==2.1.1` - Neural network compatibility
- **numpy**: `==1.26.2` - Numerical stability
- **matplotlib**: `==3.8.2` - Visualization consistency

### Minimum Version Requirements
- **pandas**: `>=2.3.1` - Modern DataFrame API
- **scikit-learn**: `>=1.7.1` - Latest ML algorithms
- **scipy**: `>=1.15.3` - Scientific computing features
- **seaborn**: `>=0.13.2` - Statistical plotting features
- **plotly**: `>=6.2.0` - Interactive visualization features

### Version Constraints
- **setuptools**: `<66` - Compatibility with ray[rllib]

## Platform Compatibility

### Supported Platforms
- **macOS**: Primary development platform (Apple Silicon and Intel)
- **Linux**: Cluster computing environments
- **Windows**: Limited testing (should work with minor path adjustments)

### Platform-Specific Notes
- **macOS**: Uses Metal Performance Shaders for torch when available
- **Linux**: CUDA support for GPU acceleration with torch
- **Ray**: Automatically detects platform capabilities for resource allocation

## Installation Troubleshooting

### Common Issues

**Virtual Environment Not Found:**
```bash
# Create new virtual environment
uv sync
source .venv/bin/activate
```

**Package Import Errors:**
```bash
# Verify packages are installed
pip list | grep package_name

# Reinstall if missing
uv add package_name
```

**Version Conflicts:**
```bash
# Check for conflicts
uv lock --check

# Resolve conflicts
uv sync --refresh
```

**Ray Installation Issues:**
```bash
# Ray with RLlib support
uv add "ray[rllib]==2.8.0"

# Verify ray installation
python -c "import ray; print(ray.__version__)"
```

**Torch/GPU Issues:**
```bash
# CPU-only torch (if GPU issues)
uv add torch==2.1.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Verify torch installation
python -c "import torch; print(torch.__version__)"
```

## Package Documentation

### Key Package Documentation Links
- **gymnasium**: https://gymnasium.farama.org/
- **ray[rllib]**: https://docs.ray.io/en/latest/rllib/
- **stable-baselines3**: https://stable-baselines3.readthedocs.io/
- **pandas**: https://pandas.pydata.org/docs/
- **matplotlib**: https://matplotlib.org/stable/
- **plotly**: https://plotly.com/python/
- **networkx**: https://networkx.org/documentation/
- **scikit-learn**: https://scikit-learn.org/stable/

### SOUL Project Specific Usage
- **Color Schemes**: All visualization packages use unified colors from `configs/color_schemes.py`
- **Environment**: Configuration classes manage package-specific settings
- **Analysis**: Automated pipelines coordinate multiple packages for comprehensive analysis

## Testing Package Installation

### Verification Script
```python
# test_imports.py
import sys

packages = [
    'gymnasium', 'ray', 'stable_baselines3', 'pettingzoo',
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
    'sklearn', 'scipy', 'networkx', 'pyvis', 'torch'
]

for package in packages:
    try:
        __import__(package)
        print(f"✓ {package}")
    except ImportError as e:
        print(f"✗ {package}: {e}")
```

### Demo Scripts
```bash
# Test color scheme (matplotlib, plotly)
python scripts/demo_unified_colors.py

# Test data analysis (pandas, scipy, sklearn)
python data/econ_data/analysis/data_exploration.py --help

# Test network analysis (networkx, pyvis)
python data/pol_data/analysis/visualization.py --help
```

---

For package-specific issues, consult the individual package documentation or the project troubleshooting guide. 