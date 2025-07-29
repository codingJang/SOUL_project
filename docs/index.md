# SOUL Project Documentation

Welcome to the SOUL Project documentation! This project implements a multi-agent economic simulation system with comprehensive political and economic data analysis frameworks.

## 📖 Documentation Overview

### Core Documentation
- **[Data Analysis Guide](data-analysis.md)** - Comprehensive guide to political and economic data analysis pipelines
- **[Configuration Guide](configuration.md)** - Configuration system for multi-agent simulation environments
- **[Color Schemes Guide](color-schemes.md)** - Unified color scheme for consistent visualizations
- **[Citations & References](citations.md)** - Dataset citations and academic references

### Reference Materials
- **[Country Lists](reference/countries.md)** - Important countries and regional mappings
- **[Requirements](reference/requirements.md)** - Python package dependencies

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- uv package manager

### Installation
```bash
# Clone and setup
git clone <repository-url>
cd SOUL_project

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

### Running Analysis Pipelines

**Political Data Analysis:**
```bash
./scripts/run_political_analysis.sh
```

**Economic Data Analysis:**
```bash
./scripts/run_economic_analysis.sh
```

**Color Scheme Demo:**
```bash
python scripts/demo_unified_colors.py
```

## 🏗️ Project Architecture

### Core Components
1. **Multi-Agent Simulation** (`src/`)
   - Economic environment simulation
   - APPO reinforcement learning agents
   - UI components for visualization

2. **Data Analysis** (`data/`)
   - Political data analysis (IGO, DCAD datasets)
   - Economic data analysis (IMF, World Bank datasets)
   - Automated analysis pipelines

3. **Configuration** (`configs/`)
   - Environment configuration classes
   - Training configuration for RL algorithms
   - Unified color scheme management

4. **Models** (`models/`)
   - Trained RL agent checkpoints
   - Model artifacts and metadata

## 🔍 Analysis Capabilities

### Political Data Analysis
- **IGO Analysis**: Intergovernmental organization membership networks (1815-2014)
- **DCAD Analysis**: Defense cooperation agreements (1980-2010)
- **Network Visualization**: Interactive network graphs
- **Statistical Testing**: Permutation tests and significance analysis
- **Eigenanalysis**: PCA and correspondence analysis

### Economic Data Analysis
- **Time Series Analysis**: Economic indicator patterns and trends
- **Country Comparison**: Cross-country analysis with clustering
- **Data Exploration**: Comprehensive dataset structure analysis
- **Interactive Visualization**: Plotly dashboards and static plots

### Visualization Features
- **Unified Colors**: Consistent country colors across all visualizations
- **Multiple Formats**: Matplotlib, Plotly, Pyvis, and UI widget support
- **Accessibility**: Colorblind-friendly palette design
- **Interactive Elements**: Web-based network graphs and dashboards

## 🎯 Key Features

### Multi-Agent Economic Simulation
- **7-agent economic system** with heterogeneous agent types
- **APPO reinforcement learning** for agent training
- **Configurable environments** with flexible parameter systems
- **Real-time visualization** through Qt/PySide6 interface

### Data Analysis Pipelines
- **Automated preprocessing** for political and economic datasets
- **Statistical significance testing** with permutation methods
- **Comprehensive visualization** with unified color schemes
- **Flexible execution** from multiple directory locations

### Research-Ready Outputs
- **Academic citations** for all datasets used
- **Reproducible analysis** with documented pipelines
- **Publication-quality plots** with consistent styling
- **Interactive visualizations** for exploration and presentation

## 📊 Datasets

### Political Data
- **IGO Dataset**: 2-century view of international organization membership
- **DCAD Dataset**: Bilateral defense cooperation agreements
- **Coverage**: Global scope with focus on major powers

### Economic Data
- **World Bank GEM**: Quarterly GDP and economic indicators
- **IMF Datasets**: Balance of Payments, CPI, Monetary Statistics
- **Processing**: Automated cleaning and standardization

## 🛠️ Development

### Code Organization
```
SOUL_project/
├── src/                    # Multi-agent simulation code
├── data/                   # Data analysis pipelines
├── configs/                # Configuration management
├── scripts/                # Execution scripts
├── models/                 # Trained models
└── docs/                   # Documentation
```

### Development Workflow
1. **Environment Setup**: Use uv for package management
2. **Configuration**: Modify configs for different scenarios
3. **Analysis**: Run pipelines or individual scripts
4. **Visualization**: Generate plots with unified colors
5. **Model Training**: Train RL agents with APPO

### Best Practices
- **Virtual Environment**: Always use `.venv/bin/activate`
- **Package Management**: Install with `uv add <package>`
- **Color Consistency**: Use unified color scheme for all plots
- **Documentation**: Update docs when adding features

## 📚 Research Context

This project supports research in:
- **Multi-agent economic modeling**
- **International relations network analysis**
- **Economic indicator pattern recognition**
- **Reinforcement learning in economic systems**

### Academic Applications
- Political science research on international cooperation
- Economic analysis of cross-country patterns
- Machine learning applications in social sciences
- Network analysis of global institutions

## 🤝 Contributing

### Adding New Analysis
1. Create analysis script in appropriate `data/` subdirectory
2. Follow existing patterns for configuration and output
3. Add to relevant pipeline scripts
4. Update documentation

### Adding New Visualizations
1. Use unified color scheme from `configs/color_schemes.py`
2. Support multiple output formats (static, interactive)
3. Follow accessibility guidelines
4. Document usage patterns

### Adding New Countries
1. Update color mappings in `configs/color_schemes.py`
2. Add regional classifications if needed
3. Test with demo script
4. Update reference documentation

## 📋 Support

### Troubleshooting
- **Dependencies**: Run `uv sync` to ensure packages are installed
- **Virtual Environment**: Activate with `source .venv/bin/activate`
- **Scripts**: Make executable with `chmod +x *.sh`
- **Colors**: Test with `python scripts/demo_unified_colors.py`

### Getting Help
- Check individual script help with `--help` flag
- Review script source code for detailed documentation
- Consult dataset provider guidelines for data questions
- Use the demo scripts to understand expected behavior

---

**Last Updated**: {{ current_date }}  
**Project**: SOUL Multi-Agent Economic Simulation  
**Documentation Version**: 1.0  

For questions or contributions, please refer to the individual documentation sections or contact the project maintainers. 