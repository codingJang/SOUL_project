# SOUL Project

A multi-agent economic simulation system with comprehensive political and economic data analysis frameworks.

## 🚀 Quick Start

```bash
# Setup
uv sync
source .venv/bin/activate

# Start Web Interface (Recommended)
./scripts/run_webapp.sh
# Then open http://localhost:8000 in your browser

# Or run analysis pipelines directly
./scripts/run_political_analysis.sh
./scripts/run_economic_analysis.sh

# Test unified color scheme
python scripts/demo_unified_colors.py
```

## 📖 Complete Documentation

**📋 [Full Documentation](docs/index.md)** - Complete project overview, architecture, and guides

### Key Documentation
- **[Web Interface Guide](docs/web-interface.md)** - Modern web UI for simulations
- **[Data Analysis Guide](docs/data-analysis.md)** - Political & economic analysis workflows
- **[Configuration Guide](docs/configuration.md)** - Multi-agent simulation setup
- **[Color Schemes Guide](docs/color-schemes.md)** - Unified visualization colors
- **[Citations](docs/citations.md)** - Dataset references

## 🎯 Key Features

- **Modern Web Interface** with real-time simulation visualization
- **Multi-Agent Economic Simulation** with APPO reinforcement learning
- **Political Data Analysis** (IGO, DCAD networks 1815-2014)
- **Economic Data Analysis** (World Bank, IMF indicators)
- **Unified Color Scheme** across all visualizations
- **Interactive Visualizations** with matplotlib, plotly, pyvis

## 🏗️ Project Structure

```
SOUL_project/
├── src/                    # Multi-agent simulation
├── data/                   # Analysis pipelines & datasets
├── configs/                # Configuration management
├── docs/                   # Complete documentation
├── scripts/                # Execution scripts
└── models/                 # Trained RL agents
```

## 📊 Datasets

- **IGO**: Intergovernmental Organizations (1815-2014)
- **DCAD**: Defense Cooperation Agreements (1980-2010)  
- **Economic**: World Bank GEM, IMF BOP/CPI/MFS

## 🤝 Research Applications

- International relations network analysis
- Economic indicator pattern recognition
- Multi-agent economic modeling
- Reinforcement learning in social systems

---

**For complete documentation, installation guides, API references, and examples, see [docs/](docs/)** 