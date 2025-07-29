# SOUL Project Documentation Directory

This directory contains all documentation for the SOUL project, organized for easy navigation and reference.

## 📂 Documentation Structure

```
docs/
├── index.md                    # Main project documentation entry point
├── data-analysis.md           # Comprehensive data analysis guide
├── configuration.md           # Multi-agent simulation configuration
├── color-schemes.md          # Unified color scheme guide
├── citations.md              # Dataset citations and references
├── reference/                # Reference materials
│   ├── countries.md          # Country lists and regional mappings
│   ├── requirements.md       # Python package dependencies
│   ├── requirements.txt      # Raw requirements file
│   └── important_countries.txt # Important countries list
└── README.md                 # This file
```

## 📖 Quick Navigation

### **Start Here**
- **[📋 Project Overview](index.md)** - Complete project introduction, quick start, and architecture overview

### **Core Guides**
- **[📊 Data Analysis Guide](data-analysis.md)** - Political and economic data analysis pipelines
- **[⚙️ Configuration Guide](configuration.md)** - Multi-agent simulation configuration system
- **[🎨 Color Schemes Guide](color-schemes.md)** - Unified visualization color scheme
- **[📚 Citations & References](citations.md)** - Academic dataset citations

### **Reference Materials**
- **[🌍 Country Lists](reference/countries.md)** - Country codes, regions, and color mappings
- **[📦 Requirements](reference/requirements.md)** - Python dependencies and installation guide

## 🎯 Documentation by Use Case

### **New Users / Setup**
1. [Project Overview](index.md) - Understand what SOUL project does
2. [Requirements](reference/requirements.md) - Install dependencies
3. [Quick Start](index.md#quick-start) - Run your first analysis

### **Data Analysis**
1. [Data Analysis Guide](data-analysis.md) - Complete analysis workflow
2. [Color Schemes](color-schemes.md) - Consistent visualization
3. [Country Lists](reference/countries.md) - Country reference data
4. [Citations](citations.md) - Proper dataset attribution

### **Multi-Agent Simulation**
1. [Configuration Guide](configuration.md) - Environment setup
2. [Project Architecture](index.md#project-architecture) - System overview
3. [Requirements](reference/requirements.md) - RL dependencies

### **Development**
1. [Color Schemes](color-schemes.md) - Adding visualizations
2. [Country Lists](reference/countries.md) - Adding new countries
3. [Configuration](configuration.md) - Modifying simulation parameters
4. [Requirements](reference/requirements.md) - Managing dependencies

### **Research / Academic Use**
1. [Citations](citations.md) - Proper dataset attribution
2. [Data Analysis](data-analysis.md) - Research methodology
3. [Project Overview](index.md#research-context) - Academic applications

## 📋 Quick Reference

### **Running Analysis Pipelines**
```bash
# Political data analysis
./scripts/run_political_analysis.sh

# Economic data analysis  
./scripts/run_economic_analysis.sh

# Color scheme demo
python scripts/demo_unified_colors.py
```

### **Key Configuration Files**
- `configs/color_schemes.py` - Unified color management
- `configs/environment_config.py` - Simulation environment
- `pyproject.toml` - Project dependencies

### **Important Countries (Political Focus)**
```
USA, CHN, JPN, GER, UKG, FRN, RUS, KOR, IND, AUL
```

### **Main Datasets**
- **IGO**: Intergovernmental Organizations (1815-2014)
- **DCAD**: Defense Cooperation Agreements (1980-2010)
- **World Bank GEM**: Economic indicators
- **IMF**: Balance of Payments, CPI, Monetary Statistics

## 🔧 Documentation Maintenance

### **Adding New Documentation**
1. Create markdown file in appropriate location
2. Follow existing documentation structure and style
3. Update this README.md with navigation links
4. Update [index.md](index.md) if it's major documentation

### **Updating Existing Documentation**
1. Maintain consistency with existing style
2. Update cross-references when moving content
3. Test all code examples and commands
4. Update version information when applicable

### **Style Guidelines**
- Use clear, descriptive headings
- Include code examples for technical content
- Cross-reference related documentation
- Use emojis sparingly for section headers
- Include troubleshooting sections for complex topics

## 🏷️ Document Categories

### **Complete Guides** (Comprehensive documentation)
- [Data Analysis Guide](data-analysis.md) - 355 lines, complete workflows
- [Color Schemes Guide](color-schemes.md) - Comprehensive implementation guide
- [Project Overview](index.md) - Complete project introduction

### **Configuration Reference** (Technical specifications)
- [Configuration Guide](configuration.md) - Dataclass specifications
- [Requirements](reference/requirements.md) - Package specifications

### **Quick Reference** (Lookup tables and lists)
- [Country Lists](reference/countries.md) - Reference tables
- [Citations](citations.md) - Academic references

### **Raw Data Files** (Source files)
- `reference/requirements.txt` - Machine-readable requirements
- `reference/important_countries.txt` - Raw country list

## 📊 Documentation Stats

| Document | Type | Lines | Focus |
|----------|------|-------|-------|
| [index.md](index.md) | Overview | ~300 | Project introduction |
| [data-analysis.md](data-analysis.md) | Guide | 355 | Analysis workflows |
| [color-schemes.md](color-schemes.md) | Guide | ~250 | Visualization consistency |
| [configuration.md](configuration.md) | Reference | 145 | System configuration |
| [citations.md](citations.md) | Reference | 76 | Academic citations |
| [countries.md](reference/countries.md) | Reference | ~200 | Country data |
| [requirements.md](reference/requirements.md) | Reference | ~200 | Dependencies |

## 🤝 Contributing to Documentation

### **Documentation Improvements Welcome**
- Clarifications and examples
- Additional troubleshooting content  
- Cross-platform installation notes
- Academic use case examples

### **Submitting Updates**
1. Edit relevant markdown files
2. Test all code examples
3. Update navigation if needed
4. Maintain consistent style

---

**Documentation Version**: 1.0  
**Last Updated**: Current  
**Maintained by**: SOUL Project Team  

For questions about documentation or to suggest improvements, please refer to the main project repository or contact the maintainers. 