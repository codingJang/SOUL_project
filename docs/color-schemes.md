# Unified Color Scheme for SOUL Project

This document explains how to use the unified color scheme across all visualizations in the SOUL project.

## Overview

The unified color scheme ensures consistent country colors across:
- **Matplotlib** static plots (economic/political analysis)
- **Plotly** interactive visualizations  
- **Pyvis** network graphs
- **UI widgets** (Qt/PySide6)

## Quick Start

```python
from configs.color_schemes import get_country_color, get_country_colors

# Get color for a single country
usa_color = get_country_color('USA', 'hex')          # '#1f77b4'
usa_rgb = get_country_color('USA', 'matplotlib')     # (0.12, 0.47, 0.71)
usa_plotly = get_country_color('USA', 'plotly')     # 'rgb(31, 119, 180)'

# Get colors for multiple countries
countries = ['USA', 'CHN', 'JPN', 'GER']
colors = get_country_colors(countries, 'matplotlib')
```

## Supported Color Formats

| Format | Usage | Example Output |
|--------|-------|----------------|
| `'hex'` | General use, web colors | `'#1f77b4'` |
| `'rgb'` | Matplotlib (normalized 0-1) | `(0.12, 0.47, 0.71)` |
| `'rgba'` | With alpha channel | `(0.12, 0.47, 0.71, 1.0)` |
| `'plotly'` | Plotly graphs | `'rgb(31, 119, 180)'` |
| `'matplotlib'` | Same as 'rgb' | `(0.12, 0.47, 0.71)` |

## Country Color Mappings

### Important Countries (Political Analysis)
- **USA**: Blue `#1f77b4` - Stable, trustworthy
- **CHN**: Red `#d62728` - Traditional Chinese color  
- **JPN**: Orange `#ff7f0e` - Distinctive and warm
- **GER**: Green `#2ca02c` - Environmental leadership
- **UKG**: Purple `#9467bd` - Royal association
- **FRN**: Brown `#8c564b` - Earthy, sophisticated
- **RUS**: Pink/Magenta `#e377c2` - Distinctive
- **KOR**: Gray `#7f7f7f` - Modern, technological
- **IND**: Olive `#bcbd22` - Traditional Indian colors
- **AUL**: Cyan `#17becf` - Ocean association

### Regional Colors
- **North America**: Blue `#1f77b4`
- **South America**: Light Green `#98df8a`
- **Europe**: Purple `#9467bd`
- **Asia**: Orange `#ff7f0e`
- **Africa**: Brown `#cd853f`
- **Oceania**: Cyan `#17becf`
- **Middle East**: Pink `#e377c2`
- **Other**: Gray `#7f7f7f`

## Usage Examples

### Matplotlib Plots

```python
import matplotlib.pyplot as plt
from configs.color_schemes import get_country_colors, create_country_color_map

countries = ['USA', 'CHN', 'JPN']
colors = get_country_colors(countries, 'matplotlib')

# Time series plot
for i, country in enumerate(countries):
    plt.plot(years, data[country], color=colors[i], label=country)

# Bar chart with color mapping
color_map = create_country_color_map(countries, 'matplotlib')
bars = plt.bar(countries, values, color=[color_map[c] for c in countries])
```

### Plotly Interactive Plots

```python
import plotly.graph_objects as go
from configs.color_schemes import get_country_colors

countries = ['USA', 'CHN', 'JPN']
colors = get_country_colors(countries, 'plotly')

fig = go.Figure()
for i, country in enumerate(countries):
    fig.add_trace(go.Scatter(
        x=years, y=data[country],
        name=country,
        line=dict(color=colors[i], width=2)
    ))
```

### UI Widgets

```python
# Updated widget methods support countries parameter
countries = ['USA', 'CHN', 'JPN']

# Bar plot with unified colors
widget.ShowPlot(values, len(countries), countries=countries)

# Line plot with unified colors  
widget.ShowHistoryPlot(time_series_data, len(countries), 
                       label='Economic Trends', countries=countries)

# Or manually get colors
colors = get_country_colors(countries, 'matplotlib')
widget.ShowPlot(values, len(countries), colors=colors)
```

### Network Visualizations (Pyvis)

```python
import networkx as nx
from pyvis.network import Network
from configs.color_schemes import get_country_color

# Create network
net = Network()

# Add nodes with unified colors
for country in countries:
    color = get_country_color(country, 'hex')
    net.add_node(country, color=color, title=country)
```

## Fallback Colors

Unknown countries automatically get consistent fallback colors:

```python
unknown_color = get_country_color('XYZ')  # Returns consistent color for 'XYZ'
```

## Adding New Countries

To add new countries, edit `configs/color_schemes.py`:

```python
self.extended_country_colors = {
    # Add your new country mappings
    'NEW': '#123456',  # New country code and color
    # ...
}
```

## Testing the Color Scheme

Run the demonstration script to see all features:

```bash
cd /path/to/SOUL_project
source .venv/bin/activate
python scripts/demo_unified_colors.py
```

This generates:
- Matplotlib demo plots showing all visualization types
- Interactive Plotly plots  
- Color consistency tables
- Example outputs in `data/results/color_demo/`

## Migration Guide

### From Existing Code

**Before:**
```python
# Hard-coded colors
colors = ['red', 'blue', 'green']
plt.plot(x, y, color='red')

# Seaborn palette
sns.set_palette("husl")
```

**After:**
```python
# Unified colors
from configs.color_schemes import get_country_colors
colors = get_country_colors(['USA', 'CHN', 'JPN'], 'matplotlib')
plt.plot(x, y, color=colors[0])

# No need for sns.set_palette
```

### Economic Analysis Files
- ✅ `data/econ_data/analysis/economic_visualization.py` - Updated
- ✅ `data/econ_data/analysis/country_comparison.py` - Updated

### Political Analysis Files  
- ✅ `data/pol_data/analysis/eigenanalysis.py` - Updated
- ✅ `data/pol_data/analysis/visualization.py` - Color-blind friendly updates

### UI Components
- ✅ `src/UI/item_widget.py` - Updated with countries parameter

## Design Principles

1. **Accessibility**: Colors chosen for colorblind accessibility
2. **Cultural Sensitivity**: Country colors reflect cultural associations where appropriate
3. **Consistency**: Same country = same color across all visualizations
4. **Extensibility**: Easy to add new countries and regions
5. **Flexibility**: Multiple format outputs for different libraries

## Troubleshooting

### Import Errors
Ensure the configs directory is in your Python path:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'configs'))
```

### Color Not Found
Unknown countries get fallback colors automatically. Check spelling of country codes.

### Visualization Issues
Different libraries may require different color formats. Use the appropriate format parameter:
- Matplotlib: `'matplotlib'` or `'rgb'`
- Plotly: `'plotly'` 
- Web/CSS: `'hex'`

## Best Practices

1. **Always specify format**: Use the correct format for your visualization library
2. **Use country codes**: Stick to standard 3-letter country codes when possible
3. **Test with demo**: Run the demo script when making changes
4. **Document new countries**: Add comments explaining color choices for new countries
5. **Consider accessibility**: Test color combinations for colorblind users

---

For questions or issues with the color scheme, check the demo script output or review the `configs/color_schemes.py` implementation. 