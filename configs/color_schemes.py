#!/usr/bin/env python3
"""
Unified Color Schemes for SOUL Project

This module provides consistent color mappings for countries across all
visualization types including matplotlib, plotly, pyvis, and UI widgets.

The color scheme is designed with accessibility in mind and provides:
- Consistent country-to-color mapping
- Support for multiple visualization libraries
- Regional color groupings
- Accessibility-friendly color palettes
- Fallback colors for new countries
"""

import colorsys
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.colors as mcolors
import numpy as np


class ColorSchemeManager:
    """Manages unified color schemes for country visualizations."""
    
    def __init__(self):
        """Initialize with default color schemes."""
        self._initialize_country_colors()
        self._initialize_regional_colors()
        self._initialize_fallback_palette()
    
    def _initialize_country_colors(self) -> None:
        """Initialize specific colors for important countries."""
        # Primary important countries with distinct, accessible colors
        self.important_country_colors = {
            'USA': '#1f77b4',  # Blue - stable, trustworthy
            'CHN': '#d62728',  # Red - traditional Chinese color
            'JPN': '#ff7f0e',  # Orange - distinctive and warm
            'GER': '#2ca02c',  # Green - environmental leadership
            'UKG': '#9467bd',  # Purple - royal association
            'FRN': '#8c564b',  # Brown - earthy, sophisticated
            'RUS': '#e377c2',  # Pink/Magenta - distinctive
            'KOR': '#7f7f7f',  # Gray - modern, technological
            'IND': '#bcbd22',  # Olive - traditional Indian colors
            'AUL': '#17becf',  # Cyan - ocean association
        }
        
        # Extended color mapping for common economic analysis countries
        self.extended_country_colors = {
            # North America
            'US': '#1f77b4',   # Same as USA
            'CA': '#aec7e8',   # Light blue (related to US)
            'MX': '#ffbb78',   # Light orange
            
            # Europe
            'DE': '#2ca02c',   # Same as GER
            'FR': '#8c564b',   # Same as FRN
            'IT': '#ff9896',   # Light red
            'ES': '#c5b0d5',   # Light purple
            'UK': '#9467bd',   # Same as UKG
            'GB': '#9467bd',   # Same as UKG
            'NL': '#c49c94',   # Light brown
            'SE': '#f7b6d3',   # Light pink
            'NO': '#c7c7c7',   # Light gray
            
            # Asia Pacific
            'CN': '#d62728',   # Same as CHN
            'JP': '#ff7f0e',   # Same as JPN
            'KR': '#7f7f7f',   # Same as KOR
            'IN': '#bcbd22',   # Same as IND
            'AU': '#17becf',   # Same as AUL
            'TH': '#1fbecf',   # Cyan variant
            'SG': '#98df8a',   # Light green
            'MY': '#ff1744',   # Bright red
            
            # South America
            'BR': '#98df8a',   # Green - forest association
            'AR': '#87ceeb',   # Sky blue
            'CL': '#dda0dd',   # Plum
            'CO': '#f0e68c',   # Khaki
            
            # Africa
            'ZA': '#cd853f',   # Peru brown
            'NG': '#228b22',   # Forest green
            'EG': '#ffd700',   # Gold
            'KE': '#dc143c',   # Crimson
            
            # Middle East
            'SA': '#800080',   # Purple
            'AE': '#4169e1',   # Royal blue
            'IL': '#00ced1',   # Dark turquoise
            'TR': '#ff6347',   # Tomato
        }
        
        # Merge the dictionaries
        self.country_colors = {**self.important_country_colors, **self.extended_country_colors}
    
    def _initialize_regional_colors(self) -> None:
        """Initialize color schemes for regional groupings."""
        self.regional_colors = {
            'North America': '#1f77b4',  # Blue
            'South America': '#98df8a',  # Light green
            'Europe': '#9467bd',         # Purple
            'Asia': '#ff7f0e',           # Orange
            'Africa': '#cd853f',         # Brown
            'Oceania': '#17becf',        # Cyan
            'Middle East': '#e377c2',    # Pink
            'Other': '#7f7f7f',          # Gray
        }
    
    def _initialize_fallback_palette(self) -> None:
        """Initialize fallback color palette for unknown countries."""
        # Generate a diverse, accessible color palette
        self.fallback_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5',
            '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
            '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194'
        ]
        self._fallback_index = 0
    
    def get_country_color(self, country_code: str, format_type: str = 'hex') -> str:
        """
        Get color for a specific country.
        
        Args:
            country_code: ISO country code or country abbreviation
            format_type: Color format ('hex', 'rgb', 'rgba', 'plotly')
        
        Returns:
            Color in requested format
        """
        # Normalize country code
        country_code = str(country_code).upper().strip()
        
        # Get hex color (fallback if not found)
        hex_color = self.country_colors.get(country_code)
        if hex_color is None:
            hex_color = self._get_fallback_color(country_code)
        
        return self._convert_color_format(hex_color, format_type)
    
    def get_country_colors(self, country_codes: List[str], format_type: str = 'hex') -> List[str]:
        """
        Get colors for multiple countries.
        
        Args:
            country_codes: List of country codes
            format_type: Color format ('hex', 'rgb', 'rgba', 'plotly')
        
        Returns:
            List of colors in requested format
        """
        return [self.get_country_color(code, format_type) for code in country_codes]
    
    def get_regional_color(self, region: str, format_type: str = 'hex') -> str:
        """
        Get color for a region.
        
        Args:
            region: Region name
            format_type: Color format ('hex', 'rgb', 'rgba', 'plotly')
        
        Returns:
            Color in requested format
        """
        hex_color = self.regional_colors.get(region, self.regional_colors['Other'])
        return self._convert_color_format(hex_color, format_type)
    
    def _get_fallback_color(self, country_code: str) -> str:
        """Get a fallback color for unknown countries."""
        # Use country code hash to get consistent color for same country
        hash_val = hash(country_code) % len(self.fallback_palette)
        return self.fallback_palette[hash_val]
    
    def _convert_color_format(self, hex_color: str, format_type: str) -> Union[str, Tuple]:
        """Convert hex color to different formats."""
        if format_type == 'hex':
            return hex_color
        
        # Convert hex to RGB
        rgb = mcolors.hex2color(hex_color)
        
        if format_type == 'rgb':
            return rgb
        elif format_type == 'rgba':
            return rgb + (1.0,)  # Add alpha channel
        elif format_type == 'plotly':
            # Plotly prefers RGB string format
            rgb_255 = tuple(int(c * 255) for c in rgb)
            return f'rgb({rgb_255[0]}, {rgb_255[1]}, {rgb_255[2]})'
        elif format_type == 'matplotlib':
            return rgb  # Matplotlib accepts RGB tuples
        else:
            raise ValueError(f"Unsupported format_type: {format_type}")
    
    def create_color_map(self, countries: List[str], format_type: str = 'hex') -> Dict[str, str]:
        """
        Create a mapping of countries to colors.
        
        Args:
            countries: List of country codes
            format_type: Color format for the mapping
        
        Returns:
            Dictionary mapping country codes to colors
        """
        return {country: self.get_country_color(country, format_type) 
                for country in countries}
    
    def get_matplotlib_cmap(self, countries: List[str]) -> mcolors.ListedColormap:
        """
        Create a matplotlib colormap for the given countries.
        
        Args:
            countries: List of country codes
        
        Returns:
            Matplotlib ListedColormap
        """
        colors = self.get_country_colors(countries, 'matplotlib')
        return mcolors.ListedColormap(colors)
    
    def display_color_palette(self, countries: Optional[List[str]] = None) -> None:
        """
        Display the color palette for visual inspection.
        
        Args:
            countries: Optional list of specific countries to display
        """
        import matplotlib.pyplot as plt
        
        if countries is None:
            countries = list(self.important_country_colors.keys())
        
        colors = self.get_country_colors(countries, 'hex')
        
        fig, ax = plt.subplots(figsize=(12, max(2, len(countries) * 0.3)))
        
        y_pos = np.arange(len(countries))
        bars = ax.barh(y_pos, [1] * len(countries), color=colors)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(countries)
        ax.set_xlabel('Color')
        ax.set_title('Country Color Palette')
        
        # Add color codes as text
        for i, (country, color) in enumerate(zip(countries, colors)):
            ax.text(0.5, i, color, ha='center', va='center', 
                   fontweight='bold', color='white' if sum(mcolors.hex2color(color)) < 1.5 else 'black')
        
        plt.tight_layout()
        plt.show()


# Global instance for easy import
color_scheme = ColorSchemeManager()


# Convenience functions for easy usage
def get_country_color(country_code: str, format_type: str = 'hex') -> str:
    """Get color for a country (convenience function)."""
    return color_scheme.get_country_color(country_code, format_type)


def get_country_colors(country_codes: List[str], format_type: str = 'hex') -> List[str]:
    """Get colors for multiple countries (convenience function)."""
    return color_scheme.get_country_colors(country_codes, format_type)


def create_country_color_map(countries: List[str], format_type: str = 'hex') -> Dict[str, str]:
    """Create country-to-color mapping (convenience function)."""
    return color_scheme.create_color_map(countries, format_type)


# Regional mapping for countries (can be extended)
COUNTRY_REGION_MAP = {
    'USA': 'North America', 'US': 'North America', 'CA': 'North America', 'MX': 'North America',
    'BR': 'South America', 'AR': 'South America', 'CL': 'South America', 'CO': 'South America',
    'DE': 'Europe', 'GER': 'Europe', 'FR': 'Europe', 'FRN': 'Europe', 'IT': 'Europe', 
    'ES': 'Europe', 'UK': 'Europe', 'UKG': 'Europe', 'GB': 'Europe', 'NL': 'Europe',
    'CN': 'Asia', 'CHN': 'Asia', 'JP': 'Asia', 'JPN': 'Asia', 'IN': 'Asia', 'IND': 'Asia',
    'KR': 'Asia', 'KOR': 'Asia', 'TH': 'Asia', 'SG': 'Asia', 'MY': 'Asia',
    'AU': 'Oceania', 'AUL': 'Oceania', 'NZ': 'Oceania',
    'EG': 'Africa', 'ZA': 'Africa', 'NG': 'Africa', 'KE': 'Africa',
    'SA': 'Middle East', 'AE': 'Middle East', 'IL': 'Middle East', 'TR': 'Middle East',
}


def get_region_for_country(country_code: str) -> str:
    """Get region for a country code."""
    return COUNTRY_REGION_MAP.get(country_code.upper(), 'Other')


if __name__ == "__main__":
    # Demo the color scheme
    print("SOUL Project Color Scheme Demo")
    print("=" * 40)
    
    # Show important countries
    important = ['USA', 'CHN', 'JPN', 'GER', 'UKG', 'FRN', 'RUS', 'KOR', 'IND', 'AUL']
    print("\nImportant Countries:")
    for country in important:
        color = get_country_color(country)
        print(f"  {country}: {color}")
    
    # Show regional colors
    print("\nRegional Colors:")
    for region, color in color_scheme.regional_colors.items():
        print(f"  {region}: {color}")
    
    # Display the palette
    try:
        color_scheme.display_color_palette(important)
    except ImportError:
        print("\nMatplotlib not available for palette display") 