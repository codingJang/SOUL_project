# Country Lists and Regional Mappings

This document provides reference information about countries and regional classifications used throughout the SOUL project.

## Important Countries (Political Analysis)

The following countries are considered "important" for focused political analysis based on their significance in international relations:

```
KOR - South Korea
FRN - France  
RUS - Russia
JPN - Japan
GER - Germany
CHN - China
UKG - United Kingdom
USA - United States
AUL - Australia
IND - India
```

### Country Code Mappings

| Abbreviation | Country Code | DCAD Dataset Name | IGO Dataset Name |
|--------------|--------------|-------------------|------------------|
| USA | 2 | United States | United States |
| CHN | 710 | China | China |
| JPN | 740 | Japan | Japan |
| GER | 255 | Germany | Germany |
| UKG | 200 | United Kingdom | United Kingdom |
| FRN | 220 | France | France |
| RUS | 365 | Russia | Russia |
| KOR | 732 | South Korea | Korea, South |
| IND | 750 | India | India |
| AUL | 900 | Australia | Australia |

## Regional Classifications

Countries are grouped into the following regions for analysis purposes:

### North America
- **USA** - United States (`#1f77b4` - Blue)
- **US** - United States (alternate code)
- **CA** - Canada (`#aec7e8` - Light Blue)
- **MX** - Mexico (`#ffbb78` - Light Orange)

### South America
- **BR** - Brazil (`#98df8a` - Green)
- **AR** - Argentina (`#87ceeb` - Sky Blue)
- **CL** - Chile (`#dda0dd` - Plum)
- **CO** - Colombia (`#f0e68c` - Khaki)

### Europe
- **DE** - Germany (`#2ca02c` - Green)
- **GER** - Germany (alternate code)
- **FR** - France (`#8c564b` - Brown)
- **FRN** - France (alternate code)
- **IT** - Italy (`#ff9896` - Light Red)
- **ES** - Spain (`#c5b0d5` - Light Purple)
- **UK** - United Kingdom (`#9467bd` - Purple)
- **UKG** - United Kingdom (alternate code)
- **GB** - Great Britain (`#9467bd` - Purple)
- **NL** - Netherlands (`#c49c94` - Light Brown)
- **SE** - Sweden (`#f7b6d3` - Light Pink)
- **NO** - Norway (`#c7c7c7` - Light Gray)

### Asia
- **CN** - China (`#d62728` - Red)
- **CHN** - China (alternate code)
- **JP** - Japan (`#ff7f0e` - Orange)
- **JPN** - Japan (alternate code)
- **KR** - South Korea (`#7f7f7f` - Gray)
- **KOR** - South Korea (alternate code)
- **IN** - India (`#bcbd22` - Olive)
- **IND** - India (alternate code)
- **TH** - Thailand (`#1fbecf` - Cyan variant)
- **SG** - Singapore (`#98df8a` - Light Green)
- **MY** - Malaysia (`#ff1744` - Bright Red)

### Oceania
- **AU** - Australia (`#17becf` - Cyan)
- **AUL** - Australia (alternate code)
- **NZ** - New Zealand (`#17becf` - Cyan)

### Africa
- **ZA** - South Africa (`#cd853f` - Peru Brown)
- **NG** - Nigeria (`#228b22` - Forest Green)
- **EG** - Egypt (`#ffd700` - Gold)
- **KE** - Kenya (`#dc143c` - Crimson)

### Middle East
- **SA** - Saudi Arabia (`#800080` - Purple)
- **AE** - United Arab Emirates (`#4169e1` - Royal Blue)
- **IL** - Israel (`#00ced1` - Dark Turquoise)
- **TR** - Turkey (`#ff6347` - Tomato)

## Color Scheme

Each country has a consistent color across all visualizations:

### Primary Countries (Political Focus)
- **USA**: Blue `#1f77b4` - Represents stability and trustworthiness
- **CHN**: Red `#d62728` - Traditional Chinese color association
- **JPN**: Orange `#ff7f0e` - Distinctive and warm
- **GER**: Green `#2ca02c` - Environmental leadership association
- **UKG**: Purple `#9467bd` - Royal association
- **FRN**: Brown `#8c564b` - Earthy, sophisticated
- **RUS**: Pink/Magenta `#e377c2` - Distinctive choice
- **KOR**: Gray `#7f7f7f` - Modern, technological
- **IND**: Olive `#bcbd22` - Traditional Indian color inspiration
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

## Usage in Code

### Getting Country Colors
```python
from configs.color_schemes import get_country_color, get_country_colors

# Single country
usa_color = get_country_color('USA', 'hex')  # '#1f77b4'

# Multiple countries
colors = get_country_colors(['USA', 'CHN', 'JPN'], 'matplotlib')
```

### Regional Analysis
```python
from configs.color_schemes import COUNTRY_REGION_MAP, get_region_for_country

# Get region for a country
region = get_region_for_country('USA')  # 'North America'

# Check all mappings
for country, region in COUNTRY_REGION_MAP.items():
    print(f"{country}: {region}")
```

## Data Sources

### IGO Dataset Countries
The IGO dataset includes membership data for intergovernmental organizations from 1815-2014, covering all countries that were members of IGOs with 3+ nation-states.

### DCAD Dataset Countries  
The DCAD dataset covers bilateral defense cooperation agreements for all independent countries from 1980-2010, with this analysis focusing on agreements from 2000 onwards.

### Economic Dataset Countries
Economic analysis includes countries with sufficient data availability in:
- World Bank Global Economic Monitor
- IMF Balance of Payments
- IMF Consumer Price Index
- IMF Monetary and Financial Statistics

## Adding New Countries

To add a new country to the analysis:

1. **Add to color scheme** (`configs/color_schemes.py`):
   ```python
   self.extended_country_colors = {
       'NEW': '#123456',  # New country code and color
       # ...
   }
   ```

2. **Add regional mapping**:
   ```python
   COUNTRY_REGION_MAP = {
       'NEW': 'Appropriate Region',
       # ...
   }
   ```

3. **Test with demo script**:
   ```bash
   python scripts/demo_unified_colors.py
   ```

4. **Update this documentation** with the new country information.

## Notes

- Country codes may vary between datasets (2-letter, 3-letter, numeric)
- Some countries have multiple abbreviations for historical reasons
- Color assignments consider accessibility and cultural appropriateness
- Regional groupings are based on common geopolitical classifications
- The "important countries" list focuses on major powers in international relations

---

For questions about country classifications or to request additions, please refer to the main project documentation or the color scheme configuration files. 