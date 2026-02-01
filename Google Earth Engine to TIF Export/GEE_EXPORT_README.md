# Google Earth Engine Fire Analysis & Export

Professional toolkit for processing multi-temporal satellite imagery through Google Earth Engine to generate burn scar analysis and training data for deep learning models.

**Author**: Tushar Thokdar

## 📋 Overview

This script provides a complete workflow for:

- Processing Sentinel-2 satellite imagery via Google Earth Engine
- Calculating burn indices (dNBR) and severity classifications
- Exporting multi-band GeoTIFF files
- Analyzing burn scar statistics
- Creating RGB visualizations for data understanding

For a deep technical analysis of the algorithms used here, see the [Technical Deep Dive](../Prithvi_Technical_Deep_Dive.md).

**Key Features**:

- ✅ No chip creation during export (processed later)
- ✅ Two export formats: Basic (8-band) and Prithvi (13-band)
- ✅ Interactive map visualization
- ✅ Comprehensive statistical analysis
- ✅ RGB chip generation for understanding data

## 🎯 Workflow Overview

```
1. Configure Analysis Area & Dates
   ↓
2. Process Sentinel-2 Imagery (GEE)
   ↓
3. Calculate dNBR & Classify Severity
   ↓
4. Export to Google Drive
   ↓
5. Download & Analyze TIF Files
   ↓
6. Generate RGB Visualizations
   ↓
7. Ready for Model Training
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install earthengine-api geemap rasterio numpy pandas matplotlib
```

### 1. Initialize Google Earth Engine

```python
from gee_export_clean import initialize_gee, Config

# First time setup - authenticate
initialize_gee('your-project-id')
```

### 2. Configure Your Analysis

```python
config = Config()

# Update project ID
config.GEE_PROJECT_ID = 'your-gee-project-id'

# Define area of interest (west, south, east, north)
config.AOI_BOUNDS = [-121.90, 39.80, -121.50, 40.30]

# Set temporal windows
config.PRE_FIRE_START = '2024-05-15'
config.PRE_FIRE_END = '2024-06-30'
config.POST_FIRE_START = '2024-09-15'
config.POST_FIRE_END = '2024-10-30'

# Adjust export settings
config.EXPORT_FOLDER = 'Fire_Analysis_Output'
```

### 3. Run Analysis

```python
from gee_export_clean import run_fire_analysis

results = run_fire_analysis(config)

# Display interactive map (in Jupyter/Colab)
results['map']
```

## 📤 Export Formats

### Basic Analysis (8 bands)

Perfect for initial analysis and visualization:

| Band | Description       | Range   |
| ---- | ----------------- | ------- |
| 1    | dNBR (continuous) | -1 to 1 |
| 2    | Severity class    | 0-5     |
| 3-5  | Pre-fire RGB      | 0-1     |
| 6-8  | Post-fire RGB     | 0-1     |

**Use Case**: Statistical analysis, RGB visualization, understanding patterns

### Prithvi Format (13 bands)

Designed for Prithvi EO 2.0 model training:

| Bands | Description | Spectral Channels         |
| ----- | ----------- | ------------------------- |
| 0-5   | Pre-fire    | B2, B3, B4, B8A, B11, B12 |
| 6-11  | Post-fire   | B2, B3, B4, B8A, B11, B12 |
| 12    | Label       | Severity class (0-5)      |

**Use Case**: Direct input for model training (process with dataset generator)

**Spectral Band Details**:

- B2: Blue (490nm)
- B3: Green (560nm)
- B4: Red (665nm)
- B8A: NIR (865nm)
- B11: SWIR1 (1610nm)
- B12: SWIR2 (2190nm)

## 📊 Post-Processing

### Statistical Analysis

After export completes, analyze your data:

```python
from gee_export_clean import TIFAnalyzer

# Load and analyze
analyzer = TIFAnalyzer('/path/to/Fire_Analysis_Basic.tif')
analyzer.print_summary()
```

**Output Example**:

```
📊 BURN SCAR ANALYSIS SUMMARY
======================================================================

🗺️  Total Area Analyzed:
   432,567 acres (175,041 hectares)

🔥 Severity Distribution:
Class                     Acres           Hectares        %
----------------------------------------------------------------------
Regrowth                     12,345           4,996      2.85%
Unburned                    234,567          94,921     54.23%
Low Severity                 89,012          36,017     20.58%
Moderate-Low Severity        56,789          22,983     13.13%
Moderate-High Severity       28,901          11,695      6.68%
High Severity                10,953           4,433      2.53%
----------------------------------------------------------------------

📈 dNBR Statistics:
   Mean:   0.2345
   Median: 0.1876
   Std:    0.1923
   Min:    -0.1234
   Max:    0.8765
```

### RGB Visualization

Create visual chips to understand your data:

```python
from gee_export_clean import create_rgb_chips

create_rgb_chips(
    tif_path='/path/to/Fire_Analysis_Basic.tif',
    output_dir='rgb_visualizations',
    chip_size=512,
    max_chips=10
)
```

This creates 2x2 subplot images showing:

- Top-left: Pre-fire RGB
- Top-right: Post-fire RGB
- Bottom-left: dNBR index (color-coded)
- Bottom-right: Severity classification

**Perfect for**:

- Understanding data quality
- Presentations and reports
- Identifying interesting regions
- Quality control

## ⚙️ Configuration Guide

### Area of Interest (AOI)

Define your study area as a bounding box:

```python
# Format: [west, south, east, north] in decimal degrees
config.AOI_BOUNDS = [-121.90, 39.80, -121.50, 40.30]
```

**Tips**:

- Use Google Earth to find coordinates
- Keep area reasonable (< 100km²) to avoid timeout
- Ensure it covers your fire perimeter

### Temporal Windows

Critical for good results:

```python
# Pre-fire: Vegetation peak, minimal clouds
config.PRE_FIRE_START = '2024-05-15'  # Late spring
config.PRE_FIRE_END = '2024-06-30'    # Early summer

# Post-fire: After burning, before rain/snow
config.POST_FIRE_START = '2024-09-15'  # Post-fire
config.POST_FIRE_END = '2024-10-30'    # Before winter
```

**Best Practices**:

- Pre-fire: Choose green vegetation period (spring/early summer)
- Post-fire: Wait 2-4 weeks after fire containment
- Avoid rainy/snowy seasons
- Use 4-6 week windows for better composites

### Cloud Filtering

```python
config.CLOUD_THRESHOLD_PRE = 20   # Strict for pre-fire
config.CLOUD_THRESHOLD_POST = 30  # Slightly relaxed for post
```

**Rationale**: Post-fire imagery often has smoke/haze, so slightly higher threshold is acceptable.

### Severity Thresholds

Customize burn severity classification:

```python
config.SEVERITY_THRESHOLDS = {
    'regrowth': -0.1,      # Vegetation increase
    'unburned': 0.1,       # Minimal change
    'low': 0.27,           # Low severity burn
    'moderate_low': 0.44,  # Moderate-low severity
    'moderate_high': 0.66  # Moderate-high severity
    # > 0.66 = High severity
}
```

Based on USGS standards, but adjustable for your region.

## 🗺️ Severity Classification

### dNBR Scale

| Class | dNBR Range   | Description   | Visual Indicator |
| ----- | ------------ | ------------- | ---------------- |
| 0     | < -0.1       | Regrowth      | Dark green       |
| 1     | -0.1 to 0.1  | Unburned      | Light green/aqua |
| 2     | 0.1 to 0.27  | Low Severity  | Yellow           |
| 3     | 0.27 to 0.44 | Moderate-Low  | Orange           |
| 4     | 0.44 to 0.66 | Moderate-High | Red              |
| 5     | > 0.66       | High Severity | Dark red         |

### Understanding dNBR

**dNBR = NBR_pre - NBR_post**

Where NBR = (NIR - SWIR2) / (NIR + SWIR2)

- **Positive values**: Vegetation loss (burn)
- **Negative values**: Vegetation increase (regrowth)
- **Near zero**: No significant change

## 📁 Output Files

After export completes (~10-15 minutes):

```
Google Drive/
└── Fire_Analysis_Output/
    ├── Fire_Analysis_Basic.tif          # 8-band file
    └── Prithvi_PrePost_Training_Data.tif # 13-band file
```

**File Sizes**:

- Typical size: 50-200 MB depending on area
- Resolution: 20m per pixel (Sentinel-2)

## 🔄 Complete Workflow Example

### Google Colab Notebook

```python
# 1. Setup
from google.colab import drive
drive.mount('/content/drive')

from gee_export_clean import *

# 2. Configure
config = Config()
config.GEE_PROJECT_ID = 'your-project-id'
config.AOI_BOUNDS = [-121.90, 39.80, -121.50, 40.30]
config.PRE_FIRE_START = '2024-05-15'
config.PRE_FIRE_END = '2024-06-30'
config.POST_FIRE_START = '2024-09-15'
config.POST_FIRE_END = '2024-10-30'

# 3. Run export
results = run_fire_analysis(config)

# 4. View map
results['map']

# ====== WAIT FOR EXPORT (10-15 min) ======

# 5. Analyze results
tif_path = '/content/drive/MyDrive/Fire_Analysis_Output/Fire_Analysis_Basic.tif'
analyzer = TIFAnalyzer(tif_path)
analyzer.print_summary()

# 6. Create visualizations
create_rgb_chips(
    tif_path,
    output_dir='rgb_chips',
    chip_size=512,
    max_chips=10
)

# 7. The Prithvi format file is ready for the dataset generator!
# Use it with prithvi_data_generation_clean.py
```

## 🎨 Visualization Guide

### Interactive Map Layers

The generated map includes:

1. **Pre-Fire RGB**: Natural color composite before fire
2. **Post-Fire RGB**: Natural color composite after fire
3. **dNBR Index**: Continuous burn intensity map
4. **Severity Classes**: Categorical classification
5. **Analysis Area**: Boundary of study region

Toggle layers on/off to compare changes.

### Color Schemes

**dNBR Gradient**:

- 🔵 Blue → White → 🟡 Yellow → 🟠 Orange → 🔴 Red → ⚫ Dark Red
- Represents: No burn → Increasing severity

**Severity Classes**:

- 🟢 Dark Green: Regrowth
- 🔷 Aquamarine: Unburned
- 🟡 Yellow: Low burn
- 🟠 Orange: Moderate burn
- 🔴 Red: High burn
- ⚫ Dark Red: Extreme burn

## 🐛 Troubleshooting

### Common Issues

**Issue**: `Exception: Project not found`

```python
# Solution: Authenticate first
ee.Authenticate()
ee.Initialize(project='your-project-id')
```

**Issue**: Export task fails

- Check area size (reduce if > 100km²)
- Verify date ranges have available imagery
- Ensure cloud threshold isn't too strict

**Issue**: "No imagery found"

- Expand date ranges
- Increase cloud threshold
- Check if AOI is over ocean/invalid area

**Issue**: RGB chips look too dark

- Adjust brightness multiplier in code:

```python
# In create_rgb_chips function
axes[0, 0].imshow(np.transpose(pre_rgb, (1, 2, 0)) * 5)  # Increase multiplier
```

### Export Monitoring

Check export status:

```python
# List all tasks
import ee
tasks = ee.batch.Task.list()
for task in tasks[:5]:
    print(f"{task.config['description']}: {task.state}")
```

## 📚 Technical Details

### Sentinel-2 Processing

- **Source**: Copernicus S2_SR_HARMONIZED
- **Cloud Masking**: SCL band-based
- **Compositing**: Median (reduces noise)
- **Scaling**: Reflectance / 10000 → [0, 1]

### NBR Calculation

```
NBR = (NIR - SWIR2) / (NIR + SWIR2)
    = (B8A - B12) / (B8A + B12)
```

**Why NBR?**

- NIR: High in healthy vegetation
- SWIR2: High in burned areas
- NBR decreases significantly after burning

### Export Settings

- **CRS**: WGS84 (EPSG:4326)
- **Format**: GeoTIFF (Cloud-Optimized)
- **Compression**: LZW
- **NoData**: Automatically handled

## 🔗 Integration with Training Pipeline

After export, use with the dataset generator:

```python
# Step 1: Export from GEE (this script)
# → Generates: Prithvi_PrePost_Training_Data.tif (13 bands)

# Step 2: Process with dataset generator
from prithvi_data_generation_clean import generate_dataset, Config

Config.INPUT_TIF = "path/to/Prithvi_PrePost_Training_Data.tif"
Config.OUTPUT_DIR = "training_dataset"
num_chips = generate_dataset()

# Step 3: Train your model!
```

## 📝 Best Practices

1. **Data Quality**
   - Always inspect RGB chips before training
   - Check class distribution in analysis
   - Verify temporal windows capture change

2. **Area Selection**
   - Focus on complete burn scars
   - Include unburned areas for context
   - Avoid partial fires at edges

3. **Temporal Strategy**
   - Pre-fire: Peak vegetation (May-July)
   - Post-fire: 2-4 weeks after containment
   - Avoid seasons with snow/heavy rain

4. **Processing Efficiency**
   - Start with smaller areas to test
   - Use appropriate cloud thresholds
   - Monitor export tasks

5. **Validation**
   - Create RGB chips for all datasets
   - Review statistics for anomalies
   - Compare with known fire perimeters

## 📊 Example Applications

### Research

- Fire severity mapping
- Burn scar delineation
- Temporal change analysis
- Model training data generation

### Operations

- Damage assessment
- Recovery monitoring
- Risk mapping
- Resource allocation

### Education

- Remote sensing tutorials
- GEE workshops
- Deep learning demonstrations

## 📄 License

This tool is provided for research and educational purposes.

## 🤝 Contributing

Improvements and suggestions welcome!

## 📧 Support

For issues or questions, please consult:

- GEE Documentation: https://developers.google.com/earth-engine
- Sentinel-2 Info: https://sentinel.esa.int/web/sentinel/missions/sentinel-2

---

**Last Updated**: January 29, 2026  
**Version**: 1.0.0
