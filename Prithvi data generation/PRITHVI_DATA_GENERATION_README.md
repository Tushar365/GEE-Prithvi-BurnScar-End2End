# Prithvi EO 2.0 Burn Scar Dataset Generator

Professional toolkit for generating training data from multi-temporal satellite imagery for burn scar detection and severity classification using the Prithvi EO 2.0 model.

**Author**: Tushar Thokdar

## 📋 Overview

This script processes multi-temporal satellite imagery (pre-fire and post-fire) to create high-quality training datasets for deep learning models. It includes automatic quality filtering, data cleaning, and comprehensive analysis tools.

For details on the **Delta Channel Algorithm** implemented in this module, see the [Technical Deep Dive](../Prithvi_Technical_Deep_Dive.md).

## 🎯 Features

- **Automated Data Processing**: Converts GeoTIFF imagery into training-ready NumPy arrays
- **Quality Filtering**: Automatic removal of low-quality chips based on configurable thresholds
- **Multi-temporal Support**: Handles pre-fire, post-fire, and optional change detection (delta) channels
- **Data Cleaning**: Robust handling of NaN, Inf, and invalid values
- **Class Balancing Analysis**: Built-in dataset statistics and class distribution reporting
- **Scalable**: Processes large satellite imagery efficiently with progress tracking

## 📊 Input Data Specification

**Required Format**: GeoTIFF with 13 bands

- Bands 1-6: Pre-fire reflectance (6 spectral bands)
- Bands 7-12: Post-fire reflectance (6 spectral bands)
- Band 13: Burn severity labels (1-5)

**Label Encoding**:

- 1 → Unburned (mapped to 0)
- 2 → Low Severity (mapped to 1)
- 3 → Moderate-Low Severity (mapped to 2)
- 4 → Moderate-High Severity (mapped to 3)
- 5 → High Severity (mapped to 4)
- 255 → Ignore (invalid/unlabeled pixels)

## 🚀 Quick Start

### Installation

```bash
pip install rasterio numpy tqdm
```

### Basic Usage

```python
from prithvi_data_generation_clean import generate_dataset, analyze_dataset, Config

# Configure paths
Config.INPUT_TIF = "path/to/your/satellite_imagery.tif"
Config.OUTPUT_DIR = "prithvi_dataset"

# Generate dataset
num_chips = generate_dataset()

# Analyze results
analyze_dataset(Config.OUTPUT_DIR)
```

### Google Colab Usage

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Update configuration
Config.INPUT_TIF = "/content/drive/MyDrive/data_in_TIFF/Prithvi_PrePost_Training_Data.tif"
Config.OUTPUT_DIR = "prithvi_dataset"

# Run generation
num_chips = generate_dataset()
analyze_dataset(Config.OUTPUT_DIR)

# Backup to Drive
!zip -r prithvi_dataset.zip prithvi_dataset/
!cp prithvi_dataset.zip /content/drive/MyDrive/
```

## ⚙️ Configuration

### Key Parameters

```python
class Config:
    # File paths
    INPUT_TIF = "path/to/input.tif"
    OUTPUT_DIR = "prithvi_dataset"

    # Processing
    TILE_SIZE = 224                # Chip size (pixels)
    IGNORE_VALUE = 255             # Value for invalid pixels

    # Quality thresholds
    MIN_VALID_PIXELS = 0.95        # Require 95% valid reflectance
    MIN_LABELED_PIXELS = 0.01      # Require 1% labeled pixels

    # Optional features
    INCLUDE_DELTA = True           # Include change detection channel
```

### Adjusting Quality Thresholds

For different data quality requirements:

```python
# Stricter quality control
Config.MIN_VALID_PIXELS = 0.98
Config.MIN_LABELED_PIXELS = 0.05

# More permissive (more chips, lower quality)
Config.MIN_VALID_PIXELS = 0.90
Config.MIN_LABELED_PIXELS = 0.005
```

## 📁 Output Structure

```
prithvi_dataset/
├── temporal_images/
│   ├── chip_000000.npy    # Shape: (3, 6, 224, 224) with delta
│   ├── chip_000001.npy    # Or (2, 6, 224, 224) without delta
│   └── ...
└── masks/
    ├── chip_000000.npy    # Shape: (224, 224), dtype: uint8
    ├── chip_000001.npy    # Values: 0-4 (classes), 255 (ignore)
    └── ...
```

### Data Format Details

**Temporal Images** (`temporal_images/chip_*.npy`):

- Shape: `(3, 6, 224, 224)` with delta or `(2, 6, 224, 224)` without
- Dimensions: `[time_steps, bands, height, width]`
- Time steps: 0=pre-fire, 1=post-fire, 2=delta (if enabled)
- Bands: 6 spectral channels per time step
- Values: Float32, range [0, 1] for reflectance, [-1, 1] for delta

**Masks** (`masks/chip_*.npy`):

- Shape: `(224, 224)`
- Type: uint8
- Values: 0-4 (burn severity classes), 255 (ignore)

## 🔍 Quality Control

The script applies multiple quality filters:

1. **Valid Pixel Ratio**: Ensures sufficient non-corrupted pixels
2. **Label Coverage**: Requires minimum labeled area
3. **NaN/Inf Handling**: Automatic cleaning of invalid values
4. **Reflectance Scaling**: Auto-detection and normalization

Chips failing quality checks are automatically excluded from the dataset.

## 📈 Dataset Analysis

The built-in analysis provides:

- **Class Distribution**: Pixel counts and percentages per class
- **Sample Verification**: Shape and range validation
- **Statistics Summary**: Total chips and pixels

Example output:

```
📈 Class Distribution:
Class           Count        Percentage
----------------------------------------
Unburned        1,369,197    27.00%
Low Severity    771,342      15.21%
Moderate-Low    924,961      18.24%
Moderate-High   822,241      16.22%
High Severity   1,182,633    23.32%
----------------------------------------
Total           5,070,374    100.00%
```

## 🛠️ Advanced Usage

### Custom Processing Pipeline

```python
# Create custom configuration
class CustomConfig(Config):
    TILE_SIZE = 256
    MIN_VALID_PIXELS = 0.99
    INCLUDE_DELTA = False

# Generate with custom settings
num_chips = generate_dataset(CustomConfig())
```

### Batch Processing Multiple Files

```python
import os

input_files = [
    "region1.tif",
    "region2.tif",
    "region3.tif"
]

for i, input_file in enumerate(input_files):
    Config.INPUT_TIF = input_file
    Config.OUTPUT_DIR = f"prithvi_dataset_region{i+1}"

    num_chips = generate_dataset()
    analyze_dataset(Config.OUTPUT_DIR)
```

## 📝 Best Practices

1. **Memory Management**: Process large files in chunks if memory is limited
2. **Quality Thresholds**: Adjust based on your specific use case and data quality
3. **Delta Channel**: Enable for change detection tasks, disable for simpler workflows
4. **Validation**: Always run analysis after generation to verify class balance
5. **Backup**: Zip and save datasets to cloud storage after generation

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ValueError: Expected 13 bands`

- **Solution**: Verify input TIF has correct band structure (6 pre + 6 post + 1 label)

**Issue**: No chips generated

- **Solution**: Lower quality thresholds or check input data validity

**Issue**: Memory error

- **Solution**: Process smaller regions or increase available RAM

**Issue**: Imbalanced classes

- **Solution**: This is normal for burn scar data; consider class weighting during training

## 📚 Citation

If you use this dataset generation tool, please cite:

```bibtex
@software{prithvi_dataset_generator,
  title={Prithvi EO 2.0 Burn Scar Dataset Generator},
  author={Data Science Team},
  year={2026},
  version={1.0}
}
```

## 📄 License

This tool is provided as-is for research and educational purposes.

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Last Updated**: January 29, 2026  
**Version**: 1.0.0
