# AORC Annual Maximum Precipitation Analysis Tool

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python tool for retrieving and analyzing annual maximum precipitation events from the NOAA Analysis of Record for Calibration (AORC) dataset. This tool extracts precipitation data for any user-defined watershed, calculates rolling precipitation totals, identifies annual maximum events, and produces professional Excel summaries and precipitation maps.

**Developed by [Mohsen Tahmasebi Nasab, PhD](https://www.hydromohsen.com/)**

---

## Table of Contents

- [Features](#features)
- [Data Source](#data-source)
- [Methodology](#methodology)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Files](#output-files)
- [Example](#example)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Cloud-based data access**: Directly retrieves AORC precipitation data from NOAA's public AWS S3 bucket (no local data storage required)
- **Flexible watershed input**: Supports GeoJSON and other OGR-compatible formats
- **Configurable analysis period**: Analyze any time range from 1979 to present
- **Variable storm durations**: Calculate annual maxima for any duration (24-hour, 48-hour, 72-hour, etc.)
- **Professional outputs**:
  - Excel summary with multiple sheets (Annual Maxima, Metadata, Statistics)
  - High-resolution precipitation maps for each annual maximum event
- **StormHub compatible**: Follows conventions used in NOAA's StormHub framework
- **Comprehensive logging**: Detailed log files for debugging and record-keeping

---

## Data Source

This tool uses the **NOAA Analysis of Record for Calibration (AORC) v1.1** dataset:

- **Spatial Resolution**: 1 km × 1 km
- **Temporal Resolution**: Hourly
- **Coverage**: Continental United States (CONUS)
- **Period of Record**: 1979 – Present
- **Source**: `s3://noaa-nws-aorc-v1-1-1km`

For more information about AORC, visit the [NOAA AORC documentation](https://hydrology.nws.noaa.gov/aorc-historic/).

---

## Methodology

### How Annual Maximum Events Are Identified

1. **Rolling Sum Calculation**: For each hourly time step, the tool calculates a rolling sum of precipitation over the specified duration (e.g., 24 hours). This produces an N-hour cumulative precipitation value at each grid cell for every time step.

2. **Spatial Averaging**: At each time step, the tool calculates the spatial mean of the N-hour cumulative precipitation across all grid cells within the watershed.

3. **Annual Maximum Selection**: For each year, the tool identifies the time step with the highest watershed-average N-hour precipitation. This becomes that year's annual maximum event.

### Output Statistics Explained

All statistics are calculated from the precipitation grid at the time of the annual maximum event:

| Statistic | Description |
|-----------|-------------|
| **Mean Precipitation** | The spatial average of N-hour cumulative precipitation across all grid cells in the watershed. This is the value used to identify the annual maximum event. |
| **Min Precipitation** | The lowest N-hour cumulative precipitation value among all grid cells in the watershed (the "driest" spot). |
| **Max Precipitation** | The highest N-hour cumulative precipitation value among all grid cells in the watershed (the "wettest" spot). |
| **Total Precipitation** | The sum of all grid cell values. Note: This is a numerical sum without physical meaning (not a volume calculation). |

### Example Interpretation

For a 24-hour annual maximum event with:
- **Mean: 2.5 inches** → On average across the watershed, 2.5 inches fell in 24 hours
- **Min: 1.8 inches** → The driest location in the watershed received 1.8 inches in 24 hours
- **Max: 3.2 inches** → The wettest location in the watershed received 3.2 inches in 24 hours

### Precipitation Maps (PNG Files)

Each PNG map displays the **N-hour cumulative precipitation depth** at each 1 km × 1 km grid cell for the annual maximum event. The color scale represents the total precipitation (in inches) accumulated over the storm duration at each location.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Git (optional, for cloning the repository)

### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/mohsennasab/aorc-annual-max-precip.git
cd aorc-annual-max-precip
```

Or download the ZIP file and extract it to your desired location.

### Step 2: Create a Virtual Environment

It is strongly recommended to use a virtual environment to avoid dependency conflicts.

**On Windows:**

```bash
# Create the virtual environment
python -m venv aorc_ams

# Activate the virtual environment
aorc_ams\Scripts\activate
```

**On macOS/Linux:**

```bash
# Create the virtual environment
python3 -m venv aorc_ams

# Activate the virtual environment
source aorc_ams/bin/activate
```

### Step 3: Install Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import xarray, geopandas, s3fs; print('All dependencies installed successfully!')"
```

---

## Configuration

Before running the analysis, you need to update the user parameters in `aorc_annual_max.py`. Open the file in your preferred text editor and modify the **USER SETTINGS** section (lines 36-66):

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `WATERSHED_PATH` | Path to your watershed polygon file (GeoJSON etc.) | `"path/to/watershed.geojson"` |
| `START_YEAR` | First year of analysis | `2000` |
| `END_YEAR` | Last year of analysis | `2024` |
| `STORM_DURATION_HOURS` | Rolling window duration in hours | `24` |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `OUTPUT_DIR` | Directory for output files | `"aorc_annual_max_output"` |
| `CREATE_PLOTS` | Generate precipitation maps | `True` |
| `PLOT_DPI` | Resolution of output maps | `300` |
| `EXCEL_FILENAME` | Name of the summary Excel file | `"aorc_annual_max_precipitation_summary.xlsx"` |
| `USE_LOCAL_AORC` | Use local AORC files instead of S3 | `False` |
| `LOCAL_AORC_PATH` | Path to local AORC files (if `USE_LOCAL_AORC=True`) | `"/path/to/local/aorc"` |
| `BUFFER_WATERSHED_DEGREES` | Buffer around watershed for data extraction | `0.1` |

### Example Configuration

```python
# =============================================================================
# USER SETTINGS - MODIFY THESE PARAMETERS
# =============================================================================

# Input parameters
WATERSHED_PATH = r"C:\Projects\MyWatershed\basin.geojson"

# Time period settings
START_YEAR = 2000
END_YEAR = 2023
STORM_DURATION_HOURS = 24  # 24-hour annual maximum

# Output settings
OUTPUT_DIR = "output_my_watershed"
CREATE_PLOTS = True
PLOT_DPI = 300
```

### Using a StormHub Config File (Alternative)

If you have a StormHub-style configuration file, you can use it instead:

```python
CONFIG_FILE = "path/to/config.json"
WATERSHED_PATH = None  # Will be read from config file
```

---

## Usage

### Running the Analysis

1. Ensure your virtual environment is activated:

   **Windows:**
   ```bash
   aorc_ams\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source aorc_ams/bin/activate
   ```

2. Run the script:

   ```bash
   python aorc_annual_max.py
   ```

3. Monitor the progress in the terminal. The analysis will:
   - Load your watershed geometry
   - Connect to NOAA's S3 bucket and download relevant AORC data
   - Calculate rolling precipitation totals
   - Identify annual maximum events for each year
   - Generate Excel summary and precipitation maps

### Expected Runtime

Runtime depends on:
- Length of analysis period (number of years)
- Size of watershed
- Internet connection speed

Typical runtime: **5-30 minutes** for a medium-sized watershed over 20 years.

---

## Output Files

All outputs are saved to the directory specified by `OUTPUT_DIR`. The default structure is:

```
aorc_annual_max_output/
├── aorc_annual_max_precipitation_summary.xlsx
├── analysis.log
└── plots/
    ├── annual_max_precipitation_2000_24hr.png
    ├── annual_max_precipitation_2001_24hr.png
    ├── annual_max_precipitation_2002_24hr.png
    └── ...
```

### Excel Summary (`aorc_annual_max_precipitation_summary.xlsx`)

The Excel file contains three sheets:

1. **Annual_Maxima_Summary**: Detailed information for each annual maximum event
   - Year
   - Start/End time (UTC)
   - Duration (hours)
   - Mean, Total, Min, Max precipitation (inches and mm)
   - Path to corresponding plot file

2. **Metadata**: Analysis configuration and parameters
   - Analysis date
   - Watershed file path
   - Analysis period
   - Storm duration
   - Data source information

3. **Statistics**: Summary statistics across all years
   - Mean, Median, Min, Max, Standard Deviation of annual maxima

### Precipitation Maps

Each annual maximum event has a corresponding PNG map showing:
- Spatial distribution of precipitation (color-coded)
- Watershed boundary
- Event date and time
- Summary statistics (mean, max, min precipitation)

---

## Example

### Sample Workflow

```python
# 1. Set your watershed path
WATERSHED_PATH = r"data/LowerMinnesota.geojson"

# 2. Define analysis period
START_YEAR = 2000
END_YEAR = 2024

# 3. Set storm duration (24-hour annual maximum)
STORM_DURATION_HOURS = 24

# 4. Run the script
# $ python aorc_annual_max.py
```

### Sample Console Output

```
2024-01-15T10:30:00Z | INFO | ================================================================================
2024-01-15T10:30:00Z | INFO | ANNUAL MAXIMUM PRECIPITATION ANALYSIS
2024-01-15T10:30:00Z | INFO | ================================================================================
2024-01-15T10:30:00Z | INFO | Analysis period: 2000 - 2024
2024-01-15T10:30:00Z | INFO | Storm duration: 24 hours
2024-01-15T10:30:01Z | INFO | Loading watershed geometry from: data/LowerMinnesota.geojson
2024-01-15T10:30:01Z | INFO | Loaded watershed with 1 feature(s)
2024-01-15T10:30:02Z | INFO | Generated 25 AORC dataset paths for years 2000-2024
...
2024-01-15T10:45:30Z | INFO | ANALYSIS SUMMARY
2024-01-15T10:45:30Z | INFO | Years analyzed: 25
2024-01-15T10:45:30Z | INFO | Mean annual maximum: 2.34 inches
2024-01-15T10:45:30Z | INFO | Median annual maximum: 2.21 inches
2024-01-15T10:45:30Z | INFO | Results saved to: aorc_annual_max_output
2024-01-15T10:45:30Z | INFO | Analysis completed successfully!
```

---

## Troubleshooting

### Common Issues

**1. "No AORC datasets found on S3"**
- Check your internet connection
- Verify the analysis years are within the AORC data availability (1979-present)
- Try a smaller year range first

**2. "Watershed file not found"**
- Verify the path to your watershed file is correct
- Use raw strings (`r"..."`) for Windows paths to handle backslashes

**3. Memory errors**
- Reduce the analysis period (fewer years)
- Use a smaller watershed
- Increase the `BUFFER_WATERSHED_DEGREES` value slightly

**4. "No precipitation variable found"**
- This is rare but may occur if AORC data format changes
- Check the log file for available variable names

**5. Slow performance**
- First-time access downloads data from S3, which can be slow
- Consider using local AORC files for repeated analyses

### Getting Help

If you encounter issues:
1. Check the `analysis.log` file in your output directory
2. Open an issue on GitHub with the log file and error message
3. Contact the developer (see below)

---

## Requirements

See `requirements.txt` for a complete list of dependencies. Key packages include:

- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `xarray` - N-dimensional labeled arrays
- `geopandas` - Geospatial data handling
- `rioxarray` - Rasterio extension for xarray
- `s3fs` - S3 filesystem interface
- `matplotlib` - Visualization
- `openpyxl` - Excel file creation

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Mohsen Tahmasebi Nasab, PhD**

- Website: [https://www.hydromohsen.com/](https://www.hydromohsen.com/)
- GitHub: [hydromohsen](https://github.com/hydromohsen)

---

*Last updated: January 2025*
