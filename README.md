# AORC Precipitation Analysis Tools

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python toolkit for retrieving and analyzing precipitation events from the NOAA Analysis of Record for Calibration (AORC) dataset. This repository includes two complementary tools for watershed-based precipitation analysis.

**Developed by [Mohsen Tahmasebi Nasab, PhD](https://www.hydromohsen.com/)**

---

## Table of Contents

- [Overview](#overview)
- [Tool Comparison](#tool-comparison)
- [Data Source](#data-source)
- [Methodology](#methodology)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Files](#output-files)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

---

## Overview

This toolkit provides two analysis scripts for different precipitation event identification strategies:

| Tool | Script | Purpose |
|------|--------|---------|
| **Annual Maximum** | `aorc_annual_max.py` | Identifies the single largest precipitation event per year |
| **Events Over Threshold** | `aorc_events_over_thresh.py` | Identifies all events exceeding a user-defined threshold |

Both tools share a common workflow: they extract AORC precipitation data for a user-defined watershed, calculate rolling precipitation totals, identify events based on their respective criteria, and produce professional Excel summaries and precipitation maps.

---

## Tool Comparison

### Annual Maximum (`aorc_annual_max.py`)

**Use Case:** Flood frequency analysis, annual exceedance probability studies, or when you need exactly one representative storm per year.

**How It Works:**
- For each calendar year, finds the single time step with the highest watershed-average N-hour precipitation
- Returns exactly one event per year (e.g., 25 events for a 25-year analysis)
- Event naming: `annual_max_precipitation_YYYY_XXhr.png`

**Key Parameters:**
```python
START_YEAR = 2000
END_YEAR = 2024
STORM_DURATION_HOURS = 24  # Rolling window (24, 48, 72, etc.)
```

### Events Over Threshold (`aorc_events_over_thresh.py`)

**Use Case:** Storm catalog development, partial duration series analysis, or when you need to capture all significant storms regardless of whether they're the annual maximum.

**How It Works:**
- Identifies all non-overlapping events where the watershed-average rolling precipitation exceeds a user-defined threshold
- Can return multiple events per year or zero events in dry years
- Groups contiguous threshold exceedances into single events, selecting the peak within each group
- Event naming: `YYYY_E01_YYYYMMDDTHHmm` format (e.g., `2023_E02_20230715T1800`)

**Key Parameters:**
```python
START_YEAR = 2000
END_YEAR = 2024
STORM_DURATION_HOURS = 72  # Rolling window (24, 48, 72, etc.)
ROLLING_THRESHOLD_INCHES = 2.0  # Minimum mean precipitation to qualify as an event
```

### When to Use Each Tool

| Scenario | Recommended Tool |
|----------|------------------|
| Flood frequency curve fitting (Bulletin 17C) | Annual Maximum |
| FFRD storm catalog development | Events Over Threshold |
| Identifying all major storms in a period | Events Over Threshold |
| Annual exceedance probability analysis | Annual Maximum |
| Comparing storm magnitudes across years | Either (depends on need) |
| Partial duration series analysis | Events Over Threshold |

---

## Data Source

Both tools use the **NOAA Analysis of Record for Calibration (AORC) v1.1** dataset:

| Property | Value |
|----------|-------|
| Spatial Resolution | 1 km × 1 km |
| Temporal Resolution | Hourly |
| Coverage | Continental United States (CONUS) |
| Period of Record | 1979 – Present |
| Source | `s3://noaa-nws-aorc-v1-1-1km` |

For more information about AORC, visit the [NOAA AORC documentation](https://registry.opendata.aws/noaa-nws-aorc/).

---

## Methodology

### Rolling Sum Calculation (Both Tools)

1. **Rolling Sum Calculation**: For each hourly time step, the tool calculates a rolling sum of precipitation over the specified duration (e.g., 72 hours). This produces an N-hour cumulative precipitation value at each grid cell for every time step.

2. **Spatial Averaging**: At each time step, the tool calculates the spatial mean of the N-hour cumulative precipitation across all grid cells within the watershed.

### Event Selection

**Annual Maximum Tool:**
- For each calendar year, identifies the single time step with the highest watershed-average N-hour precipitation
- Guarantees exactly one event per year

**Events Over Threshold Tool:**
- Identifies contiguous periods where the watershed-average rolling precipitation ≥ threshold
- Groups consecutive threshold exceedances into single events
- Selects the peak (maximum) time step within each contiguous group
- Ensures non-overlapping events by treating each contiguous exceedance as one event

### Output Statistics Explained

All statistics are calculated from the precipitation grid at the time of the event:

| Statistic | Description |
|-----------|-------------|
| **Mean Precipitation** | Spatial average of N-hour cumulative precipitation across all watershed grid cells |
| **Min Precipitation** | Lowest N-hour cumulative value among all watershed grid cells (driest spot) |
| **Max Precipitation** | Highest N-hour cumulative value among all watershed grid cells (wettest spot) |
| **Total Precipitation** | Sum of all grid cell values (numerical sum, not a volume calculation) |

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

**On Windows:**
```bash
python -m venv aorc_env
aorc_env\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv aorc_env
source aorc_env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import xarray, geopandas, s3fs; print('All dependencies installed successfully!')"
```

---

## Configuration

Both scripts use a similar configuration structure. Open the desired script and modify the **USER SETTINGS** section.

### Common Parameters (Both Tools)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `WATERSHED_PATH` | Path to watershed polygon file | `"path/to/watershed.geojson"` |
| `START_YEAR` | First year of analysis | `2000` |
| `END_YEAR` | Last year of analysis | `2024` |
| `STORM_DURATION_HOURS` | Rolling window duration in hours | `72` |
| `OUTPUT_DIR` | Directory for output files | `"output"` |
| `CREATE_PLOTS` | Generate precipitation maps | `True` |
| `PLOT_DPI` | Resolution of output maps | `300` |

### Events Over Threshold – Additional Parameter

| Parameter | Description | Example |
|-----------|-------------|---------|
| `ROLLING_THRESHOLD_INCHES` | Minimum mean precipitation to qualify as an event | `2.0` |

### Example Configurations

**Annual Maximum (24-hour events):**
```python
WATERSHED_PATH = r"data/MyWatershed.geojson"
START_YEAR = 2000
END_YEAR = 2024
STORM_DURATION_HOURS = 24
OUTPUT_DIR = "output_annual_max_24hr"
```

**Events Over Threshold (72-hour events ≥ 2 inches):**
```python
WATERSHED_PATH = r"data/MyWatershed.geojson"
START_YEAR = 2000
END_YEAR = 2024
STORM_DURATION_HOURS = 72
ROLLING_THRESHOLD_INCHES = 2.0
OUTPUT_DIR = "output_events_72hr_2in"
```

---

## Usage

### Running the Annual Maximum Analysis

```bash
# Activate virtual environment
source aorc_env/bin/activate  # Linux/macOS
aorc_env\Scripts\activate     # Windows

# Run analysis
python aorc_annual_max.py
```

### Running the Events Over Threshold Analysis

```bash
# Activate virtual environment
source aorc_env/bin/activate  # Linux/macOS
aorc_env\Scripts\activate     # Windows

# Run analysis
python aorc_events_over_thresh.py
```

### Expected Runtime

Runtime depends on the length of the analysis period, watershed size, and internet connection speed. Typical runtime: **5–30 minutes** for a medium-sized watershed over 20 years.

---

## Output Files

### Annual Maximum Output Structure

```
aorc_annual_max_output/
├── aorc_annual_max_precipitation_summary.xlsx
├── analysis.log
└── plots/
    ├── annual_max_precipitation_2000_24hr.png
    ├── annual_max_precipitation_2001_24hr.png
    └── ...
```

### Events Over Threshold Output Structure

```
aorc_events_output/
├── aorc_events_over_thresh_summary.xlsx
├── analysis_events_over_thresh.log
└── plots/
    ├── aorc_precip_2007_E01_20070415T0600_72hr_mean2p34.png
    ├── aorc_precip_2007_E02_20070823T1200_72hr_mean3p12.png
    └── ...
```

### Excel Summary Contents

Both tools produce Excel files with three sheets:

1. **Annual_Maxima_Summary** (or Events Summary): Detailed information for each event
   - Year, Event ID (threshold tool only)
   - Start/End time (UTC)
   - Duration (hours)
   - Mean, Total, Min, Max precipitation (inches and mm)
   - Path to corresponding plot file
   - Threshold value (threshold tool only)
   - Event number within year (threshold tool only)

2. **Metadata**: Analysis configuration and parameters

3. **Statistics**: Summary statistics across all events

### Precipitation Maps

Each event has a corresponding PNG map showing:
- Spatial distribution of precipitation (color-coded)
- Watershed boundary
- Event date and time
- Summary statistics (mean, max, min precipitation)

The Events Over Threshold tool uses a consistent color scale across all plots for easier visual comparison between events.

---

## Examples

### Example 1: Annual Maximum Analysis

```python
# Configuration
WATERSHED_PATH = r"data/LowerMinnesota.geojson"
START_YEAR = 2000
END_YEAR = 2024
STORM_DURATION_HOURS = 24

# Output: 25 events (one per year)
# Each event represents the largest 24-hour storm for that year
```

### Example 2: Events Over Threshold Analysis

```python
# Configuration
WATERSHED_PATH = r"data/UpperTennessee.geojson"
START_YEAR = 2007
END_YEAR = 2024
STORM_DURATION_HOURS = 72
ROLLING_THRESHOLD_INCHES = 2.0

# Output: Variable number of events
# All 72-hour storms with mean precipitation ≥ 2.0 inches
# Some years may have multiple events; dry years may have none
```

### Sample Console Output (Events Over Threshold)

```
2024-01-15T10:30:00Z | INFO | ================================================================================
2024-01-15T10:30:00Z | INFO | EVENTS OVER THRESHOLD PRECIPITATION ANALYSIS
2024-01-15T10:30:00Z | INFO | ================================================================================
2024-01-15T10:30:00Z | INFO | Analysis period: 2007 - 2024
2024-01-15T10:30:00Z | INFO | Storm duration: 72 hours
2024-01-15T10:30:00Z | INFO | Threshold: 2.0 inches
...
2024-01-15T10:45:30Z | INFO | Found 47 events over 2.0 inches
2024-01-15T10:45:30Z | INFO | Event 2007_E01_20070415T0600: 2.34 inches on 2007-04-15 06:00 UTC
2024-01-15T10:45:30Z | INFO | Event 2007_E02_20070823T1200: 3.12 inches on 2007-08-23 12:00 UTC
...
```

---

## Troubleshooting

### Common Issues

**1. "No AORC datasets found on S3"**
- Check your internet connection
- Verify the analysis years are within AORC data availability (1979–present)
- Try a smaller year range first

**2. "Watershed file not found"**
- Verify the path to your watershed file is correct
- Use raw strings (`r"..."`) for Windows paths to handle backslashes

**3. Memory errors**
- Reduce the analysis period (fewer years)
- Use a smaller watershed
- Increase the `BUFFER_WATERSHED_DEGREES` value slightly

**4. "No events found over X inches" (threshold tool)**
- Your threshold may be too high for the watershed/duration combination
- Try lowering `ROLLING_THRESHOLD_INCHES`
- Check that your watershed file is valid

**5. Slow performance**
- First-time access downloads data from S3, which can be slow
- Consider using local AORC files for repeated analyses (`USE_LOCAL_AORC = True`)

### Getting Help

If you encounter issues:
1. Check the log file in your output directory (`analysis.log` or `analysis_events_over_thresh.log`)
2. Open an issue on GitHub with the log file and error message
3. Contact the developer (see below)

---

## Requirements

See `requirements.txt` for a complete list. Key packages include:

- `numpy` – Numerical computing
- `pandas` – Data manipulation
- `xarray` – N-dimensional labeled arrays
- `geopandas` – Geospatial data handling
- `rioxarray` – Rasterio extension for xarray
- `s3fs` – S3 filesystem interface
- `matplotlib` – Visualization
- `openpyxl` – Excel file creation

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Contact

**Mohsen Tahmasebi Nasab, PhD**

- Website: [https://www.hydromohsen.com/](https://www.hydromohsen.com/)
- GitHub: [hydromohsen](https://github.com/mohsennasab)

---

*Last updated: February 2025*
