"""
Events Over Threshold Precipitation Analysis Tool
==========================================

This script retrieves precipitation events from the NOAA AORC dataset for a given 
watershed polygon and time period. It follows StormHub conventions and produces 
Excel summaries and PNG maps for each event over the user specified threshold.

Developed by Mohsen Tahmasebi Nasab
"""

import os
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import json

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import rioxarray
import s3fs
from shapely.geometry import shape, mapping
from affine import Affine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# USER SETTINGS - MODIFY THESE PARAMETERS
# =============================================================================

# Input parameters
WATERSHED_PATH = r"D:\Projects\FFRD_Upper_Tennessee\GIS\Data\upper-tennessee_huc04.shp"  # Path to watershed polygon (GeoJSON, shapefile, etc.)


# Alternative: Use config.json format like StormHub
CONFIG_FILE = None  # "path/to/config.json"  # Set to None to use WATERSHED_PATH directly

# Time period settings
START_YEAR = 2000
END_YEAR = 2024
STORM_DURATION_HOURS = 72  # Rolling window size (24, 48, 72, etc.)
ROLLING_THRESHOLD_INCHES = 3.0  # Threshold for mean rolling precipitation (inches)

# Output settings
OUTPUT_DIR = r"D:\Projects\FFRD_Upper_Tennessee\AORC_Annual_Max_Precip\EventsOverThresholdOutput"
CREATE_PLOTS = True
PLOT_DPI = 300
EXCEL_FILENAME = "aorc_events_over_thresh_summary.xlsx"

# AORC dataset settings
AORC_BASE_URL = "s3://noaa-nws-aorc-v1-1-1km"
USE_LOCAL_AORC = False  # Set to True if using local AORC files
LOCAL_AORC_PATH = "/path/to/local/aorc"  # Only used if USE_LOCAL_AORC is True

# Processing settings
CHUNK_SIZE = "auto"  # xarray chunk size for memory management
BUFFER_WATERSHED_DEGREES = 0.1  # Buffer around watershed for data extraction

# =============================================================================
# CONSTANTS (Following StormHub conventions)
# =============================================================================

AORC_PRECIP_VARIABLE = "APCP_surface"
AORC_X_VAR = "longitude" 
AORC_Y_VAR = "latitude"
MM_TO_INCH_CONVERSION_FACTOR = 0.03937007874015748

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str) -> None:
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "analysis_events_over_thresh.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%SZ',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_watershed_geometry(watershed_path: str = None, config_file: str = None) -> gpd.GeoDataFrame:
    """
    Load watershed geometry from file or config.
    
    Args:
        watershed_path: Direct path to geometry file
        config_file: Path to StormHub-style config file
        
    Returns:
        GeoDataFrame with watershed geometry in EPSG:4326
    """
    if config_file and os.path.exists(config_file):
        logging.info(f"Loading watershed from config file: {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)
        watershed_path = config['watershed']['geometry_file']
        logging.info(f"Using watershed file from config: {watershed_path}")
    
    if not watershed_path or not os.path.exists(watershed_path):
        raise FileNotFoundError(f"Watershed file not found: {watershed_path}")
    
    logging.info(f"Loading watershed geometry from: {watershed_path}")
    gdf = gpd.read_file(watershed_path)
    
    if len(gdf) != 1:
        logging.warning(f"Watershed file contains {len(gdf)} features. Using the first one.")
        gdf = gdf.iloc[[0]]
    
    # Ensure geometry is in WGS84 (EPSG:4326) for AORC data
    if gdf.crs != "EPSG:4326":
        logging.info(f"Reprojecting watershed from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
    
    return gdf

def get_aorc_paths(start_year: int, end_year: int, base_url: str) -> List[str]:
    """
    Generate AORC dataset paths for the specified year range.
    
    Args:
        start_year: Starting year
        end_year: Ending year (inclusive)
        base_url: Base URL for AORC data
        
    Returns:
        List of AORC dataset paths
    """
    paths = []
    for year in range(start_year, end_year + 1):
        if USE_LOCAL_AORC:
            path = os.path.join(LOCAL_AORC_PATH, f"{year}.zarr")
        else:
            path = f"{base_url}/{year}.zarr"
        paths.append(path)
    
    logging.info(f"Generated {len(paths)} AORC dataset paths for years {start_year}-{end_year}")
    return paths

def load_aorc_data(paths: List[str], watershed_gdf: gpd.GeoDataFrame, 
                   start_time: datetime, end_time: datetime) -> xr.Dataset:
    """
    Load AORC precipitation data for the specified watershed and time period.
    
    Args:
        paths: List of AORC dataset paths
        watershed_gdf: Watershed geometry
        start_time: Start of time period
        end_time: End of time period
        
    Returns:
        xarray Dataset with AORC precipitation data
    """
    logging.info(f"Loading AORC data from {len(paths)} datasets")
    logging.info(f"Time period: {start_time} to {end_time}")
    
    # Get watershed bounds with buffer
    bounds = watershed_gdf.total_bounds
    buffered_bounds = [
        bounds[0] - BUFFER_WATERSHED_DEGREES,  # minx
        bounds[1] - BUFFER_WATERSHED_DEGREES,  # miny  
        bounds[2] + BUFFER_WATERSHED_DEGREES,  # maxx
        bounds[3] + BUFFER_WATERSHED_DEGREES   # maxy
    ]
    
    logging.info(f"Watershed bounds (with buffer): {buffered_bounds}")
    
    if USE_LOCAL_AORC:
        # Load local zarr files
        datasets = []
        for path in paths:
            if os.path.exists(path):
                ds = xr.open_zarr(path, chunks=CHUNK_SIZE)
                datasets.append(ds)
            else:
                logging.warning(f"Local AORC file not found: {path}")
        
        if not datasets:
            raise FileNotFoundError("No local AORC datasets found")
            
        ds = xr.concat(datasets, dim='time')
    else:
        # Load from S3
        s3 = s3fs.S3FileSystem(anon=True)
        
        # Filter paths to only existing datasets
        existing_paths = []
        for path in paths:
            zarr_path = path.replace("s3://", "")
            if s3.exists(zarr_path):
                existing_paths.append(path)
            else:
                logging.warning(f"AORC dataset not found: {path}")
        
        if not existing_paths:
            raise FileNotFoundError("No AORC datasets found on S3")
        
        logging.info(f"Found {len(existing_paths)} existing AORC datasets")
        
        # Create S3 file system mappings
        fileset = []
        for path in existing_paths:
            s3_map = s3fs.S3Map(root=path, s3=s3, check=False)
            fileset.append(s3_map)
        
        # Try consolidated=True first, fallback to False (StormHub pattern)
        try:
            logging.info("Attempting to open AORC data with consolidated=True")
            ds = xr.open_mfdataset(fileset, engine="zarr", chunks=CHUNK_SIZE, consolidated=True)
        except Exception as e:
            logging.warning(f"Consolidated=True failed: {e}")
            logging.info("Retrying with consolidated=False")
            ds = xr.open_mfdataset(fileset, engine="zarr", chunks=CHUNK_SIZE, consolidated=False)
    
    # Log dataset information
    logging.info(f"Dataset variables: {list(ds.data_vars)}")
    logging.info(f"Dataset time range: {ds.time.min().values} to {ds.time.max().values}")
    logging.info(f"Dataset spatial extent: lon({ds.longitude.min().values:.3f}, {ds.longitude.max().values:.3f}), "
                 f"lat({ds.latitude.min().values:.3f}, {ds.latitude.max().values:.3f})")
    
    # Check if precipitation variable exists
    precip_vars = [AORC_PRECIP_VARIABLE, 'APCP', 'precip', 'precipitation']
    precip_var = None
    for var in precip_vars:
        if var in ds.data_vars:
            precip_var = var
            break
    
    if precip_var is None:
        available_vars = list(ds.data_vars)
        raise ValueError(f"No precipitation variable found. Available variables: {available_vars}")
    
    logging.info(f"Using precipitation variable: {precip_var}")
    
    # Select only precipitation data
    ds = ds[[precip_var]]
    
    # Subset spatially
    ds = ds.sel(
        longitude=slice(buffered_bounds[0], buffered_bounds[2]),
        latitude=slice(buffered_bounds[1], buffered_bounds[3])
    )
    
    # Subset temporally
    ds = ds.sel(time=slice(start_time, end_time))
    
    logging.info(f"Subsetted dataset shape: {ds[precip_var].shape}")
    logging.info(f"Subsetted time range: {ds.time.min().values} to {ds.time.max().values}")
    
    return ds

def calculate_rolling_precipitation(ds: xr.Dataset, window_hours: int, 
                                   watershed_gdf: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Calculate rolling precipitation sums over the specified window.
    
    Args:
        ds: AORC dataset with precipitation
        window_hours: Rolling window size in hours
        watershed_gdf: Watershed geometry for clipping
        
    Returns:
        DataArray with rolling precipitation sums
    """
    # Get precipitation variable name
    precip_var = list(ds.data_vars)[0]
    precip_data = ds[precip_var]
    
    logging.info(f"Calculating {window_hours}-hour rolling precipitation")
    logging.info(f"Original precipitation data shape: {precip_data.shape}")
    
    # Set up CRS for the data array
    precip_data = precip_data.rio.set_crs("EPSG:4326")
    
    # Clip to watershed geometry  
    watershed_geom = watershed_gdf.geometry.iloc[0]
    try:
        precip_clipped = precip_data.rio.clip([watershed_geom], drop=True, all_touched=True)
        logging.info(f"Clipped precipitation data shape: {precip_clipped.shape}")
    except Exception as e:
        logging.warning(f"Error clipping to watershed: {e}")
        logging.info("Using buffered bounds instead of exact clipping")
        bounds = watershed_gdf.total_bounds
        precip_clipped = precip_data.sel(
            longitude=slice(bounds[0], bounds[2]),
            latitude=slice(bounds[1], bounds[3])
        )
    
    # Calculate rolling sum
    rolling_precip = precip_clipped.rolling(time=window_hours, center=False).sum()
    
    # Remove NaN values from the beginning (due to rolling window)
    rolling_precip = rolling_precip.isel(time=slice(window_hours-1, None))
    
    logging.info(f"Rolling precipitation shape: {rolling_precip.shape}")
    return rolling_precip

def find_threshold_events(rolling_precip: xr.DataArray,
                         start_year: int, end_year: int,
                         window_hours: int,
                         threshold_inches: float) -> List[Dict]:
    """
    Find all non-overlapping precipitation events where the rolling mean exceeds a threshold.
    
    Args:
        rolling_precip: Rolling precipitation data
        start_year: Starting year
        end_year: Ending year
        window_hours: Rolling window size in hours
        threshold_inches: Threshold for mean rolling precipitation (inches)
        
    Returns:
        List of dictionaries with event information
    """
    events = []
    
    # Convert inches threshold to mm (data units)
    threshold_mm = threshold_inches / MM_TO_INCH_CONVERSION_FACTOR
    
    logging.info(f"Finding all events where rolling mean >= {threshold_inches:.2f} inches")
    
    # Restrict to requested analysis years
    data = rolling_precip.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31T23:59:59"))
    
    if len(data.time) == 0:
        logging.warning("No data available for the requested analysis period")
        return events
    
    # Calculate mean precipitation over watershed for each time step
    mean_precip = data.mean(dim=[AORC_X_VAR, AORC_Y_VAR], skipna=True)
    
    if mean_precip.isnull().all():
        logging.warning("All precipitation values are NaN for the requested analysis period")
        return events
    
    # Identify contiguous threshold-exceedance segments (non-overlapping events)
    mean_series = mean_precip.to_series()
    exceed = mean_series >= threshold_mm
    exceed = exceed.fillna(False)
    
    # Group by consecutive True/False runs
    run_id = (exceed != exceed.shift()).cumsum()
    true_runs = run_id[exceed]
    
    if true_runs.empty:
        logging.warning("No threshold exceedance events found")
        return events
    
    # Track event numbering per year for clearer outputs
    per_year_counter: Dict[int, int] = {}
    
    for rid in true_runs.unique():
        segment = mean_series[run_id == rid]
        if segment.empty:
            continue
        
        # Choose the single rolling window time that maximizes mean precipitation for this event
        peak_time = pd.Timestamp(segment.idxmax())
        peak_value_mm = float(segment.loc[peak_time])
        peak_value_inches = peak_value_mm * MM_TO_INCH_CONVERSION_FACTOR
        
        # Calculate start and end times for the storm
        end_time = peak_time
        start_time = end_time - pd.Timedelta(hours=window_hours)
        
        # Get the precipitation grid for this event
        precip_grid = data.sel(time=peak_time, method='nearest')
        
        # Calculate additional statistics
        total_precip_mm = float(precip_grid.sum(skipna=True).values)
        total_precip_inches = total_precip_mm * MM_TO_INCH_CONVERSION_FACTOR
        min_precip_inches = float(precip_grid.min(skipna=True).values) * MM_TO_INCH_CONVERSION_FACTOR
        max_precip_inches = float(precip_grid.max(skipna=True).values) * MM_TO_INCH_CONVERSION_FACTOR
        
        year = int(end_time.year)
        per_year_counter[year] = per_year_counter.get(year, 0) + 1
        event_num = per_year_counter[year]
        event_id = f"{year}_E{event_num:02d}_{end_time.strftime('%Y%m%dT%H%M')}"
        
        event_info = {
            'event_id': event_id,
            'year': year,
            'event_number_in_year': event_num,
            'start_time': start_time,
            'end_time': end_time,
            'max_time': end_time,
            'mean_precip_mm': peak_value_mm,
            'mean_precip_inches': peak_value_inches,
            'total_precip_mm': total_precip_mm,
            'total_precip_inches': total_precip_inches,
            'min_precip_inches': min_precip_inches,
            'max_precip_inches': max_precip_inches,
            'precip_grid': precip_grid,
            'duration_hours': window_hours,
            'threshold_inches': threshold_inches
        }
        
        events.append(event_info)
        
        logging.info(f"Event {event_id}: {peak_value_inches:.2f} inches "
                    f"on {end_time.strftime('%Y-%m-%d %H:%M')} UTC")
    
    # Sort by time for consistent outputs
    events.sort(key=lambda x: x['end_time'])
    return events

def get_ncl_whitebluegreenyellowred_cmap(n_bins: int = 256) -> LinearSegmentedColormap:
    """Create a colormap similar to NCL WhiteBlueGreenYellowRed."""
    # NCL WhiteBlueGreenYellowRed (approximate control points)
    colors = [
        "#ffffff",  # white
        "#b7e3ff",  # light blue
        "#2b83ba",  # blue
        "#1a9850",  # green
        "#fee08b",  # yellow
        "#f46d43",  # orange-red
        "#d73027"   # red
    ]
    return LinearSegmentedColormap.from_list("ncl_whitebluegreenyellowred", colors, N=n_bins)

def create_precipitation_map(event_info: Dict, watershed_gdf: gpd.GeoDataFrame, 
                           output_path: str, vmin: float = None, vmax: float = None) -> str:
    """
    Create a precipitation map for an annual maximum event.
    
    Args:
        event_info: Event information dictionary
        watershed_gdf: Watershed geometry
        output_path: Output file path
        
    Returns:
        Path to created map file
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get precipitation data and convert to inches
    precip_grid = event_info['precip_grid'] * MM_TO_INCH_CONVERSION_FACTOR
    
    # Create custom colormap (similar to precipitation maps)
    cmap = get_ncl_whitebluegreenyellowred_cmap(n_bins=256)
    
    # Plot precipitation
    im = precip_grid.plot(ax=ax, cmap=cmap, add_colorbar=False, alpha=0.8, vmin=vmin, vmax=vmax)

    # Plot watershed boundary
    watershed_gdf.boundary.plot(ax=ax, color='black', linewidth=2, label='Watershed')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, extend='max')
    cbar.set_label(f'{event_info["duration_hours"]}-hour Precipitation (inches)', fontsize=12)

    
    # Formatting
    title_prefix = "Event"
    if 'event_id' in event_info:
        title_prefix = f'Event {event_info["event_id"]}'
    ax.set_title(f'{title_prefix} {event_info["duration_hours"]}-hour Precipitation - {event_info["year"]}\n'
                f'Event End: {event_info["end_time"].strftime("%Y-%m-%d %H:%M")} UTC\n'
                f'Mean: {event_info["mean_precip_inches"]:.2f} in, '
                f'Max: {event_info["max_precip_inches"]:.2f} in', 
                fontsize=14, pad=20)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add statistics text box
    stats_text = (f'Duration: {event_info["duration_hours"]} hours\n'
                 f'Mean Precipitation: {event_info["mean_precip_inches"]:.2f} in\n'
                 f'Min Precipitation: {event_info["min_precip_inches"]:.2f} in\n'
                 f'Max Precipitation: {event_info["max_precip_inches"]:.2f} in')
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Created precipitation map: {output_path}")
    return output_path

def create_excel_summary(annual_maxima: List[Dict], output_path: str, 
                        watershed_path: str, plot_paths: Dict[int, str]) -> str:
    """
    Create Excel summary of annual maximum events.
    
    Args:
        annual_maxima: List of annual maximum event dictionaries
        output_path: Output Excel file path
        watershed_path: Path to watershed file
        plot_paths: Dictionary mapping years to plot file paths
        
    Returns:
        Path to created Excel file
    """
    # Prepare data for Excel
    excel_data = []
    for event in annual_maxima:
        year = event['year']
        event_key = event.get('event_id', year)
        row = {
            'Event_ID': event.get('event_id', f'{year}'),
            'Year': year,
            'Start_Time_UTC': event['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'End_Time_UTC': event['end_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'Duration_Hours': event['duration_hours'],
            'Mean_Precipitation_Inches': round(event['mean_precip_inches'], 3),
            'Total_Precipitation_Inches': round(event['total_precip_inches'], 3),
            'Min_Precipitation_Inches': round(event['min_precip_inches'], 3),
            'Max_Precipitation_Inches': round(event['max_precip_inches'], 3),
            'Mean_Precipitation_MM': round(event['mean_precip_mm'], 2),
            'Total_Precipitation_MM': round(event['total_precip_mm'], 2),
            'Plot_File_Path': plot_paths.get(event_key, 'N/A')
        }
        if 'threshold_inches' in event:
            row['Threshold_Inches'] = round(event['threshold_inches'], 3)
        if 'event_number_in_year' in event:
            row['Event_Number_In_Year'] = event['event_number_in_year']
        excel_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(excel_data)
    
    # Create Excel writer with multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main summary sheet
        df.to_excel(writer, sheet_name='Annual_Maxima_Summary', index=False)
        
        # Metadata sheet
        metadata = {
            'Parameter': [
                'Analysis_Date',
                'Watershed_File', 
                'Analysis_Period',
                'Storm_Duration_Hours',
                'Total_Years_Analyzed',
                'AORC_Data_Source',
                'Output_Directory'
            ],
            'Value': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                watershed_path,
                f'{START_YEAR} - {END_YEAR}',
                STORM_DURATION_HOURS,
                len(annual_maxima),
                'NOAA AORC v1.1 (1km)',
                OUTPUT_DIR
            ]
        }
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        # Statistics sheet
        if annual_maxima:
            stats_data = {
                'Statistic': [
                    'Mean_Annual_Maximum_Inches',
                    'Median_Annual_Maximum_Inches', 
                    'Min_Annual_Maximum_Inches',
                    'Max_Annual_Maximum_Inches',
                    'Std_Annual_Maximum_Inches'
                ],
                'Value': [
                    round(df['Mean_Precipitation_Inches'].mean(), 3),
                    round(df['Mean_Precipitation_Inches'].median(), 3),
                    round(df['Mean_Precipitation_Inches'].min(), 3),
                    round(df['Mean_Precipitation_Inches'].max(), 3),
                    round(df['Mean_Precipitation_Inches'].std(), 3)
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    logging.info(f"Created Excel summary: {output_path}")
    return output_path

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_events_over_thresh_precipitation():
    """
    Main function to analyze annual maximum precipitation events.
    """
    # Setup
    setup_logging(OUTPUT_DIR)
    logging.info("=" * 80)
    logging.info("ANNUAL MAXIMUM PRECIPITATION ANALYSIS")
    logging.info("=" * 80)
    logging.info(f"Analysis period: {START_YEAR} - {END_YEAR}")
    logging.info(f"Storm duration: {STORM_DURATION_HOURS} hours")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    
    try:
        # Load watershed geometry
        watershed_gdf = load_watershed_geometry(WATERSHED_PATH, CONFIG_FILE)
        logging.info(f"Loaded watershed with {len(watershed_gdf)} feature(s)")
        
        # Generate AORC dataset paths
        aorc_paths = get_aorc_paths(START_YEAR, END_YEAR, AORC_BASE_URL)
        
        # Define time period with some padding
        start_time = datetime(START_YEAR, 1, 1)
        end_time = datetime(END_YEAR + 1, 1, 1)  # Include through end of last year
        
        # Load AORC data
        aorc_data = load_aorc_data(aorc_paths, watershed_gdf, start_time, end_time)
        
        # Calculate rolling precipitation
        rolling_precip = calculate_rolling_precipitation(
            aorc_data, STORM_DURATION_HOURS, watershed_gdf
        )
        
        # Find annual maximum events
        events_over_thresh = find_threshold_events(
            rolling_precip, START_YEAR, END_YEAR, STORM_DURATION_HOURS, ROLLING_THRESHOLD_INCHES
        )
        
        if not events_over_thresh:
            logging.error(f"No events found over {ROLLING_THRESHOLD_INCHES} inches!")
            return
        
        logging.info(f"Found {len(events_over_thresh)} events over {ROLLING_THRESHOLD_INCHES} inches")
        
        # Create output directories
        plots_dir = os.path.join(OUTPUT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Compute a consistent plotting scale (inches) across all events
        plot_vmin = 0.0
        plot_vmax = 8.0 # 8 in is approx a 100-yr 72-hr precip for much of the upper tennessee HUC4

        # Create precipitation maps
        plot_paths = {}
        if CREATE_PLOTS:
            logging.info("Creating precipitation maps...")
            for event in events_over_thresh:
                event_id = event.get('event_id', str(event['year']))
                mean_in = event['mean_precip_inches']
                mean_in_str = f"{mean_in:.2f}".replace(".", "p")
                plot_filename = f"aorc_precip_{event_id}_{STORM_DURATION_HOURS}hr_mean{mean_in_str}.png"
                plot_path = os.path.join(plots_dir, plot_filename)
                
                create_precipitation_map(event, watershed_gdf, plot_path, vmin=plot_vmin, vmax=plot_vmax)
                plot_paths[event_id] = plot_path
        
        # Create Excel summary
        excel_path = os.path.join(OUTPUT_DIR, EXCEL_FILENAME)
        watershed_path_for_excel = CONFIG_FILE if CONFIG_FILE else WATERSHED_PATH
        create_excel_summary(events_over_thresh, excel_path, watershed_path_for_excel, plot_paths)
        
        # Summary statistics
        logging.info("=" * 50)
        logging.info("ANALYSIS SUMMARY")
        logging.info("=" * 50)
        
        mean_values = [event['mean_precip_inches'] for event in events_over_thresh]
        logging.info(f"Years analyzed: {len(events_over_thresh)}")
        logging.info(f"Mean event maximum: {np.mean(mean_values):.2f} inches")
        logging.info(f"Median event maximum: {np.median(mean_values):.2f} inches")
        logging.info(f"Min event maximum: {np.min(mean_values):.2f} inches")
        logging.info(f"Max event maximum: {np.max(mean_values):.2f} inches")
        
        # List event maxima
        logging.info("Maximum Events:")
        for event in sorted(events_over_thresh, key=lambda x: x['end_time']):
            label = event.get('event_id', str(event['year']))
            logging.info(f"  {label}: {event['mean_precip_inches']:.2f} inches "
                        f"on {event['end_time'].strftime('%Y-%m-%d %H:%M')} UTC")
        
        logging.info(f"Results saved to: {OUTPUT_DIR}")
        logging.info(f"Excel summary: {excel_path}")
        if CREATE_PLOTS:
            logging.info(f"Precipitation maps: {plots_dir}")
        
        logging.info("Analysis completed successfully!")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    analyze_events_over_thresh_precipitation()
