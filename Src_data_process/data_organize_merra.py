"""Organize MERRA2 files into CSV with path and datetime information."""

import os
import glob
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import random


def extract_datetime_from_filename(filename):
    """Extract datetime from merra2_YYYYMMDD_HH_MM.nc format.
    
    Args:
        filename: e.g., 'merra2_19800101_00_00.nc'
    
    Returns:
        str: ISO format datetime string (YYYY-MM-DD HH:MM:SS)
    """
    # Extract pattern: merra2_YYYYMMDD_HH_MM
    match = re.search(r'merra2_(\d{8})_(\d{2})_(\d{2})', filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        hour = match.group(2)       # HH
        minute = match.group(3)     # MM
        
        dt = datetime.strptime(f"{date_str}{hour}{minute}", "%Y%m%d%H%M")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return None


def organize_merra2_files(directory, output_dir=None):
    """Organize MERRA2 files into CSV with absolute paths and datetimes.
    
    Args:
        directory: folder containing merra2_*.nc files
        output_dir: output directory for CSV files (defaults to input directory)
    
    Returns:
        pd.DataFrame: dataframe with Path and Datetime columns
    """
    if output_dir is None:
        output_dir = directory
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all merra2 files matching pattern
    pattern = os.path.join(directory, "merra2_*.nc")
    files = sorted(glob.glob(pattern))
    
    print(f"Found {len(files)} MERRA2 files in {directory}")
    
    if not files:
        print("No files found matching pattern merra2_*.nc")
        return None
    
    # Build dataframe
    data = []
    for filepath in files:
        filename = os.path.basename(filepath)
        datetime_str = extract_datetime_from_filename(filename)
        
        if datetime_str:
            abs_path = os.path.abspath(filepath)
            data.append({
                "Path": abs_path,
                "Datetime": datetime_str
            })
    
    df = pd.DataFrame(data)
    print(f"Successfully extracted {len(df)} records")
    
    # Save full path CSV
    full_csv_path = os.path.join(output_dir, "full_path.csv")
    df.to_csv(full_csv_path, index=False)
    print(f"Saved full path CSV: {full_csv_path}")
    
    # Generate sample CSV with 100 random rows (or fewer if dataset is smaller)
    sample_size = min(100, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    sample_csv_path = os.path.join(output_dir, "sample_path.csv")
    sample_df.to_csv(sample_csv_path, index=False)
    print(f"Saved sample CSV ({sample_size} rows): {sample_csv_path}")
    
    return df


if __name__ == "__main__":
    # Define input and output directories directly
    input_dir = "/N/scratch/tnn3/DATA/nasa-merra2/merra2_extend"
    output_dir = "/N/slate/tnn3/DucHGA/Foundation/Data/merra/base"
    
    df = organize_merra2_files(input_dir, output_dir)
