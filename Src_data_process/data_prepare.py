import pandas as pd
import os
from datetime import timedelta
from typing import List, Union
import numpy as np
from pandarallel import pandarallel


# Initialize pandarallel with progress bar
pandarallel.initialize(progress_bar=True, nb_workers=8)


def add_label_paths_with_lead_time(df: pd.DataFrame,
                                   list_lead_time: List[int],
                                   step: Union[str, timedelta],
                                   datetime_col: str = "Datetime",
                                   path_col: str = "Path",) -> pd.DataFrame:
    """
    Generate Label_path by calculating future datetime and constructing expected filename.
    Check if generated file exists; set to NaN if not found.
    
    Args:
        df: Input dataframe with datetime and path columns
        list_lead_time: List of lead time multipliers (e.g., [1, 2, 3])
        step: Time interval between samples ('3h' for MERRA2, or timedelta)
        datetime_col: Name of datetime column (default: "Datetime")
        path_col: Name of path column (default: "Path")
    
    Returns:
        Dataframe with new Label_path and Lead_time columns for each lead_time value.
        Label_path is NaN if the generated file does not exist.
    """
    
    # Convert datetime column and step
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    def _generate_label_path(row):
        """Generate label path from current path + lead_time; check if exists"""
        # Extract directory and filename

        label_path = row[path_col][: -17]  # Remove last 17 chars (e.g., "YYYYMMDD_HH_MM.nc")
        label_datetime = row[datetime_col] + row["_lead_time"] * step
        label_path = f"{label_path}{label_datetime.strftime('%Y%m%d_%H_%M')}.nc"
        
        # Return path only if file exists, otherwise NaN
        return label_path if os.path.exists(label_path) else np.nan
    
    # Build result: create rows for each lead_time value
    result_parts = []
    for lead_time in list_lead_time:
        print(f"Processing lead_time={lead_time}...")
        temp_df = df.copy()
        temp_df["_lead_time"] = lead_time  # Temporary column for calculation
        # Generate paths in parallel with progress bar
        temp_df["Label_path"] = temp_df.parallel_apply(_generate_label_path, axis=1)
        temp_df["Lead_time"] = lead_time  # Final lead_time column
        temp_df.drop(columns=["_lead_time"], inplace=True)  # Remove temporary column
        result_parts.append(temp_df)
    
    # Concatenate all results
    return pd.concat(result_parts, ignore_index=True)


# ============================================================================
# CONFIGURATION: Edit these parameters
# ============================================================================

if __name__ == "__main__":
    # Input/Output paths
    CSV_INPUT_PATH = "/N/slate/tnn3/DucHGA/meteor-foundation/Data/merra/base/full_path.csv"
    CSV_OUTPUT_PATH = "/N/slate/tnn3/DucHGA/meteor-foundation/Data/merra/dataset/sample.csv"
    
    # Lead time and time step configuration
    LIST_LEAD_TIME = [1]  # Lead time multipliers
    STEP = timedelta(hours=3)  # Time step (e.g., "3h" for MERRA2)
    
    # Column names
    DATETIME_COL = "Datetime"
    PATH_COL = "Path"
    
    # ========================================================================
    # Execute: Read, process, and save
    # ========================================================================
    
    # Read input CSV
    print(f"Reading: {CSV_INPUT_PATH}")
    df = pd.read_csv(CSV_INPUT_PATH)
    # Sort by datetime for consistency
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    print(f"Loaded {len(df)} rows\n")
    
    # Process data: generate Label_path and Lead_time columns
    result = add_label_paths_with_lead_time(
        df, LIST_LEAD_TIME, STEP,
        datetime_col=DATETIME_COL, path_col=PATH_COL
    )
    
    # Summary
    print(f"Processing complete: {len(result)} rows created")
    print(f"Lead times: {LIST_LEAD_TIME}\nStep: {STEP}\n")
    
    # Save output
    result.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Saved: {CSV_OUTPUT_PATH}")
