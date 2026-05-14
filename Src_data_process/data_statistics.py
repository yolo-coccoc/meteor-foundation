import sys
import os

sys.path.insert(0, '/N/slate/tnn3/DucHGA/TC-formation/Src_model')
import collections
import math
import traceback

import pandas as pd
import xarray as xr
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from Utils.New_features import vorticity, divergence, meshgrid

# Global placeholder for the worker's shared memory context
WORKER_CONFIG = {}

def worker_init(config_dict):
    """
    Initializes the shared memory context for a worker.
    This ensures heavy objects (like lat/lon grids) are serialized ONLY ONCE per worker,
    rather than being re-serialized for every single file.
    """
    global WORKER_CONFIG
    WORKER_CONFIG = config_dict


def process_file_optimized(file_path):
    """Process a single NetCDF file using the pre-initialized WORKER_CONFIG."""
    try:
        cfg = WORKER_CONFIG
        
        # Open dataset with time decoding disabled for faster parsing
        ds = xr.open_dataset(file_path, engine='netcdf4', decode_times=False)
        
        num_channels = cfg['num_channels']
        spatial_size = cfg['spatial_size']
        
        # 1. PRE-ALLOCATE MEMORY: Avoids dynamic list stacking overhead
        data_2d = np.empty((num_channels, spatial_size), dtype=np.float32)
        idx = 0
        
        # 2. Extract Pressure Variables
        num_levels = len(cfg['level_indices'])
        for var in cfg['pressure_vars']:
            if var in ds.data_vars:
                # Direct slice using pre-calculated sorted indices
                data = ds[var].squeeze().values[cfg['level_indices'], :, :]
                data_2d[idx : idx + num_levels, :] = data.reshape(num_levels, -1)
            else:
                data_2d[idx : idx + num_levels, :] = np.nan
            idx += num_levels
            
        # 3. Extract and Compute VOR / DIV
        if cfg['u_var'] and cfg['v_var']:
            if cfg['u_var'] in ds.data_vars and cfg['v_var'] in ds.data_vars:
                u_data = ds[cfg['u_var']].squeeze().values[cfg['level_indices'], :, :]
                v_data = ds[cfg['v_var']].squeeze().values[cfg['level_indices'], :, :]
                
                # Use PRE-COMPUTED grids to save massive CPU cycles
                vor = vorticity(u_data, v_data, cfg['lat_grid'], cfg['lon_grid'])
                div = divergence(u_data, v_data, cfg['lat_grid'], cfg['lon_grid'])
                
                data_2d[idx : idx + num_levels, :] = vor.reshape(num_levels, -1)
                idx += num_levels
                
                data_2d[idx : idx + num_levels, :] = div.reshape(num_levels, -1)
                idx += num_levels
            else:
                data_2d[idx : idx + num_levels * 2, :] = np.nan
                idx += num_levels * 2
                
        # 4. Extract Single-level Variables
        for var in cfg['single_level_vars']:
            if var in ds.data_vars:
                data_2d[idx, :] = ds[var].squeeze().values.ravel()
            else:
                data_2d[idx, :] = np.nan
            idx += 1
            
        ds.close()
        
        # 5. Vectorized Masking & Fast Accumulation
        valid_mask = np.isfinite(data_2d)
        data_2d[~valid_mask] = 0.0  # Zero out NaNs so they don't affect sums
        
        # Cast to float64 strictly for summation to prevent overflow
        data_64 = data_2d.astype(np.float64)
        
        sum_val = data_64.sum(axis=1)
        sum_sq_val = (data_64 * data_64).sum(axis=1)
        count_val = valid_mask.sum(axis=1)
        
        return sum_val, sum_sq_val, count_val
        
    except Exception as e:
        return f"Error in {os.path.basename(file_path)}: {e}\n{traceback.format_exc()}"


def calculate_statistics_ignore_nan(nc_directory: str, num_workers: int = None):
    if num_workers is None:
        num_workers = min(8, cpu_count())
        
    nc_files = sorted([os.path.join(nc_directory, f) for f in os.listdir(nc_directory) if f.endswith('.nc')])
    if not nc_files:
        print(f"Lỗi: Không tìm thấy file .nc nào trong '{nc_directory}'")
        return None

    print("Phát hiện các biến và tính toán bộ lưới tiền xử lý...")
    with xr.open_dataset(nc_files[0], engine='netcdf4') as ds_first:
        lat = ds_first['lat'].values if 'lat' in ds_first.coords else ds_first['latitude'].values
        lon = ds_first['lon'].values if 'lon' in ds_first.coords else ds_first['longitude'].values
        spatial_size = len(lat) * len(lon)
        
        # Identify dimensions
        level_dim_name = next((dim for dim in ds_first.dims if dim in ['lev', 'level', 'isobaricInhPa', 'isobaricInPa']), None)
        
        # Pre-calculate deterministic level mapping (100 to 1000 hPa, sorted descending)
        if level_dim_name:
            all_levels = ds_first[level_dim_name].values
            valid_mask = (all_levels >= 100) & (all_levels <= 1000)
            valid_levels = all_levels[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            
            # Sort descending (e.g., 1000, 850, 500...)
            sort_args = np.argsort(valid_levels)[::-1]
            filtered_levels = valid_levels[sort_args]
            filtered_level_indices = valid_indices[sort_args]
        else:
            filtered_levels = []
            filtered_level_indices = []

        # Identify U/V
        u_var = next((v for v in ds_first.data_vars if v in ['U', 'ugrdprs']), None)
        v_var = next((v for v in ds_first.data_vars if v in ['V', 'vgrdprs']), None)
        
        # Categorize vars
        pressure_vars = sorted([v for v in ds_first.data_vars if level_dim_name in ds_first[v].dims])
        single_level_vars = sorted([v for v in ds_first.data_vars if level_dim_name not in ds_first[v].dims])

    # Pre-compute spatial grids for derived calculations
    if len(filtered_levels) > 0 and u_var and v_var:
        lat_grid, lon_grid = meshgrid(lat, lon, len(filtered_levels))
    else:
        lat_grid, lon_grid = None, None

    # Construct variable labels deterministically
    variable_labels = []
    for var in pressure_vars:
        variable_labels.extend([f"{var}_{int(lv)}" for lv in filtered_levels])
    
    if u_var and v_var and len(filtered_levels) > 0:
        variable_labels.extend([f"VOR_{int(lv)}" for lv in filtered_levels])
        variable_labels.extend([f"DIV_{int(lv)}" for lv in filtered_levels])
        
    variable_labels.extend(single_level_vars)
    num_channels = len(variable_labels)
    
    print(f"Tổng số biến phát hiện: {num_channels} kênh")
    
    # Build Master Configuration for Workers
    master_config = {
        'num_channels': num_channels,
        'spatial_size': spatial_size,
        'pressure_vars': pressure_vars,
        'single_level_vars': single_level_vars,
        'level_indices': filtered_level_indices,
        'u_var': u_var,
        'v_var': v_var,
        'lat_grid': lat_grid,
        'lon_grid': lon_grid
    }

    # Initialize accumulators
    total_sum    = np.zeros(num_channels, dtype=np.float64)
    total_sum_sq = np.zeros(num_channels, dtype=np.float64)
    valid_counts = np.zeros(num_channels, dtype=np.int64)

    print(f"\nBắt đầu xử lý {len(nc_files)} file với {num_workers} workers...")
    
    # Initialize pool with the shared configuration
    with Pool(processes=num_workers, initializer=worker_init, initargs=(master_config,)) as pool:
        for result in tqdm(pool.imap_unordered(process_file_optimized, nc_files, chunksize=4),
                          total=len(nc_files), desc="Xử lý", unit="file"):
            if isinstance(result, str):
                print(f"\n[WARNING] {result}") # Caught an exception in the worker
            else:
                s_sum, s_sq, v_cnt = result
                total_sum += s_sum
                total_sum_sq += s_sq
                valid_counts += v_cnt

    # Vectorized Final Calculation
    counts = np.maximum(valid_counts, 1).astype(np.float64)
    mean = total_sum / counts
    var = (total_sum_sq / counts) - (mean * mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    no_data = (valid_counts == 0)
    mean[no_data] = np.nan
    std[no_data]  = np.nan

    results = collections.OrderedDict()
    for i, label in enumerate(variable_labels):
        results[label] = {
            'Mean': float(mean[i]),
            'Std':  float(std[i]),
            'Valid Pixels': int(valid_counts[i])
        }
    return results


def save_results_to_excel(results: dict, output_path: str):
    if not results:
        return
    rows = []
    for var, v in results.items():
        mean, std = v['Mean'], v['Std']
        variation = (std / mean) if (not np.isnan(mean) and not np.isnan(std) and abs(mean) > 1e-12) else np.nan
            
        rows.append({
            'Variable': var,
            'Mean': mean,
            'Std': std,
            'Variation': variation,
            'Valid Pixels': v['Valid Pixels'],
        })
        
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"✅ Đã lưu Excel: {output_path}")
    except Exception as e:
        csv_path = os.path.splitext(output_path)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Đã lưu CSV: {csv_path} (Lỗi Excel: {e})")


if __name__ == '__main__':
    # --- CẤU HÌNH ---
    NC_FOLDER_PATH = '/N/scratch/tnn3/DATA/nasa-merra2/merra2_extend'
    OUTPUT_EXCEL_PATH = '/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/merra_extend_statistics.xlsx'
    NUM_WORKERS = min(32, cpu_count())

    statistics_results = calculate_statistics_ignore_nan(NC_FOLDER_PATH, num_workers=NUM_WORKERS)

    if statistics_results:
        save_results_to_excel(statistics_results, OUTPUT_EXCEL_PATH)