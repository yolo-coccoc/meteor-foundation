"""MERRA-2 dataset utilities for PyTorch / Lightning.

This module provides Dataset and DataModule implementations to load
MERRA-2 reanalysis data prepared as NetCDF files (and a companion `.pt`
format loader). It handles common preprocessing steps like spatial
cropping, resizing and applies z-score normalization using precomputed
per-variable mean/std statistics stored in an Excel file.

Main components:
- `MerraFull`: torch.utils.data.Dataset that reads NetCDF files via
    xarray, extracts single-level and pressure-level variables, applies
    preprocessing/postprocessing and returns normalized tensors.
- `MerraFull_pt`: thin subclass expecting tensors saved with `torch.save`.
- `MerraDataModule`: LightningDataModule that constructs train/val/test
    datasets and exposes DataLoader factories.

The file also defines small helper functions for cropping and
resizing tensors.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# MERRA-2 Variable Configuration
# ============================================================================

# Single-level variables (surface quantities with no vertical levels)
# These are expected to exist in the NetCDF dataset with shape
# (time, latitude, longitude) or (latitude, longitude) after squeezing.
SINGLE_VAR = ["PHIS", "PS", "SLP"]

# Multi-level variables defined on pressure levels. For each variable we
# will extract values at a set of pressure levels defined in PRESS_LEVEL.
PRESS_VAR = ["H", "OMEGA", "QI", "QL", "QV", "RH", "T", "U", "V"]

# Pressure levels (in hPa) used to index the vertical dimension. Order
# matters because the normalization statistics expect the flattened
# channel ordering to follow LIST_VAR below.
PRESS_LEVEL = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725,
               700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100]

# Number of pressure levels we will extract for each multi-level variable
NUM_LEVELS = len(PRESS_LEVEL)

# LIST_VAR is the flattened channel order expected by the statistics
# spreadsheet. It begins with single-level fields followed by each
# multi-level variable expanded across pressure levels (e.g. "H_1000").
LIST_VAR = [f"{var}" for var in SINGLE_VAR]
LIST_VAR += [f"{var}_{level}" for var in PRESS_VAR for level in PRESS_LEVEL]


# ============================================================================
# Preprocessing and Postprocessing Functions
# ============================================================================

def preprocess_crop_nc(ds, lat_min, lat_max, lon_min, lon_max):
    """
    Crop NetCDF dataset to specified latitude and longitude range.
    
    Args:
        ds: xarray.Dataset object
        lat_min, lat_max: Latitude range (degrees)
        lon_min, lon_max: Longitude range (degrees)
    
    Returns:
        Cropped xarray.Dataset
    """
    # xarray selection returns a view of the dataset sliced by latitude
    # and longitude. This is useful to reduce IO and memory if training
    # on a regional subset of the globe.
    return ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))


def postprocess_resize_array(tensor, target_height, target_width):
    """
    Resize torch tensor to target spatial dimensions using bilinear interpolation.
    
    Args:
        tensor: Input tensor with shape (channels, height, width)
        target_height: Target height
        target_width: Target width
    
    Returns:
        Resized torch tensor with shape (channels, target_height, target_width)
    """
    # If the spatial size already matches, return early. Otherwise use
    # bilinear interpolation on the spatial axes. `F.interpolate`
    # expects a batch dimension, so temporarily unsqueeze and then
    # remove the batch dim before returning.
    if tensor.shape[-2:] == (target_height, target_width):
        return tensor  # Already correct size

    tensor = tensor.unsqueeze(0)
    resized = F.interpolate(tensor, size=(target_height, target_width), mode='bilinear', align_corners=False)
    return resized.squeeze(0)

class MerraFull(Dataset):
    """Dataset that loads MERRA-2 fields from NetCDF files.

    Each sample row in the CSV passed to `merra_path` must contain two
    columns: `Path` (the input NetCDF file) and `Label_path` (the
    corresponding target NetCDF file). The dataset reads files lazily
    in `__getitem__` using xarray, extracts channels according to
    `SINGLE_VAR` and `PRESS_VAR`/`PRESS_LEVEL`, applies optional
    `pre_process` functions to the xarray Dataset and optional
    `post_process` functions to the final tensor, then returns a
    z-score normalized `torch.float32` tensor.
    """

    def __init__(self, merra_path: Path,
                 stat_path: Path = '/N/slate/tnn3/DucHGA/meteor-foundation/Data/merra/base/merra_extend_statistics.xlsx',
                 dataset: str = "train",
                 pre_process=None,
                 post_process=None):
        """
        Args:
            merra_path: CSV file with 'Path' (input) and 'Label_path' (target) columns
            stat_path: Excel file with mean/std statistics for normalization
            dataset: Dataset split name (train/val/test)
            pre_process: Function or list of functions to apply after opening NetCDF file
            post_process: Function or list of functions to apply after creating numpy array
        """
        super().__init__()
        
        # Load CSV listing input/target file pairs. Drop rows where the
        # target is missing. The `[:100]` slice is a small-sample
        # restriction present in the original script and can be removed
        # for full-dataset runs.
        print(f"[MerraFull] Loading {dataset} dataset from {merra_path}")
        self.data_table = pd.read_csv(merra_path).dropna(subset=["Label_path"]).reset_index(drop=True)[:100]

        # Load normalization statistics (expected Excel file has rows
        # indexed by variable names matching LIST_VAR and columns
        # 'Mean' and 'Std'). We reindex to LIST_VAR to guarantee the
        # channel ordering matches the stacked data produced below.
        stat = pd.read_excel(stat_path, index_col="Variable")
        stat = stat.loc[LIST_VAR]  # Ensure correct order and indexing

        # Convert series to arrays shaped (channels, 1, 1) so they can
        # broadcast across spatial axes when normalizing tensors.
        self.mean = stat["Mean"].to_numpy().reshape(-1, 1, 1)
        self.std = stat["Std"].to_numpy().reshape(-1, 1, 1)
        
        # Normalize pre_process and post_process to lists
        self.pre_process = self._normalize_processors(pre_process)
        self.post_process = self._normalize_processors(post_process)
    
    def _normalize_processors(self, processors):
        """Convert single function or list of functions to list."""
        if processors is None:
            return []
        elif callable(processors):
            return [processors]
        elif isinstance(processors, (list, tuple)):
            return list(processors)
        else:
            raise TypeError(f"Expected function, list of functions, or None, got {type(processors)}")

    def _load_and_normalize(self, path: str) -> np.ndarray:
        """Load NetCDF file, extract fields and apply z-score normalization.

        Steps:
        1. Open the dataset with xarray.
        2. Optionally apply Dataset-level preprocessors (e.g. cropping).
        3. Extract single-level variables and expand multi-level
            variables according to `PRESS_LEVEL`.
        4. Stack channels into a numpy array with shape
            (channels, height, width).
        5. Apply z-score normalization using preloaded statistics.
        6. Replace NaNs with numeric values, convert to `torch.float32`.
        7. Flip the latitude axis so the tensor uses a consistent
            orientation (e.g. north-to-south → top-to-bottom).
        8. Optionally apply tensor-level postprocessors (e.g. resizing).
        """
        try:
            with xr.open_dataset(path) as ds:
                # Apply pre-processing functions
                for processor in self.pre_process:
                    ds = processor(ds)
                
                # Collect features in the expected flattened channel
                # order. Each entry appended to `features` should be a 2D
                # array (latitude, longitude).
                features = []
                # Single-level variables: append each 2D field
                for var in SINGLE_VAR:
                    features.append(ds[var].squeeze().data)
                # Multi-level variables: each variable contributes
                # `NUM_LEVELS` 2D fields, one per pressure level.
                for var in PRESS_VAR:
                    features.extend(ds[var].squeeze().data[:NUM_LEVELS])

            # Stack features into (channels, H, W) and normalize using
            # broadcasting with precomputed mean/std.
            data = np.stack(features, axis=0)
            data = (data - self.mean) / self.std
            # Replace NaN/inf introduced by division or missing data
            data = np.nan_to_num(data)
            data = torch.from_numpy(data).to(torch.float32)

            # Flip the latitude axis so tensors are oriented with the
            # first row corresponding to the northernmost latitude.
            data = torch.flip(data, dims=(-2,))

            # Run any user-provided postprocessing (e.g. resizing).
            for processor in self.post_process:
                data = processor(data)

            return data
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.data_table)

    def __getitem__(self, idx: int) -> tuple:
        """Return (input, target) tensors."""
        row = self.data_table.iloc[idx]
        input_data = self._load_and_normalize(row["Path"])
        target_data = self._load_and_normalize(row["Label_path"])
        return input_data, target_data

class MerraFull_pt(MerraFull):
    """Load MERRA-2 data from .pt files with z-score normalization."""

    def _load_and_normalize(self, path: str) -> torch.Tensor:
        """Load pre-saved tensor from `.pt` file and normalize.

        This loader assumes `path` is a Torch-saved tensor with the same
        channel ordering and spatial dimensions used when computing the
        statistics. It applies the same normalization and postprocessing
        pipeline as `MerraFull`.
        """
        try:
            data = torch.load(path).numpy()
            data = (data - self.mean) / self.std
            data = np.nan_to_num(data)
            data = torch.from_numpy(data).to(torch.float32)
            
            for processor in self.post_process:
                data = processor(data)
            
            return data
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")
            raise

class MerraDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for MERRA-2 dataset."""

    def __init__(self, 
                 dataset_class=MerraFull,
                 train_path: str = "train.csv",
                 val_path: str = "val.csv",
                 test_path: str = "test.csv",
                 batch_size: int = 32,
                 num_workers: int = None,
                 pin_memory: bool = torch.cuda.is_available(),
                 pre_process=None,
                 post_process=None,
                 **kwargs):
        """
        Args:
            dataset_class: Dataset class (default: MerraFull)
            train_path: Training CSV file path
            val_path: Validation CSV file path
            test_path: Test CSV file path
            batch_size: Batch size for dataloaders
            num_workers: Number of data loading workers (default: CPU count)
            pin_memory: Pin memory for GPU transfer
            pre_process: Preprocessing function(s) to apply
            post_process: Postprocessing function(s) to apply
            **kwargs: Additional arguments passed to dataset_class
        """
        super().__init__()

        # Create datasets
        self.train_dataset = dataset_class(merra_path=train_path, dataset="train", 
                                           pre_process=pre_process, post_process=post_process, **kwargs)
        self.val_dataset = dataset_class(merra_path=val_path, dataset="val",
                                         pre_process=pre_process, post_process=post_process, **kwargs)
        self.test_dataset = dataset_class(merra_path=test_path, dataset="test",
                                          pre_process=pre_process, post_process=post_process, **kwargs)

        # Configuration
        self.batch_size = batch_size
        self.num_workers = num_workers or os.cpu_count()
        self.pin_memory = pin_memory

    def _create_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Helper: Create DataLoader with consistent configuration."""
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                         num_workers=self.num_workers, pin_memory=self.pin_memory)

    def train_dataloader(self) -> DataLoader:
        return self._create_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._create_loader(self.test_dataset)


if __name__ == "__main__":

    # # Create preprocessing function that crops to a specific region
    # def crop_to_region(ds):
    #     return preprocess_crop_nc(ds, lat_min=0, lat_max=30, lon_min=100, lon_max=150)
    
    # # Create postprocessing function that resizes to specific dimensions
    # def resize_to_60x80(array):
    #     return postprocess_resize_array(array, target_height=60, target_width=80)

    # # Test: Load and print batch shapes
    # dm_with_processing = MerraDataModule(
    #     train_path='/N/slate/tnn3/DucHGA/meteor-foundation/Data/merra/dataset/sample.csv',
    #     val_path='/N/slate/tnn3/DucHGA/meteor-foundation/Data/merra/dataset/sample.csv',
    #     test_path='/N/slate/tnn3/DucHGA/meteor-foundation/Data/merra/dataset/sample.csv',
    #     pre_process=crop_to_region,
    #     post_process=resize_to_60x80
    # )

    # train_loader = dm_with_processing.train_dataloader()

    
    # for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
    #     # Print shapes for a quick sanity check. In many environments the
    #     # loader may perform IO on worker processes so printing is useful
    #     # for verifying that batching and preprocessing work as expected.
    #     print(f"[Test] Batch {batch_idx}: Input {input_batch.shape}, Target {target_batch.shape}")
    #     if batch_idx == 0:
    #         break


    def resize_to_60x80(array):
        return postprocess_resize_array(array, target_height=60, target_width=80)

    # Test: Load and print batch shapes
    dm_with_processing = MerraDataModule(
        dataset_class=MerraFull_pt,
        train_path='/N/slate/tnn3/DucHGA/meteor-foundation/Data/merra/dataset/sample_dataset_pt/train.csv',
        val_path='/N/slate/tnn3/DucHGA/meteor-foundation/Data/merra/dataset/sample_dataset_pt/val.csv',
        test_path='/N/slate/tnn3/DucHGA/meteor-foundation/Data/merra/dataset/sample_dataset_pt/test.csv',
        pre_process=None,
        post_process=resize_to_60x80
    )

    train_loader = dm_with_processing.train_dataloader()

    
    for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
        # Print shapes for a quick sanity check. In many environments the
        # loader may perform IO on worker processes so printing is useful
        # for verifying that batching and preprocessing work as expected.
        print(f"[Test] Batch {batch_idx}: Input {input_batch.shape}, Target {target_batch.shape}")
        if batch_idx == 0:
            break