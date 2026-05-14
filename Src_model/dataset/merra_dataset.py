"""MERRA-2 Dataset for PyTorch Lightning with z-score normalization."""

import os
import pandas as pd
import numpy as np
import torch
import xarray as xr
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# MERRA-2 Variable Configuration
# ============================================================================

# Single-level variables (surface)
SINGLE_VAR = ["PS", "SLP", "PHIS"]

# Multi-level variables (pressure levels)
PRESS_VAR = ["H", "OMEGA", "QI", "QL", "QV", "RH", "T", "U", "V"]

# Pressure levels (hPa)
PRESS_LEVEL = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725,
               700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100]

# Number of pressure levels
NUM_LEVELS = len(PRESS_LEVEL)

# All variables: "PS", "SLP", ..., "H_1000", "H_975", ..., "V_100"
LIST_VAR = [f"{var}" for var in SINGLE_VAR]
LIST_VAR += [f"{var}_{level}" for var in PRESS_VAR for level in PRESS_LEVEL]


class MerraFull(Dataset):
    """Load MERRA-2 data from NetCDF files with z-score normalization."""

    def __init__(self, merra_path: Path,
                 stat_path: Path = '/N/slate/tnn3/DucHGA/Foundation/Data/merra/base/merra_extend_statistics.xlsx',
                 dataset: str = "train"):
        """
        Args:
            merra_path: CSV file with 'Path' (input) and 'Label_path' (target) columns
            stat_path: Excel file with mean/std statistics for normalization
            dataset: Dataset split name (train/val/test)
        """
        super().__init__()
        
        # Load data table with Path and Label_path columns
        print(f"[MerraFull] Loading {dataset} dataset from {merra_path}")
        self.data_table = pd.read_csv(merra_path).dropna(subset=["Label_path"]).reset_index(drop=True)[:100]

        # Load normalization statistics
        stat = pd.read_excel(stat_path, index_col="Variable")
        stat = stat.loc[LIST_VAR]  # Ensure correct order and indexing

        # Extract mean and std arrays
        self.mean = stat["Mean"].to_numpy().reshape(-1, 1, 1)
        self.std = stat["Std"].to_numpy().reshape(-1, 1, 1)

    def _load_and_normalize(self, path: str) -> np.ndarray:
        """Load NetCDF file and apply z-score normalization."""
        try:
            with xr.open_dataset(path) as ds:
                features = []
                # Single-level variables
                for var in SINGLE_VAR:
                    features.append(ds[var].squeeze().data)
                # Multi-level variables (select first NUM_LEVELS)
                for var in PRESS_VAR:
                    features.extend(ds[var].squeeze().data[:NUM_LEVELS])

            # Normalize: (x - mean) / std
            data = np.stack(features, axis=0)
            data = (data - self.mean) / self.std
            return np.nan_to_num(data)
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
        return (torch.tensor(input_data, dtype=torch.float32),
                torch.tensor(target_data, dtype=torch.float32))


class MerraDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for MERRA-2 dataset."""

    def __init__(self, dataset_class=MerraFull,
                 train_path: str = "train.csv",
                 val_path: str = "val.csv",
                 test_path: str = "test.csv",
                 batch_size: int = 32,
                 num_workers: int = None,
                 pin_memory: bool = True,
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
            **kwargs: Additional arguments passed to dataset_class
        """
        super().__init__()

        # Create datasets
        print(f"[MerraDataModule] Creating train dataset from {train_path}")
        self.train_dataset = dataset_class(merra_path=train_path, dataset="train", **kwargs)
        print(f"[MerraDataModule] Creating val dataset from {val_path}")
        self.val_dataset = dataset_class(merra_path=val_path, dataset="val", **kwargs)
        print(f"[MerraDataModule] Creating test dataset from {test_path}")
        self.test_dataset = dataset_class(merra_path=test_path, dataset="test", **kwargs)

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

    # Test: Load and print batch shapes
    dm = MerraDataModule(
        train_path='/N/slate/tnn3/DucHGA/Foundation/Data/merra/dataset/sample.csv',
        val_path='/N/slate/tnn3/DucHGA/Foundation/Data/merra/dataset/sample.csv',
        test_path='/N/slate/tnn3/DucHGA/Foundation/Data/merra/dataset/sample.csv',
    )

    print("[Test] Getting train dataloader...")
    train_loader = dm.train_dataloader()
    print(f"[Test] Dataloader created with {len(train_loader)} batches")
    
    for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
        print(f"[Test] Batch {batch_idx}: Input {input_batch.shape}, Target {target_batch.shape}")
        if batch_idx == 0:
            break
    
    print("[Test] Completed successfully")
