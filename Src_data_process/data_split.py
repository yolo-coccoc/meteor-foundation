"""Split a dataset CSV into train/val/test using year-based holdout and a validation ratio.

Expected CSV columns: Path, Datetime, Label_path, Lead_time

The script splits by year ranges first, then splits train+val into train and val by ratio.
Output files are written in the same directory as the input CSV.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def split_by_year_ranges(
    df: pd.DataFrame,
    datetime_column: str,
    train_val_start_year: int,
    train_val_end_year: int,
    test_start_year: int,
    test_end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if datetime_column not in df.columns:
        raise ValueError(f"Datetime column '{datetime_column}' not found in dataframe columns: {list(df.columns)}")

    df[datetime_column] = pd.to_datetime(df[datetime_column], errors="coerce")
    if df[datetime_column].isna().any():
        raise ValueError("Some datetime values could not be parsed. Check the Datetime column formatting.")

    years = df[datetime_column].dt.year
    train_val_mask = (years >= train_val_start_year) & (years <= train_val_end_year)
    test_mask = (years >= test_start_year) & (years <= test_end_year)

    train_val_df = df.loc[train_val_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    return train_val_df, test_df


def split_train_val(df: pd.DataFrame, val_ratio: float, random_seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("Validation ratio must be between 0 and 1.")

    train_df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    val_size = int(len(train_df) * val_ratio)
    if val_size == 0 and len(train_df) > 0:
        raise ValueError("Validation ratio is too small for the available train+val dataset size.")

    val_df = train_df.iloc[:val_size].reset_index(drop=True)
    train_df = train_df.iloc[val_size:].reset_index(drop=True)
    return train_df, val_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    base_name: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def main() -> None:
    input_csv = Path("/N/slate/tnn3/DucHGA/meteor-foundation/Data/merra/dataset/sample_dataset_pt/full.csv")
    datetime_column = "Datetime"
    train_val_start_year = 1980
    train_val_end_year = 2016
    test_start_year = 2017
    test_end_year = 2024
    val_ratio = 0.2
    random_seed = 42

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    train_val_df, test_df = split_by_year_ranges(
        df,
        datetime_column=datetime_column,
        train_val_start_year=train_val_start_year,
        train_val_end_year=train_val_end_year,
        test_start_year=test_start_year,
        test_end_year=test_end_year,
    )

    train_df, val_df = split_train_val(train_val_df, val_ratio=val_ratio, random_seed=random_seed)

    output_dir = input_csv.parent
    print(output_dir)
    base_name = input_csv.stem
    train_path, val_path, test_path = save_splits(train_df, val_df, test_df, output_dir, base_name)

    print(f"Saved train: {train_path}")
    print(f"Saved val:   {val_path}")
    print(f"Saved test:  {test_path}")


if __name__ == "__main__":
    main()
