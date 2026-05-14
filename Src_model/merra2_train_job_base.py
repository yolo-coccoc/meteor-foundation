import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

sys.path.insert(0, "/N/slate/tnn3/DucHGA/meteor-foundation/Src_model")

from dataset.merra_dataset import MerraDataModule, MerraFull
from model.ClimaX import ClimaX
from progress.Progress import RegressionModule
from progress.Callback import save_checkpoint_callback
from utils.Seed import set_all_seeds


def setup_output_directory(args) -> str:
    """Set up output directory using formal version counting.
    
    Args:
        args: Command-line arguments
    """

    if args.mode < 0:
        version = 0
        while True:
            out_dir = os.path.join(args.out_dir, 
                                    f"{args.project}_s{args.seed}",
                                    f"Step_{args.step}_v{version}")
            if os.path.isdir(out_dir):
                version += 1
            else:
                break

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return out_dir
    elif args.mode >= 0:
        version = args.mode
        return os.path.join(args.out_dir, 
                            f"{args.project}_s{args.seed}",
                            f"Step_{args.step}_v{version}")
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Use negative for new training or >= 0 for loading existing version.")


def save_config(args, out_dir: str):
    """Save arguments to config.json with creation timestamp."""
    config = vars(args).copy()
    config["time_creation"] = datetime.now().isoformat()
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def setup_wandb(args, out_dir: str):
    """Initialize WandB logger."""
    os.environ["WANDB_API_KEY"] = '3b59eddf5201c6c82ed66a6f97c3b2a813ba8929'
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY environment variable not set")
    wandb.login(key=wandb_api_key)
    
    # Extract timestamp from output directory name
    run_name = os.path.basename(out_dir)
    return WandbLogger(project=args.project,
                       name=f"{run_name}_s{args.seed}")


def main(args):
    # Set seed for reproducibility
    set_all_seeds(args.seed)

    # Setup output directory
    out_dir = setup_output_directory(args)
    print(f"Output directory: {out_dir}")

    # Prepare dataset
    ds = MerraDataModule(
        dataset_class=MerraFull,
        train_path=os.path.join(args.inp_dir, f"Step_{args.step}", "train.csv"),
        val_path=os.path.join(args.inp_dir, f"Step_{args.step}", "val.csv"),
        test_path=os.path.join(args.inp_dir, f"Step_{args.step}", "test.csv"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print("Dataset prepared")

    # Prepare model (ClimaX foundation model)
    model = ClimaX(
        in_channels=278,  # MERRA-2 variable count
        out_channels=1,   # Regression output
        patch_size=2,
        depth=12,
    )
    print("Model prepared")

    # Training phase (only if mode < 0)
    lightning_model = None
    if args.mode < 0:
        save_config(args, out_dir)
        
        wandb_logger = setup_wandb(args, out_dir)

        trainer = Trainer(
            logger=wandb_logger,
            log_every_n_steps=100,
            max_epochs=args.max_epochs,
            callbacks=save_checkpoint_callback(out_dir),
            accelerator="auto",
            devices=1,
        )
        
        lightning_model = RegressionModule(
            model,
            export_result=args.export_result,
            optimizer_kwargs={
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            out_dir=out_dir,
        )

        trainer.fit(lightning_model, datamodule=ds)
        print("Training completed")
    
    # Testing phase (always run all checkpoints)
    print("Testing all checkpoints...")
    test_results = {}
    
    for checkpoint_name in args.checkpoint:
        checkpoint_path = os.path.join(out_dir, "checkpoints", f"{checkpoint_name}.ckpt")
        print(f"Loading model from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}, skipping...")
            continue
        
        # Load lightning module with checkpoint
        lightning_model = RegressionModule.load_from_checkpoint(
            checkpoint_path,
            model=model,
            export_result=f"{args.export_result}_{checkpoint_name}" if args.export_result else None,
            optimizer_kwargs={
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            out_dir=out_dir,
        )
        print(f"Model loaded from checkpoint: {checkpoint_name}")
        
        # Run testing
        test_trainer = Trainer(
            accelerator="auto",
            devices=1,
        )
        results = test_trainer.test(lightning_model, datamodule=ds)
        test_results[checkpoint_name] = results
        print(f"Testing completed for {checkpoint_name}")
    
    print(f"All checkpoints tested. Results saved to {out_dir}")
    return test_results


if __name__ == "__main__":
    parser = ArgumentParser(description="Train ClimaX foundation model on MERRA-2 data")

    # Project settings
    parser.add_argument("--project", type=str, default="merra2_climax", help="WandB project name")

    # Dataset arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--step", type=int, default=1, help="Step size")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")

    # Mode and paths
    parser.add_argument("--mode", type=int, default=-1, help="Mode: -1 for new training, >=0 for load existing model version")
    parser.add_argument("--checkpoint", type=str, nargs="*", default=("last", "best_r2", "best_rmse"), help="Checkpoints to save and test")
    parser.add_argument("--inp_dir", type=str, default="./Data/merra/dataset", help="Input dataset directory")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--export_result", type=str, default="regression_results", help="Export test results to file (xlsx)")

    args = parser.parse_args()
    
    main(args)
