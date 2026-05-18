import os
import torch

import lightning as L
import numpy as np
import pandas as pd

from torch.nn import MSELoss
from torch.optim import AdamW

from utils.metrics import mae, pearson_r, r2_score, rmse

class RegressionModule(L.LightningModule):
    """Lightning module for regression with MSE loss and regression metrics logging."""

    def __init__(self,
                 model,
                 export_result=None,
                 out_dir=None,
                 optimizer_kwargs = {"lr": 1e-4, "weight_decay": 1e-2},):
        
        super().__init__()
        self.model = model
        self.loss_func = MSELoss()

        self.out_dir = out_dir
        self.export_result = export_result
        self.optimizer_kwargs = optimizer_kwargs

        # Initialize prediction and true label lists for each phase
        self.pred_lists = {"train": [], "val": [], "test": []}
        self.true_lists = {"train": [], "val": [], "test": []}

    def _compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> dict:
        """Compute PearsonR, R2, RMSE, and MAE."""
        return {
            "cc": pearson_r(true, pred),
            "r2": r2_score(true, pred),
            "rmse": rmse(true, pred),
            "mae": mae(true, pred),
        }

    def _log_step_metrics(self, loss: torch.Tensor, true: np.ndarray, pred: np.ndarray, prefix: str):
        """Log metrics for a single step."""
        metrics = self._compute_metrics(true, pred)
        self.log(f"{prefix}_step_loss", loss, prog_bar=True, on_epoch=True)
        for key, value in metrics.items():
            self.log(f"{prefix}_step_{key}", value, on_epoch=True)

    def _log_epoch_metrics(self, phase: str):
        """Log metrics for an entire epoch."""
        pred = np.array(self.pred_lists[phase])
        true = np.array(self.true_lists[phase])
        metrics = self._compute_metrics(true, pred)

        for key, value in metrics.items():
            self.log(f"{phase}_epoch_{key}", value, prog_bar=True, on_epoch=True)

        # Reset lists
        self.pred_lists[phase] = []
        self.true_lists[phase] = []

    def training_step(self, batch, batch_idx):
        inputs, true = batch
        pred = self.model(inputs).squeeze()
        loss = self.loss_func(pred, true)
        pred_np = pred.cpu().detach().numpy()
        true_np = true.cpu().detach().numpy()

        self._log_step_metrics(loss, true_np, pred_np, "train")
        self.pred_lists["train"].extend(pred_np)
        self.true_lists["train"].extend(true_np)

        return loss

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def validation_step(self, batch, batch_idx):
        inputs, true = batch
        pred = self.model(inputs).squeeze()
        loss = self.loss_func(pred, true)
        pred_np = pred.cpu().detach().numpy()
        true_np = true.cpu().detach().numpy()

        self._log_step_metrics(loss, true_np, pred_np, "val")
        self.pred_lists["val"].extend(pred_np)
        self.true_lists["val"].extend(true_np)

        return loss

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def test_step(self, batch, batch_idx):
        inputs, true = batch
        pred = self.model(inputs).squeeze()
        pred_np = pred.cpu().detach().numpy()
        true_np = true.cpu().detach().numpy()

        self.pred_lists["test"].extend(pred_np)
        self.true_lists["test"].extend(true_np)

    def on_test_epoch_end(self):
        pred = np.array(self.pred_lists["test"])
        true = np.array(self.true_lists["test"])

        scoreboard = pd.DataFrame()
        scoreboard.loc["metrics", "cc"] = pearson_r(true, pred)
        scoreboard.loc["metrics", "r2"] = r2_score(true, pred)
        scoreboard.loc["metrics", "rmse"] = rmse(true, pred)
        scoreboard.loc["metrics", "mae"] = mae(true, pred)

        if self.export_result:
            scoreboard.to_excel(os.path.join(self.out_dir, f"{self.export_result}.xlsx"))
        else:
            print(scoreboard)

        self.pred_lists["test"] = []
        self.true_lists["test"] = []

    def predict_step(self, batch, batch_idx):
        inputs, _ = batch
        return self.model(inputs).squeeze()

    def configure_optimizers(self):
        return AdamW(self.parameters(), **self.optimizer_kwargs)
