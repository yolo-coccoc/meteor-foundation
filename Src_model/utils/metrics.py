import numpy as np

__all__ = [
    "rmse",
    "mae",
    "pearson_r",
    "r2_score",
]


def rmse(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Root mean square error over flattened prediction and target arrays."""
    predictions = predictions.flatten()
    targets = targets.flatten()
    result = np.sqrt(np.mean((predictions - targets) ** 2) + eps)
    return np.array(result)


def mae(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Mean absolute error over flattened prediction and target arrays."""
    predictions = predictions.flatten()
    targets = targets.flatten()
    result = np.mean(np.abs(predictions - targets))
    return np.array(result)


def pearson_r(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Pearson correlation coefficient over flattened prediction and target arrays."""
    predictions = predictions.flatten()
    targets = targets.flatten()
    predictions_mean = np.mean(predictions)
    targets_mean = np.mean(targets)
    predictions_centered = predictions - predictions_mean
    targets_centered = targets - targets_mean
    covariance = np.mean(predictions_centered * targets_centered)
    predictions_std = np.std(predictions)
    targets_std = np.std(targets)
    result = covariance / (predictions_std * targets_std + eps)
    return np.array(result)


def r2_score(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Coefficient of determination (R^2) over flattened prediction and target arrays."""
    predictions = predictions.flatten()
    targets = targets.flatten()
    target_mean = np.mean(targets)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - target_mean) ** 2)
    result = 1.0 - ss_res / (ss_tot + eps)
    return np.array(result)
