import random
import numpy as np
import torch
import os

def set_all_seeds(seed):
    """
    Sets the random seed for Python's random module, NumPy, and PyTorch (CPU and CUDA).
    """
    os.environ['PYTHONHASHSEED'] = str(seed) # For Python hash-based operations
    random.seed(seed) # Python's built-in random module
    np.random.seed(seed) # NumPy's random number generator
    torch.manual_seed(seed) # PyTorch's CPU random number generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # PyTorch's CUDA random number generator for all devices
    torch.backends.cudnn.deterministic = True # Ensure deterministic CuDNN operations
    torch.backends.cudnn.benchmark = False # Disable CuDNN benchmark for determinism