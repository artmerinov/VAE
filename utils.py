import torch


def set_random_seed(seed: int):
    """
    Fixes random state for reproducibility. 
    """
    torch.manual_seed(seed)
