import torch
from torch import nn


def get_module_device(module: nn.Module) -> torch.device:
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.

    Returns:
        torch.device: The device of the module.
    """
    try:
        next(module.parameters())
    except StopIteration:
        raise ValueError("The input module should contain parameters.")

    if next(module.parameters()).is_cuda:
        return torch.device(next(module.parameters()).get_device())

    return torch.device("cpu")
