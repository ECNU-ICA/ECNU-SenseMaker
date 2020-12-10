import numpy as np
import torch


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
