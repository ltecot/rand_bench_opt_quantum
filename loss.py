# Custom loss functions for qml

import torch

# L2 Loss for comparing two states
def L2_state_loss(pred, target):
    return torch.sqrt(torch.sum((pred - target) ** 2))