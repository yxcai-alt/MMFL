import torch
import torch.nn.functional as F
import numpy as np


def Loss_fn(X_featue, adj, output, target, Mask, Loss_F, label_weight, device):
    loss = Loss_F(output[Mask], target[Mask])

    return loss