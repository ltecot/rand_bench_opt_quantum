# Problems for quantum optimization
# All problems must implement a step() function to optimize for the problem
# All problems must implement an eval() function to evaluate performance without optimization.
# Both functions should return a diagnostics dict. (IE with loss)
# All needed objects (optimizer, loss, circuit, etc.) are passed in constructor.
# Interface intentionally kept vague to account for flexibility in types of learning problems.

import torch
from torch.utils.data import Dataset
import numpy as np

class RandomState():
    """Problem where we're going from the default inital state to a random  state."""

    def __init__(self, qnode, loss_fn, optimizer, num_qubits):
        """
        Arguments:
            purity (float): Purity of the target state
        """
        self.num_qubits = num_qubits
        v = torch.randn(2**num_qubits, dtype=torch.cfloat)
        self.target = v / torch.sqrt(torch.sum(v ** 2))
        self.opt = optimizer
        self.qnode = qnode
        self.loss_fn = loss_fn

    def step(self):
        pred = self.qnode()
        self.opt.zero_grad()
        loss = self.loss_fn(pred, self.target)
        loss.backward()
        self.opt.step()
        return {"loss": loss.item()}

    def eval(self, idx):
        pred = self.qnode()
        return (pred, self.target, self.loss_fn(pred, self.target))