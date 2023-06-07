# Problems for quantum optimization
# All problems must implement a step() function to optimize for the problem
# All problems must implement an eval() function to evaluate performance without optimization.
# Both functions should return a diagnostics dict. (IE with loss)
# All needed objects (optimizer, loss, circuit, etc.) are passed in constructor.
# Interface intentionally kept vague to account for flexibility in types of learning problems.
# Useful link: setting initial states in quantum circuits
# https://docs.pennylane.ai/en/stable/code/api/pennylane.QubitStateVector.html

import torch
from torch.utils.data import Dataset
import numpy as np

class RandomMixedState():
    """Dataset where we're going from the all-0 state to a random mixed state."""

    def __init__(self, num_qubits, purity):
        """
        Arguments:
            purity (float): Purity of the target state
        """
        self.purity = purity
        self.num_qubits = num_qubits
        v = np.random.normal(0, 1, num_qubits)  # random vector, sampled from normal dist
        # create a random Bloch vector with the specified purity
        self.target = torch.autograd.Variable(
            torch.tensor(np.sqrt(2 * purity - 1) * v / np.sqrt(np.sum(v ** 2))),
            requires_grad=False
        )
        self.input = torch.zeros(num_qubits)

    def step(self):
        return 1

    def eval(self, idx):
        return (self.input, self.target)