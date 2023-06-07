# Datasets for quantum optimization

# TODO: Rethink this structure. Some "data" is like a hamiltonian rather than input-output pairs.
# Maybe just make it a class that takes in a circuit + optimizer, then have an "step" function
# Probably also make something for data with actual datasets with multiple inputs?
# Maybe have step with a counter and some max_step thing to keep track.

# All datasets must be based off the abstract class torch.utils.data.Dataset
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
# https://docs.pennylane.ai/en/stable/code/api/pennylane.QubitStateVector.html

import torch
from torch.utils.data import Dataset
import numpy as np

class RandomMixedState(Dataset):
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

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return (self.input, self.target)