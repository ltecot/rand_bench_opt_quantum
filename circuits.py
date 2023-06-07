# File for quantum circuits

import pennylane as qml
import torch
import itertools
import numpy as np

# -------- MODULAR CIRCUITS CLASSES --------
# Quantum circuits, implemented in modular classes 
# All classes must maintain a self.parameters list which contains all trainable parameters.
# All classes must maintain a circuit(x) function which executes the qml circuit on input x.
# TODO: Maybe make modular between pytorch and numpy autograd?

class RandomizedCnotCircuit():
    """Circuit with universal single qubit gates followed by randomized cnots"""
    def __init__(self, num_qubits, num_layers):
        """
        Arguments:
            num_layers (int): number of layers
            num_qubits (int): number of qubits
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # TODO: Add random sampling of these pairs
        self.cnot_pairs = itertools.combinations(range(self.num_qubits), 2)
        # TODO: Smarter initalization? Maybe make init a thing that happens in main instead?
        params = np.random.normal(0, np.pi, (self.num_qubits, 3))
        self.parameters = torch.tensor(params, requires_grad=True)

    def circuit(self, x):
        for i in range(self.num_qubits):
            qml.RX(self.parameters[i, 0], wires=i)
            qml.RY(self.parameters[i, 1], wires=i)
            qml.RZ(self.parameters[i, 2], wires=i)
        for pair in self.cnot_pairs:
            qml.CNOT(wires=pair)


# -------- RETURN MEASURMENT CIRCUIT FUNCTIONS --------
# These circuits are intended to be the final measurment of a system.
# These functions are what you pass into the Qnode constructor in the main file.
# Takes in a modular circuit class as an input

def StateMeasurment(circ):
    circ.circuit()
    return qml.state()
