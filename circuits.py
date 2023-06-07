# File for quantum circuits

import pennylane as qml
import torch
import itertools
import numpy as np

# -------- MODULAR CIRCUITS CLASSES --------
# Trainable quantum circuits, implemented in modular classes 
# All classes must maintain a self.parameters list which contains all trainable parameters.
# All classes must maintain a circuit() function which executes the qml circuit.
# TODO: Maybe make modular between pytorch and numpy autograd?
# TODO: Change circuit() to take arbitrary arguments, by adding **kwargs to circuit modifiers.
# TODO: Most pennylane functions take params as inputs. Not doing this might mess up numpy autograd.
#       Probably will need to add params to kwargs, and just maintain classes as an easy way to 
#       keep track of the trainable parameters + their dict config that you pass into the function.

# note: https://docs.pennylane.ai/en/stable/code/api/pennylane.RandomLayers.html
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

    def circuit(self):
        for i in range(self.num_qubits):
            qml.RX(self.parameters[i, 0], wires=i)
            qml.RY(self.parameters[i, 1], wires=i)
            qml.RZ(self.parameters[i, 2], wires=i)
        for pair in self.cnot_pairs:
            qml.CNOT(wires=pair)

# -------- RETURN MEASURMENT CIRCUIT FUNCTIONS --------
# These circuit modifiers add on the final measurment of a system.
# Takes in a circuit function as an input

def StateMeasurment(circuit):
    circuit()
    return qml.state()

# -------- STATE PREPARATION CIRCUIT FUNCTIONS --------
# These circuit modifiers can be used to add state preparation operations.
# Takes in a circuit function as an input
