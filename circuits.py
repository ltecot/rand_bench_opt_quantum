# File for quantum circuit functions

import pennylane as qml
import torch
import itertools
import numpy as np

# -------- MODULAR CIRCUITS CLASSES --------
# Trainable quantum circuits, implemented in modular classes 
# All classes must maintain a self.params list which contains all trainable parameters.
# All classes must take in optional state preperation + measurment circuits in the constructor
# All classes must maintain a circuit(params, x, A) function which executes the qml circuit.
# TODO: Make template class instead of these docs

class RandomLayers():
    """ Basically just implements qml.RandomLayers but in this interface
        https://docs.pennylane.ai/en/stable/code/api/pennylane.RandomLayers.html """
    
    def __init__(self, num_qubits, num_layers, num_params, ratio_imprim, seed, state_circuit=None, measure_circuit=None):
        """
        Arguments:
            num_qubits (int): number of qubits
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_params = num_params
        self.ratio_imprim = ratio_imprim
        self.seed = seed
        # self.state_circuit = state_circuit
        # self.measure_circuit = measure_circuit

    def params_shape(self):
        return (self.num_layers, self.num_params)

    def circuit(self, params):
        qml.RandomLayers(weights=params, wires=range(self.num_qubits), 
                         ratio_imprim=self.ratio_imprim, seed=self.seed)
        # return qml.state()

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
        params = np.random.normal(0, np.pi, (self.num_layers, self.num_qubits, 3))
        self.params = torch.tensor(params, requires_grad=True)

    def circuit(self):
        for l in range(self.num_layers):
            for i in range(self.num_qubits):
                qml.RX(self.params[l, i, 0], wires=i)
                qml.RY(self.params[l, i, 1], wires=i)
                qml.RZ(self.params[l, i, 2], wires=i)
            for pair in self.cnot_pairs:
                qml.CNOT(wires=pair)
        return qml.state()

# TODO: Remove below, just have it be part of the problems.py classes
#       Can keep some utility functions for special types of quantum loss that require extended circuits.

# -------- RETURN MEASURMENT CIRCUIT FUNCTIONS --------
# These circuit modifiers add on the final measurment of a system.
# Pass these into the constructor of the modular circuits above

# def StateMeasurment(circuit):
#     circuit()
#     return qml.state()

# -------- STATE PREPARATION CIRCUIT FUNCTIONS --------
# These circuit modifiers can be used to add state preparation operations.
# Pass these into the constructor of the modular circuits above

# https://docs.pennylane.ai/en/stable/code/api/pennylane.QubitStateVector.html