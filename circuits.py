# File for quantum circuit functions

import pennylane as qml
import torch
import itertools
import numpy as np

# -------- MODULAR CIRCUITS CLASSES --------
# Trainable quantum circuits, implemented in modular classes 
# All classes must maintain a params_shape() functon that returns the shape of the input params.
# All classes must maintain a circuit(params) function which executes the qml model circuit.
# TODO: Make template class instead of these docs

class RandomLayers():
    """ Basically just a wrapper, implements qml.RandomLayers but in this interface
        https://docs.pennylane.ai/en/stable/code/api/pennylane.RandomLayers.html """
    
    def __init__(self, num_qubits, num_layers, num_params, ratio_imprim, seed, state_circuit=None, measure_circuit=None):
        """
        Arguments:
            arg (type): description
            TODO
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_params = num_params
        self.ratio_imprim = ratio_imprim
        self.seed = seed

    def params_shape(self):
        return (self.num_layers, self.num_params)

    def circuit(self, params):
        qml.RandomLayers(weights=params, wires=range(self.num_qubits), 
                         ratio_imprim=self.ratio_imprim, seed=self.seed)

class FullCnotCircuit():
    """Circuit with universal single qubit gates followed by all possible combinations of cnots"""
    def __init__(self, num_qubits, num_layers):
        """
        Arguments:
            arg (type): description
            TODO
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # TODO: Add random sampling of these pairs, for a different type of circuit class.
        self.cnot_pairs = itertools.combinations(range(self.num_qubits), 2)

    def params_shape(self):
        return (self.num_layers, self.num_qubits, 3)

    def circuit(self, params):
        for l in range(self.num_layers):
            for i in range(self.num_qubits):
                qml.RX(params[l, i, 0], wires=i)
                qml.RY(params[l, i, 1], wires=i)
                qml.RZ(params[l, i, 2], wires=i)
            for pair in self.cnot_pairs:
                qml.CNOT(wires=pair)