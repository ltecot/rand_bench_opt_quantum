# File for quantum circuit functions

import pennylane as qml
import torch
import itertools
import numpy as np

import util as qo_util

# -------- MODULAR CIRCUITS CLASSES --------
# Trainable quantum circuits, implemented in modular classes 
# All classes must maintain a params_shape() functon that returns the shape of the input params.
# All classes must maintain a circuit(params) function which executes the qml model circuit.
# TODO: Make template class instead of these docs

class QcbmAnsatz():
    """ Chosen ansatz from the below paper
        https://dx.doi.org/10.1088/2058-9565/acd578 """
    
    def __init__(self, num_qubits, num_layers):
        """
        Arguments:
            arg (type): description
            TODO
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def params_shape(self):
        return (self.num_layers, 2 * self.num_qubits + (self.num_qubits - 1))

    def circuit(self, params):
        for l in range(self.num_layers):
            for i in range(self.num_qubits):
                qml.RX(params[l, 2*i], wires=i)
                qml.RZ(params[l, 2*i+1], wires=i)
            for i in range(self.num_qubits-1):
                p_ind = 2 * self.num_qubits + i
                qml.QubitUnitary(qo_util.RXX(params[l, p_ind]), wires=[i, i+1])

class RandomLayers():
    """ Basically just a wrapper, implements qml.RandomLayers but in this interface
        https://docs.pennylane.ai/en/stable/code/api/pennylane.RandomLayers.html """
    
    def __init__(self, num_qubits, num_layers, num_params, ratio_imprim, seed, adjoint_fix=False):
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
        self.adjoint_fix = adjoint_fix

    def params_shape(self):
        return (self.num_layers, self.num_params)

    def circuit(self, params):
        # Double adjoint is an annoying way to get around bug where adjoint isn't defined.
        # Can disable whenever not using QNSPSA (not sure how badly this affects compute time)
        if self.adjoint_fix:
            qml.adjoint(qml.adjoint(  
            qml.RandomLayers(weights=params, wires=range(self.num_qubits), 
                            ratio_imprim=self.ratio_imprim, seed=self.seed)
            ))
        else:
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