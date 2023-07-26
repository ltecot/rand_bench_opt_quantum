# Custom utility functions, for use in various QML problems

import torch
import pennylane as qml
import math

# Hamiltonian for ising model (no external field)
def ising_hamiltonian(num_qubits):
    z_coeffs = [-1. for i in range(num_qubits)]
    z_obs = [qml.PauliZ(i) @ qml.PauliZ((i+1) % num_qubits) for i in range(num_qubits)]
    return qml.Hamiltonian(z_coeffs, z_obs)

# Hamiltonian for transverse ising model
def transverse_ising_hamiltonian(num_qubits, h=0.5):
    z_coeffs = [-1. for i in range(num_qubits)]
    z_obs = [qml.PauliZ(i) @ qml.PauliZ((i+1) % num_qubits) for i in range(num_qubits)]
    x_coeffs = [-h for i in range(num_qubits)]
    x_obs = [qml.PauliX(i) for i in range(num_qubits)]
    return qml.Hamiltonian(z_coeffs + x_coeffs, z_obs + x_obs)

# L2 Loss for comparing two states
def L2_state_loss(pred, target):
    diff = (pred - target)
    return torch.sqrt(torch.sum(diff.conj() * diff))

# Fitness utilities, typically for NES
# Basically is a log-ordering with zero mean and zero sum 
def fitness_utilities(n):
    inds = torch.arange(n) + 1
    utilities = math.log((n / 2) + 1) - torch.log(inds)
    utilities[utilities < 0] = 0
    utilities = utilities / torch.sum(utilities)
    utilities = utilities - (1 / n)
    return utilities