# Custom utility functions, for use in various QML problems

import torch
from torch.autograd import Function
import pennylane as qml
import math
import scipy.linalg
import numpy as np
import copy

# ------------------ HAMILTONIANS ------------------

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

# Hamiltonian for the Quantum Heisenberg model
def heisenberg_2d_hamiltonian(num_qubits, j=1.0, h=0.5):
    # Raise an exception if num_qubits is not a perfect square
    if not math.sqrt(num_qubits).is_integer():
        raise ValueError("Number of qubits must be a perfect square")
    # Create the Hamiltonian
    coeffs = []
    obs = []
    # Enumrate over all qubits in grid, add interactions with qubits that are one index above in each direction
    # Has closed border (AKA periodic boundary conditions)
    nqbit_sqrt = int(math.sqrt(num_qubits))
    for i in range(nqbit_sqrt):
        for j in range(nqbit_sqrt):
            my_qubit = i * nqbit_sqrt + j
            right_qubit = i * nqbit_sqrt + (j + 1) % nqbit_sqrt
            down_qubit = ((i + 1) % nqbit_sqrt) * nqbit_sqrt + j
            # X interaction
            coeffs.append(-j/2)
            coeffs.append(-j/2)
            obs.append(qml.PauliX(my_qubit) @ qml.PauliX(right_qubit))
            obs.append(qml.PauliX(my_qubit) @ qml.PauliX(down_qubit))
            # Y interaction
            coeffs.append(-j/2)
            coeffs.append(-j/2)
            obs.append(qml.PauliY(my_qubit) @ qml.PauliY(right_qubit))
            obs.append(qml.PauliY(my_qubit) @ qml.PauliY(down_qubit))
            # Z interaction
            coeffs.append(-j/2)
            coeffs.append(-j/2)
            obs.append(qml.PauliZ(my_qubit) @ qml.PauliZ(right_qubit))
            obs.append(qml.PauliZ(my_qubit) @ qml.PauliZ(down_qubit))
            # External field
            coeffs.append(-h/2)
            obs.append(qml.PauliZ(my_qubit))
    return qml.Hamiltonian(coeffs, obs)

# Randomized 1-and-2 Interation Hamiltonian
# Samples random Pauli X, Y, or Z (uniformly sampled) for random qubits and weighs randomly
# Will sample random pairs of qubits and do the tensor between either a Pauli X, Y, or Z (uniformly sampled)
# Also weighs each spin pair interaction randomly
def randomized_hamiltonian(num_qubits, num_random_singles, num_random_doubles, rand_seed):
    np.random.seed(rand_seed)
    coeffs = np.random.normal(0, np.pi, num_random_singles + num_random_doubles).tolist()
    single_inds = np.random.randint(low=0, high=num_qubits, size=num_random_singles).tolist()
    single_inds_xyz = np.random.randint(low=0, high=3, size=num_random_singles).tolist()
    double_inds_first = np.random.randint(low=0, high=num_qubits, size=num_random_doubles)
    double_inds_diff = np.random.randint(low=1, high=num_qubits, size=num_random_doubles)
    double_inds_first_xyz = np.random.randint(low=0, high=3, size=num_random_doubles).tolist()
    double_inds_second_xyz = np.random.randint(low=0, high=3, size=num_random_doubles).tolist()
    double_inds_second = np.remainder(double_inds_first + double_inds_diff, num_qubits).tolist()
    paulis = [[qml.PauliX(i), qml.PauliY(i), qml.PauliZ(i)] for i in range(num_qubits)]
    single_obs = [paulis[i][j] for i, j in zip(single_inds, single_inds_xyz)]
    double_obs = [paulis[i][ii] @ paulis[j][jj] for (i, ii, j, jj) in zip(double_inds_first, double_inds_first_xyz, 
                                                                          double_inds_second, double_inds_second_xyz)]
    return qml.Hamiltonian(coeffs, single_obs + double_obs)

# ------------------ PROBABILITY DISTS ------------------

def random_dist(num_qubits, rand_seed):
    np.random.seed(rand_seed)
    probs = np.random.normal(0, np.pi, 2 ** num_qubits)
    probs_abs = torch.abs(torch.tensor(probs))
    norm_probs = probs_abs / torch.sum(probs_abs)
    return norm_probs

def parity_dist(num_qubits, target_num):
    cardinality = torch.tensor([bin(i).count("1") for i in range(2 ** num_qubits)])
    probs = (cardinality == target_num).float()
    norm_probs = probs / torch.sum(probs)
    return norm_probs

# ------------------ LOSS ------------------

# L2 Loss for comparing two states
def L2_state_loss(pred, target):
    diff = (pred - target)
    return torch.sqrt(torch.sum(diff.conj() * diff))

# Negative log likelihood for quantum generative modeling
def nll_loss(pred_probs, target_probs, eps=1e-8):
    p_lp = target_probs * torch.log(torch.maximum(pred_probs, eps * torch.ones(pred_probs.size())))
    return -torch.sum(p_lp)

# ------------------ MISC ------------------

# Fitness utilities, typically for NES
# Basically is a log-ordering with zero mean and zero sum 
def fitness_utilities(n):
    inds = torch.arange(n) + 1
    utilities = math.log((n / 2) + 1) - torch.log(inds)
    utilities[utilities < 0] = 0
    utilities = utilities / torch.sum(utilities)
    utilities = utilities - (1 / n)
    return utilities

# Merge source information in destination dict
def merge_dict_to_dest(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge_dict_to_dest(value, node)
        else:
            destination[key] = value

# For sweep configs, merging dicts
# If there are conflicts, source overwrites dest
def merge_dict(destination, source):
    new_dest = copy.deepcopy(destination)
    merge_dict_to_dest(source, new_dest)
    return new_dest
