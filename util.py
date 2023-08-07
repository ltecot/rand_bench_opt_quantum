# Custom utility functions, for use in various QML problems

import torch
from torch.autograd import Function
import pennylane as qml
import math
import scipy.linalg
import numpy as np
import copy

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

# Randomized Hamiltonia
# Will sample random pairs of qubits and do the tensor between either a controlled X, Y, or Z (uniformly sampled)
# Also weighs each spin pair interaction 
def randomized_hamiltonian(num_qubits, num_random_singles, num_random_doubles):
    z_coeffs = [-1. for i in range(num_random_singles + num_random_doubles)]
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
def merge_dict(source, destination):
    new_dest = copy.deepcopy(destination)
    merge_dict_to_dest(source, new_dest)
    return new_dest


# # Pytorch matrix sqrt
# # https://github.com/steveli/pytorch-sqrtm
# class MatrixSquareRoot(Function):
#     """Square root of a positive definite matrix.

#     NOTE: matrix square root is not differentiable for matrices with
#           zero eigenvalues.
#     """
#     @staticmethod
#     def forward(ctx, input):
#         m = input.detach().cpu().numpy().astype(np.float_)
#         sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
#         ctx.save_for_backward(sqrtm)
#         return sqrtm

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = None
#         if ctx.needs_input_grad[0]:
#             sqrtm, = ctx.saved_tensors
#             sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
#             gm = grad_output.data.cpu().numpy().astype(np.float_)

#             # Given a positive semi-definite matrix X,
#             # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
#             # matrix square root dX^{1/2} by solving the Sylvester equation:
#             # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
#             grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

#             grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
#         return grad_input

# sqrtm = MatrixSquareRoot.apply