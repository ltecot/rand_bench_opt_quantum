import sweep_configs
import pprint
import numpy as np
import torch
import math
import pennylane as qml
from pennylane import numpy as plnp
import util as qo_util

num_qubits = 20
num_random_singles = 50
num_random_doubles = 50
rand_seed_1 = 42
rand_seed_2 = 43

hamiltonian_1 = qo_util.randomized_hamiltonian(num_qubits, num_random_singles, num_random_doubles, rand_seed_1)
hamiltonian_2 = qo_util.randomized_hamiltonian(num_qubits, num_random_singles, num_random_doubles, rand_seed_2)

print(hamiltonian_1.compare(hamiltonian_2))

# rs_model = 42
# interface = "torch"
# params_shape = (3,3)

# b = 32 + 24

# np.random.seed(rs_model)
# if not rs_model:
#     rs_model = np.random.randint(1e8)
# else:
#     rs_model = rs_model

# np.random.seed(rs_model)
# params = np.random.normal(0, np.pi, params_shape)
# if interface == "torch":
#     rg = False
#     params = torch.tensor(params, requires_grad=rg).float()
# elif interface == "numpy":  # WARNING: All our code uses pytorch. Only use numpy for pennylane native optimizers and compatible problems.
#     # plnp.random.seed(rs_model)
#     # params = plnp.random.normal(0, plnp.pi, params_shape)
#     params = plnp.copy(params)
# else:
#     raise Exception("Need to give a valid ML library interface option")

# print(params)

# for i in range(2, 6):
#     print(qo_util.fitness_utilities(i))

# fitness = torch.tensor([3, 4, 2, 0, 1, 5])
# util_inds = torch.argsort(fitness)
# order = []
# for i in range(6):
#     j = util_inds[i]
#     order.append(j)
# print(order)

# num_qubits = 2
# dev = qml.device("default.qubit", wires=num_qubits)
# @qml.qnode(dev)
# def cost(params):
#     qml.RandomLayers(weights=params, wires=num_qubits, seed=42)
#     return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# params = plnp.random.normal(0, plnp.pi, (2, 4))
# opt = qml.QNSPSAOptimizer(stepsize=5e-2)
# for i in range(51):
#     params, loss = opt.step_and_cost(cost, params)
#     if i % 10 == 0:
#         print(f"Step {i}: cost = {loss:.4f}")

# pprint.pprint(sweep_configs.spsa_hs)

# nums = set()
# while len(nums) < 100:
#     nums.add(np.random.randint(1e4))
# print(list(nums))

# print(0.05 * (50 + 1) ** 0.602)

# def utilities(fitness):
#     ordering = torch.argsort(fitness) + 1
#     utilities = math.log((torch.numel(fitness) / 2) + 1) - torch.log(ordering)
#     utilities[utilities < 0] = 0
#     utilities = utilities / torch.sum(utilities)
#     utilities = utilities - (1 / torch.numel(fitness))
#     return utilities

# fitness = torch.tensor([1, 2, 3, 4, 5, 6, 7])
# utilities = utilities(fitness)
# print(utilities)
# print(torch.sum(utilities))

# print((9 + 3 * math.log(30)) / (5 * (30 ** 0.5)))