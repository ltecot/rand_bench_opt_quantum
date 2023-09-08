import util as qo_util
print(qo_util.random_dist(5, 41))

# import pennylane as qml
# from copy import copy
# import circuits as qo_circuits
# import numpy as np
# import torch

# dev = qml.device("lightning.qubit", wires=5)

# qmodel = qo_circuits.RandomLayers(5, 5, 5, 0.3, seed=42, adjoint_fix=False)

# def dummy_circuit(params):
#     qmodel.circuit(params)
#     return qml.expval(qml.PauliZ(0))
# qnode = qml.QNode(dummy_circuit, dev, interface="torch")

# def get_state_overlap(params1, params2):
#     def get_operations(qnode, params):
#         qnode.construct([params], {})
#         return qnode.tape.operations
#     def get_overlap_tape(qnode, params1, params2):
#         op_forward = get_operations(qnode, params1)
#         op_inv = get_operations(qnode, params2)
#         with qml.tape.QuantumTape() as tape:
#             for op in op_forward:
#                 qml.apply(op)
#             for op in reversed(op_inv):
#                 qml.adjoint(copy(op))
#             qml.probs(wires=qnode.tape.wires.labels)
#         return tape
#     tape = get_overlap_tape(qnode, params1, params2)
#     return qml.execute([tape], dev, None)[0][0]

# params_1 = np.random.normal(0, np.pi, qmodel.params_shape())
# params_1 = torch.tensor(params_1, requires_grad=False).float()
# params_2 = np.random.normal(0, np.pi, qmodel.params_shape())
# params_2 = torch.tensor(params_2, requires_grad=False).float()
# print("Perfect overlap: ", get_state_overlap(params_1, params_1))
# print("Random state overlap: ", get_state_overlap(params_1, params_2))

# import util as qo_util
# dist = qo_util.cardinality_dist(3, target_num=1)
# print(dist)
# print(qo_util.nll_loss(dist, dist))

# import sweep_configs
# import pprint
# import numpy as np
# import torch
# import math
# import pennylane as qml
# from pennylane import numpy as np
# import util as qo_util

# print([bin(i) for i in range(2 ** 3)])
# print([bin(i).count("1") for i in range(2 ** 3)])

# import pennylane as qml
# from pennylane import numpy as np

# num_qubits = 2
# dev = qml.device("default.qubit", wires=num_qubits)
# @qml.qnode(dev)
# def cost(params):
#     qml.RX(params[0], wires=0)
#     qml.CRY(params[1], wires=[0, 1])
#     return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# params = np.random.rand(2)
# opt = qml.QNSPSAOptimizer(stepsize=5e-2)
# for i in range(51):
#     params, loss = opt.step_and_cost(cost, params)
#     if i % 10 == 0:
#         print(f"Step {i}: cost = {loss:.4f}")

# num_qubits = 20
# num_random_singles = 50
# num_random_doubles = 50
# rand_seed_1 = 42
# rand_seed_2 = 43

# hamiltonian_1 = qo_util.randomized_hamiltonian(num_qubits, num_random_singles, num_random_doubles, rand_seed_1)
# hamiltonian_2 = qo_util.randomized_hamiltonian(num_qubits, num_random_singles, num_random_doubles, rand_seed_2)

# print(hamiltonian_1.compare(hamiltonian_2))

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

# params = np.random.normal(0, np.pi, (2, 4))
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