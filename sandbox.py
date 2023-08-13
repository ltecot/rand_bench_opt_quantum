# import sweep_configs
# import pprint
# import numpy as np
# import torch
# import math
# import pennylane as qml
# from pennylane import numpy as np
# import util as qo_util

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ
from qiskit.providers.ibmq.runtime import UserMessenger, ProgramBackend
import scipy.optimize as opt
from qiskit.algorithms.optimizers import NELDER_MEAD, ADAM, GradientDescent, L_BFGS_B
from math import pi
import numpy as np
import pandas as pd 
import typing
import itertools
import math 

from qiskit.circuit.library import *

class QCBM: 
    '''Class for generating an arbitrary parametrized circuit for a QCBM. 
    Args:
        n_qubits (int): width of the circuit 
        n_layers (int): depth of the circuit 
        topology (list[list]): list containing 4 inner lists. [Single qubit gate names: 'h', 'rx', 'ry', 'rz'], 
        [Single qubit gate index], [Entangling qubit gate names: 'crx', 'xx'], 
        [Entangled qubit gate index (control, output)] 
    '''
    def __init__(self, n_qubits: int, n_layers: int, topology: typing.List[list]):
        self.n_qubits = n_qubits
        self.n_layers = n_layers 
        self.single_gates = topology[0]
        self.single_qubit_idx = topology[1]
        self.entangled_gates = topology[2]
        self.entangled_qubit_idx = topology[3]
        self.extra_gates = topology[4]
        self.extra_qubit_idx = topology[5]
        self.total_param_number = (len(self.single_gates) * n_layers) + (len(self.entangled_gates) * n_layers) + len(self.extra_gates)

    def CreateCircuit(self, theta_values: list):
        '''Returns the circuit ansatz for the QCBM.
        Args: 
            theta_values (list): list of parameter values for each parametrized gate in the circuit.
        Returns: 
            circ: the QCBM circuit object. '''
        q = QuantumRegister(self.n_qubits, 'q')
        c = ClassicalRegister(self.n_qubits, 'c')
        circ = QuantumCircuit(q, c) 

        for layer in range(self.n_layers): 

            if self.n_layers >= 2: 
                if layer == (self.n_layers - 1): 
                    end_idx= len(theta_values) - ((self.n_qubits * 3) + len(self.entangled_gates))
                    theta_split = theta_values[end_idx:]
                else: 
                    initial = int(((len(theta_values) - self.n_qubits) /self.n_layers) * layer)
                    output = int(((len(theta_values) - self.n_qubits)/self.n_layers) * (layer + 1))
                    theta_split = theta_values[initial : output]
            else: 
                initial = int((len(theta_values)/self.n_layers) * layer)
                output = int((len(theta_values)/self.n_layers) * (layer + 1))
                theta_split = theta_values[initial : output]

            if self.n_layers >= 2 and layer == (self.n_layers - 1):
                for gate,idx,theta in zip(self.extra_gates, self.extra_qubit_idx, theta_split[: len(self.extra_gates)]):
                    if gate == "h": 
                        circ.h(q[idx])
                    if gate == "ry": 
                        circ.ry(theta, q[idx])
                    if gate == "rx": 
                        circ.rx(theta, q[idx])
                    if gate == "rz": 
                        circ.rz(theta, q[idx])
                    if gate == "none": 
                        continue 
                for gate, idx, theta in zip(self.entangled_gates, self.entangled_qubit_idx, theta_split[len(self.extra_gates): ]): 
                    if gate == "cry": 
                        circ.cry(theta, q[idx[0]], q[idx[1]])
                    if gate == "xx": 
                        circ.append(RXXGate(theta), [q[idx[0]], q[idx[1]]] )
                    if gate == "none": 
                        continue 
            else: 
                for gate,idx,theta in zip(self.single_gates, self.single_qubit_idx, theta_split[: len(self.single_gates)]):
                    if gate == "h": 
                        circ.h(q[idx])
                    if gate == "ry": 
                        circ.ry(theta, q[idx])
                    if gate == "rx": 
                        circ.rx(theta, q[idx])
                    if gate == "rz": 
                        circ.rz(theta, q[idx])
                    if gate == "none": 
                        continue 

                for gate, idx, theta in zip(self.entangled_gates, self.entangled_qubit_idx, theta_split[len(self.single_gates): ]): 
                    if gate == "cry": 
                        circ.cry(theta, q[idx[0]], q[idx[1]])
                    if gate == "xx": 
                        circ.append(RXXGate(theta), [q[idx[0]], q[idx[1]]] )
                    if gate == "none": 
                        continue 
        
        for idx in range(self.n_qubits): 
            circ.measure(q[idx], c[idx])     
        return circ


def CountstoProb(samples: typing.Dict):
    total_shots = np.sum(list(samples.values()))
    for bitstring, count in samples.items(): 
        samples[bitstring] = count / total_shots 
    return samples 

def dec2bin(number: int, length: int) -> typing.List[int]:
    bit_str = bin(number)
    bit_str = bit_str[2 : len(bit_str)]  
    bit_string = [int(x) for x in list(bit_str)]
    if len(bit_string) < length:
        len_zeros = length - len(bit_string)
        bit_string = [int(x) for x in list(np.zeros(len_zeros))] + bit_string
    return bit_string

def KL(training_distribution: typing.Dict, model_distribution: typing.Dict, n_qubits: int): 
    distributions = dists_check(training_distribution, model_distribution, n_qubits)
    training_distribution = distributions[0]
    model_distribution = distributions[1]
    return np.dot(np.asarray(list(training_distribution.values())), np.log(np.asarray(list(training_distribution.values()))/np.asarray(list(model_distribution.values()))))

def dists_check(distribution1: typing.Dict, distribution2: typing.Dict, n_qubits: int):             
    new_dist1 = {}
    new_dist2 = {}
      
    for state in range(2**n_qubits): 
        bitstring = ''
        bit_list = dec2bin(state, n_qubits)
        for bit in bit_list: 
            bitstring = bitstring + str(bit)
        bitstring = bitstring[len(bitstring) :: -1]
        
        if bitstring in distribution1.keys(): 
            if distribution1[bitstring] == 0: 
                new_dist1[bitstring] = float(1e-16)
            else: 
                new_dist1[bitstring] = distribution1[bitstring]
            
        else: 
            new_dist1[bitstring] = float(1e-16)
            
        if bitstring in distribution2.keys(): 
            if distribution2[bitstring] == 0: 
                new_dist2[bitstring] = float(1e-16)
            else: 
                new_dist2[bitstring] = distribution2[bitstring]
        else: 
            new_dist2[bitstring] = float(1e-16)

    dist1_keys, dist1_values = zip(*sorted(zip(new_dist1.keys(), new_dist1.values())))
    dist2_keys, dist2_values = zip(*sorted(zip(new_dist2.keys(), new_dist2.values())))   
    
    return dict(zip(list(dist1_keys),list(dist1_values))), dict(zip(list(dist2_keys), list(dist2_values)))

def counts_to_queries(counts: dict) -> list:
    queries = []

    for i, j in counts.items(): 
        for num in range(int(j * 10000)):
            queries.append(i)

    return queries

def bin2dec(x: typing.List[int]) -> int:
    dec = 0
    coeff = 1
    for i in range(len(x)):
        dec = dec + coeff * x[len(x) - 1 - i]
        coeff = coeff * 2
    return dec

def comb(n: int, k: int)-> int:
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def main(backend: ProgramBackend, user_messenger: UserMessenger, args_list: typing.Dict): 
    n_qubits = args_list.get('n_qubits')
    n_layers = args_list.get('n_layers')
    topology = args_list.get('topology')
    initial_thetas = args_list.get('initial_thetas')
    n_shots = args_list.get('n_shots')
    training_seed = args_list.get('training_seed')
    training_distribution_specs = args_list.get('training_distribution_specs')
    optimizer_specs = args_list.get('optimizer_specs')
    cost_function = args_list.get("cost_function")


    qcbm = QCBM(n_qubits, n_layers, topology)
    training_distribution = training_distribution_specs["training_distribution"]
    
    
    theta_list = []
    cost_list = []
    iteration_list = []
    
    def runqcbm(parameters):
        theta_list.append(parameters)
        circuit = qcbm.CreateCircuit(parameters)
        job = execute(circuit, backend, shots = n_shots, seed_transpiler = training_seed, seed_simulator = training_seed)
        counts = job.result().get_counts()
        model_distribution = CountstoProb(counts)
        cost_value = cost_function(training_distribution, model_distribution.copy(), n_qubits)
        cost_list.append(cost_value)
        iteration_list.append(1)
        if len(iteration_list) % len(parameters) == 0: 
            print(cost_value)
        return cost_value     

    np.random.seed(training_seed)
    optimizer_name = optimizer_specs["name"]
    optimizer = optimizer_name(maxiter=optimizer_specs["maxiter"], eps=optimizer_specs["eps"], lr = optimizer_specs["lr"] )

    result = optimizer.minimize(fun = runqcbm, x0 = initial_thetas) 

    return theta_list, cost_list,



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