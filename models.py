# Quantum circuits
# All must be based off the abstract class of torch.nn.Modules
# https://pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html
# All parameters should be of Parameter class and stored to model, as below, so circuit.parameters() works
# https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
# Note that these just implement the circuit. You will need to pass the module and a qml device into a qnode.

import pennylane as qml
import torch
import itertools

class RandomizedCnotLayer(torch.nn.Module):
    """Randomized circuit"""
    def __init__(self, num_qubits, num_layers):
        """
        Arguments:
            num_layers (int): number of layers
            num_qubits (int): number of qubits
        """
        super().__init__()
        # self.a = torch.nn.Parameter(torch.randn(()))
        # TODO: Smarter initalization? Maybe make init a thing that happens in main instead?
        self.params = torch.nn.Parameter(torch.randn(()))
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # TODO: Add random sampling of these pairs
        self.cnot_pairs = itertools.combinations(range(self.num_qubits), 2)
        # self.dev = qml.device("default.qubit", wires=num_qubits)
        # self.qnode = qml.QNode(self.circuit, self.dev)

    # def circuit(self, x):
    #     return x

    def forward(self, x):
        for i in range(self.num_qubits):
            qml.RX(params[i, 0], wires=i)
            qml.RY(params[i, 1], wires=i)
            qml.RZ(params[i, 2], wires=i)
        for pair in self.cnot_pairs:
            qml.CNOT(wires=pair)

# array of Pauli matrices (will be useful later)
# Paulis = Variable(torch.zeros([3, 2, 2], dtype=torch.complex128), requires_grad=False)
# Paulis[0] = torch.tensor([[0, 1], [1, 0]])
# Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])
# Paulis[2] = torch.tensor([[1, 0], [0, -1]])

# number of layers in the circuit
# nr_layers = 2

# a layer of the circuit ansatz
# def layer(params, j):
#     for i in range(nr_qubits):
#         qml.RX(params[i, j, 0], wires=i)
#         qml.RY(params[i, j, 1], wires=i)
#         qml.RZ(params[i, j, 2], wires=i)

#     qml.CNOT(wires=[0, 1])
#     qml.CNOT(wires=[0, 2])
#     qml.CNOT(wires=[1, 2])

# dev = qml.device("default.qubit", wires=3)

# randomly initialize parameters from a normal distribution
params = np.random.normal(0, np.pi, (nr_qubits, nr_layers, 3))
params = Variable(torch.tensor(params), requires_grad=True)

@qml.qnode(dev, interface="torch")
def circuit(params, A):

    # repeatedly apply each layer in the circuit
    for j in range(nr_layers):
        layer(params, j)

    # returns the expectation of the input matrix A on the first qubit
    return qml.expval(qml.Hermitian(A, wires=0))