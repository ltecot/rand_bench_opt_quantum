# Models / quantum circuits
# All must be based off the abstract class of torch.nn.Modules
# https://pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html

import pennylane as qml
import torch

# TODO: Give more precise name
class RandomizedCircuit(torch.nn.Module):
    """Randomized circuit"""
    def __init__(self, device):
        """
        Arguments:
            num_layers (int): number of layers
            num_qubits (int): number of qubits
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.circuit = None

    def circuit(self, x):
        return x

    def forward(self, x):
        return self.circuit(x)

# array of Pauli matrices (will be useful later)
Paulis = Variable(torch.zeros([3, 2, 2], dtype=torch.complex128), requires_grad=False)
Paulis[0] = torch.tensor([[0, 1], [1, 0]])
Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])
Paulis[2] = torch.tensor([[1, 0], [0, -1]])

# number of layers in the circuit
nr_layers = 2

# a layer of the circuit ansatz
def layer(params, j):
    for i in range(nr_qubits):
        qml.RX(params[i, j, 0], wires=i)
        qml.RY(params[i, j, 1], wires=i)
        qml.RZ(params[i, j, 2], wires=i)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 2])

dev = qml.device("default.qubit", wires=3)

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