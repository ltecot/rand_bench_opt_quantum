purity = 0.66

# we generate a three-dimensional random vector by sampling
# each entry from a standard normal distribution
v = np.random.normal(0, 1, 3)

# create a random Bloch vector with the specified purity
bloch_v = Variable(
    torch.tensor(np.sqrt(2 * purity - 1) * v / np.sqrt(np.sum(v ** 2))),
    requires_grad=False
)

# array of Pauli matrices (will be useful later)
Paulis = Variable(torch.zeros([3, 2, 2], dtype=torch.complex128), requires_grad=False)
Paulis[0] = torch.tensor([[0, 1], [1, 0]])
Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])
Paulis[2] = torch.tensor([[1, 0], [0, -1]])

# number of qubits in the circuit
nr_qubits = 3
# number of layers in the circuit
nr_layers = 2