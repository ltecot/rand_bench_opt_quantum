import pennylane as qml
from pennylane import numpy as np

# qml.about()

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

coeffs = [0.2, -0.543, 0.4514]
obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2),
            qml.PauliX(3) @ qml.PauliZ(1)]
H = qml.Hamiltonian(coeffs, obs)
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)
@qml.qnode(dev)
def cost(params, num_qubits=1):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=range(num_qubits))
    for i in range(num_qubits):
        qml.Rot(*params[i], wires=0)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[2, 0])
        qml.CNOT(wires=[3, 1])
    return qml.expval(H)
params = np.random.normal(0, np.pi, (num_qubits, 3), requires_grad=True)

max_iterations = 100
opt = qml.SPSAOptimizer(maxiter=max_iterations)
for _ in range(max_iterations):
    params, energy = opt.step_and_cost(cost, params, num_qubits=num_qubits)
print(energy)