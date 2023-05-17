import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
import wandb

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

parser = argparse.ArgumentParser(description='Optimize certified area')
parser.add_argument('--dataset', type=str)  # According to datasets.py
parser.add_argument('--optimizer', type=str)  # According to optimzers.py file
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--rand_seed', type=int, default=None)  # Pseudo-random if not given
# Option-specific Arguments
# Optimizers
parser.add_argument('--learning_rate', type=float)

args = parser.parse_args()
np.random.seed(args.rand_seed)
config = vars(args)

# Data

# Model

# Optimizer
opt = torch.optim.SGD([params], lr=learning_rate)
# Loss


# the final stage of optimization isn't always the best, so we keep track of
# the best parameters along the way
best_cost = cost_fn(params)
best_params = np.zeros((nr_qubits, nr_layers, 3))

print("Cost after 0 steps is {:.4f}".format(cost_fn(params)))

# optimization begins
wandb.init(project="quantum_optimization", config=config)
for n in range(steps):
    opt.zero_grad()
    loss = cost_fn(params)
    loss.backward()
    # print(params.grad.flatten())
    if random_per_param_descent == True:
        if greedy_param_descent:
            max_id = torch.argmax(torch.abs(params.flatten()))  # Get max val index
            idxs = torch.arange(params.flatten().size(0))
            idxs = torch.cat([idxs[0:max_id], idxs[max_id+1:]])
        else:
            idxs = torch.randperm(params.flatten().size(0))  # Get random indicies
        params.grad.flatten()[idxs[NUM_RANDOM_PARAMS:]] *= 0  # Set all grads after selected number to be zero
    opt.step()

    # keeps track of best parameters
    if loss < best_cost:
        best_cost = loss
        best_params = params

    if random_per_param_descent == True:
        wandb.log({"loss": loss, "grad_percent": n * NUM_RANDOM_PARAMS / params.flatten().size(0)})
    else:
        wandb.log({"loss": loss, "grad_percent": n})

    # Keep track of progress every 10 steps
    if n % 10 == 9 or n == steps - 1:
        print("Cost after {} steps is {:.4f}".format(n + 1, loss))

# calculate the Bloch vector of the output state
output_bloch_v = np.zeros(3)
for l in range(3):
    output_bloch_v[l] = circuit(best_params, Paulis[l])

# print results
print("Target Bloch vector = ", bloch_v.numpy())
print("Output Bloch vector = ", output_bloch_v)
