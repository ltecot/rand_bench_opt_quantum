# File for running quantum optimization experiments
# TODO: Add option to re-run this for multiple wandDB runs.
# TODO: Make file to process multiple wandDB runs into plots (or some saved file maybe)
# TODO: May need further modifications to optimizer calls once we add gradient-free methods

import json
import argparse
import pennylane as qml
import numpy as np
import torch
import wandb

import optimizers as qo_optim
import loss as qo_loss
import datasets as qo_data
import models as qo_models

parser = argparse.ArgumentParser(description='Optimize certified area')
parser.add_argument('--dataset', type=str)  # According to datasets.py
parser.add_argument('--optimizer', type=str)  # According to optimzers.py file
parser.add_argument('--rand_seed', type=int, default=None)  # Pseudo-random if not given
parser.add_argument('--print_interval', type=int, default=50)  # Mostly just to see progress in terminal
parser.add_argument('--num_qubits', type=int, default=5)
# Option-specific Arguments
# Dataset
parser.add_argument('--data_size', type=float) # For random sampling problems. For now we define and generate all data before training.
# Optimizers
parser.add_argument('--epochs', type=int, default=1000) # For single-item datasets, this is the number of iterations
parser.add_argument('--learning_rate', type=float)

args = parser.parse_args()
np.random.seed(args.rand_seed)

# TODO: Add extra options using if-statement with args as they are added
# Data
dataset = qo_data.RandomMixedState()
data_loader = dataset
# Model
model = qo_models.RandomEntanglment()
# Optimizer
opt = torch.optim.SGD([circuit.params], lr=args.learning_rate)
# Loss
# TODO: Some losses may need to modify the quantum circuit itself, so may need to pass that as an argument.
cost_fn = torch.nn.MSELoss()

# optimization begins
config = vars(args)
wandb.init(project="quantum_optimization", config=config)  # Init wandDB logs, make sure you're logged into the right team
# wandb.run.log_code(".")  # Uncomment if you want to save code. Git hash should be saved by default.
for e in range(args.epochs):
    for i, (data, target) in enumerate(data_loader):  # For now just one at a time, maybe add batching later
        pred = model(data)
        opt.zero_grad()
        loss = cost_fn(pred, target)
        loss.backward()
        opt.step()
        wandb_log = {"loss": loss,
                     "epoch": e}
        wandb.log(wandb_log)
        # Keep track of progress every 10 steps
        if (e * len(data_loader) + i) % args.print_interval == 0:
            print(json.dumps(wandb_log, indent=4))
    # TODO: Test eval. Right now we just do compilation but will need to add for future tasks.


# Random legacy code, delete when clearly no longer needed for easy reference

# print("Cost after {} steps is {:.4f}".format(n + 1, loss))

# calculate the Bloch vector of the output state
# output_bloch_v = np.zeros(3)
# for l in range(3):
#     output_bloch_v[l] = circuit(best_params, Paulis[l])

# print results
# print("Target Bloch vector = ", bloch_v.numpy())
# print("Output Bloch vector = ", output_bloch_v)

# if random_per_param_descent == True:
#             if greedy_param_descent:
#                 max_id = torch.argmax(torch.abs(params.flatten()))  # Get max val index
#                 idxs = torch.arange(params.flatten().size(0))
#                 idxs = torch.cat([idxs[0:max_id], idxs[max_id+1:]])
#             else:
#                 idxs = torch.randperm(params.flatten().size(0))  # Get random indicies
#             params.grad.flatten()[idxs[NUM_RANDOM_PARAMS:]] *= 0  # Set all grads after selected number to be zero
        
# if random_per_param_descent == True:
#         wandb.log({"loss": loss, "grad_percent": n * NUM_RANDOM_PARAMS / params.flatten().size(0)})
#     else:
#         wandb.log({"loss": loss, "grad_percent": n})