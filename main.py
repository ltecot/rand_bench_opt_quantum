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
# TODO: Account for problems that have datasets rather than specific problems.
parser.add_argument('--steps', type=int, default=1000) 
parser.add_argument('--learning_rate', type=float)

args = parser.parse_args()
np.random.seed(args.rand_seed)

# TODO: Add extra options using if-statement with args as they are added
# Model
dev = qml.device("default.qubit", wires=3)
circuit = qo_models.RandomEntanglment()
model = qml.QNode(circuit, dev)
# Loss
cost_fn = torch.nn.MSELoss()
# Optimizer
opt = torch.optim.SGD([circuit.params], lr=args.learning_rate)
# Problem
dataset = qo_data.RandomMixedState()

# optimization begins
config = vars(args)
wandb.init(project="quantum_optimization", config=config)  # Init wandDB logs, make sure you're logged into the right team
# wandb.run.log_code(".")  # Uncomment if you want to save code. Git hash should be saved by default.
for i in range(args.steps):
    pred = model(data)
    opt.zero_grad()
    loss = cost_fn(pred, target)
    loss.backward()
    opt.step()
    wandb_log = {"loss": loss,
                 "step": i}
    wandb.log(wandb_log)
    # Keep track of progress every 10 steps
    if (e * len(data_loader) + i) % args.print_interval == 0:
        print(json.dumps(wandb_log, indent=4))
# TODO: Test eval. Right now we just do compilation but will need to add for future tasks.