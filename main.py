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
import problems as qo_problems
import circuits as qo_circuits

parser = argparse.ArgumentParser(description='Optimize certified area')
parser.add_argument('--dataset', type=str)  # According to datasets.py
parser.add_argument('--optimizer', type=str)  # According to optimzers.py file
parser.add_argument('--rand_seed', type=int, default=None)  # Pseudo-random if not given
parser.add_argument('--print_interval', type=int, default=10)  # Mostly just to see progress in terminal
parser.add_argument('--num_qubits', type=int, default=3)
# Option-specific Arguments
# Dataset
parser.add_argument('--data_size', type=float) # For random sampling problems. For now we define and generate all data before training.
# Optimizers
# TODO: Account for problems that have datasets rather than specific problems.
parser.add_argument('--steps', type=int, default=1000) 
parser.add_argument('--learning_rate', type=float, default=0.01)

args = parser.parse_args()
np.random.seed(args.rand_seed)

# TODO: Add extra options using if-statement with args as they are added
# Model
dev = qml.device("default.qubit", wires=args.num_qubits)
qmodel = qo_circuits.RandomizedCnotCircuit(args.num_qubits, 1)
qnode = qml.QNode(qmodel.circuit, dev, interface="torch")
# Loss
loss_fn = qo_loss.L2_state_loss
# Optimizer
opt = torch.optim.SGD([qmodel.params], lr=args.learning_rate)
# Problem
q_problem = qo_problems.RandomState(qnode, loss_fn, opt, args.num_qubits)

# optimization begins
print(qml.draw(qmodel.circuit)())  # TODO: Probably also save this to wandb too
config = vars(args)
# wandb.init(project="quantum_optimization", config=config)  # Init wandDB logs, make sure you're logged into the right team
# wandb.run.log_code(".")  # Uncomment if you want to save code. Git hash should be saved by default.
for i in range(args.steps):
    log = q_problem.step()
    log["step"] = i
    # wandb.log(log)
    # Keep track of progress every 10 steps
    if i % args.print_interval == 0:
        # print(json.dumps(log, indent=4))
        print(log)
# TODO: Test eval. Right now we just do compilation but will need to add for future tasks.