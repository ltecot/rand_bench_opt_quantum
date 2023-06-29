# File for running quantum optimization experiments
# TODO: Add option to re-run this for multiple wandDB runs.
# TODO: Make file to process multiple wandDB runs into plots (or some saved file maybe)

import json
import argparse
import pennylane as qml
import numpy as np
import torch
import wandb

import optimizers as qo_optim
import util as qo_util
import problems as qo_problems
import circuits as qo_circuits

parser = argparse.ArgumentParser(description='Optimize certified area')
parser.add_argument('--rand_seed', type=int, default=42)  # Global rand seed. Pseudo-random if not given
parser.add_argument('--rand_seed_model', type=int, default=42)  # Seed for specifically model generation
parser.add_argument('--rand_seed_problem', type=int, default=42)  # Seed for specifically problem generation
parser.add_argument('--print_interval', type=int, default=50)  # Mostly just to see progress in terminal
parser.add_argument('--num_qubits', type=int, default=3)  # Number of qubits
parser.add_argument('--interface', type=str, default="torch")  # ML learning library to use
parser.add_argument('--no_wandb', action=argparse.BooleanOptionalAction)  # To turn off wandb for debug
# Problems
parser.add_argument('--problem', type=str, default="transverse_ising")  # Type of problem to run circuit + optimizer on.
parser.add_argument('--loss', type=str)  # Type of loss. Are specific to each problem, see problems.py for each class' options.
# Optimizers
parser.add_argument('--optimizer', type=str, default="spsa")  # Type of optimizer
parser.add_argument('--steps', type=int, default=1000) 
parser.add_argument('--learning_rate', type=float, default=0.001)
# Model Circuit
parser.add_argument('--model', type=str, default="rand_cnot") # Type of circuit "model" to use.
parser.add_argument('--num_layers', type=int, default=2) # For models that have layers, the number of them.
parser.add_argument('--num_params', type=int, default=10) # Number of parameters in used model. If multiple layers, it's number per layer.
parser.add_argument('--ratio_imprim', type=float, default=0.3) # For randomized models, # of 2-qubit gates divided by number of 1-qubit gates.

args = parser.parse_args()
np.random.seed(args.rand_seed)

# ------------------ Model & Params ------------------

# MODEL / CIRCUIT
if args.model == "rand_cnot":
    qmodel = qo_circuits.RandomizedCnotCircuit(args.num_qubits, args.num_layers)
elif args.model == "rand_layers":
    np.random.seed(args.rand_seed_model)
    qmodel = qo_circuits.RandomLayers(args.num_qubits, args.num_layers, 
                                      args.num_params, args.ratio_imprim, 
                                      seed=np.random.randint(1e16))
else:
    raise Exception("Need to give a valid model option")

# INTERFACE / PARAMS
# Some optimizers / problems are incompatible with some interfaces, so be careful when setting this.
if args.interface == "torch":
    params = np.random.normal(0, np.pi, qmodel.params_shape())
    if args.optimizer == "spsa":
        params = torch.tensor(params, requires_grad=False)  # Some optimizers don't work with required grads
    else:
        params = torch.tensor(params, requires_grad=True)
elif args.interface == "numpy":  # WARNING: All our code uses pytorch. Only use numpy for pennylane native optimizers and compatible problems.
    params = np.random.normal(0, np.pi, qmodel.params_shape())
else:
    raise Exception("Need to give a valid ML library interface option")

# ------------------ Optimizer ------------------

if args.optimizer == "sgd":
    opt = qo_optim.PytorchSGD(params, args.learning_rate)
elif args.optimizer == "spsa":
    opt = qml.SPSAOptimizer(maxiter=args.steps)
else:
    raise Exception("Need to give a valid optimizer option")

# ------------------ Problem ------------------

if args.problem == "random_state":
    q_problem = qo_problems.RandomState(qmodel.circuit, params, opt, args)
elif args.problem == "transverse_ising":
    q_problem = qo_problems.HamiltonianMinimization(qmodel.circuit, params, opt, 
                                                    qo_util.transverse_ising_hamiltonian(args.num_qubits), args)
else:
    raise Exception("Need to give a valid problem option")

# ------------------ Training ------------------

print(qml.draw(qmodel.circuit)(params))  # TODO: Probably also save this to wandb too
if not args.no_wandb:
    config = vars(args)
    wandb.init(project="quantum_optimization", config=config)  # Init wandDB logs, make sure you're logged into the right team
# wandb.run.log_code(".")  # Uncomment if you want to save code. Git hash should be saved by default.
for i in range(args.steps):
    log = q_problem.step()
    log["step"] = i
    if not args.no_wandb: wandb.log(log)
    # Keep track of progress every few steps
    if i % args.print_interval == 0:
        print(log)
print(q_problem.eval())