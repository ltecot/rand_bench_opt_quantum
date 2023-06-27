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
import loss as qo_loss
import problems as qo_problems
import circuits as qo_circuits

parser = argparse.ArgumentParser(description='Optimize certified area')
parser.add_argument('--rand_seed', type=int, default=None)  # Pseudo-random if not given
parser.add_argument('--print_interval', type=int, default=50)  # Mostly just to see progress in terminal
parser.add_argument('--num_qubits', type=int, default=3)  # Number of qubits
parser.add_argument('--interface', type=str, default="torch")  # ML learning library to use
parser.add_argument('--no_wandb', action=argparse.BooleanOptionalAction)  # To turn off wandb for debug
# Problems
parser.add_argument('--problem', type=str)  # Type of problem to run circuit + optimizer on.
parser.add_argument('--loss', type=str)  # Type of loss. Are specific to each problem, see problems.py for each class' options.
# Optimizers
parser.add_argument('--optimizer', type=str, default="spsa")  # Type of optimizer
parser.add_argument('--steps', type=int, default=10000) 
parser.add_argument('--learning_rate', type=float, default=0.001)
# Model Circuit
parser.add_argument('--model', type=str, default="rand_layers") # Type of circuit "model" to use.
parser.add_argument('--num_layers', type=int, default=2) # For models that have layers, the number of them.
parser.add_argument('--num_params', type=int, default=10) # Number of parameters in used model. If multiple layers, it's number per layer.
parser.add_argument('--ratio_imprim', type=float, default=0.3) # For randomized models, # of 2-qubit gates divided by number of 1-qubit gates.

args = parser.parse_args()
np.random.seed(args.rand_seed)
# TODO: Maybe change to optional seperate model + optimizer + problem seeds? For better controlled comparisons

# ------------------ Model & Params ------------------
# dev = qml.device("default.qubit", wires=args.num_qubits)

# MODEL / CIRCUIT
if args.model == "rand_cnot":
    qmodel = qo_circuits.RandomizedCnotCircuit(args.num_qubits, args.num_layers)
elif args.model == "rand_layers":
    qmodel = qo_circuits.RandomLayers(args.num_qubits, args.num_layers, 
                                      args.num_params, args.ratio_imprim, 
                                      seed=np.random.randint(1e16))
else:
    raise Exception("Need to give a valid model option")

# INTERFACE / PARAMS
if args.interface == "torch":
    params = np.random.normal(0, np.pi, qmodel.params_shape())
    if args.optimizer == "spsa":
        params = torch.tensor(params, requires_grad=False)  # Some optimizers don't work with required grads
    else:
        params = torch.tensor(params, requires_grad=True)
else:
    raise Exception("Need to give a valid ML library interface option")

# qnode = qml.QNode(qmodel.circuit, dev, interface=args.interface)
# ------------------ Loss ------------------
# TODO: Changing this to live in the problem's class instead.
#       Having flexibility to make it modular is proving problematic.
# loss_fn = qo_loss.L2_state_loss
# ------------------ Optimizer ------------------

if args.optimizer == "sgd":
    opt = torch.optim.SGD([params], lr=args.learning_rate)
elif args.optimizer == "spsa":
    opt = qml.SPSAOptimizer(maxiter=args.steps)
else:
    raise Exception("Need to give a valid optimizer option")

# ------------------ Problem ------------------

q_problem = qo_problems.RandomState(qmodel.circuit, params, opt, args)

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
        # print(json.dumps(log, indent=4))
        print(log)
print(q_problem.eval())