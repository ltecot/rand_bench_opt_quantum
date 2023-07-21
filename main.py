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
parser.add_argument('--print_interval', type=int, default=1)  # Mostly just to see progress in terminal
parser.add_argument('--num_qubits', type=int, default=3)  # Number of qubits
parser.add_argument('--interface', type=str, default="torch")  # ML learning library to use
parser.add_argument('--no_wandb', action=argparse.BooleanOptionalAction)  # To turn off wandb for debug
# ------------------------------------- Problems -------------------------------------
parser.add_argument('--problem', type=str, default="transverse_ising")  # Type of problem to run circuit + optimizer on.
parser.add_argument('--loss', type=str)  # Type of loss. Are specific to each problem, see problems.py for each class' options.
# ------------------------------------- Optimizers -------------------------------------
parser.add_argument('--optimizer', type=str, default="spsa")  # Type of optimizer
parser.add_argument('--steps', type=int, default=500)  # Steps in the learning problem
parser.add_argument('--learning_rate', type=float, default=1e-1)  # Learning rate. Be careful cause this can be different scales for different optimizers.
parser.add_argument('--est_shots', type=int, default=1)  # Number of rand vectors to use in estimating a gradient / doing an update step
parser.add_argument('--stddev', type=float, default=1e-1)  # Standard deviation of random sampled vectors.
# Guided Evolutonary Strategies
parser.add_argument('--explore_tradeoff', type=float, default=0.5)  # Percent to bias to indentity covariance. Alpha in GES.
parser.add_argument('--grad_scale', type=float, default=1)  # Scale modifier of estimated gradients. Beta in GES.
parser.add_argument('--grad_memory', type=int, default=10)  # Number of vectors to remember for biased sampling. k in GES.
# QNSPSA
parser.add_argument('--metric_reg', type=float, default=0.001)  # Percent of identity to add to 2nd order metric so it's positive definite.
# ------------------------------------- Model Circuit -------------------------------------
parser.add_argument('--model', type=str, default="full_cnot") # Type of circuit "model" to use.
parser.add_argument('--num_layers', type=int, default=2) # For models that have layers, the number of them.
parser.add_argument('--num_params', type=int, default=10) # Number of parameters in used model, if it is changeable. If multiple layers, it's number per layer.
parser.add_argument('--ratio_imprim', type=float, default=0.3) # For randomized models, # of 2-qubit gates divided by number of 1-qubit gates.

args = parser.parse_args()
np.random.seed(args.rand_seed)
if not args.rand_seed_model:
    args.rand_seed_model = np.random.randint(1e16)
if not args.rand_seed_problem:
    args.rand_seed_problem = np.random.randint(1e16)

# ------------------ Model & Params ------------------

# MODEL / CIRCUIT
if args.model == "full_cnot":
    qmodel = qo_circuits.FullCnotCircuit(args.num_qubits, args.num_layers)
elif args.model == "rand_layers":
    np.random.seed(args.rand_seed_model)
    qmodel = qo_circuits.RandomLayers(args.num_qubits, args.num_layers, 
                                      args.num_params, args.ratio_imprim, 
                                      seed=args.rand_seed_model)
else:
    raise Exception("Need to give a valid model option")

# INTERFACE / PARAMS
# Some optimizers / problems are incompatible with some interfaces, so be careful when setting this.
if args.interface == "torch":
    params = np.random.normal(0, np.pi, qmodel.params_shape())
    params = torch.tensor(params, requires_grad=True)
elif args.interface == "numpy":  # WARNING: All our code uses pytorch. Only use numpy for pennylane native optimizers and compatible problems.
    params = np.random.normal(0, np.pi, qmodel.params_shape())
else:
    raise Exception("Need to give a valid ML library interface option")

# ------------------ Optimizer ------------------

if args.optimizer == "sgd":
    opt = qo_optim.PytorchSGD(params, args.learning_rate)
elif args.optimizer == "ges":
    opt = qo_optim.GES(torch.numel(params), args.learning_rate, args.explore_tradeoff, 
                       args.grad_scale, args.stddev ** 2, args.grad_memory, args.est_shots)
elif args.optimizer == "xnes":
    opt = qo_optim.xNES(param_len=torch.numel(params), stddev_init=args.stddev, num_shots=args.est_shots, nu_sigma=None, nu_b=None, nu_mu=1)
elif args.optimizer == "spsa":
    opt = qo_optim.SPSA(param_len=torch.numel(params), maxiter=args.steps, num_shots=args.est_shots, 
                        alpha=0.602, c=0.2, gamma=0.101, A=None, a=None)
elif args.optimizer == "pl_spsa":
    # maxiter=None, alpha=0.602, gamma=0.101, c=0.2, A=None, a=None
    opt = qml.SPSAOptimizer(maxiter=args.steps, alpha=0.602, gamma=0.101, c=0.2, A=None, a=None)
elif args.optimizer == "pl_qnspsa":
    # stepsize=0.001, regularization=0.001, finite_diff_step=0.01, resamplings=1, blocking=True, history_length=5, seed=None
    opt = qml.QNSPSAOptimizer(stepsize=args.learning_rate, regularization=args.metric_reg, finite_diff_step=args.stddev, 
                              resamplings=args.est_shots, blocking=True, history_length=5, seed=None)
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
    log["num_shots"] = i * args.est_shots  # For optimizers that use shots
    if not args.no_wandb: wandb.log(log)
    # Keep track of progress every few steps
    if i % args.print_interval == 0:
        print(log)
print(q_problem.eval())