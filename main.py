# File for running quantum optimization experiments
# TODO: Add option to re-run this for multiple wandDB runs.
# TODO: Make file to process multiple wandDB runs into plots (or some saved file maybe)
# TODO: For now pennylane optimizers seem not to work

import json
import argparse
import ast
import pennylane as qml
import numpy as np
from pennylane import numpy as plnp
import torch
import wandb

import optimizers as qo_optim
import util as qo_util
import problems as qo_problems
import circuits as qo_circuits
import sweep_configs as qo_sc

parser = argparse.ArgumentParser()
parser.add_argument('--rand_seed', type=int, default=42)  # Global rand seed. Pseudo-random if not given
parser.add_argument('--rand_seed_model', type=int, default=42)  # Seed for specifically model generation
# parser.add_argument('--rand_seed_problem', type=int, default=42)  # Seed for specifically problem generation
parser.add_argument('--print_interval', type=int, default=1)  # Mostly just to see progress in terminal
parser.add_argument('--num_qubits', type=int, default=2)  # Number of qubits
parser.add_argument('--interface', type=str, default="torch")  # ML learning library to use
parser.add_argument('--no_wandb', action=argparse.BooleanOptionalAction)  # To turn off wandb for debug
parser.add_argument('--wandb_sweep', action=argparse.BooleanOptionalAction)  # Instead use a wandb sweep config for the run. All options used here must be provided by the config
parser.add_argument('--wandb_config', type=str, default="")  # Sweep config to use. Make sure a config of this name exists in sweep_configs.py
# ------------------------------------- Problems -------------------------------------
parser.add_argument('--problem', type=str, default="transverse_ising")  # Type of problem to run circuit + optimizer on.
# ------------------------------------- Model Circuit -------------------------------------
parser.add_argument('--model', type=str, default="rand_layers") # Type of circuit "model" to use.
parser.add_argument('--num_layers', type=int, default=3) # For models that have layers, the number of them.
parser.add_argument('--num_params', type=int, default=10) # Number of parameters in used model, if it is changeable. If multiple layers, it's number per layer.
parser.add_argument('--ratio_imprim', type=float, default=0.3) # For randomized models, # of 2-qubit gates divided by number of 1-qubit gates.
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
# NES
parser.add_argument('--nu_mu', type=float, default=1)  # Exponential control constant of NES for mu (mean / parameters)
parser.add_argument('--nu_sigma', type=float, default=None)  # Exponential control constant of NES for sigma (variance for xNES, sigma vector for sNES)
parser.add_argument('--nu_b', type=float, default=None)  # Exponential control constant of xNES for B (normalized covariance)
# SPSA
parser.add_argument('--alpha', type=float, default=1)  # Exponential decay exponent of the learning rate for SPSA
parser.add_argument('--gamma', type=float, default=1)  # Exponential decay exponent of the step size for SPSA
# ADAM SPSA
parser.add_argument('--beta', type=float, default=0.999)  # Multiplicative decay / weighting of the first-moment / momentum
parser.add_argument('--lmd', type=float, default=0.9)  # Exponential decay exponent of the momentum weighting for methods using decaying momentum.
parser.add_argument('--zeta', type=float, default=0.999)  # Multiplicative decay / weighting of the second-moment / curvature correction
# QNSPSA
parser.add_argument('--metric_reg', type=float, default=0.001)  # Percent of identity to add to 2nd order metric so it's positive definite.

def main(args=None):
    
    if not args:
        wandb.init(project="quantum_optimization")
        args = wandb.config
        log_wandb = True
    elif not args.no_wandb:
        config = vars(args)
        wandb.init(project="quantum_optimization", config=config)  # Init wandDB logs, make sure you're logged into the right team
        log_wandb = True
    else:
        log_wandb = False
    
    np.random.seed(args.rand_seed)
    if not args.rand_seed_model:
        rs_model = np.random.randint(1e8)
    else:
        rs_model = args.rand_seed_model
    # if not args.rand_seed_problem:
    #     args.rand_seed_problem = np.random.randint(1e8)

    # ------------------ Model & Params ------------------

    # MODEL / CIRCUIT
    if args.model == "full_cnot":
        qmodel = qo_circuits.FullCnotCircuit(args.num_qubits, args.num_layers)
    elif args.model == "rand_layers":
        adjoint_fix = (args.optimizer == "pl_qnspsa")
        qmodel = qo_circuits.RandomLayers(args.num_qubits, args.num_layers, 
                                        args.num_params, args.ratio_imprim, 
                                        seed=rs_model, adjoint_fix=adjoint_fix)
    else:
        raise Exception("Need to give a valid model option")

    # INTERFACE / PARAMS
    # Some optimizers / problems are incompatible with some interfaces, so be careful when setting this.
    if args.interface == "torch":
        params = np.random.normal(0, np.pi, qmodel.params_shape())
        rg = True if args.optimizer == "sgd" else False
        params = torch.tensor(params, requires_grad=rg).float()
    elif args.interface == "numpy":  # WARNING: All our code uses pytorch. Only use numpy for pennylane native optimizers and compatible problems.
        params = plnp.random.normal(0, plnp.pi, qmodel.params_shape())
    else:
        raise Exception("Need to give a valid ML library interface option")

    # ------------------ Optimizer ------------------
    # Default example run copies are included
    # Make sure to add shot_num for accurate tracking of number of quantum circuit evals

    if args.optimizer == "sgd":
        # python main.py --optimizer=sgd --learning_rate=0.1 --no_wandb
        opt = qo_optim.PytorchSGD(params, args.learning_rate)
        shot_num = 1
    elif args.optimizer == "ges":
        # python main.py --optimizer=ges --learning_rate=0.1 --explore_tradeoff=0.5 --grad_scale=2 --stddev=0.1 --grad_memory=10 --est_shots=1 --no_wandb
        opt = qo_optim.GES(torch.numel(params), args.learning_rate, args.explore_tradeoff, 
                        args.grad_scale, args.stddev ** 2, args.grad_memory, args.est_shots)
        shot_num = 2 * args.est_shots
    elif args.optimizer == "xnes":
        # python main.py --optimizer=xnes --stddev=0.1 --est_shots=2 --nu_sigma=0.01 --nu_b=0.001 --nu_mu=0.1 --no_wandb
        opt = qo_optim.xNES(param_len=torch.numel(params), stddev_init=args.stddev, num_shots=args.est_shots, 
                            nu_sigma=args.nu_sigma, nu_b=args.nu_b, nu_mu=args.nu_mu)
        shot_num = args.est_shots
    elif args.optimizer == "snes":
        # python main.py --optimizer=snes --stddev=0.1 --est_shots=2 --nu_sigma=0.01 --nu_mu=0.1 --no_wandb
        opt = qo_optim.sNES(param_len=torch.numel(params), stddev_init=args.stddev, num_shots=args.est_shots, 
                            nu_sigma=args.nu_sigma, nu_mu=args.nu_mu)
        shot_num = args.est_shots
    elif args.optimizer == "spsa":
        # python main.py --optimizer=spsa --stddev=0.2 --est_shots=1 --alpha=0.602 --gamma=0.101 --no_wandb
        opt = qo_optim.SPSA(param_len=torch.numel(params), maxiter=args.steps, num_shots=args.est_shots, 
                            alpha=args.alpha, c=args.stddev, gamma=args.gamma, A=None, a=None)
        shot_num = 2 * args.est_shots
    elif args.optimizer == "adamspsa":
        # python main.py --optimizer=adamspsa --stddev=0.2 --est_shots=1 --alpha=0.602 --gamma=0.101 --no_wandb
        opt = qo_optim.SPSA(param_len=torch.numel(params), maxiter=args.steps, num_shots=args.est_shots, 
                            alpha=args.alpha, c=args.stddev, gamma=args.gamma, a=args.learning_rate,
                            beta=args.beta, lmd=args.lmd, zeta=args.zeta)
        shot_num = 2 * args.est_shots
    elif args.optimizer == "2spsa":
        # python main.py --optimizer=2spsa --learning_rate=0.01 --metric_reg=0.001 --stddev=0.01 --est_shots=1 --no_wandb
        opt = qo_optim.SPSA_2(param_len=torch.numel(params), stepsize=args.learning_rate, regularization=args.metric_reg, 
                              finite_diff_step=args.stddev, num_shots=args.est_shots, blocking=True, history_length=5)
        shot_num = 5 * args.est_shots  # 2 for grad, 2 extra for metric, 1 for blocking
    elif args.optimizer == "pl_spsa":
        # python main.py --optimizer=pl_spsa --stddev=0.2 --est_shots=1 --alpha=0.602 --gamma=0.101 --interface=numpy --no_wandb
        opt = qml.SPSAOptimizer(maxiter=args.steps, alpha=args.alpha, gamma=args.gamma, c=args.stddev, A=None, a=None)
        shot_num = 2
    elif args.optimizer == "pl_qnspsa":
        # python main.py --optimizer=pl_qnspsa --learning_rate=0.01 --metric_reg=0.001 --stddev=0.01 --est_shots=1 --interface=numpy --no_wandb
        opt = qml.QNSPSAOptimizer(stepsize=args.learning_rate, regularization=args.metric_reg, finite_diff_step=args.stddev, 
                                resamplings=args.est_shots, blocking=True, history_length=5, seed=None)
        shot_num = 7 * args.est_shots  # 2 for grad, 4 for metric, 1 for blocking
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
    # wandb.run.log_code(".")  # Uncomment if you want to save code. Git hash should be saved by default.
    for i in range(args.steps):
        log = q_problem.step()
        log["step"] = i
        log["num_shots"] = i * shot_num  # For optimizers that use shots
        if log_wandb: 
            wandb.log(log)
        # Keep track of progress every few steps
        if i % args.print_interval == 0:
            print(log)
    print(q_problem.eval())
    if log_wandb:
        wandb.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.wandb_sweep:
        main(args)
    else:
        config_str = "qo_sc." + args.wandb_config
        sweep_id = wandb.sweep(
            sweep=eval(config_str), 
            project='quantum_optimization'
        )
        wandb.agent(sweep_id, function=main)