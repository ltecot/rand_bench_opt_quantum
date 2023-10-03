# File for running gradient norm scaling experiments.

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
parser.add_argument('--interface', type=str, default="torch")  # ML learning library to use
parser.add_argument('--device', type=str, default="lightning.qubit")  # Quantum computing device to use
parser.add_argument('--no_wandb', action=argparse.BooleanOptionalAction)  # To turn off wandb for debug
parser.add_argument('--wandb_sweep', action=argparse.BooleanOptionalAction)  # Instead use a wandb sweep config for the run. All options used here must be provided by the config
parser.add_argument('--wandb_config', type=str, default="")  # Sweep config to use. Make sure a config of this name exists in sweep_configs.py
parser.add_argument('--wandb_agent', action=argparse.BooleanOptionalAction)  # Create a new agent on an existing sweep created by this program.
parser.add_argument('--sweep_id', type=str, default="")  # Sweep ID to use for creating a new agent.
# ------------------------------------- Problems -------------------------------------
# parser.add_argument('--problem', type=str, default="randomized_hamiltonian")  # Type of problem to run circuit + optimizer on.
parser.add_argument('--num_qubits', type=int, default=10)  # Number of qubits
parser.add_argument('--num_random_singles', type=int, default=20)  # Number of random single-qubit Paulis in the hamiltonian
parser.add_argument('--num_random_doubles', type=int, default=20)  # Number of random two-qubit tensored Paulis in the hamiltonian
# ------------------------------------- Model Circuit -------------------------------------
# parser.add_argument('--model', type=str, default="rand_layers") # Type of circuit "model" to use.
parser.add_argument('--num_layers', type=int, default=20) # For models that have layers, the number of them.
parser.add_argument('--num_params', type=int, default=2) # Number of parameters in used model, if it is changeable. If multiple layers, it's number per layer.
parser.add_argument('--ratio_imprim', type=float, default=0.5) # For randomized models, # of 2-qubit gates divided by number of 1-qubit gates.
# ------------------------------------- Scaling Params -------------------------------------
parser.add_argument('--scaling_choice', choices=['full_scaling', 'num_qubits', 'num_layers'],
                    default='full_scaling',
                    const='full_scaling',
                    nargs='?',) # Choice of scaling variable. Will overwrite the arg variable value.
parser.add_argument('--scaling_max', type=int, default=50) # Scale scaling choice from 1 to this value.


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
    if not args.rand_seed:
        rand_seed = np.random.randint(1e8)
    else:
        rand_seed = args.rand_seed

    for i in range(2, args.scaling_max + 1):
        if args.scaling_choice == "full_scaling":
            args.num_qubits = i
            args.num_layers = i
            args.num_random_singles = i
            args.num_random_doubles = i
        elif args.scaling_choice == "num_qubits":
            args.num_qubits = i
        elif args.scaling_choice == "num_layers":
            args.num_layers = i
        else:
            raise Exception("Need to give a valid scaling choice option")

        print(f"Running with {args.scaling_choice} = {i}")

        # ------------------ Device ------------------

        dev = qml.device(args.device, wires=args.num_qubits)

        # ------------------ Circuit / Model ------------------

        qmodel = qo_circuits.RandomLayers(args.num_qubits, args.num_layers, 
                                        args.num_params, args.ratio_imprim, 
                                        seed=rand_seed, adjoint_fix=False)

        # ------------------ Interface / Params ------------------
        # Some optimizers / problems are incompatible with some interfaces, so be careful when setting this.

        np.random.seed(rand_seed)
        # params = np.random.normal(0, np.pi, qmodel.params_shape())
        params = np.random.uniform(-2*np.pi, 2*np.pi, qmodel.params_shape())
        params = torch.tensor(params, requires_grad=True)
        params.retain_grad()

        # ------------------ Problem ------------------

        q_ham = qo_util.randomized_hamiltonian(args.num_qubits, args.num_random_singles, args.num_random_doubles, rand_seed)
        q_problem = qo_problems.HamiltonianMinimization(qmodel.circuit, params, None, q_ham, dev, args)

        # ------------------ Training ------------------

        if params.grad:
            params.grad.zero_()
        log = q_problem.eval()
        energy = log['energy']
        energy.backward()
        grad = params.grad
        log["scale"] = i
        log["l0"] = grad.norm(0).item()
        log["l1"] = grad.norm(1).item()
        log["l2"] = grad.norm(2).item()
        log["linf"] = grad.norm(float('inf')).item()
        log["grad_check"] = torch.autograd.gradcheck(q_problem.qnode, params)
        if log_wandb: 
            wandb.log(log)
        print(log)

    if log_wandb:
        wandb.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    if args.wandb_agent:
        wandb.agent(args.sweep_id, function=main)
    elif args.wandb_sweep:
        sweep_id = wandb.sweep(
            sweep=qo_sc.sweep_configs[args.wandb_config], 
            project='quantum_optimization'
        )
        wandb.agent(sweep_id, function=main)
    else:
        main(args)