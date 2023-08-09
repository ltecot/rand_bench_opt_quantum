# Problems for quantum optimization
# All problems must implement a step() function to optimize for the problem
# All problems must implement an eval() function to evaluate performance without optimization.
# Both functions should return a diagnostics dict. (IE with loss)
# All needed objects (optimizer, circuit, args, etc.) are passed in constructor.
# This class needs to do everything (IE modifying circuit to work properly with loss and data input, create Qnode, etc.)
# TODO: Make template class instead of these docs

import torch
from torch.utils.data import Dataset
import numpy as np
import pennylane as qml

import util as qo_util

class HamiltonianMinimization():
    """Problem to minimize any hamiltonian observable"""

    def __init__(self, model_circuit, params, optimizer, hamiltonian, args):
        """
        Arguments:
            arg (type): description
            TODO
        """
        self.params = params
        self.opt = optimizer

        def full_circuit(params):
            model_circuit(params)
            return qml.expval(hamiltonian)
        
        dev = qml.device("lightning.qubit", wires=args.num_qubits)
        self.qnode = qml.QNode(full_circuit, dev, interface=args.interface)

    def step(self):
        new_params, loss = self.opt.step_and_cost(self.qnode, self.params)
        self.params = new_params
        return {"loss": loss.item()}

    def eval(self):
        energy = self.qnode(self.params)
        return {"energy": energy}


class RandomState():
    """Problem where we're going from the default inital state to a random  state.
       TODO: Pennylane optimizers seem to not like optimizing non-qnode cost functions. Figure out how to fix or don't use their optimizers for now."""

    def __init__(self, model_circuit, params, optimizer, args, target=None):
        """
        Arguments:
            arg (type): description
            TODO
        """
        self.params = params
        self.opt = optimizer
        if not target:
            np.random.seed(args.rand_seed_problem)
            v = torch.randn(2**args.num_qubits, dtype=torch.cfloat)
            target = v / torch.sqrt(torch.sum(v.conj() * v))
        self.target = target
        # TODO: Change this to use projector. Right now it seems broken in Pennylane
        # projector = qml.Projector(target, wires=list(range(args.num_qubits)))

        def full_circuit(params):
            model_circuit(params)
            return qml.state()
            # return qml.expval(qml.Projector(target, wires=list(range(args.num_qubits))))  # Need to make (1 - expval)^2
        
        dev = qml.device("lightning.qubit", wires=args.num_qubits)
        self.qnode = qml.QNode(full_circuit, dev, interface=args.interface)

    def _full_loss(self, params):
        """Full end-to-end forward pass, taking params and maybe features as input.
           We pass thing function into the optimizer"""
        pred = self.qnode(params)
        return qo_util.L2_state_loss(pred, self.target)

    def step(self):
        new_params, loss = self.opt.step_and_cost(self._full_loss, self.params)
        # new_params, loss = self.opt.step_and_cost(self.qnode, self.params)
        self.params = new_params
        return {"loss": loss.item().real,
                "loss_imag": loss.item().imag,}  # Should be zero but just for diagnostics

    def eval(self):
        pred = self.qnode(self.params)
        return {"prediction": pred,
                "target": self.target, 
                "loss": qo_util.L2_state_loss(pred, self.target)}