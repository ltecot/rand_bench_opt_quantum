# Problems for quantum optimization
# All problems must implement a step() function to optimize for the problem
# All problems must implement an eval() function to evaluate performance without optimization.
# Both functions should return a diagnostics dict. (IE with loss)
# All needed objects (optimizer, circuit, args, etc.) are passed in constructor.
# This class needs to do everything (IE modifying circuit to work properly with loss, create Qnode, etc.)

import torch
from torch.utils.data import Dataset
import numpy as np
import pennylane as qml

import loss as qo_loss

class IsingModel():
    pass

class RandomState():
    """Problem where we're going from the default inital state to a random  state."""

    def __init__(self, model_circuit, params, optimizer, args):
        """
        Arguments:
            purity (float): Purity of the target state
        """
        # self.model_circuit = model_circuit
        self.params = params
        self.opt = optimizer
        v = torch.randn(2**num_qubits, dtype=torch.cfloat)
        self.target = v / torch.sqrt(torch.sum(v.conj() * v))

        def full_circuit(params):
            model_circuit(params)
            return qml.state()
        
        dev = qml.device("default.qubit", wires=args.num_qubits)
        self.qnode = qml.QNode(full_circuit, dev, interface=args.interface)

    def _full_loss(self, params):
        """Full end-to-end forward pass, taking params and maybe features as input.
           We pass thing function into the optimizer"""
        pred = self.qnode(params)
        return qo_loss.L2_state_loss(pred, self.target)

    def step(self):
        pred = self.qnode()
        new_params, loss = self.opt.step_and_cost(self._full_loss, self.params)
        self.params = new_params
        # self.opt.zero_grad()
        # loss = self.loss_fn(pred, self.target)
        # loss.backward()
        # self.opt.step()
        return {"loss": loss.item().real,
                "loss_imag": loss.item().imag,}  # Should be zero but just for diagnostics

    def eval(self):
        pred = self.qnode()
        return {"prediction": pred,
                "target": self.target, 
                "loss": self.loss_fn(pred, self.target)}