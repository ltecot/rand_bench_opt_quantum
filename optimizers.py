# File for optimizers.
# All optimizers should have step(objective_fn, *args, **kwargs) + step_and_cost(objective_fn, *args, **kwargs) functions similar to that of pennylane QML interface
# Example: https://docs.pennylane.ai/en/stable/code/api/pennylane.SPSAOptimizer.html
# TODO: Make template class instead of these docs

import torch



class ES():
    """Custom Implemented Evolutionary Strategy"""

    def __init__(self):
        """
        Arguments:
            arg (type): description
            TODO
        """
        # TODO

class PytorchSGD():
    """Interface for Pytorch implemented SGD"""

    def __init__(self, params, lr):
        """
        Arguments:
            arg (type): description
            TODO
        """
        self.params = params
        self.opt = torch.optim.SGD([params], lr=lr)

    def step_and_cost(self, objective_fn, *args, **kwargs):
        self.opt.zero_grad()
        loss = objective_fn(*args, **kwargs)
        loss.backward()
        self.opt.step()
        return self.params, loss

    def step(self, objective_fn, *args, **kwargs):
        p, l = self.step_and_cost(objective_fn, *args, **kwargs)
        return p
