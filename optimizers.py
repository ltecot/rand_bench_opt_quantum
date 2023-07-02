# File for optimizers.
# All optimizers should have step(objective_fn, *args, **kwargs) + step_and_cost(objective_fn, *args, **kwargs) functions similar to that of pennylane QML interface
# Example: https://docs.pennylane.ai/en/stable/code/api/pennylane.SPSAOptimizer.html
# TODO: Make template class instead of these docs

import torch
import numpy as np

class GES():
    """Custom Implemented Guided Evolutionary Strategy"""

    def __init__(self, param_len, lr, alpha, beta, variance, k, P):
        """
        Arguments:
            arg (type): description
            TODO
        """
        self.n = param_len
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.variance = variance
        self.k = k
        self.P = P
        self.U = torch.zeros(param_len, k)
        self.u_write_ind = 0  # So we can just keep track of the oldest gradient row to replace
        if self.alpha < 0 or self.alpha > 1:
            raise Exception("GES: Need to give valid alpha hyperparameter range.")

    def step_and_cost(self, objective_fn, params, *args, **kwargs):
        loss = objective_fn(params, *args, **kwargs)
        new_params = self.step(objective_fn, params, *args, **kwargs)
        return new_params, loss

    def step(self, objective_fn, params, *args, **kwargs):
        covar = (self.alpha / self.n) * torch.eye(self.n) + ((1 - self.alpha) / self.k) * torch.mm(self.U, torch.t(self.U))
        mvg_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(self.n), covariance_matrix=self.variance * covar)
        grad_est = torch.zeros(self.n)
        for i in range(self.P):
            eps = mvg_dist.sample()
            fp = objective_fn(params + torch.reshape(eps, params.shape), *args, **kwargs)
            fn = objective_fn(params - torch.reshape(eps, params.shape), *args, **kwargs)
            grad_est += eps * (fp - fn)
        grad_est *= self.beta / (2 * self.P)
        self.U[self.u_write_ind] = grad_est
        self.u_write_ind = (self.u_write_ind + 1) % self.k
        return self.params - self.lr * grad_est

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

    def step_and_cost(self, objective_fn, params, *args, **kwargs):
        self.opt.zero_grad()
        loss = objective_fn(*args, **kwargs)
        loss.backward()
        self.opt.step()
        return self.params, loss

    def step(self, objective_fn, params, *args, **kwargs):
        p, l = self.step_and_cost(objective_fn, *args, **kwargs)
        return p
