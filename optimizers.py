# File for optimizers.
# All optimizers should have step(objective_fn, *args, **kwargs) + step_and_cost(objective_fn, *args, **kwargs) functions similar to that of pennylane QML interface
# Example: https://docs.pennylane.ai/en/stable/code/api/pennylane.SPSAOptimizer.html
# TODO: Make template class instead of these docs

import torch
import numpy as np

class SPSA():
    """SPSA"""

    def __init__(self, param_len, num_shots=1, alpha=0.602, c=0.2, gamma=0.101, maxiter=None, A=None, a=None): 
        """
        Arguments:
            arg (type): description
            TODO
        """
        if not maxiter and not A:
            raise Exception("SPSA: Need to give valid maxiter or A")
        self.param_len = param_len
        self.num_shots = num_shots
        self.c = c
        self.gamma = gamma
        self.alpha = alpha
        self.A = A  
        self.a = a  
        if not A:
            self.A = maxiter * 0.1
        if not a:
            self.a = 0.05 * (self.A + 1) ** alpha
        self.k = 0  # Step count

    def step(self, objective_fn, params, *args, **kwargs):
        self.k += 1
        ck = self.c / (self.k ** self.gamma)
        grad_est = torch.zeros(self.param_len)
        for i in range(self.num_shots):
            eps = 2 * torch.bernoulli(0.5 * torch.ones(self.param_len)) - 1  # Equal prob -1 or 1 per element
            fp = objective_fn(params + torch.reshape(eps, params.shape), *args, **kwargs)
            fn = objective_fn(params - torch.reshape(eps, params.shape), *args, **kwargs)
            grad_est += (fp - fn) * (eps ** -1)
        grad_est *= 1 / (2 * ck * self.num_shots)
        ak = self.a / ((self.A + self.k) ** self.alpha)
        return params - ak * torch.reshape(grad_est, params.shape)

    def step_and_cost(self, objective_fn, params, *args, **kwargs):
        loss = objective_fn(params, *args, **kwargs)
        new_params = self.step(objective_fn, params, *args, **kwargs)
        return new_params, loss

class xNES():
    """Exponential natural evolution strategies"""

    def __init__(self, param_len, stddev, num_shots):
        """
        Arguments:
            arg (type): description
            TODO
        """
        if num_shots < 2:
            raise Exception("xNES: Need 2 or more shots per update step")
        self.n = param_len
        self.stddev = stddev
        self.B = torch.eye(param_len)

    def step_and_cost(self, objective_fn, params, *args, **kwargs):
        loss = objective_fn(params, *args, **kwargs)
        new_params = self.step(objective_fn, params, *args, **kwargs)
        return new_params, loss

    def step(self, objective_fn, params, *args, **kwargs):
        U, _ = torch.linalg.qr(self.grad_subspace)
        a = np.sqrt(self.variance * self.alpha / self.n)
        b = np.sqrt(self.variance * (1. - self.alpha) / self.k)
        grad_est = torch.zeros(self.n)
        for i in range(self.P):
            if self.gs_write_ind < self.k:
                eps = np.sqrt(self.variance / self.n) * torch.normal(mean=torch.zeros(self.n), std=1.0)  # To properly initalize U
            else:
                eps = a * torch.normal(mean=torch.zeros(self.n), std=1.0) + b * torch.mv(U, torch.normal(mean=torch.zeros(min(self.n, self.k)), std=1.0))
            fp = objective_fn(params + torch.reshape(eps, params.shape), *args, **kwargs)
            fn = objective_fn(params - torch.reshape(eps, params.shape), *args, **kwargs)
            grad_est += eps * (fp - fn)
        grad_est *= self.beta / (2 * self.variance * self.P)
        self.grad_subspace[:, self.gs_write_ind % self.k] = grad_est
        self.gs_write_ind += 1
        return params - self.lr * torch.reshape(grad_est, params.shape)

class GES():
    """Guided Evolutionary Strategy"""

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
        self.grad_subspace = torch.zeros(param_len, k)
        self.gs_write_ind = 0  # So we can just keep track of the oldest gradient row to replace
        if self.alpha < 0 or self.alpha > 1:
            raise Exception("GES: Need to give valid alpha hyperparameter range.")

    def step_and_cost(self, objective_fn, params, *args, **kwargs):
        loss = objective_fn(params, *args, **kwargs)
        new_params = self.step(objective_fn, params, *args, **kwargs)
        return new_params, loss

    def step(self, objective_fn, params, *args, **kwargs):
        U, _ = torch.linalg.qr(self.grad_subspace)
        a = np.sqrt(self.variance * self.alpha / self.n)
        b = np.sqrt(self.variance * (1. - self.alpha) / self.k)
        grad_est = torch.zeros(self.n)
        for i in range(self.P):
            if self.gs_write_ind < self.k:
                eps = np.sqrt(self.variance / self.n) * torch.normal(mean=torch.zeros(self.n), std=1.0)  # To properly initalize U
            else:
                eps = a * torch.normal(mean=torch.zeros(self.n), std=1.0) + b * torch.mv(U, torch.normal(mean=torch.zeros(min(self.n, self.k)), std=1.0))
            fp = objective_fn(params + torch.reshape(eps, params.shape), *args, **kwargs)
            fn = objective_fn(params - torch.reshape(eps, params.shape), *args, **kwargs)
            grad_est += eps * (fp - fn)
        grad_est *= self.beta / (2 * self.variance * self.P)
        self.grad_subspace[:, self.gs_write_ind % self.k] = grad_est
        self.gs_write_ind += 1
        return params - self.lr * torch.reshape(grad_est, params.shape)

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
