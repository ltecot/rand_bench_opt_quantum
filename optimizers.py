# File for optimizers.
# All optimizers should have step(objective_fn, *args, **kwargs) + step_and_cost(objective_fn, *args, **kwargs) functions similar to that of pennylane QML interface
# Example: https://docs.pennylane.ai/en/stable/code/api/pennylane.SPSAOptimizer.html
# TODO: Make template class instead of these docs

import torch
import numpy as np
import pennylane as qml
import math
from copy import copy

import util as qo_util

class SPSA_2():
    """2nd order Hessian SPSA
    TODO: Add option to switch to Fubini-Study (QNSPSA). For now just use pennylane version."""

    def __init__(self, param_len, num_shots=1, stepsize=0.001, regularization=0.001, finite_diff_step=0.01, blocking=True, history_length=5): 
        """
        Arguments:
            arg (type): description
            TODO
        """
        self.param_len = param_len
        self.num_shots = num_shots
        self.stepsize = stepsize
        self.regularization = regularization
        self.finite_diff_step = finite_diff_step
        self.blocking = blocking
        self.history_length = history_length
        self.metric_avg = torch.eye(param_len)
        self.k = 1
        if self.blocking:
            self.loss_history = torch.zeros(history_length)

    def step(self, objective_fn, params, *args, **kwargs):
        # ck = self.c / (self.k ** self.gamma)
        grad_est = torch.zeros(self.param_len)
        metric_est = torch.zeros(self.param_len, self.param_len)
        for i in range(self.num_shots):
            # 1st Order
            eps = 2 * torch.bernoulli(0.5 * torch.ones(self.param_len)) - 1  # Equal prob -1 or 1 per element
            fp = objective_fn(params + torch.reshape(self.finite_diff_step * eps, params.shape), *args, **kwargs)
            fn = objective_fn(params - torch.reshape(self.finite_diff_step * eps, params.shape), *args, **kwargs)
            grad_est += (fp - fn) * eps
            # 2nd Order
            eps_1 = eps  # Re-use prior sample
            eps_2 = 2 * torch.bernoulli(0.5 * torch.ones(self.param_len)) - 1  # Equal prob -1 or 1 per element
            # fp = objective_fn(params + torch.reshape(self.finite_diff_step * eps_1, params.shape), *args, **kwargs)
            # fn = objective_fn(params - torch.reshape(self.finite_diff_step * eps_1, params.shape), *args, **kwargs)
            fp2 = objective_fn(params + torch.reshape(self.finite_diff_step * (eps_1 + eps_2), params.shape), *args, **kwargs)
            fn2 = objective_fn(params - torch.reshape(self.finite_diff_step * (eps_1 - eps_2), params.shape), *args, **kwargs)
            metric_est += (fp2 - fp - fn2 + fn) * (torch.outer(eps_1, eps_2) + torch.outer(eps_2, eps_1))
        grad_est *= 1 / (2 * self.finite_diff_step * self.num_shots)
        metric_est *= 1 / (4 * (self.finite_diff_step ** 2) * self.num_shots)
        self.metric_avg = 1 / (self.k + 1) * metric_est + self.k / (self.k + 1) * self.metric_avg
        # TODO: Maybe add sqrt(A^t A) below. Matrix should already be positive semi-definite and A^t = A so it seems pointless, but maybe I'm missing something.
        pos_def_metric_avg = self.metric_avg + self.regularization * torch.eye(self.param_len)
        # Solves the equation [x_{t+1} = x_t - lr * metric_matrix^{-1} * grad_x] by putting it into the form [A x_{t+1} = B]
        # This is done to avoid finding the inverse of the metric matrix and potentially running into numerical stability issues.
        new_params_vec = torch.linalg.solve(
            pos_def_metric_avg,
            (-self.stepsize * grad_est + torch.mv(pos_def_metric_avg, params.flatten())),
        )
        new_params = new_params_vec.reshape(params.shape)
        if self.blocking:
            # Assumes current params input is the same as last step's output.
            loss_curr = self.loss_history[(self.k - 2) % self.history_length]
            loss_next = objective_fn(new_params, *args, **kwargs)
            tol = 2 * torch.std(self.loss_history) if self.k > self.history_length else 2 * torch.std(self.loss_history[:self.k-1])
            if loss_curr + tol < loss_next:
                new_params = params  # Cancel update if it doesn't meet threshold
                self.loss_history[(self.k - 1) % self.history_length] = loss_curr
            else:
                self.loss_history[(self.k - 1) % self.history_length] = loss_next  # Add new loss to history if update is succesful
        self.k += 1
        return new_params

    def step_and_cost(self, objective_fn, params, *args, **kwargs):
        loss = objective_fn(params, *args, **kwargs)
        new_params = self.step(objective_fn, params, *args, **kwargs)
        return new_params, loss

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
            fp = objective_fn(params + torch.reshape(ck * eps, params.shape), *args, **kwargs)
            fn = objective_fn(params - torch.reshape(ck * eps, params.shape), *args, **kwargs)
            grad_est += (fp - fn) * eps
        grad_est *= 1 / (2 * ck * self.num_shots)
        ak = self.a / ((self.A + self.k) ** self.alpha)
        return params - ak * torch.reshape(grad_est, params.shape)

    def step_and_cost(self, objective_fn, params, *args, **kwargs):
        loss = objective_fn(params, *args, **kwargs)
        new_params = self.step(objective_fn, params, *args, **kwargs)
        return new_params, loss

class AdamSPSA():
    """AdamSPSA
       Only difference is instead of doing bias correction, we just initalize in a way that removes all bias."""

    def __init__(self, param_len, num_shots=1, alpha=0.602, c=0.2, gamma=0.101, beta=0.999, lmd=0.5, zeta=0.999, maxiter=None, a=None, delta=1e-6): 
        """
        Arguments:
            arg (type): description
            TODO
        """
        if not maxiter and not a:
            raise Exception("SPSA: Need to give valid maxiter or a")
        self.param_len = param_len
        self.num_shots = num_shots
        self.c = c
        self.gamma = gamma  # Sample step size decay
        self.alpha = alpha  # Learning rate decay
        self.beta = beta  # Momentum / first order metric decay
        self.lmd = lmd  # Step decay for Momentum / first order metric decay
        self.zeta = zeta  # Integral / second order metric decay
        self.a = a
        if not a:
            self.a = 0.05 * (maxiter * 0.2 + 1) ** alpha  # Because we removed A this is probably a weird estimate, so probably don't use.
        self.k = 0  # Step count
        self.first_moment = torch.zeros(self.param_len)
        self.second_moment = torch.zeros(self.param_len)
        self.delta = delta

    def step(self, objective_fn, params, *args, **kwargs):
        self.k += 1
        ck = self.c / (self.k ** self.gamma)
        beta_k = self.beta / (self.k ** self.lmd)
        ak = self.a / (self.k ** self.alpha)
        grad_est = torch.zeros(self.param_len)
        for i in range(self.num_shots):
            eps = 2 * torch.bernoulli(0.5 * torch.ones(self.param_len)) - 1  # Equal prob -1 or 1 per element
            fp = objective_fn(params + torch.reshape(ck * eps, params.shape), *args, **kwargs)
            fn = objective_fn(params - torch.reshape(ck * eps, params.shape), *args, **kwargs)
            grad_est += (fp - fn) * eps
        grad_est *= 1 / (2 * ck * self.num_shots)
        if self.k == 1:
            self.first_moment = grad_est
            self.second_moment = (grad_est ** 2)
        else:
            self.first_moment = beta_k * self.first_moment + (1 - beta_k) * grad_est
            self.second_moment = self.zeta * self.second_moment + (1 - self.zeta) * (grad_est ** 2)
        return params - ak * torch.reshape(self.first_moment / (torch.sqrt(self.second_moment) + self.delta), params.shape)

    def step_and_cost(self, objective_fn, params, *args, **kwargs):
        loss = objective_fn(params, *args, **kwargs)
        new_params = self.step(objective_fn, params, *args, **kwargs)
        return new_params, loss

class xNES():
    """Exponential natural evolution strategies"""

    def __init__(self, param_len, stddev_init=1, num_shots=2, nu_mu=1, nu_sigma=None, nu_b=None):
        """
        Arguments:
            arg (type): description
            TODO
        """
        if num_shots and num_shots < 2:
            raise Exception("xNES: Need 2 or more shots per update step")
        self.n = param_len
        self.stddev = stddev_init
        self.B = torch.eye(param_len)
        self.num_shots = num_shots
        self.nu_sigma = nu_sigma
        self.nu_b = nu_b
        self.nu_mu = nu_mu
        if not nu_sigma:
            self.nu_sigma = (9 + 3 * math.log(param_len)) / (5 * (param_len ** 1.5))
        if not nu_b:
            self.nu_b = (9 + 3 * math.log(param_len)) / (5 * (param_len ** 1.5))
        if not num_shots:
            self.num_shots = 4 + math.floor(3 * math.log(param_len))
        self.utilities = qo_util.fitness_utilities(num_shots)

    def step_and_cost(self, objective_fn, params, *args, **kwargs):
        loss = objective_fn(params, *args, **kwargs)
        new_params = self.step(objective_fn, params, *args, **kwargs)
        return new_params, loss

    def step(self, objective_fn, params, *args, **kwargs):
        fitnesses = torch.zeros(self.num_shots)
        sk_list = []
        for i in range(self.num_shots):
            sk = torch.normal(mean=torch.zeros(self.n), std=1.0)
            zk = params + torch.reshape(self.stddev * torch.mv(self.B.t(), sk), params.shape)
            fitnesses[i] = objective_fn(zk, *args, **kwargs)
            sk_list.append(sk)
        util_inds = torch.argsort(fitnesses)
        d_delta = torch.zeros(self.n)
        d_M = torch.zeros(self.n, self.n)
        for i in range(self.num_shots):
            j = util_inds[i]
            d_delta += self.utilities[i] * sk_list[j]
            d_M += self.utilities[i] * (torch.outer(sk_list[j], sk_list[j]) - torch.eye(self.n))
        d_stddev = torch.trace(d_M) / self.n
        d_B = d_M - (d_stddev * torch.eye(self.n))
        new_params = params + torch.reshape(self.nu_mu * self.stddev * torch.mv(self.B, d_delta), params.shape)
        self.stddev = self.stddev * torch.exp(self.nu_sigma / 2 * d_stddev)
        self.B = torch.mm(self.B, torch.matrix_exp(self.nu_b / 2 * d_B))
        return new_params

class sNES():
    """Seperable exponential natural evolution strategies"""

    def __init__(self, param_len, stddev_init=1, nu_mu=1, num_shots=None, nu_sigma=None):
        """
        Arguments:
            arg (type): description
            TODO
        """
        if num_shots and num_shots < 2:
            raise Exception("sNES: Need 2 or more shots per update step")
        self.n = param_len
        self.sigma = torch.ones(param_len) * stddev_init
        self.num_shots = num_shots
        self.nu_sigma = nu_sigma
        self.nu_mu = nu_mu
        if not nu_sigma:
            self.nu_sigma = (9 + 3 * math.log(param_len)) / (5 * (param_len ** 0.5))
        if not num_shots:
            self.num_shots = 4 + math.floor(3 * math.log(param_len))
        self.utilities = qo_util.fitness_utilities(num_shots)

    def step_and_cost(self, objective_fn, params, *args, **kwargs):
        loss = objective_fn(params, *args, **kwargs)
        new_params = self.step(objective_fn, params, *args, **kwargs)
        return new_params, loss

    def step(self, objective_fn, params, *args, **kwargs):
        fitnesses = torch.zeros(self.num_shots)
        sk_list = []
        for i in range(self.num_shots):
            sk = torch.normal(mean=torch.zeros(self.n), std=1.0)
            zk = params + torch.reshape(self.sigma * sk, params.shape)
            fitnesses[i] = objective_fn(zk, *args, **kwargs)
            sk_list.append(sk)
        # utilities = qo_util.fitness_utilities(fitnesses)
        util_inds = torch.argsort(fitnesses)
        d_mu = torch.zeros(self.n)
        d_sigma = torch.zeros(self.n)
        for i in range(self.num_shots):
            j = util_inds[i]
            d_mu += self.utilities[i] * sk_list[j]
            d_sigma += self.utilities[i] * ((sk_list[j] ** 2) - 1)
        new_params = params + torch.reshape(self.nu_mu * self.sigma * d_mu, params.shape)
        self.sigma = self.sigma * torch.exp(self.nu_sigma / 2 * d_sigma)
        return new_params

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
        loss = objective_fn(params, *args, **kwargs)
        loss.backward()
        self.opt.step()
        return self.params, loss

    def step(self, objective_fn, params, *args, **kwargs):
        p, l = self.step_and_cost(objective_fn, *args, **kwargs)
        return p
