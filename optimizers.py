# File for optimizers.
# All optimizers should have step(objective_fn, *args, **kwargs) + step_and_cost(objective_fn, *args, **kwargs) functions similar to that of pennylane QML interface
# Example: https://docs.pennylane.ai/en/stable/code/api/pennylane.SPSAOptimizer.html
# TODO: Make template class instead of these docs

import torch
import numpy as np
import pennylane as qml

class QNSPSA():
    """Quantum natural SPSA
        Has option to switch between Hessian (so technically 2-SPSA) and true QN-SPSA"""

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

    def get_operations(qnode, params):
        qnode.construct([params], {})
        return qnode.tape.operations

    def get_overlap_tape(qnode, params1, params2):
        op_forward = get_operations(qnode, params1)
        op_inv = get_operations(qnode, params2)

        with qml.tape.QuantumTape() as tape:
            for op in op_forward:
                qml.apply(op)
            for op in reversed(op_inv):
                qml.adjoint(copy(op))
            qml.probs(wires=qnode.tape.wires.labels)
        return tape

    def get_state_overlap(tape):
        return qml.execute([tape], dev, None)[0][0]

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

    def __init__(self, param_len, stddev, num_shots, nu_sigma, nu_b, nu_delta):
        """
        Arguments:
            arg (type): description
            TODO
        """
        if num_shots < 2:
            raise Exception("xNES: Need 2 or more shots per update step")
        self.n = param_len
        self.stddev = stddev
        self.B = torch.eye(param_len) / stddev
        self.num_shots = num_shots
        self.nu_sigma = nu_sigma
        self.nu_b = nu_b
        self.nu_delta = nu_delta

    def _utilities(self, fitness):
        ordering = torch.argsort(fitness) + 1
        utilities = torch.log((self.num_shots / 2) + 1) - torch.log(ordering)
        utilities[utilities < 0] = 0
        utilities = utilities / torch.sum(utilities)
        utilities = utilities - (1 / self.num_shots)
        return utilities

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
            fitnesses[i] = objective_fn(params + torch.reshape(zk, params.shape), *args, **kwargs)
            sk_list.append(sk)
        utilities = self._utilities(fitnesses)
        d_delta = torch.zeros(self.n)
        d_M = torch.zeros(self.n, self.n)
        for i in range(self.num_shots):
            d_delta += utilities[i] * sk_list[i]
            d_M += utilities[i] * (torch.outer(sk_list[i], sk_list[i]) - torch.eye(self.n))
        d_stddev = torch.trace(d_M) / self.n
        d_B = d_M - (d_stddev * torch.eye(self.n))
        new_params = params + torch.reshape(self.nu_delta * self.stddev * torch.mv(self.B, d_delta), params.shape)
        self.stddev = self.stddev * torch.exp(self.nu_sigma / 2 * d_stddev)
        self.B = torch.mm(self.B, torch.matrix_exp(self.nu_B / 2 * d_B))
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
        loss = objective_fn(*args, **kwargs)
        loss.backward()
        self.opt.step()
        return self.params, loss

    def step(self, objective_fn, params, *args, **kwargs):
        p, l = self.step_and_cost(objective_fn, *args, **kwargs)
        return p
