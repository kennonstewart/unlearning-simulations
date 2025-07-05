import numpy as np
from numpy.linalg import solve

class MemoryPair:
    """
    Incremental / decremental ERM with (ε,δ)-unlearning guarantees
    inspired by Sekhari et al. 2021.
    """
    def __init__(self, d, loss, grad, hess, lam):
        self.d       = d
        self.loss    = loss
        self.grad    = grad
        self.hess    = hess
        self.lam     = lam
        self.n   = 0
        self.w   = np.zeros(d)
        self.H   = None

    def fit(self, data, *, lr=0.01, steps=500, tol=1e-8):
        w = self.w.copy()
        for _ in range(steps):
            idx = np.random.randint(len(data))
            g   = self.grad(w, data[idx])
            w  -= lr * g
            if np.linalg.norm(g) < tol: break
        H = np.mean([self.hess(w, z) for z in data], axis=0)
        self.w, self.H, self.n = w, H, len(data)

    def delete(self, U, *, add_noise=False, eps=1.0, delta=1e-6, M=1.0, L=1.0):
        m = len(U)
        if m == 0: return
        sum_grad_U  = np.sum([self.grad(self.w, z)  for z in U], axis=0)
        sum_hess_U  = np.sum([self.hess(self.w, z)  for z in U], axis=0)
        H_new = (self.n * self.H - sum_hess_U) / (self.n - m)
        delta_w = solve(H_new,  sum_grad_U) / (self.n - m)
        w_new   = self.w + delta_w
        if add_noise:
            gamma   = 2 * (M * L**2) * m**2 / (self.lam**3 * self.n**2)
            sigma   = gamma / eps * np.sqrt(2 * np.log(1.25 / delta))
            w_new  += np.random.normal(scale=sigma, size=self.d)
        self.w, self.H, self.n = w_new, H_new, self.n - m

    def insert(self, V):
        k = len(V)
        if k == 0: return
        sum_grad_V = np.sum([self.grad(self.w, z)  for z in V], axis=0)
        sum_hess_V = np.sum([self.hess(self.w, z)  for z in V], axis=0)
        H_new = (self.n * self.H + sum_hess_V) / (self.n + k)
        delta_w = -solve(H_new, sum_grad_V) / (self.n + k)
        self.w, self.H, self.n = self.w + delta_w, H_new, self.n + k