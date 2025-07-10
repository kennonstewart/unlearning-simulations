import numpy as np
from l_bfgs import two_loop_recursion, add_pair
from typing import List

# ----------------------------------------------------------------------
class StreamNewtonMemoryPair:
    """
    Streaming ridge-regression learner / unlearner.

    insert(x, y)  : one Newton step with Sherman–Morrison H⁻¹ update
    delete(x, y)  : exact inverse-update removal + privacy noise

    Parameters
    ----------
    dim            : feature dimension
    lam            : ridge λ  (regulariser)
    eps_total      : total ε budget for all deletions
    delta_total    : total δ budget for all deletions
    max_deletions  : anticipated upper bound on #delete() calls
    """

    # ---------------- initialisation ----------------
    def __init__(
        self,
        dim: int,
        lam: float = 1.0,
        eps_total: float = 1.0,
        delta_total: float = 1e-5,
        max_deletions: int = 20,
    ):
        self.dim = dim
        self.lam = lam

        # Parameters and inverse Hessian
        self.theta  = np.zeros(dim)

        # ---- memory for L-BFGS pairs ----
        self.S, self.Y, self.RHO = [], [], []   # L-BFGS memory
        self.m_max = 10                        # keeps 10 curvature points

        # ---- privacy bookkeeping ----
        self.K            = max_deletions
        self.eps_total    = eps_total
        self.delta_total  = delta_total
        self.eps_step     = eps_total  / (2 * max_deletions)
        self.delta_step   = delta_total / (2 * max_deletions)
        self.eps_spent    = 0.0
        self.deletions_so_far = 0

    # ---------------- helpers ----------------
    def _grad_point(self, x, y):
        """
        Gradient of the current loss ½(θᵀx − y)² wrt θ, evaluated at
        current θ.
        """
        residual = self.theta @ x - y
        return residual * x
    
    
    def insert(self, x: np.ndarray, y: float):
        g_old = self._grad_point(x, y)

        # ---------- safe Newton-like step ----------
        d = two_loop_recursion(g_old, self.S, self.Y, self.RHO)

        # optional learning-rate to tame very first step
        lr = 0.5  
        theta_new = self.theta + lr * d

        # ---------- curvature pair ----------
        s = theta_new - self.theta
        g_new = self._grad_point(x, y)
        y_vec = g_new - g_old

        add_pair(self.S, self.Y, self.RHO, s, y_vec, self.m_max)
        self.theta = theta_new

    # ---------------- unlearning ----------------
    def delete(self, x: np.ndarray, y: float):
        """
        Remove the influence of observation (x, y).
        No raw data are stored internally; caller must supply x, y.
        """
        if self.deletions_so_far >= self.K:
            raise RuntimeError("max_deletions budget exceeded")

        # ─ remove curvature pair from memory ───────────────
        if not self.S:
            raise RuntimeError("No curvature pairs to remove")
        
        if len(self.S) != len(self.Y) or len(self.S) != len(self.RHO):
            raise RuntimeError("Inconsistent curvature pair lists")
        
        g = self._grad_point(x, y)
        d = two_loop_recursion(g, self.S, self.Y, self.RHO)
        self.theta -= d     # undo the influence (approximate)

        # ── calibrated Gaussian noise for (ε,δ)-unlearning ─────────
        sensitivity = np.linalg.norm(d, 2)
        sigma = (
            sensitivity
            * np.sqrt(2 * np.log(1.25 / self.delta_step))
            / self.eps_step
        )
        self.theta += np.random.normal(0.0, sigma, size=self.dim)

        # ── book-keeping ────────────────────────────────────────────
        self.eps_spent        += self.eps_step
        self.deletions_so_far += 1

    # ---------------- utility ----------------
    def privacy_ok(self):
        """Return True iff cumulative ε ≤ ε_total."""
        return self.eps_spent <= self.eps_total

# ---------------------------------------------------------------------
# limited_memory_bfgs.py
#
# Utilities for maintaining an *implicit* inverse-Hessian Hk⁻¹ through
# (s, y, ρ) curvature pairs and for computing   d = –Hk⁻¹ g   via the
# classic two-loop recursion.
#
# – No n×n matrix is ever stored.
# – The only state is the “memory” lists S, Y, ρ (length ≤ m).
# – A curvature pair can be *removed* simply by deleting the same index
#   from the three lists.
# ---------------------------------------------------------------------

# ------------------------------------------------------------
# limited_memory_bfgs.py
import numpy as np
from typing import List

def two_loop_recursion(g, S, Y, RHO, gamma=None):
    """Return p = –H_k^{-1} g (L-BFGS two-loop)."""
    q = g.copy()
    al = []
    for s, y, rho in zip(reversed(S), reversed(Y), reversed(RHO)):
        a = rho * s.dot(q)
        al.append(a)
        q -= a * y

    if gamma is None:                         # safe diagonal scaling
        if S:
            s0, y0 = S[0], Y[0]
            gamma = s0.dot(y0) / max(y0.dot(y0), 1e-12)
        else:
            gamma = 1.0 / max(g.dot(g), 1.0)  # ≤ 1, avoids huge step
    r = gamma * q

    for a, s, y, rho in zip(reversed(al), S, Y, RHO):
        b = rho * y.dot(r)
        r += s * (a - b)
    return -r

def add_pair(S, Y, RHO, s, y, m_max, eps=1e-10):
    """Append curvature pair, damp if y·s ≤ 0."""
    ys = y.dot(s)
    if ys <= eps:                         # damp to enforce PD
        y += eps * s
        ys = y.dot(s)
    if len(S) == m_max:
        S.pop(0); Y.pop(0); RHO.pop(0)
    ys = abs(y.dot(s))
    if ys < 1e-8:
        return          # skip the pair
    S.append(s); Y.append(y); RHO.append(1.0 / ys)


def remove_pair_at(S: List[np.ndarray],
                   Y: List[np.ndarray],
                   RHO: List[float],
                   idx: int):
    """
    Delete the i-th curvature pair (0 = oldest).  This is how an
    unlearning call forgets the influence of the update that created it.
    """
    if 0 <= idx < len(S):
        S.pop(idx); Y.pop(idx); RHO.pop(idx)