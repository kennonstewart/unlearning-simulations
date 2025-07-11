# ---------------------------------------------------------------------
# l_bfgs.py
#
# Lightweight Limited‑memory BFGS helper that keeps all curvature
# bookkeeping inside a single class.
# ---------------------------------------------------------------------

from typing import List
import numpy as np
import logging
logger = logging.getLogger(__name__)


class LimitedMemoryBFGS:
    """
    Compact container for L‑BFGS curvature pairs and the two‑loop
    recursion.  Maintains at most ``m_max`` pairs and never stores the
    full inverse‑Hessian explicitly.

        >>> lbfgs = LimitedMemoryBFGS(m_max=10)
        >>> lbfgs.add_pair(s, y)
        >>> d = lbfgs.direction(g)

    Parameters
    ----------
    m_max : int
        Maximum number of (s, y, ρ) pairs to keep.
    """

    def __init__(self, m_max: int = 10):
        self.S: List[np.ndarray] = []
        self.Y: List[np.ndarray] = []
        self.RHO: List[float] = []
        self.m_max = m_max

    # -----------------------------------------------------------------
    # public helpers
    # -----------------------------------------------------------------
    def __len__(self) -> int:  # ``len(lbfgs)``
        return len(self.S)

    def add_pair(self, s_new: np.ndarray, y_new: np.ndarray):
        """
        Append a new curvature pair (s, y), discarding the oldest if the
        memory is full and ensuring yᵀs > 0 to preserve PD-ness.
        """
        ys = float(y_new.dot(s_new))
        if ys <= 1e-12:           # damp to enforce positive curvature
            y_new = y_new + 1e-10 * s_new
            ys = float(y_new.dot(s_new))

        if len(self.S) == self.m_max:
            self.S.pop(0); self.Y.pop(0); self.RHO.pop(0)

        if abs(ys) < 1e-8:        # skip degenerate pair
            return

        logger.info(
            "lbfgs_add_pair",
            extra={
                "s_norm": float(np.linalg.norm(s_new)),
                "y_norm": float(np.linalg.norm(y_new)),
                "rho": 1.0 / ys,
                "memory_len": len(self.S) + 1 if len(self.S) < self.m_max else self.m_max,
            },
        )

        self.S.append(s_new.copy())
        self.Y.append(y_new.copy())
        self.RHO.append(1.0 / ys)

    def remove_pair_at(self, idx: int):
        """Delete the *idx*-th curvature pair (0 = oldest)."""
        if 0 <= idx < len(self.S):
            self.S.pop(idx); self.Y.pop(idx); self.RHO.pop(idx)

    def direction(self, g: np.ndarray, gamma: float ) -> np.ndarray:
        """
        Return the descent direction  d = −H_k^{-1} g  using the classic
        two‑loop recursion based on the currently stored curvature pairs.
        """
        if g.ndim != 1:
            raise ValueError("Gradient `g` must be a 1‑D array")

        q = g.copy()
        alphas: list[float] = []

        # ---------- first (backward) loop ----------
        for s_i, y_i, rho_i in zip(reversed(self.S),
                                   reversed(self.Y),
                                   reversed(self.RHO)):
            alpha_i = rho_i * s_i.dot(q)
            alphas.append(alpha_i)
            q -= alpha_i * y_i

        # ---------- apply initial scaling H₀ = γI ----------
        if gamma is None:
            if self.S:
                s0, y0 = self.S[0], self.Y[0]
                gamma = float(s0.dot(y0) / max(y0.dot(y0), 1e-12))
            else:
                gamma = 1.0
        r = gamma * q

        # ---------- second (forward) loop ----------
        for alpha_i, s_i, y_i, rho_i in zip(reversed(alphas),
                                            self.S,
                                            self.Y,
                                            self.RHO):
            beta_i = rho_i * y_i.dot(r)
            r += s_i * (alpha_i - beta_i)

        logger.debug(
            "lbfgs_direction_computed",
            extra={"grad_norm": float(np.linalg.norm(g)), "mem_pairs": len(self.S)},
        )

        return -r

# ---------------------------------------------------------------------
# Convenience wrappers (deprecated).  New code should rely on the class
# but the stubs remain to avoid import errors.
# ---------------------------------------------------------------------
def two_loop_recursion(*_a, **_k):
    raise RuntimeError(
        "two_loop_recursion() is deprecated. Instantiate "
        "`LimitedMemoryBFGS` and call `.direction()` instead."
    )

def add_pair(*_a, **_k):
    raise RuntimeError(
        "add_pair() is deprecated. Instantiate `LimitedMemoryBFGS` and "
        "call `.add_pair()` instead."
    )

def remove_pair_at(*_a, **_k):
    raise RuntimeError(
        "remove_pair_at() is deprecated. Instantiate `LimitedMemoryBFGS` "
        "and call `.remove_pair_at()` instead."
    )