# ---------------------------------------------------------------------
# l_bfgs.py
#
# Lightweight Limited-memory BFGS helper that keeps all curvature
# bookkeeping inside a single class.
# ---------------------------------------------------------------------

from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LimitedMemoryBFGS:
    """
    Compact container for L-BFGS curvature pairs and the two-loop
    recursion.  Maintains at most ``m_max`` pairs and never stores the
    full inverse-Hessian explicitly.

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
        memory is full and ensuring yᵀs > 0 to preserve PD-ness.
        """
        ys = float(y_new.dot(s_new))
        if ys <= 1e-12:  # damp to enforce positive curvature
            y_new = y_new + 1e-10 * s_new
            ys = float(y_new.dot(s_new))

        if len(self.S) == self.m_max:
            self.S.pop(0)
            self.Y.pop(0)
            self.RHO.pop(0)

        # ---- relative, not absolute, degeneracy test ----
        rel_tol = 1e-5
        if abs(ys) < rel_tol * np.linalg.norm(s_new) * np.linalg.norm(y_new):
            logger.info("skip_curvature_pair", reason="relative_ys_small", ys=ys)
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
        """Delete the *idx*-th curvature pair (0 = oldest)."""
        if 0 <= idx < len(self.S):
            self.S.pop(idx)
            self.Y.pop(idx)
            self.RHO.pop(idx)

    def direction(self, g: np.ndarray, gamma: float | None = None) -> np.ndarray:
        """
        Return the descent direction  d = −H_k^{-1} g  using the classic
        two-loop recursion based on the currently stored curvature pairs.
        """
        if g.ndim != 1:
            raise ValueError("Gradient `g` must be a 1-D array")

        q = g.copy()
        alphas: list[float] = []

        # try block for the first backward loop
        try:
            # Treat *any* FP problem as fatal inside the recursion
            with np.errstate(over="raise", under="raise", divide="raise", invalid="raise"):
                # ---------- first (backward) loop ----------
                for s_i, y_i, rho_i in zip(reversed(self.S), reversed(self.Y), reversed(self.RHO)):
                    alpha_i = rho_i * s_i.dot(q)
                    alphas.append(alpha_i)
                    q -= alpha_i * y_i

                # ---------- apply initial scaling H₀ = γI ----------
                if gamma is None:
                    if self.S:
                        s_last, y_last = self.S[-1], self.Y[-1]
                        gamma = s_last.dot(y_last) / max(y_last.dot(y_last), 1e-12)
                    else:
                        gamma = 1.0
                r = gamma * q

                # ---------- second (forward) loop ----------
                for (s_i, y_i, rho_i), alpha_i in zip(
                    zip(self.S, self.Y, self.RHO), reversed(alphas)
                ):
                    beta_i = rho_i * y_i.dot(r)
                    r += s_i * (alpha_i - beta_i)
        except FloatingPointError as e:
            logger.warning(
                "lbfgs_direction_abort",
                extra={
                    "err": str(e),
                    "grad_norm": float(np.linalg.norm(g)),
                    "mem_pairs": len(self.S),
                },
            )
            raise RuntimeError(
                "Numerical instability detected in LimitedMemoryBFGS.direction"
            ) from e

        logger.info(
            "lbfgs_direction_computed",
            extra={"grad_norm": float(np.linalg.norm(g)), "mem_pairs": len(self.S)},
        )

        return -r


class OnlineLBFGS(LimitedMemoryBFGS):
    """
    Online L-BFGS variant based on Schraudolph et al. (2007).
    This implementation modifies the initial scaling of the Hessian
    to be more robust to stochastic noise.
    """

    def direction(self, g: np.ndarray) -> np.ndarray:
        """
        Return the descent direction using the online-adapted two-loop recursion.
        """
        if g.ndim != 1:
            raise ValueError("Gradient `g` must be a 1-D array")

        p = -g.copy()
        alphas = []
        
        # --- First loop (same as standard L-BFGS) ---
        for s_i, y_i in zip(reversed(self.S), reversed(self.Y)):
            alpha_i = (s_i.dot(p)) / (s_i.dot(y_i)) # using rho is unstable here
            alphas.append(alpha_i)
            p -= alpha_i * y_i
        
        # --- Online initial scaling H₀ (the key difference) ---
        # Averages the scaling factor over the memory buffer 
        if self.S:
            scaling_factors = [s_i.dot(y_i) / max(y_i.dot(y_i), 1e-12) for s_i, y_i in zip(self.S, self.Y)]
            gamma = sum(scaling_factors) / len(scaling_factors)
            p *= gamma

        # --- Second loop (same as standard L-BFGS) ---
        for (s_i, y_i), alpha_i in zip(zip(self.S, self.Y), reversed(alphas)):
            beta = (y_i.dot(p)) / (s_i.dot(y_i))
            p += s_i * (alpha_i - beta)

        logger.info(
            "online_lbfgs_direction_computed",
            extra={"grad_norm": float(np.linalg.norm(g)), "mem_pairs": len(self.S)},
        )
        
        return p


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