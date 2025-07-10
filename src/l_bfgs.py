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

import numpy as np
from typing import List, Tuple

# ------------------------------------------------------------
def two_loop_recursion(g: np.ndarray,
                       S: List[np.ndarray],
                       Y: List[np.ndarray],
                       RHO: List[float],
                       gamma: float = None) -> np.ndarray:
    """
    Return  d = –Hk⁻¹ g   using the L-BFGS two-loop algorithm.

    Parameters
    ----------
    g      : current gradient ∇f(x_k)
    S,Y    : lists of past   s_i = x_{i+1} – x_i,   y_i = g_{i+1} – g_i
             ordered *oldest … newest*.  len(S) = len(Y) = m (≤ M_max)
    RHO    : list of ρ_i = 1/(y_iᵀ s_i) (same length/order as S)
    gamma  : optional initial scaling for H₀ = γ I.
             If None and at least one pair exists, use
             γ = (sᵀ y)/(yᵀ y) of **oldest** pair; else γ = 1.

    Returns
    -------
    d      : descent direction –Hk⁻¹ g
    """
    q   = g.copy()
    al  = []

    # ---------- first loop (back-ward) ----------
    for s_i, y_i, rho_i in zip(reversed(S), reversed(Y), reversed(RHO)):
        alpha_i = rho_i * s_i.dot(q)
        al.append(alpha_i)
        q -= alpha_i * y_i

    # ---------- scale with H0 = γ I ----------
    if gamma is None:
        if S:
            s0, y0 = S[0], Y[0]
            gamma = s0.dot(y0) / (y0.dot(y0))
        else:
            gamma = 1.0
    r = gamma * q

    # ---------- second loop (forward) ----------
    for alpha_i, s_i, y_i, rho_i in zip(reversed(al), S, Y, RHO):
        beta_i = rho_i * y_i.dot(r)
        r += s_i * (alpha_i - beta_i)

    return -r  # descent direction
# ------------------------------------------------------------

def add_pair(S: List[np.ndarray],
             Y: List[np.ndarray],
             RHO: List[float],
             s_new: np.ndarray,
             y_new: np.ndarray,
             m_max: int):
    """
    Append a new curvature pair, discarding the oldest if memory > m_max.
    """
    if y_new.dot(s_new) <= 1e-12:              # curvature check
        return
    if len(S) == m_max:
        S.pop(0); Y.pop(0); RHO.pop(0)
    S.append(s_new)
    Y.append(y_new)
    RHO.append(1.0 / y_new.dot(s_new))


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