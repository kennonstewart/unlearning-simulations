import numpy as np

# ----------------------------------------------------------------------
# Sherman–Morrison rank-1 utilities
def sm_add_inv(H_inv, u):
    """
    H ← H + uuᵀ      ⇒     H⁻¹_new  (rank-1 *update*)
    """
    Hu     = H_inv @ u
    denom  = 1.0 + u.T @ Hu
    return H_inv - np.outer(Hu, Hu) / denom


def sm_remove_inv(H_inv, u):
    """
    H ← H − uuᵀ      ⇒     H⁻¹_new  (rank-1 *downdate*)
    Caller must ensure denom > 0 (H stays PD).
    """
    Hu     = H_inv @ u
    denom  = 1.0 - u.T @ Hu           # > 0  for stability
    if denom <= 1e-12:
        raise ValueError("Hessian downdate would destroy PD-ness")
    return H_inv + np.outer(Hu, Hu) / denom


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
        self.H_inv  = np.eye(dim) / lam          # (XᵀX + λI)⁻¹

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

    # ---------------- learning ----------------
    def insert(self, x: np.ndarray, y: float):
        """Process a new observation (x, y)."""
        # ── rank-1 update of (XᵀX + λI)⁻¹ ───────────────────────────
        self.H_inv = sm_add_inv(self.H_inv, x)

        # Newton step using *fresh* gradient
        g = self._grad_point(x, y)
        self.theta -= self.H_inv @ g

    # ---------------- unlearning ----------------
    def delete(self, x: np.ndarray, y: float):
        """
        Remove the influence of observation (x, y).
        No raw data are stored internally; caller must supply x, y.
        """
        if self.deletions_so_far >= self.K:
            raise RuntimeError("max_deletions budget exceeded")

        # ── inverse-Hessian downdate ────────────────────────────────
        self.H_inv = sm_remove_inv(self.H_inv, x)

        # Gradient of *this* point at current θ
        g = self._grad_point(x, y)
        delta_theta = self.H_inv @ g
        self.theta -= delta_theta

        # ── calibrated Gaussian noise for (ε,δ)-unlearning ─────────
        sensitivity = np.linalg.norm(delta_theta, 2)
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