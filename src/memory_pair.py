import numpy as np

# ----------------------------------------
# helper: Sherman–Morrison utilities
def sm_add_inv(H_inv, u):
    """
    Rank-1 *addition* :   H_new = H + u uᵀ
    Returns updated inverse H_inv_new.
    """
    Hu = H_inv @ u
    denom = 1.0 + u.T @ Hu
    return H_inv - np.outer(Hu, Hu) / denom

def sm_remove_inv(H_inv, u):
    """
    Rank-1 *downdate* :   H_new = H − u uᵀ
    (caller must ensure denominator > 0)
    """
    Hu = H_inv @ u
    denom = 1.0 - u.T @ Hu           # must stay positive
    return H_inv + np.outer(Hu, Hu) / denom
# ----------------------------------------


class StreamNewtonMemoryPair:
    """
    Online ridge-regression learner + unlearner with
    (ε,δ)-style Gaussian noise per deletion.

    Loss      : ½(θᵀx − y)²   (squared error)
    Regulariser: λ‖θ‖₂²
    """

    def __init__(self, dim, lam=1.0,
                 eps_total=1.0,   delta_total=1e-5,
                 max_deletions=20):
        self.dim  = dim
        self.lam  = lam

        # Model parameters & (regularised) Hessian inverse
        self.theta = np.zeros(dim)
        self.H_inv = np.eye(dim) / lam          # (XᵀX + λI)⁻¹

        # Storage: keep raw (x,y) for exact gradient recomputation
        self.data_store = {}     # id -> (x, y)
        self.deleted     = set()

        # --- privacy bookkeeping ---
        self.K          = max_deletions          # anticipate ≤K deletions
        self.eps_total  = eps_total
        self.delta_total = delta_total
        self.eps_step   = eps_total  / (2*max_deletions)   # split budget
        self.delta_step = delta_total / (2*max_deletions)
        self.eps_spent  = 0.0

    # 1st-order gradient (current θ)
    def grad(self, x, y):
        err = self.theta @ x - y
        return err * x

    # ---------------- insert ----------------
    def insert(self, idx, x, y):
        """Process a new data point (x,y)."""
        if idx in self.deleted or idx in self.data_store:
            raise ValueError("duplicate id")

        # Update inverse Hessian   H⁻¹ ← (H + x xᵀ)⁻¹  via Sherman–Morrison
        self.H_inv = sm_add_inv(self.H_inv, x)

        # One Newton step using fresh gradient
        g = self.grad(x, y)
        self.theta -= self.H_inv @ g

        # Store raw data for possible future deletion
        self.data_store[idx] = (x, y)

    # ---------------- delete ----------------
    def delete(self, idx):
        """Remove the influence of data point idx (if present)."""
        if idx in self.deleted or idx not in self.data_store:
            return

        x, y = self.data_store.pop(idx)

        # ---------------- Hessian downdate ----------------
        self.H_inv = sm_remove_inv(self.H_inv, x)

        # Re-compute gradient of this point *at current θ*
        g = self.grad(x, y)

        delta_theta = self.H_inv @ g       # influence to remove
        self.theta -= delta_theta

        # ---------------- calibrated Gaussian noise ----------------
        sensitivity = np.linalg.norm(delta_theta, 2)
        sigma = (sensitivity *
                 np.sqrt(2*np.log(1.25/self.delta_step))
                 / self.eps_step)

        self.theta += np.random.normal(0.0, sigma, size=self.dim)

        # Budget accounting
        self.eps_spent += self.eps_step
        self.deleted.add(idx)

    # ---------------- utility ----------------
    def privacy_ok(self):
        return self.eps_spent <= self.eps_total