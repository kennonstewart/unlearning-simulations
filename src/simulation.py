# sim_stream_newton.py
import numpy as np
import scipy.stats as st
import os

from memory_pair import StreamNewtonMemoryPair   # ← your class file

# ---------------------------------------------------------------------
# synthetic‐data helper
def generate_synthetic_data(n_samples=1000, n_features=10, noise=0.5):
    X = np.random.rand(n_samples, n_features)
    true_w = np.random.randn(n_features)
    y = X @ true_w + np.random.normal(0, noise, size=n_samples)
    return X, y, true_w

# ---------------------------------------------------------------------
def run_single_simulation(seed: int):
    """
    ▸ 1) train StreamNewtonMemoryPair on N_TOTAL points
    ▸ 2) delete N_DELETE of them
    ▸ 3) compare θ to closed-form ridge retrain on the remaining points
    """
    np.random.seed(seed)

    # ---------------- hyper-params ----------------
    N_TOTAL, N_FEATURES  = 4500, 5
    N_DELETE             = 500
    ALPHA                = 0.1      # ridge λ

    # ------------ generate data ------------
    X, y, _ = generate_synthetic_data(
        n_samples  = N_TOTAL,
        n_features = N_FEATURES,
        noise      = 0.5,
    )

    # ------------ create model ------------
    model = StreamNewtonMemoryPair(
        dim            = N_FEATURES,
        lam            = ALPHA,
        max_deletions  = N_DELETE,
        eps_total      = 1e6,       # huge ε ⇒ virtually no DP noise
        delta_total    = 1e-12,
    )

    # ------------ initial training ------------
    for idx in range(N_TOTAL):
        model.insert(X[idx], y[idx])

    # ------------ choose points to delete ------------
    delete_ids = np.random.choice(N_TOTAL, size=N_DELETE, replace=False)
    for idx in delete_ids:
        model.delete(X[idx], y[idx])

    w_after_delete = model.theta.copy()

    # ------------ closed-form retrain baseline ------------
    keep_mask          = np.ones(N_TOTAL, dtype=bool)
    keep_mask[delete_ids] = False
    X_keep, y_keep     = X[keep_mask], y[keep_mask]

    # ridge closed form: (XᵀX + λI)⁻¹ Xᵀy
    H   = X_keep.T @ X_keep + ALPHA * np.eye(N_FEATURES)
    w_star = np.linalg.solve(H, X_keep.T @ y_keep)

    # ------------ metric ------------
    error = np.linalg.norm(w_after_delete - w_star)
    norm  = np.linalg.norm(w_star)
    return (error / norm) * 100 if norm != 0 else 0.0

# ---------------------------------------------------------------------
if __name__ == "__main__":
    N_SIMULATIONS = 500
    print(f"🚀  Running {N_SIMULATIONS} simulations with StreamNewtonMemoryPair …")

    errors = [run_single_simulation(seed=i) for i in range(N_SIMULATIONS)]

    mean_error = np.mean(errors)
    ci_low, ci_high = st.t.interval(
        confidence = 0.95,
        df         = len(errors) - 1,
        loc        = mean_error,
        scale      = st.sem(errors),
    )

    print("\n--- ✅  Simulation Analysis ---")
    print(f"Ran {len(errors)} successful simulations.")
    print("\n📊  Incremental Deletion vs. Closed-form Retraining")
    print(f"Average relative error: {mean_error:.2f}%")
    print(f"95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]")

    # ---------------- save artefacts ----------------
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "errors.npy"), np.array(errors))

    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write("--- StreamNewtonMemoryPair Simulation ---\n")
        f.write(f"Simulations: {len(errors)}\n")
        f.write(f"Mean relative error: {mean_error:.2f}%\n")
        f.write(f"95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]\n")