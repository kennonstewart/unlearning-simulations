# sim_stream_newton.py
import numpy as np
import scipy.stats as st
import os
import itertools

from hessian_based.memory_pair import StreamNewtonMemoryPair

from event_logging import init_logging
import logging

log_dir = init_logging()
logger = logging.getLogger(__name__)
logger.info("simulation_start", extra={"log_dir": str(log_dir)})

# ---------------------------------------------------------------------
# synthetic‐data helper
def generate_synthetic_data(n_samples=1000, n_features=10, noise=0.5):
    X = np.random.rand(n_samples, n_features)
    true_w = np.random.randn(n_features)
    y = X @ true_w + np.random.normal(0, noise, size=n_samples)
    return X, y, true_w

# ---------------------------------------------------------------------
def run_single_simulation(seed: int, *, N_TOTAL: int, N_FEATURES: int, N_DELETE: int, ALPHA: float):
    """
    ▸ 1) train StreamNewtonMemoryPair on N_TOTAL points
    ▸ 2) delete N_DELETE of them
    ▸ 3) compare θ to closed-form ridge retrain on the remaining points
    """
    np.random.seed(seed)
    logger.info(
        "simulation_run_start",
        extra={
            "seed": seed,
            "N_TOTAL": N_TOTAL,
            "N_FEATURES": N_FEATURES,
            "N_DELETE": N_DELETE,
            "ALPHA": ALPHA,
        },
    )

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
        eps_total      = 1e6,
        delta_total    = 1e-12,
    )


    # ------------ initial training ------------
    for idx in range(N_TOTAL):
        model.insert(X[idx], y[idx])
    logger.info("initial_training_complete", extra={"samples": N_TOTAL})

    # print model parameters
    w_initial = model.theta.copy()

    # choose points to delete
    delete_ids = np.random.choice(N_TOTAL, size=N_DELETE, replace=False)

    for idx in delete_ids:
        model.delete(X[idx], y[idx])     # one point per call

    logger.info("deletion_phase_complete", extra={"deleted": N_DELETE})

    w_after_delete = model.theta.copy()

    # ------------ closed-form retrain baseline ------------
    keep_mask          = np.ones(N_TOTAL, dtype=bool)
    keep_mask[delete_ids] = False
    X_keep, y_keep     = X[keep_mask], y[keep_mask]

    # ridge closed form: (XᵀX + λI)⁻¹ Xᵀy
    H   = X_keep.T @ X_keep + ALPHA * np.eye(N_FEATURES)
    # closed-form ridge solution – DON'T overwrite X!
    w_star = np.linalg.solve(H, X_keep.T @ y_keep)

    # ------------ metric ------------
    error = np.linalg.norm(w_after_delete - w_star)
    norm  = np.linalg.norm(w_star)
    rel_error = (error / norm) * 100 if norm != 0 else 0.0
    logger.info(
        "simulation_run_complete",
        extra={
            "relative_error": rel_error,
            "N_TOTAL": N_TOTAL,
            "N_FEATURES": N_FEATURES,
            "N_DELETE": N_DELETE,
            "ALPHA": ALPHA,
        },
    )
    return rel_error

# ---------------------------------------------------------------------

if __name__ == "__main__":

    N_SIMULATIONS = 500

    # ------------------------------------------------------------------
    # Hyper‑parameter grid definition
    param_grid = {
        "ALPHA":    [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        "N_DELETE": [100, 250, 500, 1000],
    }

    # Fixed parameters
    N_TOTAL    = 4500
    N_FEATURES = 5

    for ALPHA, N_DELETE in itertools.product(param_grid["ALPHA"], param_grid["N_DELETE"]):
        config_name = f"alpha_{ALPHA}_del_{N_DELETE}"
        logger.info("grid_search_config_start", extra={"config": config_name})
        errors = [
            run_single_simulation(
                seed=i,
                N_TOTAL=N_TOTAL,
                N_FEATURES=N_FEATURES,
                N_DELETE=N_DELETE,
                ALPHA=ALPHA,
            )
            for i in range(N_SIMULATIONS)
        ]

        mean_error = np.mean(errors)
        ci_low, ci_high = st.t.interval(
            confidence=0.95,
            df=len(errors) - 1,
            loc=mean_error,
            scale=st.sem(errors),
        )

        print("\n--- ✅  Simulation Analysis ---")
        print(f"Config: {config_name}")
        print(f"Ran {len(errors)} successful simulations.")
        print(f"Average relative error: {mean_error:.2f}%")
        print(f"95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]")

        logger.info(
            "grid_search_config_complete",
            extra={
                "config": config_name,
                "mean_error": mean_error,
                "ci_low": ci_low,
                "ci_high": ci_high,
            },
        )

        # ---------------- save artefacts ----------------
        results_dir = os.path.join(os.path.dirname(__file__), "results", config_name)
        os.makedirs(results_dir, exist_ok=True)

        np.save(os.path.join(results_dir, "errors.npy"), np.array(errors))

        with open(os.path.join(results_dir, "summary.txt"), "w") as f:
            f.write(f"--- StreamNewtonMemoryPair Simulation ({config_name}) ---\n")
            f.write(f"Simulations: {len(errors)}\n")
            f.write(f"Mean relative error: {mean_error:.2f}%\n")
            f.write(f"95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]\n")
