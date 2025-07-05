import numpy as np
import scipy.stats as st
from memory_pair import MemoryPair
import os

def generate_synthetic_data(n_samples=1000, n_features=10, noise=0.5):
    """Generates synthetic data for a linear regression problem."""
    X = np.random.rand(
        n_samples, 
        n_features
    )
    true_w = np.random.randn(n_features)
    y = X @ true_w + np.random.normal(
        0, 
        noise, 
        n_samples
    )
    return list(zip(X, y))

def run_single_simulation(seed: int):
    """Runs one full simulation and returns the error metrics."""
    np.random.seed(seed)
    N_INITIAL_TRAIN, N_FEATURES, N_DELETE, ALPHA = 4500, 5, 500, 0.1
    data = generate_synthetic_data(
        n_samples = N_INITIAL_TRAIN, 
        n_features = N_FEATURES
    )
    initial_data, data_to_delete = data[:-N_DELETE], data[-N_DELETE:]
    
    loss = lambda w, z: 0.5 * (z[0] @ w - z[1])**2 + 0.5 * ALPHA * np.dot(w, w)
    grad = lambda w, z: (z[0] @ w - z[1]) * z[0] + ALPHA * w
    hess = lambda w, z: np.outer(z[0], z[0]) + ALPHA * np.identity(N_FEATURES)
    
    model = MemoryPair(
        d = N_FEATURES, 
        loss = loss, 
        grad = grad, 
        hess = hess, 
        lam = ALPHA
    )
    model.fit(initial_data)
    model.delete(data_to_delete)
    w_deleted_approx = model.w.copy()

    data_after_delete = initial_data[:-N_DELETE]
    model_retrained = MemoryPair(
        d = N_FEATURES, 
        loss = loss, 
        grad = grad, 
        hess = hess, 
        lam = ALPHA
    )
    model_retrained.fit(data_after_delete)
    w_deleted_retrained = model_retrained.w.copy()
    
    error = np.linalg.norm(w_deleted_approx - w_deleted_retrained)
    norm = np.linalg.norm(w_deleted_retrained)
    return (error / norm) * 100 if norm != 0 else 0

if __name__ == "__main__":
    N_SIMULATIONS = 500
    print(f"🚀 Starting {N_SIMULATIONS} simulations...")

    errors = [run_single_simulation(seed=i) for i in range(N_SIMULATIONS)]

    mean_error = np.mean(errors)
    ci = st.t.interval(
        0.95,
        len(errors)-1,
        loc=mean_error,
        scale=st.sem(errors)
    )

    print("\n--- ✅ Simulation Analysis ---")
    print(f"Ran {len(errors)} successful simulations.")
    print("\n📊 [RESULTS] Incremental Deletion vs. Retraining")
    print(f"The relative error is, on average, {mean_error:.2f}%")
    print(f"95% Confidence Interval: [{ci[0]:.2f}%, {ci[1]:.2f}%]")

    # --- Save results to /results/ directory ---
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save errors as .npy
    np.save(os.path.join(results_dir, 'errors.npy'), np.array(errors))

    # Save summary statistics as .txt
    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("--- Simulation Analysis ---\n")
        f.write(f"Ran {len(errors)} successful simulations.\n")
        f.write("[RESULTS] Incremental Deletion vs. Retraining\n")
        f.write(f"The relative error is, on average, {mean_error:.2f}%\n")
        f.write(f"95% Confidence Interval: [{ci[0]:.2f}%, {ci[1]:.2f}%]\n")